using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Dehumidifier.Simulation;

namespace Dehumidifier.Optimization;

/// <summary>
/// 纯.NET反向搜索：候选直接调用同一ONNX整机前向模型，不启动Python。
/// 每实例串行；取消在候选和前向迭代间检查，不中断正在执行的原生推理。
/// </summary>
public sealed class EnergyOptimizer : IDisposable
{
    private readonly Simulator? _simulator;
    private readonly Func<SimulationConfig,CancellationToken,Task<SimulationResult>> _evaluate;
    private readonly Func<SimulationConfig,CancellationToken,Task<SimulationResult>> _verify;
    private readonly string _resultsDirectory;
    private readonly Dictionary<string,string> _modelHashes = new();
    private readonly SemaphoreSlim _gate = new(1,1);
    private bool _disposed;
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented=true };
    private static readonly JsonSerializerOptions JsonLineOptions = new() { WriteIndented=false };

    /// <summary>modelDirectory须含本次导出的六份模型/预处理契约文件，结果写入指定根目录。</summary>
    public EnergyOptimizer(string modelDirectory,string resultsDirectory)
    {
        if(string.IsNullOrWhiteSpace(resultsDirectory)) throw new ArgumentException("结果目录不能为空。",nameof(resultsDirectory));
        _resultsDirectory=Path.GetFullPath(resultsDirectory);
        _simulator=new Simulator(modelDirectory,Path.Combine(_resultsDirectory,"best_forward"));
        _evaluate=(c,t)=>_simulator.SimulateForValidationAsync(c,cancellationToken:t);
        _verify=(c,t)=>_simulator.SimulateAsync(c,cancellationToken:t);
        foreach(var name in new[]{"xf.onnx","main.onnx","model_contract.json","preprocessing.json","preprocessing_xf.json","preprocessing_main.json"})
        {
            var file=Path.Combine(modelDirectory,name);
            if(File.Exists(file)) _modelHashes[name]=Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(file))).ToLowerInvariant();
        }
    }

    internal EnergyOptimizer(string resultsDirectory,Func<SimulationConfig,CancellationToken,Task<SimulationResult>> evaluator)
    {
        _resultsDirectory=Path.GetFullPath(resultsDirectory);
        _evaluate=evaluator; _verify=evaluator;
    }

    /// <summary>输入工况不修改。预算内无可行解返回Feasible=false；配置/基准/复算错误抛异常。</summary>
    public async Task<SearchResult> OptimizeAsync(SimulationConfig config,SearchSettings settings,
        CancellationToken cancellationToken=default,IProgress<SearchProgress>? progress=null)
    {
        ObjectDisposedException.ThrowIf(_disposed,this);
        ArgumentNullException.ThrowIfNull(config); ArgumentNullException.ThrowIfNull(settings);
        settings.Validate(); var snapshot=settings.Snapshot();
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            cancellationToken.ThrowIfCancellationRequested();
            // Validate the full configuration through the public strict parser
            // before opening a search directory or swallowing candidate errors.
            var validated=SimulationConfig.Parse(JsonSerializer.Serialize(config));
            var original=CopyConfiguration(validated,validated.ControlSetpoints,config.SourceName,config.SourcePath);
            return await SearchAsync(original,snapshot,cancellationToken,progress).ConfigureAwait(false);
        }
        catch(Exception ex) when(ex is IOException or UnauthorizedAccessException)
        { throw new OptimizationException("RESULT_SAVE_FAILED","优化结果留档失败。",ex); }
        finally { _gate.Release(); }
    }

    private async Task<SearchResult> SearchAsync(SimulationConfig original,SearchSettings settings,CancellationToken token,IProgress<SearchProgress>? progress)
    {
        var dir=Path.Combine(_resultsDirectory,$"{DateTimeOffset.Now:yyyyMMdd_HHmmss_fffffff}_optimization_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var targetConfig=CopyConfiguration(original,original.ControlSetpoints with {SupplyTemperatureTargetC=settings.SupplyTemperatureTargetC});
        var cache=new Dictionary<string,SearchCandidate>(StringComparer.Ordinal);
        var records=new List<SearchCandidate>(); SearchCandidate? best=null;
        double? fixedReference=settings.EnergyReferenceKW;
        string phase="baseline";
        var journal=new StreamWriter(Path.Combine(dir,"evaluations.jsonl"),false,new UTF8Encoding(false));
        bool faulted=false;
        double[] Canonical(double[] values)
        {
            var vector=new double[4];
            for(int i=0;i<4;i++)
            {
                var b=settings.Bounds[i];
                var index=Math.Round((values[i]-b.MinimumC)/settings.StepC,MidpointRounding.ToEven);
                vector[i]=Math.Round(Math.Clamp(b.MinimumC+index*settings.StepC,b.MinimumC,b.MaximumC),10,MidpointRounding.ToEven);
            }
            return vector;
        }
        static string Key(double[] v)=>string.Join("|",v.Select(x=>x.ToString("R",CultureInfo.InvariantCulture)));
        async Task<SearchCandidate> Evaluate(double[] values,bool quantize=true)
        {
            token.ThrowIfCancellationRequested();
            var vector=quantize?Canonical(values):values.ToArray();
            string key=Key(vector);
            if(cache.TryGetValue(key,out var cached)) return cached;
            var controls=ControlTemperatures.From(vector);
            var candidateConfig=ApplyControls(targetConfig,controls);
            CandidateAssessment assessment; string[] warnings=[];
            try
            {
                var simulation=await _evaluate(candidateConfig,token).ConfigureAwait(false);
                if(!fixedReference.HasValue)
                {
                    if(phase!="baseline") throw new OptimizationException("REFERENCE_INVALID","自动归一化基准只能由首次原配置计算确定。");
                    var physical=CandidateScoring.Assess(simulation,settings);
                    var value=physical.ThermalLoadSumKW;
                    if(!value.HasValue || !double.IsFinite(value.Value) || value<=0)
                        throw new OptimizationException("REFERENCE_INVALID","原配置六项热负荷不能作为正基准，请显式提供EnergyReferenceKW。");
                    fixedReference=value.Value;
                }
                assessment=CandidateScoring.Assess(simulation,settings,fixedReference);
                warnings=simulation.Warnings.Distinct().ToArray();
            }
            catch(Exception ex) when(IsCandidateNumericalError(ex))
            {
                if(!fixedReference.HasValue) throw new OptimizationException("REFERENCE_INVALID","首次原配置数值计算失败，无法建立固定基准，请显式提供EnergyReferenceKW。",ex);
                assessment=new CandidateAssessment {Feasible=false,Reasons=["numerical_error"],Score=1e6,EnergyReferenceKW=fixedReference,Error=$"{ex.GetType().Name}: {ex.Message}"};
            }
            bool onGrid=Canonical(vector).Zip(vector).All(x=>Math.Abs(x.First-x.Second)<1e-9);
            bool inBounds=vector.Where((x,i)=>x<settings.Bounds[i].MinimumC || x>settings.Bounds[i].MaximumC).Any()==false;
            var record=new SearchCandidate(records.Count+1,phase,controls,assessment,onGrid&&inBounds,warnings);
            cache.Add(key,record); records.Add(record);
            if(assessment.Feasible && record.EligibleForSelection && (best is null || assessment.OptimizationLoss < best.Assessment.OptimizationLoss)) best=record;
            journal.WriteLine(JsonSerializer.Serialize(record,JsonLineOptions));
            if(records.Count%100==0) journal.Flush();
            progress?.Report(new SearchProgress(records.Count,records.Count(x=>x.Assessment.Feasible),best?.Assessment.OptimizationLoss));
            return record;
        }
        var seedRuns=new List<SeedRun>();
        try
        {
            var c=targetConfig.ControlSetpoints;
            var baseline=await Evaluate([c.Cc1OutletC,c.Cc2OutletC,c.HeaterOaC,c.HeaterLowC],false).ConfigureAwait(false);
            if(settings.InitialControls is not null) { phase="initial_controls";await Evaluate(settings.InitialControls.Vector()).ConfigureAwait(false); }
            if(settings.CoarseGrid)
            {
                phase="coarse_grid";
                for(int code=0;code<81;code++)
                {
                    var point=new double[4]; int rest=code;
                    for(int i=3;i>=0;i--) { var b=settings.Bounds[i]; point[i]=b.MinimumC+(b.MaximumC-b.MinimumC)*(rest%3)/2;rest/=3; }
                    await Evaluate(point).ConfigureAwait(false);
                }
            }
            foreach(int seed in settings.Seeds)
            {
                phase=$"differential_evolution_seed_{seed}";
                var random=new Random(seed); int size=Math.Max(5,settings.PopulationMultiplier*4), calls=0;
                var population=Enumerable.Range(0,size).Select(_=>new double[4]).ToArray();
                // Latin hypercube initialization covers each marginal interval.
                for(int dimension=0;dimension<4;dimension++)
                {
                    var strata=Enumerable.Range(0,size).ToArray(); Shuffle(strata,random);
                    var b=settings.Bounds[dimension];
                    for(int i=0;i<size;i++) population[i][dimension]=b.MinimumC+(strata[i]+random.NextDouble())/size*(b.MaximumC-b.MinimumC);
                }
                population[0]=best?.Controls.Vector() ?? Canonical(baseline.Controls.Vector());
                var scores=new double[size];
                for(int i=0;i<size;i++) {scores[i]=(await Evaluate(population[i]).ConfigureAwait(false)).Assessment.Score;calls++;}
                for(int generation=0;generation<settings.MaxGenerations;generation++)
                {
                    token.ThrowIfCancellationRequested();
                    double factor=.5+.5*random.NextDouble();
                    for(int i=0;i<size;i++)
                    {
                        int bestIndex=Array.IndexOf(scores,scores.Min());
                        int a;do{a=random.Next(size);}while(a==i);
                        int b;do{b=random.Next(size);}while(b==i||b==a);
                        int required=random.Next(4);var trial=population[i].ToArray();
                        for(int dimension=0;dimension<4;dimension++) if(dimension==required || random.NextDouble()<.7)
                        {
                            var bound=settings.Bounds[dimension];
                            double value=population[bestIndex][dimension]+factor*(population[a][dimension]-population[b][dimension]);
                            trial[dimension]=value>=bound.MinimumC && value<=bound.MaximumC?value:bound.MinimumC+random.NextDouble()*(bound.MaximumC-bound.MinimumC);
                        }
                        double score=(await Evaluate(trial).ConfigureAwait(false)).Assessment.Score;calls++;
                        if(score<=scores[i]) {population[i]=trial;scores[i]=score;}
                    }
                }
                seedRuns.Add(new SeedRun(seed,settings.MaxGenerations,calls,best?.Assessment.OptimizationLoss));
            }
            if(best is not null)
            {
                phase="coordinate_refinement";
                foreach(double step in new[]{1.0,.5,settings.StepC}) for(int iteration=0;iteration<30;iteration++)
                {
                    double previous=best.Assessment.OptimizationLoss!.Value;
                    var center=best.Controls.Vector();
                    for(int dimension=0;dimension<4;dimension++) foreach(double direction in new[]{-1.0,1.0})
                    {var point=center.ToArray();point[dimension]+=direction*step;await Evaluate(point).ConfigureAwait(false);}
                    if(best.Assessment.OptimizationLoss>=previous) break;
                }
            }
            journal.Flush(); token.ThrowIfCancellationRequested();
            SimulationConfig? bestConfig=null; SimulationResult? verified=null;CandidateAssessment? verification=null;
            bool independent=false;
            if(best is not null)
            {
                bestConfig=ApplyControls(targetConfig,best.Controls);
                // This call bypasses the search cache and recomputes every rotor
                // iteration. Public production calls additionally save JSON/PNG.
                verified=await _verify(bestConfig,token).ConfigureAwait(false);
                verification=CandidateScoring.Assess(verified,settings,fixedReference);
                independent=verification.Feasible && Close(verification.OptimizationLoss,best.Assessment.OptimizationLoss,1e-9) && Close(verification.ThermalLoadSumKW,best.Assessment.ThermalLoadSumKW,1e-9);
                if(!independent) throw new OptimizationException("INDEPENDENT_VERIFICATION_FAILED","最佳候选独立复算失败，不能作为可行解交付。");
                File.WriteAllText(Path.Combine(dir,"best_config.json"),JsonSerializer.Serialize(bestConfig,JsonOptions),new UTF8Encoding(false));
            }
            var result=new SearchResult
            {
                Status=best is null?"no_feasible_candidate_found":"feasible_candidate_found",Settings=settings,
                EnergyReferenceKW=fixedReference!.Value,EnergyReferenceSource=settings.EnergyReferenceKW.HasValue?"explicit":"original_configuration_at_supply_temperature_target",
                Baseline=baseline,Best=best,MinimumThermalLoadCandidate=records.Where(x=>x.EligibleForSelection && x.Assessment.Feasible).MinBy(x=>x.Assessment.ThermalLoadSumKW),
                BestConfig=bestConfig,BestForwardResult=verified,IndependentVerification=verification,BestIndependentlyVerified=independent,
                Evaluations=records.AsReadOnly(),SeedRuns=seedRuns.AsReadOnly(),
                BestForwardJsonPath=verified?.ResultJsonPath,BestDashboardPngPath=verified?.DashboardPngPath,
                ResultJsonPath=Path.Combine(dir,"search.json"),
            };
            File.WriteAllText(result.ResultJsonPath,JsonSerializer.Serialize(result,JsonOptions),new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(dir,"model_sha256.json"),JsonSerializer.Serialize(_modelHashes,JsonOptions),new UTF8Encoding(false));
            return result;
        }
        catch(OperationCanceledException)
        {
            faulted=true;
            TryFlush(journal);
            TryWriteDiagnostic(Path.Combine(dir,"cancelled.json"),new {Status="cancelled",UniqueEvaluations=records.Count,EnergyReferenceKW=fixedReference,FeasibleCandidateRetainedButNotDelivered=best is not null});
            throw;
        }
        catch(Exception ex)
        {
            faulted=true;
            TryFlush(journal);
            TryWriteDiagnostic(Path.Combine(dir,"failed.json"),new {Status="failed",UniqueEvaluations=records.Count,Error=ex.Message});
            throw;
        }
        finally
        {
            try {journal.Dispose();}
            catch(Exception ex) when(faulted && (ex is IOException or UnauthorizedAccessException)) { /* Preserve the original cancellation/error. */ }
        }
    }

    private static bool IsCandidateNumericalError(Exception exception)=>exception is ArithmeticException || exception is SimulationException error && error.Code is "INVALID_PHYSICS" or "SOLVER_INVALID" or "MODEL_OUTPUT_INVALID";
    private static bool Close(double? a,double? b,double tolerance)=>a.HasValue && b.HasValue && Math.Abs(a.Value-b.Value)<=tolerance;
    private static void Shuffle(int[] values,Random random) {for(int i=values.Length-1;i>0;i--){int j=random.Next(i+1);(values[i],values[j])=(values[j],values[i]);}}
    private static void TryFlush(StreamWriter writer) {try{writer.Flush();}catch(Exception ex) when(ex is IOException or UnauthorizedAccessException){}}
    private static void TryWriteDiagnostic(string path,object value)
    {try{File.WriteAllText(path,JsonSerializer.Serialize(value,JsonOptions),new UTF8Encoding(false));}catch(Exception ex) when(ex is IOException or UnauthorizedAccessException){}}
    internal static SimulationConfig ApplyControls(SimulationConfig source,ControlTemperatures controls)
        => CopyConfiguration(source,source.ControlSetpoints with {Cc1OutletC=controls.Cc1OutletC,Cc2OutletC=controls.Cc2OutletC,HeaterOaC=controls.HeaterOaC,HeaterLowC=controls.HeaterLowC});
    private static SimulationConfig CopyConfiguration(SimulationConfig source,ControlSetpointsConfig controls,string? sourceName=null,string? sourcePath=null)=>new()
    {
        SchemaVersion=source.SchemaVersion,Psychrometrics=source.Psychrometrics,FlowReference=source.FlowReference,
        OutdoorAir=source.OutdoorAir,ReturnAir=source.ReturnAir,RegenOutdoorAir=source.RegenOutdoorAir,SystemParams=source.SystemParams,
        ControlSetpoints=controls,SourceName=sourceName??source.SourceName,SourcePath=sourcePath??source.SourcePath,
    };

    /// <summary>等候正在搜索的请求结束后释放模型；宿主应先取消并await，再Dispose。</summary>
    public void Dispose()
    {
        if(_disposed) return;
        _gate.Wait();
        try {if(_disposed)return;_disposed=true;_simulator?.Dispose();}
        finally {_gate.Release();}
    }
}
