using System.Text.Json.Serialization;
using Dehumidifier.Simulation;

namespace Dehumidifier.Optimization;

/// <summary>四个外层搜索变量，单位℃；流量、尺寸、入口及转速保持配置值。</summary>
public sealed record ControlTemperatures(double Cc1OutletC, double Cc2OutletC, double HeaterOaC, double HeaterLowC)
{
    internal double[] Vector() => [Cc1OutletC, Cc2OutletC, HeaterOaC, HeaterLowC];
    internal static ControlTemperatures From(double[] values) => new(values[0], values[1], values[2], values[3]);
}

/// <summary>一个控制温度的闭区间，单位℃。</summary>
public sealed record TemperatureBounds(double MinimumC, double MaximumC);

/// <summary>送风目标、固定归一化基准和搜索预算。露点容差不得超过±1℃。</summary>
public sealed class SearchSettings
{
    public required double DewPointTargetC { get; init; }
    public required double SupplyTemperatureTargetC { get; init; }
    public double DewPointToleranceC { get; init; } = 1;
    public double? SupplyTemperatureMinimumC { get; init; }
    public double? SupplyTemperatureMaximumC { get; init; }
    public double StepC { get; init; } = 0.1;
    public double? EnergyReferenceKW { get; init; }
    public double DewPointPenaltyCoefficient { get; init; } = 0.1;
    public TemperatureBounds[] Bounds { get; init; } = [new(9,13), new(13,17), new(50,140), new(50,140)];
    public int[] Seeds { get; init; } = [20261110, 20261111, 20261112];
    public int MaxGenerations { get; init; } = 40;
    public int PopulationMultiplier { get; init; } = 8;
    public bool CoarseGrid { get; init; } = true;
    public ControlTemperatures? InitialControls { get; init; }

    /// <summary>在启动计算前校验所有目标、边界及预算，失败抛ArgumentException。</summary>
    public void Validate()
    {
        double[] values = [DewPointTargetC, SupplyTemperatureTargetC, DewPointToleranceC, StepC, DewPointPenaltyCoefficient];
        if (values.Any(x => !double.IsFinite(x))) throw new ArgumentException("目标、步长及系数必须有限。");
        if (DewPointToleranceC <= 0 || DewPointToleranceC > 1) throw new ArgumentException("露点容差必须大于0且不得超过1℃。");
        if (DewPointTargetC-DewPointToleranceC <= -273.15 || SupplyTemperatureTargetC <= -273.15) throw new ArgumentException("目标温度必须高于绝对零度。");
        if (StepC <= 0 || DewPointPenaltyCoefficient <= 0) throw new ArgumentException("搜索步长与露点惩罚系数必须大于0。");
        if (EnergyReferenceKW is double reference && (!double.IsFinite(reference) || reference <= 0)) throw new ArgumentException("固定能耗基准必须是正的有限kW数值。");
        foreach (double? optional in new[] { SupplyTemperatureMinimumC, SupplyTemperatureMaximumC })
            if (optional.HasValue && (!double.IsFinite(optional.Value) || optional.Value <= -273.15)) throw new ArgumentException("附加送风温度边界必须有限且高于绝对零度。");
        if (SupplyTemperatureMinimumC is double lo && SupplyTemperatureTargetC < lo || SupplyTemperatureMaximumC is double hi && SupplyTemperatureTargetC > hi)
            throw new ArgumentException("送风温度目标必须处于附加温度边界之内。");
        if (SupplyTemperatureMinimumC.HasValue && SupplyTemperatureMaximumC.HasValue && SupplyTemperatureMinimumC > SupplyTemperatureMaximumC) throw new ArgumentException("送风温度下限不得高于上限。");
        if (Bounds is null || Bounds.Length != 4 || Bounds.Any(x => x is null)) throw new ArgumentException("须提供四个控制温度边界。");
        foreach (var bound in Bounds)
        {
            if (!double.IsFinite(bound.MinimumC) || !double.IsFinite(bound.MaximumC) || bound.MinimumC <= -273.15 || bound.MinimumC >= bound.MaximumC)
                throw new ArgumentException("每个控制温度区间必须有限、递增且高于绝对零度。");
            var count = (bound.MaximumC-bound.MinimumC)/StepC;
            if (!double.IsFinite(count) || Math.Abs(count-Math.Round(count)) > 1e-7) throw new ArgumentException("温度区间须为网格步长的整数倍。");
        }
        if (Seeds is null || Seeds.Length == 0 || Seeds.Any(x => x < 0)) throw new ArgumentException("须至少提供一个非负随机种子。");
        if (MaxGenerations < 0 || PopulationMultiplier < 2) throw new ArgumentException("代数不得为负，种群倍数至少为2。");
        if (InitialControls is not null && InitialControls.Vector().Where((v,i) => !double.IsFinite(v) || v < Bounds[i].MinimumC || v > Bounds[i].MaximumC).Any()) throw new ArgumentException("初始四控制温度必须有限且位于边界内。");
    }

    internal SearchSettings Snapshot() => new()
    {
        DewPointTargetC=DewPointTargetC, SupplyTemperatureTargetC=SupplyTemperatureTargetC,
        DewPointToleranceC=DewPointToleranceC, SupplyTemperatureMinimumC=SupplyTemperatureMinimumC,
        SupplyTemperatureMaximumC=SupplyTemperatureMaximumC, StepC=StepC, EnergyReferenceKW=EnergyReferenceKW,
        DewPointPenaltyCoefficient=DewPointPenaltyCoefficient, Bounds=Bounds.ToArray(), Seeds=Seeds.ToArray(),
        MaxGenerations=MaxGenerations, PopulationMultiplier=PopulationMultiplier, CoarseGrid=CoarseGrid,
        InitialControls=InitialControls,
    };
}

/// <summary>最终Supply与六项热负荷的可行性评价；J无量纲，Q为kW。</summary>
public sealed record CandidateAssessment
{
    public bool Feasible { get; init; }
    public string[] Reasons { get; init; } = [];
    public double Score { get; init; }
    public double? ThermalLoadSumKW { get; init; }
    public double? CoolingLoadKW { get; init; }
    public double? HeatingLoadKW { get; init; }
    public Dictionary<string,double>? EnergyBreakdownKW { get; init; }
    public double? EnergyReferenceKW { get; init; }
    public double? NormalizedEnergy { get; init; }
    public double? DewPointRegionWeight { get; init; }
    public double? WeightedNormalizedDewPointDistance { get; init; }
    public double? DewPointDistancePenalty { get; init; }
    public double? OptimizationLoss { get; init; }
    public double? SupplyTemperatureC { get; init; }
    public double? SupplyDewPointC { get; init; }
    public double? SupplyHumidityGPerKg { get; init; }
    public double? DewPointErrorC { get; init; }
    public double? SupplyTemperatureErrorC { get; init; }
    public double? MaxUnclippedRelativeHumidity { get; init; }
    public bool Converged { get; init; }
    public int Iterations { get; init; }
    public string? Error { get; init; }
}

/// <summary>每个实际独立计算的参数组合和评价。</summary>
public sealed record SearchCandidate(int Evaluation, string Phase, ControlTemperatures Controls,
    CandidateAssessment Assessment, bool EligibleForSelection, string[] Warnings);

/// <summary>搜索进度；BestLoss为无量纲评分。</summary>
public sealed record SearchProgress(int UniqueEvaluations, int FeasibleEvaluations, double? BestLoss);

/// <summary>一次随机种子搜索的预算和结果。</summary>
public sealed record SeedRun(int Seed, int Generations, int ObjectiveCallsIncludingCached, double? BestLoss);

/// <summary>搜索结果；Feasible=false表示预算内未找到符合全部硬约束的候选。</summary>
public sealed class SearchResult
{
    public int SchemaVersion { get; init; } = 3;
    public string Status { get; init; } = "no_feasible_candidate_found";
    public bool Feasible => Best is not null && BestIndependentlyVerified;
    public bool GlobalMinimumProven => false;
    public bool ActualElectricityOptimized => false;
    public string Objective => "normalized_six_thermal_load_sum_plus_piecewise_dewpoint_absolute_distance";
    public string Algorithm => "C# best/1/bin differential evolution, fixed budget, lattice cache, coordinate refinement";
    public string RandomSequenceCompatibility => "Same physical/loss contract; System.Random sequence differs from SciPy/NumPy; no identical optimizer trajectory claim.";
    public required SearchSettings Settings { get; init; }
    public required double EnergyReferenceKW { get; init; }
    public required string EnergyReferenceSource { get; init; }
    public required SearchCandidate Baseline { get; init; }
    public SearchCandidate? Best { get; init; }
    public SearchCandidate? MinimumThermalLoadCandidate { get; init; }
    public SimulationConfig? BestConfig { get; init; }
    [JsonIgnore] public SimulationResult? BestForwardResult { get; init; }
    public bool BestIndependentlyVerified { get; init; }
    public CandidateAssessment? IndependentVerification { get; init; }
    public int UniqueForwardEvaluations => Evaluations.Count;
    public int FeasibleEvaluations => Evaluations.Count(x => x.Assessment.Feasible);
    public required IReadOnlyList<SearchCandidate> Evaluations { get; init; }
    public required IReadOnlyList<SeedRun> SeedRuns { get; init; }
    public string? BestForwardJsonPath { get; init; }
    public string? BestDashboardPngPath { get; init; }
    public string? ResultJsonPath { get; internal set; }
}

/// <summary>模型、固定基准、复算或结果留档失败；无可行解作为SearchResult返回。</summary>
public sealed class OptimizationException(string code, string message, Exception? inner = null) : Exception(message, inner)
{
    public string Code { get; } = code;
}
