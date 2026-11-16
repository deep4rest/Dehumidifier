using System.Text.Json;
using Dehumidifier.Simulation;
using Dehumidifier.Optimization;
using Xunit;

namespace Dehumidifier.Optimization.Tests;

public sealed class SearchTests
{
    private static SimulationConfig Config()=>Simulator.LoadConfiguration(Path.Combine(AppContext.BaseDirectory,"examples/qy1070_case.json"));
    private static SearchSettings Settings(double reference=100)=>new()
    {
        DewPointTargetC=-45,SupplyTemperatureTargetC=-35,EnergyReferenceKW=reference,
        CoarseGrid=false,Seeds=[7],MaxGenerations=0,PopulationMultiplier=2,
    };
    private static string Temp()=>Path.Combine(Path.GetTempPath(),"Dehumidifier.Optimization.Tests",Guid.NewGuid().ToString("N"));
    private static SimulationResult Result(double dewPoint=-45,double q=100,bool converged=true,double temperature=-35)
    {
        // DptAH table uses linear segments. Restrict the fake to [-46,-44]
        // for these tests so the synthetic state encodes the requested dewpoint.
        double humidity=dewPoint<=-45?.0444+(dewPoint+45)*(.0444-.0394):.0444+(dewPoint+45)*(.0499-.0444);
        var supply=new AirState(temperature,humidity,1,101325);
        var rotor=new RotorOutputs(temperature,humidity,20,.1,80,1,dewPoint);
        return new SimulationResult
        {
            Supply=supply,IntermediateStates=new(){{"Supply",supply}},Rotor1Outputs=rotor,Rotor2Outputs=rotor,
            CoolingLoadKW=q*.8,HeatingLoadKW=q*.2,
            EnergyBreakdownKW=new(){{"CC1_kW",-q*.3},{"CC2_kW",-q*.2},{"CC3_kW",-q*.3},{"Heater_Low_kW",q*.1},{"Heater_OA_kW",q*.1},{"Heater_Post_kW",0}},
            Convergence=new ConvergenceInfo{Converged=converged,Status=converged?"converged":"not_converged",Iterations=3},
        };
    }

    [Theory]
    [InlineData(-1,.35)] [InlineData(-.50001,.35)] [InlineData(-.5,.1)]
    [InlineData(0,.1)] [InlineData(.3,.1)] [InlineData(.30001,.5)] [InlineData(1,.5)]
    public void PiecewiseSharedBoundariesBelongToMiddle(double error,double expected)
        =>Assert.Equal(expected,CandidateScoring.DewPointRegionWeight(error));

    [Theory]
    [InlineData(-1.0001)] [InlineData(1.0001)] [InlineData(double.NaN)]
    public void OutOfBandWeightIsNotDefined(double error)=>Assert.Null(CandidateScoring.DewPointRegionWeight(error));

    [Fact]
    public void PhysicalLoadAndDimensionlessLossAreSeparate()
    {
        var assessment=CandidateScoring.Assess(Result(-44.5,100),Settings(),200);
        Assert.True(assessment.Feasible);
        Assert.Equal(100,assessment.ThermalLoadSumKW);
        Assert.Equal(.5,assessment.NormalizedEnergy!.Value,12);
        Assert.Equal(.025,assessment.DewPointDistancePenalty!.Value,12);
        Assert.Equal(.525,assessment.OptimizationLoss!.Value,12);
    }

    [Fact]
    public void ReversedCoolingSignAndWrongAggregatesAreRejected()
    {
        var result=Result();result.EnergyBreakdownKW["CC3_kW"]=30;
        Assert.Contains("invalid_load_contract",CandidateScoring.Assess(result,Settings()).Reasons);
        var wrong=Result();wrong.EnergyBreakdownKW["CC3_kW"]=-31;
        Assert.False(CandidateScoring.Assess(wrong,Settings()).Feasible);
    }

    [Fact]
    public void FinalSupplyRatherThanRawRotorIsConstrained()
    {
        var result=Result(-44.9);result.IntermediateStates["AfterR2_P"]=new AirState(20,.0444,1,101325);
        Assert.True(CandidateScoring.Assess(result,Settings()).Feasible);
        var wrongTemperature=Result(-45,100,true,-34.9);
        Assert.Contains("supply_temperature_target",CandidateScoring.Assess(wrongTemperature,Settings()).Reasons);
    }

    [Fact]
    public void UnclippedSupersaturationAndUnconvergedCandidateCannotWin()
    {
        var wet=Result();wet.IntermediateStates["BadNode"]=new AirState(10,20,1,101325);
        Assert.Contains("supersaturated",CandidateScoring.Assess(wet,Settings()).Reasons);
        Assert.Contains("unconverged",CandidateScoring.Assess(Result(converged:false),Settings()).Reasons);
    }

    [Fact]
    public void SettingsRejectLooseDewpointAndBadFixedReference()
    {
        Assert.Throws<ArgumentException>(()=>new SearchSettings{DewPointTargetC=-45,SupplyTemperatureTargetC=-35,DewPointToleranceC=1.01}.Validate());
        Assert.Throws<ArgumentException>(()=>new SearchSettings{DewPointTargetC=-45,SupplyTemperatureTargetC=-35,EnergyReferenceKW=0}.Validate());
        Assert.Throws<ArgumentException>(()=>new SearchSettings{DewPointTargetC=-45,SupplyTemperatureTargetC=-35,StepC=.3}.Validate());
    }

    [Fact]
    public async Task BaselineReferenceIsFixedOnceAndCallerConfigIsUnchanged()
    {
        var original=Config();string before=JsonSerializer.Serialize(original);int calls=0;
        using var optimizer=new EnergyOptimizer(Temp(),(config,token)=>
        {calls++;return Task.FromResult(Result(q:50+config.ControlSetpoints.Cc1OutletC));});
        var result=await optimizer.OptimizeAsync(original,new SearchSettings{DewPointTargetC=-45,SupplyTemperatureTargetC=-35,CoarseGrid=false,Seeds=[7],MaxGenerations=0,PopulationMultiplier=2});
        Assert.True(result.Feasible);
        Assert.Equal(61,result.EnergyReferenceKW,12);
        Assert.All(result.Evaluations,x=>Assert.Equal(61,x.Assessment.EnergyReferenceKW!.Value,12));
        Assert.Equal(before,JsonSerializer.Serialize(original));
        Assert.True(calls>result.UniqueForwardEvaluations); // independent re-evaluation
        Assert.True(File.Exists(result.ResultJsonPath));
        Assert.Equal(-35,result.BestConfig!.ControlSetpoints.SupplyTemperatureTargetC);
        Assert.Equal(1070,result.BestConfig.SystemParams.Rotor1.DiameterMm);
        Assert.Equal(1070,result.BestConfig.SystemParams.Rotor2.DiameterMm);
        Assert.Equal(original.OutdoorAir,result.BestConfig.OutdoorAir);
        Assert.Equal(original.ReturnAir,result.BestConfig.ReturnAir);
        Assert.Equal(original.RegenOutdoorAir,result.BestConfig.RegenOutdoorAir);
        Assert.Equal(original.SystemParams,result.BestConfig.SystemParams);
        Assert.Equal(original.ControlSetpoints.Rotor1Rpm,result.BestConfig.ControlSetpoints.Rotor1Rpm);
        Assert.Equal(original.ControlSetpoints.Rotor2Rpm,result.BestConfig.ControlSetpoints.Rotor2Rpm);
        var reloaded=Simulator.LoadConfiguration(result.BestConfigPath!);
        Assert.Equal(JsonSerializer.Serialize(result.BestConfig),JsonSerializer.Serialize(reloaded));
        Assert.Equal(result.Best!.Controls.HeaterOaC,reloaded.ControlSetpoints.HeaterOaC);
        Assert.Equal(result.Best.Controls.HeaterLowC,reloaded.ControlSetpoints.HeaterLowC);
    }

    [Fact]
    public async Task BaselineFailureCannotSilentlyUseNextCandidateAsReference()
    {
        int calls=0;
        using var optimizer=new EnergyOptimizer(Temp(),(_,_)=>{calls++;throw new SimulationException("INVALID_PHYSICS","baseline failed");});
        var error=await Assert.ThrowsAsync<OptimizationException>(()=>optimizer.OptimizeAsync(Config(),new SearchSettings{DewPointTargetC=-45,SupplyTemperatureTargetC=-35}));
        Assert.Equal("REFERENCE_INVALID",error.Code);Assert.Equal(1,calls);
    }

    [Fact]
    public async Task NoFeasibleCandidateIsExplicitAndContainsNoBest()
    {
        using var optimizer=new EnergyOptimizer(Temp(),(_,_)=>Task.FromResult(Result(-43)));
        var result=await optimizer.OptimizeAsync(Config(),Settings());
        Assert.False(result.Feasible);Assert.Equal("no_feasible_candidate_found",result.Status);
        Assert.Null(result.Best);Assert.False(result.BestIndependentlyVerified);
    }

    [Fact]
    public async Task CandidateGraphFailureIsInfeasibleAndSearchCanContinue()
    {
        int calls=0;
        using var optimizer=new EnergyOptimizer(Temp(),(_,_)=>
        {calls++;if(calls==2)throw new SimulationException("MODEL_OUTPUT_INVALID","negative graph humidity");return Task.FromResult(Result());});
        var result=await optimizer.OptimizeAsync(Config(),Settings());
        Assert.True(result.Feasible);Assert.Contains(result.Evaluations,x=>x.Assessment.Reasons.Contains("numerical_error"));
    }

    [Fact]
    public async Task SearchIsActualAndUsesEnergyVersusDewpointTradeoff()
    {
        using var optimizer=new EnergyOptimizer(Temp(),(config,_)=>
        {
            double x=config.ControlSetpoints.HeaterLowC;
            double error=Math.Clamp((x-90)*.1,-.99,.99);
            return Task.FromResult(Result(-45+error,100+(x-88)*(x-88)*.005));
        });
        var result=await optimizer.OptimizeAsync(Config(),new SearchSettings
        {
            DewPointTargetC=-45,SupplyTemperatureTargetC=-35,EnergyReferenceKW=100,
            Bounds=[new(9,10),new(13,14),new(80,81),new(85,95)],Seeds=[17],MaxGenerations=3,PopulationMultiplier=2,
        });
        Assert.True(result.Feasible);Assert.True(result.Evaluations.Select(x=>x.Controls.HeaterLowC).Distinct().Count()>10);
        Assert.True(result.Best!.Assessment.OptimizationLoss<=result.MinimumThermalLoadCandidate!.Assessment.OptimizationLoss);
        Assert.InRange(result.Best.Controls.HeaterLowC,85,95);
        Assert.True(result.SeedRuns[0].ObjectiveCallsIncludingCached>8);
    }

    [Fact]
    public async Task CancellationDoesNotReturnAnUnverifiedBest()
    {
        using var cancellation=new CancellationTokenSource();int calls=0;
        using var optimizer=new EnergyOptimizer(Temp(),(_,token)=>
        {calls++;if(calls==2)cancellation.Cancel();token.ThrowIfCancellationRequested();return Task.FromResult(Result());});
        await Assert.ThrowsAnyAsync<OperationCanceledException>(()=>optimizer.OptimizeAsync(Config(),Settings(),cancellation.Token));
        Assert.InRange(calls,1,2);
    }

    [Fact]
    public async Task ChangedIndependentForwardResultCannotBeDelivered()
    {
        var counts=new Dictionary<double,int>();
        using var optimizer=new EnergyOptimizer(Temp(),(config,_)=>
        {
            double x=config.ControlSetpoints.Cc1OutletC;
            counts.TryGetValue(x,out int count);counts[x]=count+1;
            return Task.FromResult(Result(q:count==0?100+x:1000+x));
        });
        var error=await Assert.ThrowsAsync<OptimizationException>(()=>optimizer.OptimizeAsync(Config(),Settings()));
        Assert.Equal("INDEPENDENT_VERIFICATION_FAILED",error.Code);
    }

    [Fact]
    public async Task SameLossAndLoadWithDifferentDewPointCannotBeDelivered()
    {
        var seen=new HashSet<string>();
        using var optimizer=new EnergyOptimizer(Temp(),(config,_)=>
        {
            bool first=seen.Add(JsonSerializer.Serialize(config));
            // Both errors use the middle region and identical |e|, Q and J.
            return Task.FromResult(Result(dewPoint:first?-45.1:-44.9));
        });
        var error=await Assert.ThrowsAsync<OptimizationException>(()=>optimizer.OptimizeAsync(Config(),Settings()));
        Assert.Equal("INDEPENDENT_VERIFICATION_FAILED",error.Code);
    }

    [Fact]
    public async Task ChangedLoadDistributionWithSameTotalCannotBeDelivered()
    {
        var seen=new HashSet<string>();
        using var optimizer=new EnergyOptimizer(Temp(),(config,_)=>
        {
            var result=Result();
            if(!seen.Add(JsonSerializer.Serialize(config)))
            {
                result.EnergyBreakdownKW["CC1_kW"]+=1;
                result.EnergyBreakdownKW["CC2_kW"]-=1;
            }
            return Task.FromResult(result);
        });
        var error=await Assert.ThrowsAsync<OptimizationException>(()=>optimizer.OptimizeAsync(Config(),Settings()));
        Assert.Equal("INDEPENDENT_VERIFICATION_FAILED",error.Code);
    }

    [Fact]
    public async Task UnwritableResultRootUsesDocumentedErrorCode()
    {
        string path=Temp();Directory.CreateDirectory(Path.GetDirectoryName(path)!);File.WriteAllText(path,"occupied");
        using var optimizer=new EnergyOptimizer(path,(_,_)=>Task.FromResult(Result()));
        var error=await Assert.ThrowsAsync<OptimizationException>(()=>optimizer.OptimizeAsync(Config(),Settings()));
        Assert.Equal("RESULT_SAVE_FAILED",error.Code);
    }
}
