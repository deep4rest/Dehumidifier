using Dehumidifier.Simulation;

namespace Dehumidifier.Optimization;

/// <summary>与Python外层控制搜索相同的六负荷/最终Supply约束和分区评分。</summary>
public static class CandidateScoring
{
    public const double SupplyTemperatureToleranceC = 1e-4;
    private static readonly string[] Cold = ["CC1_kW","CC2_kW","CC3_kW"];
    private static readonly string[] Hot = ["Heater_Low_kW","Heater_OA_kW","Heater_Post_kW"];

    /// <summary>e=预测露点−目标。共享−0.5和+0.3端点归中间区；超±1返回null。</summary>
    public static double? DewPointRegionWeight(double errorC)
        => !double.IsFinite(errorC) || errorC < -1 || errorC > 1 ? null : errorC < -.5 ? .35 : errorC <= .3 ? .1 : .5;

    /// <summary>固定正基准归一化；null基准仅评价物理负荷/可行性，不生成J。</summary>
    public static CandidateAssessment Assess(SimulationResult result, SearchSettings settings, double? energyReferenceKW = null)
    {
        settings.Validate();
        var reference = energyReferenceKW ?? settings.EnergyReferenceKW;
        if (reference.HasValue && (!double.IsFinite(reference.Value) || reference <= 0)) throw new ArgumentException("固定基准必须为正的有限kW数值。");
        var reasons = new List<string>(); double violation = 0;
        void Reject(string reason,double amount=1) { if (!reasons.Contains(reason)) reasons.Add(reason); violation+=amount; }
        if (!result.Converged) Reject("unconverged");
        double maxRh=0;
        foreach (var state in result.IntermediateStates.Values)
        {
            if (!ValidState(state)) { Reject("invalid_state"); continue; }
            var rh=state.PressurePa*state.HumidityGPerKg/(621.98+state.HumidityGPerKg)/Psychrometrics.SaturationVaporPressureWaterPa(state.TemperatureC);
            if (!double.IsFinite(rh)) Reject("invalid_state");
            else { maxRh=Math.Max(maxRh,rh); if(rh>1+1e-6) Reject("supersaturated",Math.Min(rh-1,1e6)); }
        }
        var supply=result.Supply;
        var T=supply.TemperatureC; var W=supply.HumidityGPerKg;
        double dp=double.NaN;
        if (ValidState(supply)) dp=supply.DewPointC;
        bool validDp=double.IsFinite(dp) || (double.IsNegativeInfinity(dp) && W==0);
        if (!ValidState(supply) || !validDp) Reject("invalid_supply");
        else
        {
            double lower=settings.DewPointTargetC-settings.DewPointToleranceC, upper=settings.DewPointTargetC+settings.DewPointToleranceC;
            if(dp<lower) Reject("dew_point_below_band",double.IsFinite(dp)?(lower-dp)/10:1e6);
            if(dp>upper) Reject("dew_point_above_band",(dp-upper)/10);
            if(Math.Abs(T-settings.SupplyTemperatureTargetC)>SupplyTemperatureToleranceC) Reject("supply_temperature_target",Math.Abs(T-settings.SupplyTemperatureTargetC)/10);
            if(settings.SupplyTemperatureMinimumC is double lo && T<lo) Reject("supply_temperature",(lo-T)/10);
            if(settings.SupplyTemperatureMaximumC is double hi && T>hi) Reject("supply_temperature",(T-hi)/10);
        }
        var energy=result.EnergyBreakdownKW;
        bool loadValid=energy.Count==6 && Cold.Concat(Hot).All(k=>energy.TryGetValue(k,out double q) && double.IsFinite(q))
            && Cold.All(k=>energy[k]<=0) && Hot.All(k=>energy[k]>=0)
            && double.IsFinite(result.CoolingLoadKW) && result.CoolingLoadKW>=0 && double.IsFinite(result.HeatingLoadKW) && result.HeatingLoadKW>=0;
        if(loadValid) loadValid=Math.Abs(result.CoolingLoadKW+Cold.Sum(k=>energy[k]))<=1e-9 && Math.Abs(result.HeatingLoadKW-Hot.Sum(k=>energy[k]))<=1e-9;
        if(!loadValid) Reject("invalid_load_contract");
        double? total=loadValid ? result.CoolingLoadKW+result.HeatingLoadKW : null;
        if(total.HasValue && !double.IsFinite(total.Value)) { total=null; Reject("invalid_load"); }
        double? error=Finite(dp)?dp-settings.DewPointTargetC:null;
        double? region=error.HasValue?DewPointRegionWeight(error.Value):null;
        double? normalized=total.HasValue && reference.HasValue?total.Value/reference.Value:null;
        double? distance=region.HasValue?region.Value*Math.Abs(error!.Value):null;
        double? penalty=distance*settings.DewPointPenaltyCoefficient;
        double? loss=normalized.HasValue && penalty.HasValue?normalized.Value+penalty.Value:null;
        if(normalized.HasValue && !Finite(normalized.Value) || loss.HasValue && !Finite(loss.Value)) { Reject("invalid_objective"); normalized=null;loss=null; }
        bool feasible=reasons.Count==0;
        return new CandidateAssessment
        {
            Feasible=feasible, Reasons=reasons.ToArray(), Score=feasible?(loss.HasValue?loss.Value/(1+loss.Value):0):1+Math.Min(violation,1e6),
            ThermalLoadSumKW=total, CoolingLoadKW=Finite(result.CoolingLoadKW)?result.CoolingLoadKW:null,
            HeatingLoadKW=Finite(result.HeatingLoadKW)?result.HeatingLoadKW:null,
            EnergyBreakdownKW=loadValid?new(energy):null, EnergyReferenceKW=reference, NormalizedEnergy=normalized,
            DewPointRegionWeight=region, WeightedNormalizedDewPointDistance=distance, DewPointDistancePenalty=penalty,
            OptimizationLoss=loss, SupplyTemperatureC=Finite(T)?T:null, SupplyDewPointC=Finite(dp)?dp:null,
            SupplyHumidityGPerKg=Finite(W)?W:null, DewPointErrorC=error,
            SupplyTemperatureErrorC=Finite(T)?T-settings.SupplyTemperatureTargetC:null,
            MaxUnclippedRelativeHumidity=maxRh, Converged=result.Converged, Iterations=result.Iterations,
        };
    }

    private static bool Finite(double value)=>double.IsFinite(value);
    private static bool ValidState(AirState state)=>Finite(state.TemperatureC) && state.TemperatureC> -273.15 && Finite(state.HumidityGPerKg) && state.HumidityGPerKg>=0 && Finite(state.DryAirMassFlowKgPerS) && state.DryAirMassFlowKgPerS>=0 && Finite(state.PressurePa) && state.PressurePa>0;
}
