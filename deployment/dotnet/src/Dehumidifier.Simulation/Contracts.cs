using System.Text.Json.Serialization;

namespace Dehumidifier.Simulation;

/// <summary>带温度、含湿量、流量和压力的湿空气状态。</summary>
public sealed record AirState(
    double TemperatureC,
    double HumidityGPerKg,
    double DryAirMassFlowKgPerS,
    double PressurePa)
{
    [JsonIgnore] public double EnthalpyKJPerKg => Psychrometrics.EnthalpyKJ(TemperatureC, HumidityGPerKg);
    [JsonIgnore] public double DewPointC => Psychrometrics.DewPointWaterC(HumidityGPerKg, PressurePa);
    [JsonIgnore] public double? DewPointOrNull => double.IsNegativeInfinity(DewPointC) ? null : DewPointC;
    [JsonIgnore] public double RelativeHumidityFraction => Psychrometrics.RelativeHumidity(TemperatureC, HumidityGPerKg, PressurePa);

    public AirState AtFlow(double flow) => this with { DryAirMassFlowKgPerS = flow };
    public AirState AtPressure(double pressure) => this with { PressurePa = pressure };
}

public sealed record RotorOutputs(
    double ProcessOutletTemperatureC,
    double ProcessOutletHumidityGPerKg,
    double CoolBlowOutletTemperatureC,
    double CoolBlowOutletHumidityGPerKg,
    double RegenOutletTemperatureC,
    double RegenOutletHumidityGPerKg,
    double? ProcessOutletDewPointC);

public sealed record ConvergenceHistoryItem(
    [property: JsonPropertyName("iteration")] int Iteration,
    [property: JsonPropertyName("inputs")] Dictionary<string, double> Inputs,
    [property: JsonPropertyName("residuals")] Dictionary<string, double> Residuals);

public sealed class ConvergenceInfo
{
    [JsonPropertyName("status")] public string Status { get; init; } = "not_evaluated";
    [JsonPropertyName("converged")] public bool? Converged { get; init; }
    [JsonPropertyName("iterations")] public int Iterations { get; init; }
    [JsonPropertyName("residuals")] public Dictionary<string, double> Residuals { get; init; } = new();
    [JsonPropertyName("tolerances")] public Dictionary<string, double> Tolerances { get; init; } = new();
    [JsonPropertyName("last_evaluated_inputs")] public Dictionary<string, double> LastEvaluatedInputs { get; init; } = new();
    [JsonPropertyName("solver_parameters")] public Dictionary<string, double> SolverParameters { get; init; } = new();
    [JsonPropertyName("history")] public List<ConvergenceHistoryItem> History { get; init; } = new();
}

public sealed class SolverOptions
{
    public int MaxIterations { get; init; } = 60;
    public double Relaxation { get; init; } = 0.8;
    public double TemperatureToleranceC { get; init; } = 1e-4;
    public double HumidityToleranceGPerKg { get; init; } = 1e-6;

    internal void Validate()
    {
        if (MaxIterations <= 0) throw new SimulationException("INVALID_SOLVER_OPTIONS", "MaxIterations 必须为正整数");
        if (!double.IsFinite(Relaxation) || Relaxation <= 0 || Relaxation > 1) throw new SimulationException("INVALID_SOLVER_OPTIONS", "Relaxation 必须在 (0,1] 范围");
        if (!double.IsFinite(TemperatureToleranceC) || TemperatureToleranceC <= 0 || !double.IsFinite(HumidityToleranceGPerKg) || HumidityToleranceGPerKg <= 0)
            throw new SimulationException("INVALID_SOLVER_OPTIONS", "收敛容差必须为正的有限值");
    }
}

/// <summary>再生混风企业风量规则执行记录（Heater_OA 入口 = R1 处理出口 / 3）。</summary>
public sealed record RegenMixRuleInfo(
    string Branch,
    double ThresholdKgPerS,
    double AvailableKgPerS,
    double RegenOaConfiguredKgPerS,
    double RegenOaUsedKgPerS,
    double LeakedKgPerS,
    double HeaterOaInletKgPerS,
    double ShortfallKgPerS,
    double ThresholdM3H,
    double AvailableM3H,
    double RegenOaUsedM3H,
    double LeakedM3H,
    double HeaterOaInletM3H,
    double ShortfallM3H);

public sealed class SimulationResult
{
    public bool Converged => Convergence.Converged == true;
    public int Iterations => Convergence.Iterations;
    public ConvergenceInfo Convergence { get; set; } = new();
    public required AirState Supply { get; init; }
    public required Dictionary<string, AirState> IntermediateStates { get; init; }
    public required RotorOutputs Rotor1Outputs { get; init; }
    public required RotorOutputs Rotor2Outputs { get; init; }
    public required double CoolingLoadKW { get; init; }
    public required double HeatingLoadKW { get; init; }
    public required Dictionary<string, double> EnergyBreakdownKW { get; init; }
    public RegenMixRuleInfo? RegenMixRule { get; init; }
    public List<string> Warnings { get; init; } = new();
    public string? ResultJsonPath { get; internal set; }
    public string? DashboardPngPath { get; internal set; }
}

public sealed class SimulationException : Exception
{
    public string Code { get; }
    public SimulationException(string code, string message) : base(message) => Code = code;
    public SimulationException(string code, string message, Exception inner) : base(message, inner) => Code = code;
}
