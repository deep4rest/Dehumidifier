using System.Text.Json;
using System.Text.Json.Serialization;
using Dehumidifier.Simulation;

namespace Dehumidifier.Prediction;

/// <summary>
/// 露点与能耗预测模型v2的托管.NET入口。内部复用正式双转轮求解器和ONNX模型；
/// 温度单位℃、含湿量g/kg干空气、风量m³/h、负荷kW。
/// </summary>
/// <remarks>
/// 同一实例的请求由底层求解器串行执行。正常预测保存JSON和PNG，未收敛时也保留调试结果；
/// 调用方必须检查Converged。窗体关闭前先取消并等待任务完成，再Dispose。
/// </remarks>
public sealed class PredictionModel : IDisposable
{
    private readonly Simulator _simulator;

    /// <summary>成品名称。</summary>
    public const string ModelName = "露点与能耗预测模型v2";

    /// <summary>加载当前xf/main两台ONNX和契约文件，设置可写的结果目录。</summary>
    public PredictionModel(string modelDirectory, string resultsDirectory)
        => _simulator = new Simulator(modelDirectory, resultsDirectory);

    /// <summary>从完整schema v2 JSON文件读取工况，执行严格字段、类型及单位契约校验。</summary>
    public static SimulationConfig LoadConfiguration(string jsonPath)
        => Simulator.LoadConfiguration(jsonPath);

    /// <summary>直接解析完整schema v2 JSON文本；未知字段、非法压力和流量均报错。</summary>
    public static SimulationConfig ParseConfiguration(string json)
        => SimulationConfig.Parse(json);

    /// <summary>预测并成对保存JSON/PNG。取消检查发生在闭环迭代之间。</summary>
    public Task<SimulationResult> PredictAsync(
        SimulationConfig config,
        SolverOptions? options = null,
        CancellationToken cancellationToken = default)
        => _simulator.SimulateAsync(config, options, cancellationToken);

    /// <summary>同步预测入口。GUI应用应使用PredictAsync避免阻塞界面。</summary>
    public SimulationResult Predict(
        SimulationConfig config,
        SolverOptions? options = null,
        CancellationToken cancellationToken = default)
        => PredictAsync(config, options, cancellationToken).GetAwaiter().GetResult();

    /// <summary>解析JSON文本后使用相同预测与留档流程。</summary>
    public Task<SimulationResult> PredictJsonAsync(
        string configJson,
        SolverOptions? options = null,
        CancellationToken cancellationToken = default)
        => PredictAsync(ParseConfiguration(configJson), options, cancellationToken);

    /// <summary>JSON文本的同步预测入口。</summary>
    public SimulationResult PredictJson(
        string configJson,
        SolverOptions? options = null,
        CancellationToken cancellationToken = default)
        => PredictJsonAsync(configJson, options, cancellationToken).GetAwaiter().GetResult();

    /// <summary>构造便于宿主显示或交换的露点和六项热负荷摘要。</summary>
    public static PredictionSummary Summarize(SimulationResult result)
        => PredictionSummary.FromResult(result);

    /// <summary>等待在途计算结束后释放ONNX会话；先取消并等待宿主任务再调用。</summary>
    public void Dispose() => _simulator.Dispose();
}

/// <summary>
/// 最终Supply和六项热负荷摘要。空露点配合DryLimit=true表示零湿度干燥极限；
/// TotalThermalLoadKW为冷热负荷之和，没有折算设备COP或热源效率。
/// </summary>
public sealed record PredictionSummary(
    [property: JsonPropertyName("model_name")] string ModelName,
    [property: JsonPropertyName("converged")] bool Converged,
    [property: JsonPropertyName("iterations")] int Iterations,
    [property: JsonPropertyName("supply_T_degC")] double SupplyTemperatureC,
    [property: JsonPropertyName("supply_W_g_per_kg")] double SupplyHumidityGPerKg,
    [property: JsonPropertyName("supply_Tdp_degC")] double? SupplyDewPointC,
    [property: JsonPropertyName("dry_limit")] bool DryLimit,
    [property: JsonPropertyName("cooling_kW")] double CoolingLoadKW,
    [property: JsonPropertyName("heating_kW")] double HeatingLoadKW,
    [property: JsonPropertyName("thermal_load_sum_kW")] double TotalThermalLoadKW,
    [property: JsonPropertyName("energy_breakdown_kW")] IReadOnlyDictionary<string, double> EnergyBreakdownKW,
    [property: JsonPropertyName("warnings")] IReadOnlyList<string> Warnings,
    [property: JsonPropertyName("result_json")] string? ResultJsonPath,
    [property: JsonPropertyName("dashboard_png")] string? DashboardPngPath)
{
    /// <summary>从完整前向结果提取摘要，不改变空气状态或热负荷。</summary>
    public static PredictionSummary FromResult(SimulationResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        return new(PredictionModel.ModelName, result.Converged, result.Iterations,
            result.Supply.TemperatureC, result.Supply.HumidityGPerKg,
            result.Supply.DewPointOrNull, result.Supply.HumidityGPerKg == 0,
            result.CoolingLoadKW, result.HeatingLoadKW,
            result.CoolingLoadKW + result.HeatingLoadKW,
            new Dictionary<string, double>(result.EnergyBreakdownKW),
            result.Warnings.ToArray(), result.ResultJsonPath, result.DashboardPngPath);
    }

    /// <summary>输出标准JSON；零湿度露点写null，不写非标准Infinity。</summary>
    public string ToJson(bool indented = true)
        => JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = indented });
}
