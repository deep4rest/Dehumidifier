using System.Text.Json;
using System.Text.Json.Serialization;

namespace Dehumidifier.Simulation;

public sealed record AirStreamConfig(
    [property: JsonPropertyName("T_degC")] double TemperatureC,
    [property: JsonPropertyName("W_g_per_kg")] double HumidityGPerKg,
    [property: JsonPropertyName("flow_m3h")] double FlowM3H);

public sealed record RotorConfig(
    [property: JsonPropertyName("thickness_mm")] double ThicknessMm,
    [property: JsonPropertyName("diameter_mm")] double DiameterMm,
    [property: JsonPropertyName("process_airspeed_m_s")] double ProcessAirspeedMS,
    [property: JsonPropertyName("cool_blow_airspeed_m_s")] double CoolBlowAirspeedMS,
    [property: JsonPropertyName("cool_blow_flow_m3h")] double CoolBlowFlowM3H,
    [property: JsonPropertyName("regen_angle_deg")] double RegenAngleDeg,
    [property: JsonPropertyName("cool_blow_angle_deg")] double CoolBlowAngleDeg);

public sealed record ControlSetpointsConfig(
    [property: JsonPropertyName("cc1_T_outlet_degC")] double Cc1OutletC,
    [property: JsonPropertyName("cc2_T_outlet_degC")] double Cc2OutletC,
    [property: JsonPropertyName("heater_low_T_degC")] double HeaterLowC,
    [property: JsonPropertyName("heater_oa_T_degC")] double HeaterOaC,
    [property: JsonPropertyName("rotor1_rpm")] double Rotor1Rpm,
    [property: JsonPropertyName("rotor2_rpm")] double Rotor2Rpm);

public sealed class SimulationConfig
{
    [JsonPropertyName("schema_version")] public int SchemaVersion { get; init; }
    [JsonPropertyName("psychrometrics")] public required PsychrometricsConfig Psychrometrics { get; init; }
    [JsonPropertyName("flow_reference")] public required FlowReferenceConfig FlowReference { get; init; }
    [JsonPropertyName("outdoor_air")] public required AirStreamConfig OutdoorAir { get; init; }
    [JsonPropertyName("return_air")] public required AirStreamConfig ReturnAir { get; init; }
    [JsonPropertyName("regen_outdoor_air")] public required AirStreamConfig RegenOutdoorAir { get; init; }
    [JsonPropertyName("system_params")] public required SystemConfig SystemParams { get; init; }
    [JsonPropertyName("control_setpoints")] public required ControlSetpointsConfig ControlSetpoints { get; init; }
    [JsonIgnore] public string SourceName { get; init; } = "case";
    [JsonIgnore] public string? SourcePath { get; init; }

    [JsonIgnore]
    public double ReferenceDryAirDensityKgPerM3
        => ReferenceDensity(Psychrometrics.PressurePa);

    /// <summary>唯一的配置风量换算，按室外参考状态得到 kg 干空气/s。</summary>
    public double M3HToKgDryAirPerS(double flowM3H)
        => flowM3H * ReferenceDryAirDensityKgPerM3 / 3600.0;

    internal double ReferenceDensity(double pressurePa)
    {
        var w = OutdoorAir.HumidityGPerKg / 1000.0;
        var dryPressure = pressurePa * global::Dehumidifier.Simulation.Psychrometrics.ConfigWaterRatio / (global::Dehumidifier.Simulation.Psychrometrics.ConfigWaterRatio + w);
        return dryPressure / (287.05 * (OutdoorAir.TemperatureC + 273.15));
    }

    internal void Validate()
    {
        if (SchemaVersion != 2) throw new SimulationException("CONFIG_INVALID", "配置字段 schema_version 必须等于 2");
        if (Psychrometrics.DewPointBasis != "water") throw new SimulationException("CONFIG_INVALID", "配置字段 psychrometrics.dew_point_basis 必须为 water");
        if (FlowReference.Basis != "outdoor_air") throw new SimulationException("CONFIG_INVALID", "配置字段 flow_reference.basis 必须为 outdoor_air");
        ValidateAir(OutdoorAir, "outdoor_air"); ValidateAir(ReturnAir, "return_air"); ValidateAir(RegenOutdoorAir, "regen_outdoor_air", allowZeroFlow: true);
        ValidateRotor(SystemParams.Rotor1, "system_params.rotor1"); ValidateRotor(SystemParams.Rotor2, "system_params.rotor2");
        if (!double.IsFinite(SystemParams.Fan1KW) || SystemParams.Fan1KW <= 0 || !double.IsFinite(SystemParams.Fan2KW) || SystemParams.Fan2KW <= 0)
            throw new SimulationException("CONFIG_INVALID", "风机功率必须是正的有限值");
        var c = ControlSetpoints;
        if (!double.IsFinite(c.Rotor1Rpm) || c.Rotor1Rpm <= 0 || !double.IsFinite(c.Rotor2Rpm) || c.Rotor2Rpm <= 0)
            throw new SimulationException("CONFIG_INVALID", "转轮转速必须是正的有限值");
        var density = ReferenceDensity(Psychrometrics.PressurePa);
        if (!double.IsFinite(density) || density <= 0) throw new SimulationException("CONFIG_INVALID", "室外新风状态无法得到正的有限干空气密度");
        // 企业风量规则（2026-09-17）：Heater_OA 入口目标 = R1 处理出口 / 3；
        // 可用量 >= 目标时漏风至目标，否则按 regen_outdoor_air 补风（以配置为上限）。
        var rotor1Process = OutdoorAir.FlowM3H - SystemParams.Rotor1.CoolBlowFlowM3H;
        var f = new Dictionary<string, double>
        {
            ["rotor1_process"] = rotor1Process,
            ["rotor2_regen"] = SystemParams.Rotor2.CoolBlowFlowM3H,
            ["rotor2_process"] = OutdoorAir.FlowM3H - SystemParams.Rotor1.CoolBlowFlowM3H + ReturnAir.FlowM3H - SystemParams.Rotor2.CoolBlowFlowM3H,
            ["rotor1_regen"] = Math.Min(rotor1Process / 3.0, SystemParams.Rotor2.CoolBlowFlowM3H + RegenOutdoorAir.FlowM3H + SystemParams.Rotor1.CoolBlowFlowM3H),
        };
        foreach (var item in f) if (item.Value <= 0) throw new SimulationException("CONFIG_INVALID", $"拓扑派生流量 {item.Key}={item.Value:g} m³/h 必须 > 0");
    }

    private static void ValidateAir(AirStreamConfig air, string path, bool allowZeroFlow = false)
    {
        if (!double.IsFinite(air.TemperatureC) || air.TemperatureC <= -273.15) throw new SimulationException("CONFIG_INVALID", $"配置字段 {path}.T_degC 必须高于绝对零度");
        if (!double.IsFinite(air.HumidityGPerKg) || air.HumidityGPerKg < 0) throw new SimulationException("CONFIG_INVALID", $"配置字段 {path}.W_g_per_kg 必须 >= 0");
        if (!double.IsFinite(air.FlowM3H) || (allowZeroFlow ? air.FlowM3H < 0 : air.FlowM3H <= 0))
            throw new SimulationException("CONFIG_INVALID", $"配置字段 {path}.flow_m3h 必须 {(allowZeroFlow ? ">= 0" : "> 0")}");
    }
    private static void ValidateRotor(RotorConfig r, string path)
    {
        var values = new (string, double)[] { ("thickness_mm", r.ThicknessMm), ("diameter_mm", r.DiameterMm), ("process_airspeed_m_s", r.ProcessAirspeedMS), ("cool_blow_airspeed_m_s", r.CoolBlowAirspeedMS), ("cool_blow_flow_m3h", r.CoolBlowFlowM3H), ("regen_angle_deg", r.RegenAngleDeg), ("cool_blow_angle_deg", r.CoolBlowAngleDeg) };
        foreach (var (name, value) in values) if (!double.IsFinite(value) || value <= 0) throw new SimulationException("CONFIG_INVALID", $"配置字段 {path}.{name} 必须 > 0");
    }

    internal static SimulationConfig Parse(string json, string? sourcePath = null)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;
            RequireObject(root, "<root>");
            Exact(root, new[] { "schema_version", "psychrometrics", "flow_reference", "outdoor_air", "return_air", "regen_outdoor_air", "system_params", "control_setpoints" }, "");
            var psych = root.GetProperty("psychrometrics"); Exact(psych, new[] { "pressure_Pa", "dew_point_basis" }, "psychrometrics");
            var flow = root.GetProperty("flow_reference"); Exact(flow, new[] { "basis" }, "flow_reference");
            var system = root.GetProperty("system_params"); Exact(system, new[] { "fan1_P_kW", "fan2_P_kW", "rotor1", "rotor2" }, "system_params");
            var controls = root.GetProperty("control_setpoints"); Exact(controls, new[] { "cc1_T_outlet_degC", "cc2_T_outlet_degC", "heater_low_T_degC", "heater_oa_T_degC", "rotor1_rpm", "rotor2_rpm" }, "control_setpoints");
            var result = new SimulationConfig
            {
                SchemaVersion = Int(root, "schema_version", "schema_version"),
                Psychrometrics = new PsychrometricsConfig(Number(psych, "pressure_Pa", "psychrometrics.pressure_Pa"), String(psych, "dew_point_basis", "psychrometrics.dew_point_basis")),
                FlowReference = new FlowReferenceConfig(String(flow, "basis", "flow_reference.basis")),
                OutdoorAir = Air(root, "outdoor_air"), ReturnAir = Air(root, "return_air"), RegenOutdoorAir = Air(root, "regen_outdoor_air"),
                SystemParams = new SystemConfig(Number(system, "fan1_P_kW", "system_params.fan1_P_kW"), Number(system, "fan2_P_kW", "system_params.fan2_P_kW"), Rotor(system, "rotor1", "system_params.rotor1"), Rotor(system, "rotor2", "system_params.rotor2")),
                ControlSetpoints = new ControlSetpointsConfig(Number(controls, "cc1_T_outlet_degC", "control_setpoints.cc1_T_outlet_degC"), Number(controls, "cc2_T_outlet_degC", "control_setpoints.cc2_T_outlet_degC"), Number(controls, "heater_low_T_degC", "control_setpoints.heater_low_T_degC"), Number(controls, "heater_oa_T_degC", "control_setpoints.heater_oa_T_degC"), Number(controls, "rotor1_rpm", "control_setpoints.rotor1_rpm"), Number(controls, "rotor2_rpm", "control_setpoints.rotor2_rpm")),
                SourcePath = sourcePath, SourceName = sourcePath is null ? "case" : Path.GetFileNameWithoutExtension(sourcePath),
            };
            result.Validate();
            return result;
        }
        catch (SimulationException) { throw; }
        catch (Exception ex) { throw new SimulationException("CONFIG_INVALID", "工况 JSON 无法解析或字段类型错误", ex); }
    }

    private static AirStreamConfig Air(JsonElement root, string name)
    { var value = root.GetProperty(name); Exact(value, new[] { "T_degC", "W_g_per_kg", "flow_m3h" }, name); return new AirStreamConfig(Number(value, "T_degC", $"{name}.T_degC"), Number(value, "W_g_per_kg", $"{name}.W_g_per_kg"), Number(value, "flow_m3h", $"{name}.flow_m3h")); }
    private static RotorConfig Rotor(JsonElement root, string name, string path)
    { var value = root.GetProperty(name); Exact(value, new[] { "thickness_mm", "diameter_mm", "process_airspeed_m_s", "cool_blow_airspeed_m_s", "cool_blow_flow_m3h", "regen_angle_deg", "cool_blow_angle_deg" }, path); return new RotorConfig(Number(value, "thickness_mm", $"{path}.thickness_mm"), Number(value, "diameter_mm", $"{path}.diameter_mm"), Number(value, "process_airspeed_m_s", $"{path}.process_airspeed_m_s"), Number(value, "cool_blow_airspeed_m_s", $"{path}.cool_blow_airspeed_m_s"), Number(value, "cool_blow_flow_m3h", $"{path}.cool_blow_flow_m3h"), Number(value, "regen_angle_deg", $"{path}.regen_angle_deg"), Number(value, "cool_blow_angle_deg", $"{path}.cool_blow_angle_deg")); }
    private static void RequireObject(JsonElement value, string path) { if (value.ValueKind != JsonValueKind.Object) throw new SimulationException("CONFIG_INVALID", $"配置字段 {path} 必须是对象"); }
    private static void Exact(JsonElement value, IEnumerable<string> required, string path)
    { RequireObject(value, path); var set = required.ToHashSet(StringComparer.Ordinal); foreach (var r in set) if (!value.TryGetProperty(r, out _)) throw new SimulationException("CONFIG_INVALID", $"缺少配置字段：{(path.Length == 0 ? r : path + "." + r)}"); foreach (var p in value.EnumerateObject()) if (!set.Contains(p.Name)) throw new SimulationException("CONFIG_INVALID", $"未知配置字段：{(path.Length == 0 ? p.Name : path + "." + p.Name)}"); }
    private static double Number(JsonElement value, string name, string path) { if (!value.TryGetProperty(name, out var item) || item.ValueKind != JsonValueKind.Number || !item.TryGetDouble(out var number) || !double.IsFinite(number)) throw new SimulationException("CONFIG_INVALID", $"配置字段 {path} 必须是有限数值"); return number; }
    private static int Int(JsonElement value, string name, string path) { var number = Number(value, name, path); if (number != Math.Truncate(number)) throw new SimulationException("CONFIG_INVALID", $"配置字段 {path} 必须是整数"); return checked((int)number); }
    private static string String(JsonElement value, string name, string path) { if (!value.TryGetProperty(name, out var item) || item.ValueKind != JsonValueKind.String) throw new SimulationException("CONFIG_INVALID", $"配置字段 {path} 必须是字符串"); return item.GetString()!; }
}

public sealed record PsychrometricsConfig(
    [property: JsonPropertyName("pressure_Pa")] double PressurePa,
    [property: JsonPropertyName("dew_point_basis")] string DewPointBasis);

public sealed record FlowReferenceConfig([property: JsonPropertyName("basis")] string Basis);

public sealed record SystemConfig(
    [property: JsonPropertyName("fan1_P_kW")] double Fan1KW,
    [property: JsonPropertyName("fan2_P_kW")] double Fan2KW,
    [property: JsonPropertyName("rotor1")] RotorConfig Rotor1,
    [property: JsonPropertyName("rotor2")] RotorConfig Rotor2);
