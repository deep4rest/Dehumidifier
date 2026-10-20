using System.Drawing;
using System.Drawing.Imaging;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace Dehumidifier.Simulation;

internal sealed record SavedRun(string JsonPath, string PngPath);

internal static class RunRecorder
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    internal static SavedRun Save(SimulationConfig config, SimulationResult result, string resultsDirectory)
    {
        Directory.CreateDirectory(resultsDirectory);
        var baseName = $"{DateTimeOffset.Now:yyyyMMdd_HHmmss_fffffff}_{config.SourceName}";
        var final = Path.Combine(resultsDirectory, baseName);
        var suffix = 2;
        while (Directory.Exists(final)) final = Path.Combine(resultsDirectory, $"{baseName}_{suffix++:00}");
        var temporary = final + ".tmp";
        Directory.CreateDirectory(temporary);
        try
        {
            var jsonPath = Path.Combine(temporary, "result.json"); var pngPath = Path.Combine(temporary, "fig1_dashboard.png");
            var document = new { run_id = Path.GetFileName(final), simulated_at = DateTimeOffset.Now.ToString("O"), config_source = config.SourcePath, config, result = ToRecord(result) };
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(document, JsonOptions) + Environment.NewLine);
            WriteDashboard(result, pngPath);
            Directory.Move(temporary, final);
            return new SavedRun(Path.GetFullPath(Path.Combine(final, "result.json")), Path.GetFullPath(Path.Combine(final, "fig1_dashboard.png")));
        }
        catch
        {
            try { if (Directory.Exists(temporary)) Directory.Delete(temporary, true); } catch { }
            throw;
        }
    }

    private static object ToRecord(SimulationResult result)
    {
        var energy = result.EnergyBreakdownKW.Where(x => x.Key is "CC1_kW" or "CC2_kW" or "Heater_Low_kW" or "Heater_OA_kW").ToDictionary(x => x.Key, x => x.Value);
        return new
        {
            convergence = result.Convergence,
            supply_state = State(result.Supply),
            energy_summary_kW = new { cooling_load = result.CoolingLoadKW, heating_load = result.HeatingLoadKW },
            energy_breakdown_kW = energy,
            intermediate_states = result.IntermediateStates.ToDictionary(x => x.Key, x => State(x.Value)),
            rotor1_outputs = Rotor(result.Rotor1Outputs), rotor2_outputs = Rotor(result.Rotor2Outputs),
            regen_mix_rule = RegenMixRule(result.RegenMixRule),
            warnings = result.Warnings,
        };
    }

    private static object? RegenMixRule(RegenMixRuleInfo? rule)
    {
        if (rule is null) return null;
        return new
        {
            rule = "heater_oa_inlet = rotor1_process_outlet / 3",
            branch = rule.Branch,
            threshold_kg_da_s = rule.ThresholdKgPerS,
            available_kg_da_s = rule.AvailableKgPerS,
            regen_oa_configured_kg_da_s = rule.RegenOaConfiguredKgPerS,
            regen_oa_used_kg_da_s = rule.RegenOaUsedKgPerS,
            leaked_kg_da_s = rule.LeakedKgPerS,
            heater_oa_inlet_kg_da_s = rule.HeaterOaInletKgPerS,
            shortfall_kg_da_s = rule.ShortfallKgPerS,
            shortfall_m3h = rule.ShortfallM3H,
            threshold_m3h = rule.ThresholdM3H,
            available_m3h = rule.AvailableM3H,
            regen_oa_used_m3h = rule.RegenOaUsedM3H,
            leaked_m3h = rule.LeakedM3H,
            heater_oa_inlet_m3h = rule.HeaterOaInletM3H,
        };
    }

    private static object State(AirState state) => new
    {
        pressure_Pa = state.PressurePa,
        T_degC = state.TemperatureC,
        W_g_per_kg = state.HumidityGPerKg,
        mdot_kg_da_s = state.DryAirMassFlowKgPerS,
        Tdp_degC = double.IsNegativeInfinity(state.DewPointC) ? (double?)null : state.DewPointC,
        RH_fraction = state.RelativeHumidityFraction,
        h_kJ_per_kg_da = state.EnthalpyKJPerKg,
    };
    private static object Rotor(RotorOutputs value) => new
    {
        process_outlet_T = value.ProcessOutletTemperatureC,
        process_outlet_W = value.ProcessOutletHumidityGPerKg,
        process_outlet_Tdp = value.ProcessOutletDewPointC,
        regen_outlet_T = value.RegenOutletTemperatureC,
        regen_outlet_W = value.RegenOutletHumidityGPerKg,
        cool_blow_outlet_T = value.CoolBlowOutletTemperatureC,
        cool_blow_outlet_W = value.CoolBlowOutletHumidityGPerKg,
    };

    private static void WriteDashboard(SimulationResult result, string path)
    {
        using var bitmap = new Bitmap(1600, 1000, PixelFormat.Format24bppRgb);
        using var graphics = Graphics.FromImage(bitmap);
        graphics.Clear(Color.White);
        using var title = new Font("Microsoft YaHei", 22, FontStyle.Bold);
        using var body = new Font("Microsoft YaHei", 14, FontStyle.Regular);
        using var small = new Font("Microsoft YaHei", 11, FontStyle.Regular);
        using var black = new SolidBrush(Color.FromArgb(30, 30, 30));
        using var blue = new SolidBrush(Color.FromArgb(35, 104, 180));
        using var orange = new SolidBrush(Color.FromArgb(220, 120, 35));
        graphics.DrawString("双转轮除湿机 C# + ONNX 仿真", title, black, 45, 30);
        graphics.DrawString(result.Converged ? $"求解状态：已收敛（{result.Iterations} 次）" : $"求解状态：未收敛（{result.Iterations} 次，仅供调试）", body, result.Converged ? blue : orange, 48, 78);
        graphics.DrawString($"送风：T={result.Supply.TemperatureC:F3} ℃，W={result.Supply.HumidityGPerKg:F5} g/kg，露点={(result.Supply.DewPointOrNull is null ? "干燥极限" : result.Supply.DewPointOrNull.Value.ToString("F3"))} ℃，RH={result.Supply.RelativeHumidityFraction:P2}", body, black, 48, 125);
        graphics.DrawString($"总冷负荷：{result.CoolingLoadKW:F5} kW    总热负荷：{result.HeatingLoadKW:F5} kW", body, black, 48, 170);
        graphics.DrawString("处理侧关键节点", body, blue, 48, 235);
        var names = new[] { "OA", "AfterCC1", "AfterR1_P", "BeforeFan2", "AfterCC2", "Supply" };
        var labels = new[] { "新风入口", "表冷1后", "转轮1处理出", "混风后", "表冷2后", "送风出口" };
        var y = 280;
        for (var i = 0; i < names.Length; i++)
        {
            if (!result.IntermediateStates.TryGetValue(names[i], out var state)) continue;
            graphics.DrawString($"{labels[i],-10} T={state.TemperatureC,8:F3} ℃  W={state.HumidityGPerKg,10:F5} g/kg  Tdp={(double.IsNegativeInfinity(state.DewPointC) ? "干燥极限" : state.DewPointC.ToString("F3")),10} ℃  RH={state.RelativeHumidityFraction,7:P2}", small, black, 60, y);
            y += 42;
        }
        graphics.DrawString("热负荷明细", body, orange, 48, 560);
        y = 610;
        foreach (var key in new[] { "CC1_kW", "CC2_kW", "Heater_Low_kW", "Heater_OA_kW" })
        {
            if (result.EnergyBreakdownKW.TryGetValue(key, out var value)) { graphics.DrawString($"{key,-18} {value,12:F6} kW", body, black, 70, y); y += 40; }
        }
        graphics.DrawString("本图用于部署/数值迁移验证；不代表工程物理验收。", small, orange, 48, 900);
        bitmap.Save(path, ImageFormat.Png);
    }
}
