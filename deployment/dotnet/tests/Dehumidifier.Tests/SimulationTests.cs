using System.Text.Json;
using System.Text.Json.Nodes;
using System.Reflection;
using Dehumidifier.Simulation;
using Xunit;

namespace Dehumidifier.Tests;

public sealed class SimulationTests
{
    private static SimulationConfig LoadModifiedConfiguration(Action<JsonObject> modify)
    {
        var source = Path.Combine(AppContext.BaseDirectory, "examples", "qy1070_case.json");
        var root = JsonNode.Parse(File.ReadAllText(source))!.AsObject();
        modify(root);
        var path = Path.Combine(Path.GetTempPath(), $"dehumidifier-config-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        try { return Simulator.LoadConfiguration(path); }
        finally { File.Delete(path); }
    }

    [Fact]
    public void PsychrometricsPreserveDryLimitAndPressure()
    {
        Assert.True(double.IsNegativeInfinity(Psychrometrics.DewPointWaterC(0, 101325)));
        var dew = Psychrometrics.DewPointWaterC(Psychrometrics.SaturationHumidityRatioGPerKg(10, 80000), 80000);
        Assert.Equal(10, dew, 8);
    }

    [Fact]
    public void DewPointUsesEnterpriseDptAhTable()
    {
        // 表项整点精确命中（DptAH[i] 对应露点 -i ℃）。
        Assert.Equal(0.0, Psychrometrics.DewPointWaterC(3.764));
        Assert.Equal(-1.0, Psychrometrics.DewPointWaterC(3.466));
        Assert.Equal(-87.0, Psychrometrics.DewPointWaterC(0.0001));
        // 企业示例：露点 -0.5 ℃ 的含湿量 = 0.5*(DptAH[0]+DptAH[1])，双向一致。
        var expectedW = 0.5 * (3.764 + 3.466);
        Assert.Equal(expectedW, Psychrometrics.HumidityRatioFromDewPointC(-0.5), 12);
        Assert.Equal(-0.5, Psychrometrics.DewPointWaterC(expectedW), 9);
        // 表内 W -> Tdp -> W 精确还原。
        foreach (var w in new[] { 3.0, 1.5, 0.5, 0.1, 0.001, 0.0001 })
        {
            var tdp = Psychrometrics.DewPointWaterC(w);
            Assert.Equal(w, Psychrometrics.HumidityRatioFromDewPointC(tdp), 9);
        }
        // 干燥极限互逆。
        Assert.True(double.IsNegativeInfinity(Psychrometrics.DewPointWaterC(0.0)));
        Assert.Equal(0.0, Psychrometrics.HumidityRatioFromDewPointC(double.NegativeInfinity));
        // 低于表下限按最后一段斜率线性外推。
        Assert.Equal(-89.5, Psychrometrics.DewPointWaterC(0.00005), 9);
        // 超出表上界（正露点）按公式精确反演。
        var wSat20 = Psychrometrics.SaturationHumidityRatioGPerKg(20, 101325);
        Assert.Equal(20.0, Psychrometrics.DewPointWaterC(wSat20, 101325), 6);
    }

    [Fact]
    public void EveryEnterpriseDptAhIntegerPointRoundTrips()
    {
        var field = typeof(Psychrometrics).GetField("DptAhTable", BindingFlags.Static | BindingFlags.NonPublic);
        var table = Assert.IsType<double[]>(field?.GetValue(null));
        Assert.Equal(88, table.Length);
        for (var index = 0; index < table.Length; index++)
        {
            Assert.Equal(-index, Psychrometrics.DewPointWaterC(table[index]), 10);
            Assert.Equal(table[index], Psychrometrics.HumidityRatioFromDewPointC(-index), 12);
        }
    }

    [Fact]
    public void AssemblyVersionDistinguishesTheEnterpriseFixPackage()
    {
        var version = typeof(Simulator).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        Assert.StartsWith("1.0.1", version);
    }

    [Fact]
    public void ConfigurationRejectsUnknownFieldAndWrongSchema()
    {
        var json = "{\"schema_version\":3}";
        Assert.Throws<SimulationException>(() => Simulator.LoadConfiguration(Path.GetTempFileName()));
        var path = Path.GetTempFileName(); File.WriteAllText(path, json);
        try { var ex = Assert.Throws<SimulationException>(() => Simulator.LoadConfiguration(path)); Assert.Equal("CONFIG_INVALID", ex.Code); }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SolverDefaultsAreTheFrozenContract()
    {
        var options = new SolverOptions();
        Assert.Equal(60, options.MaxIterations); Assert.Equal(0.8, options.Relaxation); Assert.Equal(1e-4, options.TemperatureToleranceC); Assert.Equal(1e-6, options.HumidityToleranceGPerKg);
    }

    [Fact]
    public async Task CurrentModelsConvergeAndWritePairedArtifacts()
    {
        var models = Path.Combine(AppContext.BaseDirectory, "models");
        var config = Simulator.LoadConfiguration(Path.Combine(AppContext.BaseDirectory, "examples", "qy1070_case.json"));
        var results = Path.Combine(Path.GetTempPath(), "dehumidifier-dotnet-test", Guid.NewGuid().ToString("N"));
        using var simulator = new Simulator(models, results);
        var result = await simulator.SimulateAsync(config);
        Assert.True(result.Converged);
        // 2026-09-17 Windows ONNX 实测为 13 次；Python/PyTorch 基准为 12 次。
        // 此断言固定当前 Windows 行为；两侧迭代差异由迁移报告单独保留。
        Assert.Equal(13, result.Iterations);
        Assert.True(File.Exists(result.ResultJsonPath));
        Assert.True(File.Exists(result.DashboardPngPath));
        Assert.Equal(4, result.EnergyBreakdownKW.Count);
        // 企业风量规则：qy1070 走补风分支，Heater_OA 入口 = R1 处理出口 / 3 = 1111 m³/h。
        Assert.NotNull(result.RegenMixRule);
        Assert.Equal("make_up", result.RegenMixRule!.Branch);
        Assert.Equal(1111.0, result.RegenMixRule.ThresholdM3H, 6);
        Assert.Equal(1111.0, result.RegenMixRule.HeaterOaInletM3H, 6);
        Assert.Equal(277.0, result.RegenMixRule.RegenOaUsedM3H, 6);
        Assert.Equal(0.0, result.RegenMixRule.LeakedM3H, 9);
        Assert.Equal(0.0, result.RegenMixRule.ShortfallKgPerS, 9);
    }

    [Fact]
    public async Task ZeroRegenMakeupContinuesWithWarningAndRecordedShortfall()
    {
        var models = Path.Combine(AppContext.BaseDirectory, "models");
        var config = LoadModifiedConfiguration(root => root["regen_outdoor_air"]!["flow_m3h"] = 0.0);
        using var simulator = new Simulator(models, Path.Combine(Path.GetTempPath(), "dehumidifier-dotnet-zero-makeup", Guid.NewGuid().ToString("N")));

        var result = await simulator.SimulateForValidationAsync(config);

        Assert.True(result.Converged);
        Assert.NotNull(result.RegenMixRule);
        Assert.Equal("make_up", result.RegenMixRule!.Branch);
        Assert.Equal(0.0, result.RegenMixRule.RegenOaConfiguredKgPerS, 12);
        Assert.Equal(0.0, result.RegenMixRule.RegenOaUsedKgPerS, 12);
        Assert.True(result.RegenMixRule.ShortfallKgPerS > 0.0);
        Assert.Contains(result.Warnings, warning => warning.Contains("再生补风不足", StringComparison.Ordinal));
    }

    [Fact]
    public void NegativeRegenMakeupIsStillRejected()
    {
        var source = Path.Combine(AppContext.BaseDirectory, "examples", "qy1070_case.json");
        var root = JsonNode.Parse(File.ReadAllText(source))!.AsObject();
        root["regen_outdoor_air"]!["flow_m3h"] = -0.1;
        var path = Path.Combine(Path.GetTempPath(), $"dehumidifier-negative-flow-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, root.ToJsonString());
        try
        {
            var exception = Assert.Throws<SimulationException>(() => Simulator.LoadConfiguration(path));
            Assert.Equal("CONFIG_INVALID", exception.Code);
            Assert.Contains("regen_outdoor_air.flow_m3h 必须 >= 0", exception.Message, StringComparison.Ordinal);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task EnterpriseReturnAirDewPointSurvivesConfigStateAndJsonRecording()
    {
        var models = Path.Combine(AppContext.BaseDirectory, "models");
        var humidity = Psychrometrics.HumidityRatioFromDewPointC(-35.0, 101325.0);
        Assert.Equal(0.138, humidity, 12);
        var config = LoadModifiedConfiguration(root =>
        {
            root["return_air"]!["T_degC"] = 25.0;
            root["return_air"]!["W_g_per_kg"] = humidity;
        });
        var results = Path.Combine(Path.GetTempPath(), "dehumidifier-dotnet-ra-dewpoint", Guid.NewGuid().ToString("N"));
        using var simulator = new Simulator(models, results);

        var result = await simulator.SimulateAsync(config);
        var returnAir = result.IntermediateStates["RA"];

        Assert.Equal(0.138, returnAir.HumidityGPerKg, 12);
        Assert.Equal(-35.0, returnAir.DewPointC, 9);
        Assert.NotNull(result.ResultJsonPath);
        using var document = JsonDocument.Parse(File.ReadAllText(result.ResultJsonPath!));
        var recorded = document.RootElement.GetProperty("result").GetProperty("intermediate_states").GetProperty("RA");
        Assert.Equal(0.138, recorded.GetProperty("W_g_per_kg").GetDouble(), 12);
        Assert.Equal(-35.0, recorded.GetProperty("Tdp_degC").GetDouble(), 9);
    }

    [Fact]
    public async Task CancellationDoesNotPoisonReusableSimulator()
    {
        var models = Path.Combine(AppContext.BaseDirectory, "models");
        var config = Simulator.LoadConfiguration(Path.Combine(AppContext.BaseDirectory, "examples", "qy1070_case.json"));
        using var simulator = new Simulator(models, Path.Combine(Path.GetTempPath(), "dehumidifier-dotnet-cancel", Guid.NewGuid().ToString("N")));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => simulator.SimulateForValidationAsync(config, cancellationToken: cancellation.Token));
        var result = await simulator.SimulateForValidationAsync(config);
        Assert.True(result.Converged);
    }
}
