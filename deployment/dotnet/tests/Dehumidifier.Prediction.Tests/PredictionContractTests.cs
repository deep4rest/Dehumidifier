using System.Text.Json;
using System.Text.Json.Nodes;
using Dehumidifier.Prediction;
using Dehumidifier.Simulation;
using Xunit;

namespace Dehumidifier.Prediction.Tests;

public sealed class PredictionContractTests
{
    [Fact]
    public void JsonInterfaceUsesStrictConfigurationParser()
    {
        var json = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "examples", "default_case.json"));
        var root = JsonNode.Parse(json)!;
        root["control_setpoints"]!["supply_T_target_degC"] = -35.0;
        Assert.Equal(-35.0, PredictionModel.ParseConfiguration(root.ToJsonString()).ControlSetpoints.SupplyTemperatureTargetC);
        root["control_setpoints"]!["supply_temperature_typo"] = -35.0;
        var error = Assert.Throws<SimulationException>(() => PredictionModel.ParseConfiguration(root.ToJsonString()));
        Assert.Equal("CONFIG_INVALID", error.Code);
    }

    [Theory]
    [InlineData("not JSON")]
    [InlineData("{}")]
    public void InvalidJsonFailsBeforeModelCalculation(string json)
    {
        Assert.Equal("CONFIG_INVALID", Assert.Throws<SimulationException>(() => PredictionModel.ParseConfiguration(json)).Code);
    }

    [Fact]
    public void DryLimitSerializesAsNullAndNeverAsZeroDegrees()
    {
        var summary = PredictionModel.Summarize(CreateResult(0.0, true));
        Assert.True(summary.DryLimit);
        Assert.Null(summary.SupplyDewPointC);
        using var json = JsonDocument.Parse(summary.ToJson());
        Assert.Equal(JsonValueKind.Null, json.RootElement.GetProperty("supply_Tdp_degC").ValueKind);
        Assert.True(json.RootElement.GetProperty("dry_limit").GetBoolean());
    }

    [Fact]
    public void SummaryKeepsSixSignedLoadsAndPhysicalThermalSum()
    {
        var result = CreateResult(0.044, true);
        var summary = PredictionModel.Summarize(result);
        Assert.Equal(6, summary.EnergyBreakdownKW.Count);
        Assert.Equal(-3.0, summary.EnergyBreakdownKW["CC3_kW"]);
        Assert.Equal(21.0, summary.TotalThermalLoadKW);
        Assert.Equal(result.Supply.DewPointC, summary.SupplyDewPointC);
        Assert.False(summary.DryLimit);
        result.EnergyBreakdownKW["CC3_kW"] = -99;
        Assert.Equal(-3, summary.EnergyBreakdownKW["CC3_kW"]);
    }

    [Fact]
    public void UnconvergedResultIsExposedWithoutClaimingSuccess()
    {
        var summary = PredictionModel.Summarize(CreateResult(0.044, false));
        Assert.False(summary.Converged);
        Assert.Equal(60, summary.Iterations);
    }

    private static SimulationResult CreateResult(double humidity, bool converged)
    {
        var state = new AirState(-35, humidity, 1.0, 101325);
        var rotor = new RotorOutputs(15, humidity, 60, 3, 40, 10, state.DewPointOrNull);
        return new SimulationResult
        {
            Convergence = new ConvergenceInfo { Converged = converged, Iterations = converged ? 16 : 60 },
            Supply = state, Rotor1Outputs = rotor, Rotor2Outputs = rotor,
            IntermediateStates = new() { ["Supply"] = state },
            CoolingLoadKW = 6, HeatingLoadKW = 15,
            EnergyBreakdownKW = new()
            {
                ["CC1_kW"] = -1, ["CC2_kW"] = -2, ["CC3_kW"] = -3,
                ["Heater_Low_kW"] = 4, ["Heater_OA_kW"] = 5, ["Heater_Post_kW"] = 6,
            },
        };
    }
}
