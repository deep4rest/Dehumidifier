using System.Text.Json;
using System.Text.Json.Nodes;
using Dehumidifier.Simulation;
using Xunit;

namespace Dehumidifier.Tests;

public sealed class TerminalConditioningTests
{
    [Theory]
    [InlineData(null)]
    [InlineData(20.0)]
    public void MissingOrReachedTargetBypassesBothComponents(double? target)
    {
        var inlet = new AirState(20.0, 2.0, 1.5, 90000.0);
        var result = Components.TerminalConditioning(inlet, target);
        Assert.Equal(inlet, result.AfterCooling);
        Assert.Equal(inlet, result.Supply);
        Assert.Equal(0.0, result.CoolingQKW);
        Assert.Equal(0.0, result.HeatingQKW);
    }

    [Theory]
    [InlineData(15.0, true)]
    [InlineData(25.0, false)]
    public void TargetTemperatureChoosesCoolingOrHeating(double target, bool cooling)
    {
        var inlet = new AirState(20.0, 2.0, 1.5, 90000.0);
        var result = Components.TerminalConditioning(inlet, target);
        Assert.Equal(target, result.Supply.TemperatureC);
        Assert.Equal(inlet.HumidityGPerKg, result.Supply.HumidityGPerKg);
        Assert.Equal(inlet.PressurePa, result.Supply.PressurePa);
        Assert.Equal(inlet.DryAirMassFlowKgPerS, result.Supply.DryAirMassFlowKgPerS);
        Assert.Equal(inlet.DryAirMassFlowKgPerS * (result.Supply.EnthalpyKJPerKg - inlet.EnthalpyKJPerKg), result.CoolingQKW + result.HeatingQKW, 10);
        if (cooling)
        {
            Assert.True(result.CoolingQKW < 0.0);
            Assert.Equal(0.0, result.HeatingQKW);
        }
        else
        {
            Assert.Equal(0.0, result.CoolingQKW);
            Assert.True(result.HeatingQKW > 0.0);
        }
    }

    [Fact]
    public void PostCoolingPreservesCondensationAndPostHeatingKeepsItsHumidity()
    {
        var inlet = new AirState(30.0, 20.0, 1.2, 90000.0);
        var cooling = Components.TerminalConditioning(inlet, 10.0);
        Assert.Equal(Psychrometrics.SaturationHumidityRatioGPerKg(10.0, 90000.0), cooling.Supply.HumidityGPerKg, 10);
        Assert.True(cooling.Supply.HumidityGPerKg < inlet.HumidityGPerKg);
        Assert.Equal(0.0, cooling.HeatingQKW);
        var heating = Components.TerminalConditioning(cooling.Supply, 20.0);
        Assert.Equal(cooling.Supply.HumidityGPerKg, heating.Supply.HumidityGPerKg);
        Assert.Equal(0.0, heating.CoolingQKW);
        Assert.True(heating.HeatingQKW > 0.0);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(-273.15)]
    public void InvalidTerminalTargetsAreRejected(double target)
    {
        Assert.Throws<SimulationException>(() => Components.TerminalConditioning(new AirState(20.0, 2.0, 1.0, 101325.0), target));
    }

    [Theory]
    [InlineData(null)]
    [InlineData(18.0)]
    public void ConfigurationAcceptsOptionalNullableSupplyTarget(double? target)
    {
        var source = Path.Combine(AppContext.BaseDirectory, "examples", "qy1070_case.json");
        var root = JsonNode.Parse(File.ReadAllText(source))!.AsObject();
        var legacy = SimulationConfig.Parse(root.ToJsonString());
        Assert.Null(legacy.ControlSetpoints.SupplyTemperatureTargetC);
        root["control_setpoints"]!["supply_T_target_degC"] = JsonValue.Create(target);
        var configured = SimulationConfig.Parse(root.ToJsonString());
        Assert.Equal(target, configured.ControlSetpoints.SupplyTemperatureTargetC);
        root["control_setpoints"]!["supply_T_target_typo"] = 20.0;
        Assert.Throws<SimulationException>(() => SimulationConfig.Parse(root.ToJsonString()));
    }

    [Fact]
    public void RecordedResultsKeepAllSixLoadsAndTerminalStates()
    {
        var state = new AirState(20.0, 0.0444, 1.0, 101325.0);
        var rotor = new RotorOutputs(20.0, 0.0444, 70.0, 1.0, 40.0, 10.0, -45.0);
        var result = new SimulationResult
        {
            Supply = state,
            IntermediateStates = new() { ["AfterR2_P"] = state, ["AfterCC3"] = state, ["Supply"] = state },
            Rotor1Outputs = rotor, Rotor2Outputs = rotor,
            CoolingLoadKW = 6.0, HeatingLoadKW = 15.0,
            EnergyBreakdownKW = new()
            {
                ["CC1_kW"] = -1.0, ["CC2_kW"] = -2.0, ["CC3_kW"] = -3.0,
                ["Heater_Low_kW"] = 4.0, ["Heater_OA_kW"] = 5.0, ["Heater_Post_kW"] = 6.0,
            },
        };
        var json = JsonSerializer.SerializeToElement(RunRecorder.ToRecord(result));
        var loads = json.GetProperty("energy_breakdown_kW");
        Assert.Equal(6, loads.EnumerateObject().Count());
        Assert.Equal(-3.0, loads.GetProperty("CC3_kW").GetDouble());
        Assert.Equal(6.0, loads.GetProperty("Heater_Post_kW").GetDouble());
        Assert.True(json.GetProperty("intermediate_states").TryGetProperty("AfterR2_P", out _));
        Assert.True(json.GetProperty("intermediate_states").TryGetProperty("AfterCC3", out _));
    }
}
