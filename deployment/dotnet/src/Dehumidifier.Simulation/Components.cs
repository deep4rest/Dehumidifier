namespace Dehumidifier.Simulation;

internal sealed record RotorInputs(
    double ProcessInletTemperatureC,
    double ProcessInletHumidityGPerKg,
    double RegenTemperatureC,
    double CoolBlowInletTemperatureC,
    double Rpm,
    double ThicknessMm,
    double DiameterMm,
    double ProcessFlowM3H,
    double CoolBlowFlowM3H,
    double RegenFlowM3H,
    double RegenInletHumidityGPerKg,
    double PressurePa);

internal static class Components
{
    internal static (AirState State, double QKW) CoolingCoil(AirState inlet, double targetC)
    {
        if (targetC >= inlet.TemperatureC) return (inlet, 0.0);
        var saturation = Psychrometrics.SaturationHumidityRatioGPerKg(targetC, inlet.PressurePa);
        var outletHumidity = saturation < inlet.HumidityGPerKg ? saturation : inlet.HumidityGPerKg;
        var outlet = new AirState(targetC, outletHumidity, inlet.DryAirMassFlowKgPerS, inlet.PressurePa);
        return (outlet, inlet.DryAirMassFlowKgPerS * (outlet.EnthalpyKJPerKg - inlet.EnthalpyKJPerKg));
    }

    internal static (AirState State, double QKW) Heater(AirState inlet, double targetC)
    {
        if (targetC <= inlet.TemperatureC) return (inlet, 0.0);
        var outlet = new AirState(targetC, inlet.HumidityGPerKg, inlet.DryAirMassFlowKgPerS, inlet.PressurePa);
        return (outlet, inlet.DryAirMassFlowKgPerS * (outlet.EnthalpyKJPerKg - inlet.EnthalpyKJPerKg));
    }

    internal static AirState Mix(AirState first, AirState second)
    {
        if (first.PressurePa != second.PressurePa) throw new SimulationException("PRESSURE_MISMATCH", "混风两股气流压力必须一致");
        var flow = first.DryAirMassFlowKgPerS + second.DryAirMassFlowKgPerS;
        if (!double.IsFinite(flow) || flow <= 0) throw new SimulationException("INVALID_FLOW", $"混风两股流量之和 <= 0: {flow:G17}");
        var humidity = (first.DryAirMassFlowKgPerS * first.HumidityGPerKg + second.DryAirMassFlowKgPerS * second.HumidityGPerKg) / flow;
        var enthalpy = (first.DryAirMassFlowKgPerS * first.EnthalpyKJPerKg + second.DryAirMassFlowKgPerS * second.EnthalpyKJPerKg) / flow;
        return new AirState(Psychrometrics.TemperatureFromEnthalpy(enthalpy, humidity), humidity, flow, first.PressurePa);
    }
}
