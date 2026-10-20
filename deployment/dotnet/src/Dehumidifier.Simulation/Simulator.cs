using System.Text.Json;

namespace Dehumidifier.Simulation;

/// <summary>完整双转轮稳态仿真入口。</summary>
public sealed class Simulator : IDisposable
{
    private readonly OnnxRotor _xf;
    private readonly OnnxRotor _main;
    private readonly string _resultsDirectory;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public Simulator(string modelDirectory, string resultsDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory)) throw new SimulationException("MODEL_INVALID", "模型目录不能为空");
        if (string.IsNullOrWhiteSpace(resultsDirectory)) throw new SimulationException("RESULTS_INVALID", "结果目录不能为空");
        _resultsDirectory = Path.GetFullPath(resultsDirectory);
        Directory.CreateDirectory(_resultsDirectory);
        _xf = new OnnxRotor(modelDirectory, "xf");
        try { _main = new OnnxRotor(modelDirectory, "main"); }
        catch { _xf.Dispose(); throw; }
    }

    public static SimulationConfig LoadConfiguration(string jsonPath)
    {
        var source = Path.GetFullPath(jsonPath);
        if (!File.Exists(source)) throw new SimulationException("CONFIG_MISSING", $"工况文件不存在: {source}");
        return SimulationConfig.Parse(File.ReadAllText(source), source);
    }

    public async Task<SimulationResult> SimulateAsync(SimulationConfig config, SolverOptions? options = null, CancellationToken cancellationToken = default)
        => await SimulateInternalAsync(config, options, cancellationToken, saveArtifacts: true).ConfigureAwait(false);

    /// <summary>
    /// Executes the same complete system calculation without writing JSON/PNG artifacts.
    /// This is intended for host-side health checks and performance measurement; normal
    /// application calls should use <see cref="SimulateAsync"/> so every result is archived.
    /// </summary>
    public async Task<SimulationResult> SimulateForValidationAsync(SimulationConfig config, SolverOptions? options = null, CancellationToken cancellationToken = default)
        => await SimulateInternalAsync(config, options, cancellationToken, saveArtifacts: false).ConfigureAwait(false);

    private async Task<SimulationResult> SimulateInternalAsync(SimulationConfig config, SolverOptions? options, CancellationToken cancellationToken, bool saveArtifacts)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        config.Validate();
        var solver = options ?? new SolverOptions(); solver.Validate();
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                var result = Solve(config, solver, cancellationToken);
                if (!saveArtifacts) return result;
                try
                {
                    var paths = RunRecorder.Save(config, result, _resultsDirectory);
                    result.ResultJsonPath = paths.JsonPath;
                    result.DashboardPngPath = paths.PngPath;
                    return result;
                }
                catch (SimulationException) { throw; }
                catch (Exception ex) { throw new SimulationException("RESULT_SAVE_FAILED", "仿真结果留档失败", ex); }
            }, cancellationToken).ConfigureAwait(false);
        }
        finally { _gate.Release(); }
    }

    private SimulationResult Solve(SimulationConfig config, SolverOptions options, CancellationToken token)
    {
        var r1T = config.ControlSetpoints.HeaterOaC;
        var r1W = config.RegenOutdoorAir.HumidityGPerKg;
        var r2T = config.ControlSetpoints.HeaterLowC;
        SimulationResult? result = null;
        var history = new List<ConvergenceHistoryItem>();
        for (var iteration = 1; iteration <= options.MaxIterations; iteration++)
        {
            token.ThrowIfCancellationRequested();
            result = Forward(config, r1T, r1W, r2T);
            var mix3 = result.IntermediateStates["AfterMix3"];
            var r2cb = result.IntermediateStates["AfterR2_CB"];
            var newR1T = Math.Max(mix3.TemperatureC, config.ControlSetpoints.HeaterOaC);
            var newR1W = mix3.HumidityGPerKg;
            var newR2T = Math.Max(r2cb.TemperatureC, config.ControlSetpoints.HeaterLowC);
            var residuals = new Dictionary<string, double>
            {
                ["r1_T_degC"] = Math.Abs(newR1T - r1T),
                ["r1_W_g_per_kg"] = Math.Abs(newR1W - r1W),
                ["r2_T_degC"] = Math.Abs(newR2T - r2T),
            };
            if (residuals.Values.Any(x => !double.IsFinite(x))) throw new SimulationException("SOLVER_INVALID", "联立求解残差包含非有限值");
            var inputs = new Dictionary<string, double> { ["r1_T_degC"] = r1T, ["r1_W_g_per_kg"] = r1W, ["r2_T_degC"] = r2T };
            history.Add(new ConvergenceHistoryItem(iteration, new(inputs), new(residuals)));
            var converged = residuals["r1_T_degC"] <= options.TemperatureToleranceC && residuals["r1_W_g_per_kg"] <= options.HumidityToleranceGPerKg && residuals["r2_T_degC"] <= options.TemperatureToleranceC;
            result.Convergence = MakeConvergence(converged, iteration, residuals, inputs, options, history);
            if (converged) return result;
            r1T += options.Relaxation * (newR1T - r1T);
            r1W += options.Relaxation * (newR1W - r1W);
            r2T += options.Relaxation * (newR2T - r2T);
        }
        if (result is null) throw new SimulationException("SOLVER_FAILED", "求解器没有产生结果");
        result.Convergence = MakeConvergence(false, options.MaxIterations, result.Convergence.Residuals, result.Convergence.LastEvaluatedInputs, options, history);
        result.Warnings.Add("联立稳态求解未收敛；已保存最后一次已计算结果，仅供调试。");
        return result;
    }

    private static ConvergenceInfo MakeConvergence(bool converged, int iteration, Dictionary<string, double> residuals, Dictionary<string, double> inputs, SolverOptions options, List<ConvergenceHistoryItem> history)
        => new() { Status = converged ? "converged" : "not_converged", Converged = converged, Iterations = iteration, Residuals = new(residuals), Tolerances = new() { ["T_degC"] = options.TemperatureToleranceC, ["W_g_per_kg"] = options.HumidityToleranceGPerKg }, LastEvaluatedInputs = new(inputs), SolverParameters = new() { ["relax"] = options.Relaxation, ["max_iter"] = options.MaxIterations }, History = new(history) };

    private SimulationResult Forward(SimulationConfig config, double r1RegenT, double r1RegenW, double r2RegenT)
    {
        var pressure = config.Psychrometrics.PressurePa;
        var oa = new AirState(config.OutdoorAir.TemperatureC, config.OutdoorAir.HumidityGPerKg, config.M3HToKgDryAirPerS(config.OutdoorAir.FlowM3H), pressure);
        var ra = new AirState(config.ReturnAir.TemperatureC, config.ReturnAir.HumidityGPerKg, config.M3HToKgDryAirPerS(config.ReturnAir.FlowM3H), pressure);
        var regenOa = new AirState(config.RegenOutdoorAir.TemperatureC, config.RegenOutdoorAir.HumidityGPerKg, config.M3HToKgDryAirPerS(config.RegenOutdoorAir.FlowM3H), pressure);
        var p = config.SystemParams; var c = config.ControlSetpoints;
        var mR1Cb = config.M3HToKgDryAirPerS(p.Rotor1.CoolBlowFlowM3H); var mR1P = oa.DryAirMassFlowKgPerS - mR1Cb;
        var mFan2 = mR1P + ra.DryAirMassFlowKgPerS; var mR2Cb = config.M3HToKgDryAirPerS(p.Rotor2.CoolBlowFlowM3H); var mR2P = mFan2 - mR2Cb;
        // 企业风量规则（2026-09-17）：Heater_OA 入口目标 = R1 处理出口 / 3。
        // 可用量（R2 冷吹出 + R1 冷吹出）>= 目标：关闭补风并漏掉多余；
        // 可用量 < 目标：按 regen_outdoor_air 补风，补风量以配置流量为上限。
        var mTarget = mR1P / 3.0;
        var mAvailable = mR2Cb + mR1Cb;
        string branch; double mRegenOaUsed, mLeaked, mR1Rg, mShortfall;
        if (mAvailable >= mTarget)
        {
            branch = "leak"; mRegenOaUsed = 0.0; mLeaked = mAvailable - mTarget; mR1Rg = mTarget; mShortfall = 0.0;
        }
        else
        {
            branch = "make_up"; mRegenOaUsed = Math.Min(mTarget - mAvailable, regenOa.DryAirMassFlowKgPerS);
            mLeaked = 0.0; mR1Rg = mAvailable + mRegenOaUsed;
            // 缺口单独计算，避免 (available + deficit) 的浮点噪声误报。
            mShortfall = Math.Max(0.0, mTarget - mAvailable - mRegenOaUsed);
            if (mShortfall < 1e-9) mShortfall = 0.0;
        }
        if (mR1Cb >= oa.DryAirMassFlowKgPerS) throw new SimulationException("INVALID_FLOW", $"新风转轮冷吹流量必须小于新风总流量");
        if (mR2Cb >= mFan2) throw new SimulationException("INVALID_FLOW", $"主转轮冷吹流量必须小于二级风机出口流量");
        foreach (var v in new[] { mR1P, mR1Cb, mR2P, mR2Cb, mR1Rg }) if (v <= 0 || !double.IsFinite(v)) throw new SimulationException("INVALID_FLOW", "拓扑流量必须为正的有限值");
        var rho = config.ReferenceDryAirDensityKgPerM3;
        var states = new Dictionary<string, AirState> { ["OA"] = oa, ["RA"] = ra, ["RegenOA"] = regenOa };
        var energy = new Dictionary<string, double>();
        states["AfterFan1"] = oa;
        var cc1 = Components.CoolingCoil(oa, c.Cc1OutletC); states["AfterCC1"] = cc1.State; energy["CC1_kW"] = cc1.QKW;
        var r1CbIn = cc1.State.AtFlow(mR1Cb); var r1PIn = cc1.State.AtFlow(mR1P); states["R1_CB_in"] = r1CbIn; states["R1_P_in"] = r1PIn;
        var r1 = PredictRotor(_xf, new RotorInputs(r1PIn.TemperatureC, r1PIn.HumidityGPerKg, r1RegenT, r1CbIn.TemperatureC, c.Rotor1Rpm, p.Rotor1.ThicknessMm, p.Rotor1.DiameterMm, mR1P * 3600 / rho, mR1Cb * 3600 / rho, mR1Rg * 3600 / rho, r1RegenW, pressure), pressure);
        states["AfterR1_P"] = new AirState(r1.ProcessOutletTemperatureC, r1.ProcessOutletHumidityGPerKg, mR1P, pressure); states["AfterR1_CB"] = new AirState(r1.CoolBlowOutletTemperatureC, r1.CoolBlowOutletHumidityGPerKg, mR1Cb, pressure);
        var beforeFan2 = Components.Mix(states["AfterR1_P"], ra); states["BeforeFan2"] = beforeFan2; states["AfterFan2"] = beforeFan2;
        var r2CbIn = beforeFan2.AtFlow(mR2Cb); var cc2In = beforeFan2.AtFlow(mR2P); states["R2_CB_in"] = r2CbIn;
        var cc2 = Components.CoolingCoil(cc2In, c.Cc2OutletC); states["AfterCC2"] = cc2.State; energy["CC2_kW"] = cc2.QKW;
        var r2 = PredictRotor(_main, new RotorInputs(cc2.State.TemperatureC, cc2.State.HumidityGPerKg, r2RegenT, r2CbIn.TemperatureC, c.Rotor2Rpm, p.Rotor2.ThicknessMm, p.Rotor2.DiameterMm, mR2P * 3600 / rho, mR2Cb * 3600 / rho, mR2Cb * 3600 / rho, 0, pressure), pressure);
        var supply = new AirState(r2.ProcessOutletTemperatureC, r2.ProcessOutletHumidityGPerKg, mR2P, pressure); states["Supply"] = supply; states["AfterR2_CB"] = new AirState(r2.CoolBlowOutletTemperatureC, r2.CoolBlowOutletHumidityGPerKg, mR2Cb, pressure);
        var heaterLow = Components.Heater(states["AfterR2_CB"], c.HeaterLowC); states["AfterHeaterLow"] = heaterLow.State; energy["Heater_Low_kW"] = heaterLow.QKW;
        states["AfterR2_RG"] = new AirState(r2.RegenOutletTemperatureC, r2.RegenOutletHumidityGPerKg, mR2Cb, pressure);
        // 再生混风（企业风量规则）：先完全混合 R2 再生出 + R1 冷吹出；
        // leak 分支混合后漏掉多余（成分不变，仅减流量），make_up 分支按上限补风。
        var regenMixed = Components.Mix(states["AfterR2_RG"], states["AfterR1_CB"]);
        AirState mix3;
        if (branch == "leak") mix3 = regenMixed.AtFlow(mR1Rg);
        else mix3 = Components.Mix(regenMixed, regenOa.AtFlow(mRegenOaUsed));
        states["AfterMix3"] = mix3;
        var heaterOa = Components.Heater(mix3, c.HeaterOaC); states["AfterHeaterOA"] = heaterOa.State; energy["Heater_OA_kW"] = heaterOa.QKW;
        states["AfterR1_RG"] = new AirState(r1.RegenOutletTemperatureC, r1.RegenOutletHumidityGPerKg, mR1Rg, pressure);
        var warnings = new List<string> { "当前为模型迁移验证；训练域外输入和水分收支偏差仍需工程复核。" };
        if (mShortfall > 0.0)
            warnings.Add($"再生补风不足：Heater_OA 入口目标 {mTarget:F6} kg/s，实际 {mR1Rg:F6} kg/s（regen_outdoor_air 配置上限 {regenOa.DryAirMassFlowKgPerS:F6} kg/s），缺口 {mShortfall:F6} kg/s。");
        var regenMixRule = new RegenMixRuleInfo(
            Branch: branch,
            ThresholdKgPerS: mTarget, AvailableKgPerS: mAvailable,
            RegenOaConfiguredKgPerS: regenOa.DryAirMassFlowKgPerS,
            RegenOaUsedKgPerS: mRegenOaUsed, LeakedKgPerS: mLeaked,
            HeaterOaInletKgPerS: mR1Rg, ShortfallKgPerS: mShortfall,
            ThresholdM3H: mTarget * 3600 / rho, AvailableM3H: mAvailable * 3600 / rho,
            RegenOaUsedM3H: mRegenOaUsed * 3600 / rho, LeakedM3H: mLeaked * 3600 / rho,
            HeaterOaInletM3H: mR1Rg * 3600 / rho, ShortfallM3H: mShortfall * 3600 / rho);
        return new SimulationResult { Convergence = new(), Supply = supply, IntermediateStates = states, Rotor1Outputs = r1, Rotor2Outputs = r2, CoolingLoadKW = Math.Abs(cc1.QKW) + Math.Abs(cc2.QKW), HeatingLoadKW = heaterOa.QKW + heaterLow.QKW, EnergyBreakdownKW = energy, RegenMixRule = regenMixRule, Warnings = warnings };
    }

    private static RotorOutputs PredictRotor(OnnxRotor model, RotorInputs input, double pressure)
    {
        var values = model.Predict(model.Profile == "xf" ? new[] { input.ThicknessMm, input.DiameterMm, input.ProcessInletTemperatureC, input.ProcessInletHumidityGPerKg, input.ProcessFlowM3H, input.CoolBlowFlowM3H, input.Rpm, input.RegenTemperatureC, input.RegenFlowM3H, input.RegenInletHumidityGPerKg } : new[] { input.ThicknessMm, input.DiameterMm, input.ProcessInletHumidityGPerKg, input.Rpm, input.RegenTemperatureC, input.CoolBlowInletTemperatureC, input.ProcessInletTemperatureC, input.ProcessFlowM3H, input.CoolBlowFlowM3H });
        var processW = Math.Max(0, values[1]); var coolW = Math.Max(0, values[3]); var regenW = Math.Max(0, values[5]);
        return new RotorOutputs(values[0], processW, values[2], coolW, values[4], regenW, double.IsNegativeInfinity(Psychrometrics.DewPointWaterC(processW, pressure)) ? null : Psychrometrics.DewPointWaterC(processW, pressure));
    }

    public void Dispose()
    {
        if (_disposed) return;
        // Wait for an in-flight calculation before releasing ONNX sessions.  The
        // semaphore is intentionally kept alive: callers already queued behind a
        // concurrent calculation acquire it, observe _disposed, and fail cleanly.
        _gate.Wait();
        try
        {
            if (_disposed) return;
            _disposed = true;
            _xf.Dispose();
            _main.Dispose();
        }
        finally { _gate.Release(); }
        GC.SuppressFinalize(this);
    }
}
