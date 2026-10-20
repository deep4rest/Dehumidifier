using System.Text.Json;
using System.Diagnostics;
using Dehumidifier.Simulation;

static string? Value(string[] args, string name)
{
    var index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
}

var models = Value(args, "--models");
var configPath = Value(args, "--config");
var results = Value(args, "--results-dir");
var maxIterText = Value(args, "--max-iter");
var repeatText = Value(args, "--repeat");
var cancelTest = args.Contains("--cancel-test", StringComparer.OrdinalIgnoreCase);
if (models is null || configPath is null || results is null)
{
    Console.Error.WriteLine("用法: Dehumidifier.Verify.exe --models <目录> --config <工况.json> --results-dir <可写目录> [--max-iter N] [--repeat N] [--cancel-test]");
    return 2;
}

try
{
    var config = Simulator.LoadConfiguration(configPath);
    var options = maxIterText is null ? null : new SolverOptions { MaxIterations = int.Parse(maxIterText) };
    using var simulator = new Simulator(models, results);

    if (cancelTest)
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var cancelled = false;
        try { await simulator.SimulateForValidationAsync(config, options, cancellation.Token); }
        catch (OperationCanceledException) { cancelled = true; }
        var afterCancellation = await simulator.SimulateForValidationAsync(config, options);
        var cancelPayload = new
        {
            status = cancelled && afterCancellation.Converged ? "passed" : "failed",
            cancellation_observed = cancelled,
            reuse_after_cancellation_converged = afterCancellation.Converged,
            reuse_after_cancellation_iterations = afterCancellation.Iterations,
        };
        Console.WriteLine(JsonSerializer.Serialize(cancelPayload, new JsonSerializerOptions { WriteIndented = true }));
        return cancelled && afterCancellation.Converged ? 0 : 4;
    }

    if (repeatText is not null)
    {
        if (!int.TryParse(repeatText, out var repeat) || repeat <= 0) throw new SimulationException("INVALID_REPEAT", "--repeat 必须是正整数");
        var elapsed = new List<double>(repeat);
        var process = Process.GetCurrentProcess();
        var peakWorkingSet = process.WorkingSet64;
        var allConverged = true;
        SimulationResult? last = null;
        for (var i = 0; i < repeat; i++)
        {
            var watch = Stopwatch.StartNew();
            last = await simulator.SimulateForValidationAsync(config, options);
            watch.Stop();
            elapsed.Add(watch.Elapsed.TotalMilliseconds);
            process.Refresh();
            peakWorkingSet = Math.Max(peakWorkingSet, process.WorkingSet64);
            allConverged &= last.Converged;
        }
        elapsed.Sort();
        var p95Index = Math.Min(elapsed.Count - 1, (int)Math.Ceiling(elapsed.Count * 0.95) - 1);
        var repeatPayload = new
        {
            status = allConverged ? "converged" : "unconverged",
            repeat_count = repeat,
            all_converged = allConverged,
            min_ms = elapsed[0],
            mean_ms = elapsed.Average(),
            p95_ms = elapsed[p95Index],
            max_ms = elapsed[^1],
            peak_working_set_bytes = peakWorkingSet,
            last_iterations = last?.Iterations,
            last_supply_T_degC = last?.Supply.TemperatureC,
            last_supply_W_g_per_kg = last?.Supply.HumidityGPerKg,
        };
        Console.WriteLine(JsonSerializer.Serialize(repeatPayload, new JsonSerializerOptions { WriteIndented = true }));
        return allConverged ? 0 : 4;
    }

    var result = await simulator.SimulateAsync(config, options);
    var payload = new
    {
        status = result.Converged ? "converged" : "unconverged",
        converged = result.Converged,
        iterations = result.Iterations,
        cooling_kW = result.CoolingLoadKW,
        heating_kW = result.HeatingLoadKW,
        supply_T_degC = result.Supply.TemperatureC,
        supply_W_g_per_kg = result.Supply.HumidityGPerKg,
        regen_mix_branch = result.RegenMixRule?.Branch,
        regen_threshold_m3h = result.RegenMixRule?.ThresholdM3H,
        regen_oa_used_m3h = result.RegenMixRule?.RegenOaUsedM3H,
        regen_leaked_m3h = result.RegenMixRule?.LeakedM3H,
        regen_shortfall_m3h = result.RegenMixRule?.ShortfallM3H,
        result_json = result.ResultJsonPath,
        dashboard_png = result.DashboardPngPath,
    };
    Console.WriteLine(JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    return result.Converged ? 0 : 4;
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine("计算已取消");
    return 5;
}
catch (SimulationException ex)
{
    Console.Error.WriteLine($"[{ex.Code}] {ex.Message}");
    if (ex.InnerException is not null) Console.Error.WriteLine(ex.InnerException.ToString());
    return ex.Code == "RESULT_SAVE_FAILED" ? 3 : 2;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"[UNEXPECTED] {ex}");
    return 2;
}
