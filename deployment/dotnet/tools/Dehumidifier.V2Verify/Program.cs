using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Dehumidifier.Optimization;
using Dehumidifier.Prediction;
using Dehumidifier.Simulation;

Console.OutputEncoding = System.Text.Encoding.UTF8;
var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
try
{
    var flags = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 0; i < args.Length; i++)
    {
        if (args[i] is "--help" or "-h")
        {
            Console.WriteLine("V2托管DLL独立调用验证：--mode forward|search|cancel --models DIR --config FILE --results-dir DIR [--dew-point-target -45 --supply-temp-target -35 --maxiter 40 --popsize 8 --seeds 20261111,20261112,20261113 --initial-controls 9.1,16.2,98.5,91.7 --energy-reference KW --penalty-coefficient .1 --max-iter 60 --summary-output FILE]");
            return 0;
        }
        if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 == args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            throw new SimulationException("INVALID_ARGUMENT", $"参数必须为 --名称 值：{args[i]}");
        if (!flags.TryAdd(args[i], args[++i]))
            throw new SimulationException("INVALID_ARGUMENT", "参数不能重复");
    }
    var allowed = new HashSet<string> { "--mode", "--models", "--config", "--results-dir", "--dew-point-target", "--supply-temp-target", "--maxiter", "--popsize", "--seeds", "--initial-controls", "--energy-reference", "--penalty-coefficient", "--max-iter", "--summary-output" };
    foreach (var key in flags.Keys) if (!allowed.Contains(key)) throw new SimulationException("INVALID_ARGUMENT", $"未知参数：{key}");
    string Need(string name) => flags.TryGetValue(name, out var value) ? value : throw new SimulationException("INVALID_ARGUMENT", $"缺少参数：{name}");
    string? Optional(string name) => flags.GetValueOrDefault(name);
    double Number(string text) => double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
    int Integer(string text) => int.Parse(text, CultureInfo.InvariantCulture);
    var mode = Optional("--mode") ?? "forward";
    var config = PredictionModel.LoadConfiguration(Need("--config"));
    var models = Need("--models");
    var results = Need("--results-dir");
    var watch = Stopwatch.StartNew();
    object payload;
    int exitCode;
    if (mode is "forward" or "cancel")
    {
        using var model = new PredictionModel(models, results);
        var options = new SolverOptions { MaxIterations = Optional("--max-iter") is string iterations ? Integer(iterations) : 60 };
        bool cancelled = false;
        if (mode == "cancel")
        {
            using var cancellation = new CancellationTokenSource();
            cancellation.Cancel();
            try { await model.PredictAsync(config, options, cancellation.Token); }
            catch (OperationCanceledException) { cancelled = true; }
            if (!cancelled) throw new SimulationException("CANCEL_FAILED", "已取消请求没有抛出OperationCanceledException");
        }
        var prediction = await model.PredictAsync(config, options);
        payload = new { mode, cancelled, elapsed_ms = watch.Elapsed.TotalMilliseconds, prediction = PredictionModel.Summarize(prediction) };
        exitCode = prediction.Converged ? 0 : 4;
    }
    else if (mode == "search")
    {
        ControlTemperatures? initial = null;
        if (Optional("--initial-controls") is string initialText)
        {
            var values = initialText.Split(',').Select(Number).ToArray();
            if (values.Length != 4) throw new SimulationException("INVALID_ARGUMENT", "--initial-controls需要四个逗号分隔温度");
            initial = new ControlTemperatures(values[0], values[1], values[2], values[3]);
        }
        var settings = new SearchSettings
        {
            DewPointTargetC = Number(Need("--dew-point-target")),
            SupplyTemperatureTargetC = Number(Need("--supply-temp-target")),
            MaxGenerations = Optional("--maxiter") is string generations ? Integer(generations) : 40,
            PopulationMultiplier = Optional("--popsize") is string population ? Integer(population) : 8,
            Seeds = Optional("--seeds") is string seeds ? seeds.Split(',').Select(Integer).ToArray() : new[] { 20261111, 20261112, 20261113 },
            InitialControls = initial,
            EnergyReferenceKW = Optional("--energy-reference") is string reference ? Number(reference) : null,
            DewPointPenaltyCoefficient = Optional("--penalty-coefficient") is string coefficient ? Number(coefficient) : .1,
        };
        using var optimizer = new EnergyOptimizer(models, results);
        var search = await optimizer.OptimizeAsync(config, settings);
        payload = new
        {
            mode, elapsed_ms = watch.Elapsed.TotalMilliseconds,
            status = search.Status, feasible = search.Feasible,
            energy_reference_kW = search.EnergyReferenceKW,
            best_independently_verified = search.BestIndependentlyVerified,
            evaluated_count = search.Evaluations.Count,
            feasible_count = search.Evaluations.Count(c => c.Assessment.Feasible),
            best_controls = search.Best?.Controls,
            best_config_path = search.BestConfigPath,
            best_assessment = search.Best?.Assessment,
            minimum_thermal_load_candidate = search.MinimumThermalLoadCandidate,
            best_forward = search.BestForwardResult is null ? null : PredictionModel.Summarize(search.BestForwardResult),
            result_json = search.ResultJsonPath,
            global_minimum_proven = false,
        };
        exitCode = search.Feasible && search.BestIndependentlyVerified ? 0 : 6;
    }
    else throw new SimulationException("INVALID_ARGUMENT", "--mode只支持forward、search、cancel");
    var summaryJson = JsonSerializer.Serialize(payload, jsonOptions);
    if (Optional("--summary-output") is string summaryOutput)
    {
        var target = Path.GetFullPath(summaryOutput);
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        File.WriteAllText(target, summaryJson + Environment.NewLine);
    }
    Console.WriteLine(summaryJson);
    return exitCode;
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
catch (OptimizationException ex)
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
