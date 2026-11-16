using Dehumidifier.Optimization;
using Dehumidifier.Prediction;

// This sample references delivered DLL files, not the repository projects.
// Build with -p:ModelDllDirectory=<directory containing both business DLLs and dependencies>.
if (args.Length < 4)
{
    Console.Error.WriteLine("用法：V2DllConsumer forward|search <models目录> <schema-v2配置JSON> <结果目录>");
    return 2;
}
var config = PredictionModel.LoadConfiguration(args[2]);
if (args[0] == "forward")
{
    using var model = new PredictionModel(args[1], args[3]);
    var result = await model.PredictAsync(config);
    Console.WriteLine(PredictionModel.Summarize(result).ToJson());
    return result.Converged ? 0 : 4;
}
if (args[0] == "search")
{
    using var model = new EnergyOptimizer(args[1], args[3]);
    var result = await model.OptimizeAsync(config, new SearchSettings
    {
        DewPointTargetC = -45,
        SupplyTemperatureTargetC = -35,
    });
    if (!result.Feasible || !result.BestIndependentlyVerified || result.BestConfig is null)
        return 6;
    // Replay the complete QY1070 configuration, including terminal target,
    // geometry, all five flows, pressure, rotor speeds and named controls.
    using var prediction = new PredictionModel(args[1], Path.Combine(args[3], "external_forward"));
    var replay = await prediction.PredictAsync(result.BestConfig);
    var expected = result.Best!.Assessment;
    bool matches = replay.Converged
        && Math.Abs(replay.Supply.DewPointC - expected.SupplyDewPointC!.Value) <= 1e-9
        && Math.Abs(replay.Supply.TemperatureC - expected.SupplyTemperatureC!.Value) <= 1e-9
        && Math.Abs(replay.Supply.HumidityGPerKg - expected.SupplyHumidityGPerKg!.Value) <= 1e-9
        && Math.Abs(replay.CoolingLoadKW + replay.HeatingLoadKW - expected.ThermalLoadSumKW!.Value) <= 1e-9;
    Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new
    {
        feasible = result.Feasible,
        result.BestIndependentlyVerified,
        controls = result.Best?.Controls,
        assessment = result.Best?.Assessment,
        result.ResultJsonPath,
        result.BestConfigPath,
        forward_replay_matches = matches,
        forward_replay = PredictionModel.Summarize(replay),
    }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    return matches ? 0 : 7;
}
Console.Error.WriteLine("模式只支持forward或search");
return 2;
