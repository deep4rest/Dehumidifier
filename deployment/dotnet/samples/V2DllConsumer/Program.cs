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
        MaxGenerations = 1,
        PopulationMultiplier = 4,
        Seeds = new[] { 20261113 },
        InitialControls = new ControlTemperatures(9.1, 16.2, 98.5, 91.7),
    });
    Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new
    {
        feasible = result.Feasible,
        result.BestIndependentlyVerified,
        controls = result.Best?.Controls,
        assessment = result.Best?.Assessment,
        result.ResultJsonPath,
    }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    return result.Feasible && result.BestIndependentlyVerified ? 0 : 6;
}
Console.Error.WriteLine("模式只支持forward或search");
return 2;
