# v2前向预测与反向搜索DLL调用说明

当前版本**2.0.3**统一为`float64_body_float32_io_v1`：源权重不变，KAN及非负头内部双精度计算，对外输入、scaler和输出边界仍为float32。PyTorch正式加载器和ONNX导出共用这一策略，C#锁定CPU顺序执行并关闭融合/预打包。原5组工况及额外32组逐字段对照通过，原阈值和history保留；额外4组共同未收敛仅作诊断。对照Python时必须同步本次代码；旧float32历史对照另外保留，不能把旧实现说成已完全一致。以下2.0.2段落为历史。

当前补丁版本2.0.2。初次接入先看[快速使用](快速使用.md)：成品根目录提供两个双击运行示例及`接入示例`工程；每包`dll`都已含配套的两个入口和共享核心。将`dll`内文件放到宿主EXE旁，并将整个`models`目录放在EXE旁。部署契约已升级schema2，运行时会验证模型与预处理配套哈希、列序及缩放约定，请整套更新，不能只替换一个DLL或一份模型。ONNX原生库还需要VC++v14 x64运行库；自包含verify仅表示已带.NET运行时。当前严格PyTorch整机迁移仍有未通过项，具体看随包reports。

两个成品入口分别为`露点与能耗预测模型v2.dll`和`最优能耗工况搜索模型v2.dll`，都是Windows x64环境下的.NET 8托管类库。命名空间和API使用英文，引用程序集时保留中文文件名。模型推理通过ONNX Runtime完成，运行不依赖Python、PyTorch、Conda或WSL。

随包部署`Dehumidifier.Simulation.dll`、ONNX Runtime及绘图依赖、两个ONNX、三个预处理JSON和模型契约。不要只复制入口DLL。独立验证程序的自包含发布目录还包含.NET运行时；将DLL接入宿主时，宿主必须使用.NET 8与x64。托管类库不能直接用P/Invoke，也不能由.NET Framework 4.x直接引用。

本次源码和数据身份、实际测试、构建及迁移结果以交付包中的报告为准；示例代码不替代该报告。

## 前向入口

```csharp
using Dehumidifier.Prediction;
using Dehumidifier.Simulation;

using var model = new PredictionModel(
    Path.Combine(AppContext.BaseDirectory, "models"),
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "DehumidifierV2", "prediction"));
SimulationConfig config = PredictionModel.LoadConfiguration("examples/qy1070_case.json");
// 或由完整JSON文本解析：PredictionModel.ParseConfiguration(configJson)
using var cancellation = new CancellationTokenSource();
SimulationResult result = await model.PredictAsync(config, cancellationToken: cancellation.Token);
PredictionSummary summary = PredictionModel.Summarize(result);
Console.WriteLine(summary.ToJson());
```

`Predict`和`PredictJson`是同步入口；GUI事件建议使用`PredictAsync`或`PredictJsonAsync`。所有入口调用同一正式整机求解器，正常完成和未收敛都保存完整精度JSON及PNG，路径在`ResultJsonPath`和`DashboardPngPath`。先判断`Converged`，再展示预测结果；未收敛返回调试结果，不伪装成功。

主要摘要字段为送风温度、含湿量、露点、总冷/热负荷、六项热工分项、总冷热负荷和收敛信息。`SupplyDewPointC=null`且`DryLimit=true`表示零湿度干燥极限，不能显示成0℃。完整节点、转轮输出、混风规则与求解历史见`SimulationResult`或已保存JSON。

配置和单位沿用schema v2：℃、g/kg干空气、m³/h、Pa。`control_setpoints.supply_T_target_degC`控制主转轮之后串联的CC3后表冷和Heater_Post后加热；不填或null时兼容旁通。六项负荷始终保留，冷负荷分项为负，汇总为三项表冷绝对值与三项加热负荷之和。总冷热负荷没有折算COP或效率。

## 反向入口

```csharp
using Dehumidifier.Optimization;
using Dehumidifier.Prediction;

using var model = new EnergyOptimizer("models", "results/search");
var config = PredictionModel.LoadConfiguration("examples/qy1070_case.json");
var settings = new SearchSettings
{
    DewPointTargetC = -45.0,
    SupplyTemperatureTargetC = -35.0,
    DewPointToleranceC = 1.0,
    DewPointPenaltyCoefficient = 0.1,
    // 也可给InitialControls；它只是重新计算的初值，不是直接返回的答案。
};
using var cancellation = new CancellationTokenSource();
SearchResult result = await model.OptimizeAsync(config, settings, cancellation.Token);
if (result.Feasible && result.BestIndependentlyVerified)
{
    Console.WriteLine(result.Best!.Controls);
    Console.WriteLine(result.Best.Assessment.OptimizationLoss);
    Console.WriteLine(result.Best.Assessment.ThermalLoadSumKW);
    Console.WriteLine(result.ResultJsonPath);
    // 直接传完整配置，保留QY1070工况及末端目标，避免按位置抄四个温度。
    using var forward = new PredictionModel("models", "results/replay");
    var replay = await forward.PredictAsync(result.BestConfig!);
    Console.WriteLine(PredictionModel.Summarize(replay).ToJson());
    Console.WriteLine(result.BestConfigPath);
}
```

送风温度和露点必须给定；露点容差默认±1℃，不得超过±1℃，最终送风温度容差1e-4℃。搜索CC1、CC2、新风再生和主轮再生四个温度，默认步长0.1℃，流量、入口和尺寸保持原配置。末端按给定温度自动启停，不增加两个末端自由控制变量。

评分为`J=Q_sum/Q_ref+0.1*w(e)*|e|/(1℃)`，`e=最终送风露点−目标露点`。权重`[-1,-0.5)`为0.35、`[-0.5,+0.3]`为0.1、`(+0.3,+1]`为0.5；边界−0.5和+0.3归中间段。误差越界、未收敛、最终状态过饱和或非法负荷不能选解。`Q_ref`整次搜索固定，缺省来自原配置在相同送风温度目标下的总负荷；基准失败须报错，可显式提供正的`EnergyReferenceKW`。不要把无量纲J当成kW或直接用电量。

`Best`是最小J候选，`MinimumThermalLoadCandidate`另存裸负荷最低点；二者可能不同。结果保留候选和独立复算，`Feasible=false`表示没有找到可用控制参数。搜索按当前边界和预算执行，不保证全局最优；C#和SciPy随机算法不保证逐点同序，目标与验收契约保持一致。

## QY1070搜索与前向复算

两边统一读取`qy1070_case.json`：两转轮直径1070 mm，新风/回风/再生补风上限分别3500/7334/277 m³/h，两冷吹量167/667 m³/h，转速8/6 rpm。`default_case.json`是1220 mm、4200/8800/333、200/800 m³/h的另一工况，不能互换。接口仍支持显式提供其他完整工况，不会自动把它改成QY1070。

搜索直接调用前向使用的`Simulator`，只覆盖四个具名控制温度，并设置末端目标；其余模块、输入和求解参数一致。相对未启用末端的前向链路，唯一物理扩展为主转轮出口后的CC3与Heater_Post。原QY1070配置不带送风目标，因此直接预测时末端旁通；使用`BestConfig`复算会包含搜索时的末端。核对原链路时比较`AfterR2_P`；核对搜索结果时比较最终`Supply`。后加热等湿，后表冷凝结时才会降低含湿量和露点。

开发接入应直接用`result.BestConfig`或`PredictionModel.LoadConfiguration(result.BestConfigPath!)`，并保留同一套六份模型文件和共享核心DLL。仅传四个温度不能保证风量、尺寸、转速、压力、模型及末端一致。若确需逐字段映射，请按名称对应：`Cc1OutletC→cc1_T_outlet_degC`、`Cc2OutletC→cc2_T_outlet_degC`、`HeaterOaC→heater_oa_T_degC`（新风再生）、`HeaterLowC→heater_low_T_degC`（主轮再生）。`ControlTemperatures`的后两项是Oa、Low，`ControlSetpointsConfig`构造函数的后两项是Low、Oa，禁止直接照位置搬运。

露点目标−45℃且默认容差1℃表示允许−46～−44℃，不承诺恰好−45℃。但**搜索输出露点与同配置前向复算露点必须一致**，与目标容差是两种不同检查；−40℃不合格。2.0.1补强独立复算，逐项检查送风温度/湿度/露点、六负荷及迭代次数，避免只比总负荷与J漏掉差异。

可执行独立进程验证（默认QY1070）：

```powershell
conda run --no-capture-output -n dehumidifier python deployment/dotnet/validate_v2_roundtrip.py --verifier <本次构建>/v2verify/Dehumidifier.V2Verify.exe --models <本次模型目录> --output results/deployment/新的复算目录 --dew-point-target -45 --supply-temp-target -35
```

该脚本核对除四控制温度与末端目标外完整配置不变，重新启动前向进程，比较所有物理记录（含history），并记录输入、模型及DLL的SHA256。`--contrast-config deployment/dotnet/examples/default_case.json`额外诊断工况混用、末端旁通及−55℃冷凝；这些对照不作为目标达标方案。开发端若仍出现−40℃，应对照报告中的完整输入配置、字段映射及文件哈希；仅凭“QY1070”名称无法确认实际输入一致。

## 取消、错误与并发

同一实例的请求串行执行，取消在迭代/候选之间检查；不保证打断正在执行的原生推理。先取消并等待在途任务结束，再Dispose。`OperationCanceledException`表示已取消，`SimulationException.Code`和反向的`OptimizationException.Code`区分配置、模型、计算、固定基准、独立复算和留档错误；非法搜索设置还会抛出`ArgumentException`。配置未知字段会失败，不会悄悄忽略。

## 实际DLL消费示例和验证工具

`samples/V2DllConsumer`只引用交付DLL文件，不引用本仓库源码工程。使用SDK编译时通过`-p:ModelDllDirectory=绝对DLL目录`指定三业务程序集及依赖所在目录。示例的search模式执行搜索，并将完整`BestConfig`交给独立`PredictionModel`复算；差异超过1e-9时退出7。传入`examples/qy1070_case.json`完成QY1070接入验证。

```powershell
dotnet build samples/V2DllConsumer/V2DllConsumer.csproj -c Release -p:ModelDllDirectory=C:/绝对路径/lib
```

`Dehumidifier.V2Verify.exe`同时引用两个新入口，可使用以下模式：

```powershell
.\verify\Dehumidifier.V2Verify.exe --mode forward --models .\models --config .\examples\qy1070_case.json --results-dir .\results\forward --summary-output .\reports\forward.json
.\verify\Dehumidifier.V2Verify.exe --mode search --models .\models --config .\examples\qy1070_case.json --results-dir .\results\search --dew-point-target -45 --supply-temp-target -35 --summary-output .\reports\search.json
.\verify\Dehumidifier.V2Verify.exe --mode cancel --models .\models --config .\examples\qy1070_case.json --results-dir .\results\cancel
```

退出码：0成功，2配置/模型/计算错误，3留档失败，4前向未收敛，5取消，6搜索无合格方案。cancel模式先验证预取消请求，再验证同一实例可正常使用；成功退出0。输出摘要中的记录路径用于定位完整精度结果，不能只看程序是否启动。

当前训练覆盖与守恒偏差继续保留，求解收敛和DLL调用通过均不能代替整机实测精度或设备容量、结霜和联锁验收。
