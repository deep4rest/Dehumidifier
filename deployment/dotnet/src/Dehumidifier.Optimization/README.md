# 最优能耗工况搜索模型v2

这是`net8.0`托管DLL，程序集名为`最优能耗工况搜索模型v2.dll`，公开命名空间为`Dehumidifier.Optimization`。运行依赖同包的`Dehumidifier.Simulation.dll`、ONNX Runtime原生依赖和两台当前模型的六份导出文件，不启动Python，不读取训练CSV。

```csharp
using Dehumidifier.Simulation;
using Dehumidifier.Optimization;

using var optimizer = new EnergyOptimizer(
    Path.Combine(AppContext.BaseDirectory, "Dehumidifier", "models"),
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Dehumidifier", "results", "optimization"));
var config = Simulator.LoadConfiguration("examples/default_case.json");
using var cancellation = new CancellationTokenSource();
var result = await optimizer.OptimizeAsync(config, new SearchSettings
{
    DewPointTargetC = -45,
    SupplyTemperatureTargetC = -35,
    DewPointToleranceC = 1,
    DewPointPenaltyCoefficient = 0.1,
}, cancellation.Token);
if (result.Feasible)
{
    var controls = result.Best!.Controls;
    double qKW = result.Best.Assessment.ThermalLoadSumKW!.Value;
    double loss = result.Best.Assessment.OptimizationLoss!.Value;
}
```

四个搜索变量依次为CC1、CC2、新风转轮再生加热、主转轮再生加热温度。默认边界9～13、13～17、50～140、50～140℃，步长0.1℃；风量、入口、压力、转轮几何和转速保持原工况。末端CC3和Heater_Post自动按目标温度运行。基准负荷在原配置、同一目标送风温度下仅计算一次；也可提供`EnergyReferenceKW`，不逐候选改变基准。

`J = Q_sum/Q_ref + 0.1*w(e)*abs(e)/(1℃)`，其中e为最终Supply露点减目标，权重`[-1,-.5)`为.35、`[-.5,.3]`为.1、`(.3,1]`为.5。露点误差±1℃和最终温度1e-4℃仍是硬约束；检查原闭环收敛、非有限/负湿度、未裁剪RH及六项负荷合同。Q_sum是六项冷热负荷之和，单位kW；J无量纲，不代表用电量或KAN训练loss。

算法为有界best/1/bin差分进化，分层随机初始化、三种子、F每代在[.5,1)取样、交叉率.7、即时更新，再用1/.5/步长坐标细化。保留81点粗网格、原配置和可选初值；所有候选落到真实控制网格、去重缓存，按J选择可行解。与Python共用业务契约，但采用.NET `System.Random`，不是SciPy/NumPy随机序列，搜索轨迹与最终参数不保证逐位相同；不宣称全局最优。预算默认3种子、40代、popsize=8×4。没有可行点时返回`Status="no_feasible_candidate_found"`和`Feasible=false`，没有伪造的最佳参数。

每次搜索新建结果子目录，保存逐候选`evaluations.jsonl`、完整`search.json`、模型文件SHA256、最佳工况`best_config.json`。最终候选独立重新运行同一前向求解器并保存JSON/PNG；独立评分或负荷差超过1e-9则抛`INDEPENDENT_VERIFICATION_FAILED`。取消抛`OperationCanceledException`并保留取消记录；基准失效、模型或配置错误明确报错。调用方配置不修改；同一实例搜索串行，UI中应await；先取消、等待任务结束，再Dispose。

此工程新增测试在`deployment/dotnet/tests/Dehumidifier.Optimization.Tests`。这些合成状态专项验证评分、硬门禁、基准、真正搜索、取消、不可变工况和独立复算失败行为；真实ONNX及整机验证由本次交付验证器完成，不能用合成测试替代。
