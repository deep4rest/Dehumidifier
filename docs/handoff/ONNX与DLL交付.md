# ONNX 导出与 Windows DLL 交付

## 最终基线与验证边界

本次源码以主项目最终风量 KAN、企业再生混风、DptAH和零补风修复为准。Python/C#的输入、拓扑、物性与默认求解参数已经同步；NuGet业务版本1.0.1。最初上传曾混入旧工作树Python，该基线现已纠正，不应再按“风速Python待追赶C#”接手。

| 契约 | Python 与 C# 共同采用 |
| --- | --- |
| 输入 | XF 10、main 9列，处理/冷吹量为m³/h |
| 再生混风 | R1处理量的1/3为目标，R2再生出与R1冷吹出先混合，超量漏风、不足按上限补风 |
| 低温物性 | T≤0℃冰面多项式，T>0℃液态水，水汽比0.62198 |
| 露点 | 88项DptAH表，标准压标定、区间插值、低端外推、零湿度干燥极限 |
| 零补风 | 允许0；不足继续计算并记录warning和shortfall |
| 求解 | relax=0.8、max_iter=60、T容差1e-4℃、W容差1e-6 g/kg |
| 留档 | regen_mix_rule、convergence.history、solver_parameters、末次实际输入与残差 |

源码契约同步不等于跨语言逐位一致。本次需分别记录导出误差、Windows构建/运行、严格迁移和工程验收。历史记录里QY1070有Python12次/C#13次迭代差异，不能因稳态近似就宣称严格迁移通过。部署成品、模型、原始数据和历史运行报告不上传GitHub。

## 源码入口与工具链

本仓库优先 WSL/Linux 开发 Python，Windows x64 编译和验证 WinForms。推荐 Python 3.11/3.12 新建根 `.venv`；根 `requirements.txt` 是训练/仿真依赖，`deployment/dotnet/requirements-export.txt` 额外安装 `onnx`、`onnxruntime`。Python 版本与 PyTorch/ORT wheel 是否可安装要在目标机核实，不搬运另一台电脑的 `.venv`。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r deployment/dotnet/requirements-export.txt
.venv/bin/python -m pip install -e simulation/
.venv/bin/python deployment/dotnet/export_onnx.py --help
.venv/bin/python deployment/dotnet/create_reference.py --help
.venv/bin/python deployment/dotnet/package_delivery.py --help
.venv/bin/python -m pytest deployment/dotnet/tests/test_packaging.py -q
```

C# 项目使用 .NET 8 SDK，主依赖 `Microsoft.ML.OnnxRuntime 1.20.1`、`System.Drawing.Common 8.0.0`；xUnit 项目另依赖测试 SDK 17.11.1、xUnit 2.9.2、VS runner 2.8.2。DLL 目标 `net8.0`，WinForms 目标 `net8.0-windows`；图像留档依赖 Windows 的 System.Drawing，不能用 Linux 编译成功代替 Windows 运行验证。

```bash
dotnet build deployment/dotnet/src/Dehumidifier.Simulation/Dehumidifier.Simulation.csproj -c Release
```

上述主库编译无需模型。导出与运行测试需要两台最终风量KAN完整包；精确文件SHA见 `模型与数据身份.json`，模型须带外恢复。重新训练应使用六份尺寸隔离风量CSV，不能使用旧XLS风速入口。缺模型就报错；本次上传不授权全量训练，也不授权从历史模型自动回退。

## ONNX 的输入输出边界

`export_onnx.py` 只导出 KAN 神经网络图，预处理、逆变换与整机物理模型均在图外。每台图的输入名 `input`、输出名 `output`，数据类型 float32，ONNX opset 17，动态 batch：XF `[batch,10] → [batch,6]`，main `[batch,9] → [batch,6]`。

当前原始物理输入列必须保序：

```text
XF:
转轮厚度, 转轮直径, 处理入口温度(℃), 入口含湿量(g/kg),
处理风量, 冷吹风量, 转轮转速(rpm), 再生温度(℃),
再生风量, 再生入口含湿量

main:
转轮厚度, 转轮直径, 入口含湿量(g/kg), 转轮转速(rpm),
再生温度(℃), 冷吹入口温度(℃), 处理入口温度(℃),
处理风量, 冷吹风量

六个输出:
处理出口温度, 处理出口含湿量, 冷吹出口温度, 冷吹出口含湿量,
再生出口温度, 再生出口含湿量
```

物理单位：温度 ℃、含湿量 g/kg 干空气、厚度/直径 mm、转速 rpm，处理/冷吹/再生风量 m³/h。两图不能直接接受原始物理量，必须先执行 scaler：

```text
x_scaled[i] = x[i] * input_scale[i] + input_min[i]
y_transformed[j] = (y_scaled[j] - output_min[j]) / output_scale[j]
若输出列属于 log1p_output_columns：y_physical[j] = expm1(y_transformed[j])
其他列：y_physical[j] = y_transformed[j]
```

输入与目标 MinMax 范围 `[-1,1]`，`clip=false`，不能静默截断域外输入；只有处理出口含湿量做 log1p/expm1。需复核 double/float32 转换顺序：C# 先把原始输入和系数 cast float32 再乘加，Python/ONNX 导出验证只比较经同一 Python scaler 变换后的图输出，**不覆盖这一跨语言缩放差异**。

露点不是图输出：必须由处理出口含湿量及实际压力调用当前 `dew_point_water_C`。`W=0` 是合法干燥极限，露点为 `-inf`，JSON 存 `null`；不能显示为 0℃。ONNX 中不含 `power`、风机电耗、scaler、表冷/加热、混风、分流或联立迭代。

导出器对每台模型使用固定随机种子的 257 个样本，覆盖训练 min/max、中点、随机域内和约 3% 外延边界；运行 ONNX checker，并比较 PyTorch/ORT 缩放量与逆变换后的物理量。非有限输出或六输出最大物理绝对差 `>=1e-3` 会失败。这只验证固定样本上的图导出，不是模型精度或完整 C# 迁移验收。

## 导出与参考记录

确认两个完整 runtime 包存在且来源可信：

```text
artifacts/kan_xf_single/
  model/{best_model.pt,scalers.pkl,scaler_meta.json,metadata.json}
  model_metadata.json
  config_snapshot.yaml
  training_run.json
artifacts/kan_main_single/   同结构
```

从仓库根导出到不存在或空目录，`运行编号` 要替换为新的实际标识：

```bash
.venv/bin/python deployment/dotnet/export_onnx.py \
  --output results/deployment/运行编号/onnx
.venv/bin/python deployment/dotnet/create_reference.py \
  --output results/deployment/运行编号/python_reference
```

导出目录包含 `xf.onnx`、`main.onnx`、`preprocessing.json`、`preprocessing_xf.json`、`preprocessing_main.json`、`model_contract.json`。契约记录列顺序、架构、张量名/形状、源权重/scaler/元数据 SHA-256 和导出误差。参考目录含 `qy1070_case/` 与 `default_case/`，各自同目录 JSON/PNG；使用当前真实 YAML 与当前 Python 仿真代码，不改工况。必须逐份检查 `result.convergence.converged`；脚本在全部工况保存后，只要任一工况未收敛或缺少收敛评估，就用中文标识并退出 **4**，不会把参考文件生成等同于稳态成立。

`create_examples.py` 从 YAML 生成 JSON；仓库已带两份相同来源的 JSON，脚本对已有文件拒绝覆盖。工况变化后应审阅差异并主动更新 JSON，不能反复运行来掩盖配置漂移。

## Windows 构建与 DLL API

先完成前述 C# 适配和测试更新，再将本次导出的六个模型文件放入 `deployment/dotnet/src/Dehumidifier.Simulation/models/`，整个目录应被 Git 忽略。此动作是构建暂存，不上传模型。

Windows PowerShell 在仓库根运行：

```powershell
Copy-Item -Recurse -Force ".\results\deployment\运行编号\onnx" ".\deployment\dotnet\src\Dehumidifier.Simulation\models"
.\deployment\dotnet\build.ps1 -Output ".\results\deployment\运行编号_windows"
```

目标 `models` 必须事先不存在，否则 Copy-Item 会嵌套 `onnx` 子目录；已有模型暂存时应核对并明确替换六文件。构建脚本拒绝覆盖已存在的输出目录，默认也会分配新的 `results/deployment/` 目录。它依次 restore、build、test、publish 验证程序、publish WinForms、pack NuGet；首次运行要联网还原依赖。`-SkipTests` 仅供源码编译诊断，不能当验证完成。C# 测试包含企业表、补/漏风和迭代次数断言；重建必须用真实结果核验，不能改次数或放宽阈值掩盖迁移差异。

`build/Dehumidifier.Simulation.targets` 为 NuGet 消费工程复制模型，必须随源码和主包保留。它不是构建缓存。模型与 `onnxruntime.dll` 等原生依赖必须同程序目录完整部署。

主 API 是托管 C#，不是 native C ABI，不能直接对 `.dll` 使用 P/Invoke；.NET Framework 4.x 不能直接引用 `net8.0` DLL。

```csharp
using Dehumidifier.Simulation;

using var simulator = new Simulator(
    Path.Combine(AppContext.BaseDirectory, "Dehumidifier", "models"),
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Dehumidifier", "results", "simulation"));
var config = Simulator.LoadConfiguration("examples/qy1070_case.json");
using var cancellation = new CancellationTokenSource();
var result = await simulator.SimulateAsync(config, cancellationToken: cancellation.Token);
// 收敛与精度分开判断；Converged=false 也会留档最后一次计算的 JSON/PNG。
bool converged = result.Converged;
double supplyC = result.Supply.TemperatureC;
double supplyW = result.Supply.HumidityGPerKg;
double coolingKW = result.CoolingLoadKW;
double heatingKW = result.HeatingLoadKW;
string? json = result.ResultJsonPath;
string? png = result.DashboardPngPath;
```

公开类型还包括 `SimulationConfig`、`SolverOptions`、`SimulationResult`、`AirState`、`RotorOutputs`、`ConvergenceInfo`、`SimulationException`。同一 `Simulator` 的请求由 SemaphoreSlim 串行；窗体中用 await，先取消并等任务结束再 Dispose。`SimulateForValidationAsync` 使用同一求解器但不留档，仅用于验证/性能测量，常规计算仍用 `SimulateAsync`。取消在迭代间检查，不保证中断正在执行的原生推理。

两负荷为热负荷，四分项必须是 `CC1_kW`、`CC2_kW`、`Heater_Low_kW`、`Heater_OA_kW`；表冷为负，冷负荷取两项绝对值之和，热负荷为两加热器之和。凝结与自然满足零负荷应保留，不能把热负荷叫作整机直接电耗。

## 验证与严格迁移

在 Windows 对适配后的新构建运行两工况，保留标准输出与 `$LASTEXITCODE`：

```powershell
.\results\deployment\运行编号_windows\verify\Dehumidifier.Verify.exe `
  --models ".\deployment\dotnet\src\Dehumidifier.Simulation\models" `
  --config ".\deployment\dotnet\examples\qy1070_case.json" `
  --results-dir ".\results\deployment\运行编号\windows\qy1070_case"
```

default 工况同样执行。另测 `--max-iter 1` 未收敛、缺模型、非法字段/压力/流量、不可写目录和 `--cancel-test`。CLI 约定：0=收敛并留档，2=配置/模型/计算错误，3=留档失败，4=未收敛，5=取消；JSON/PNG 应成对、同目录，未收敛仍标识状态与末次实际计算输入/残差。

验证程序结果会再建运行子目录。将其实际 JSON/PNG 配对目录**复制保留**到比较暂存目录 `<windows-comparison>/<case>/`，不要移动或修改原始记录。随后在 Python 环境运行：

```bash
.venv/bin/python deployment/dotnet/validate_migration.py \
  --python-root results/deployment/运行编号/python_reference \
  --csharp-root results/deployment/运行编号/windows-comparison \
  --output results/deployment/运行编号/reports/validation.json
```

工具逐一比较送风、所有节点、两转轮六输出、四热负荷及汇总、收敛历史/末次输入/残差；若任一侧有 `regen_mix_rule` 也比较，另一侧缺字段会失败。阈值固定如下，不为通过而放宽：

| 类别 | 允许最大绝对差 |
| --- | ---: |
| 迭代次数/历史 iteration | 0 |
| 温度 | 1e-3 ℃ |
| 含湿量 | 1e-5 g/kg |
| 露点 | 1e-2 ℃ |
| 流量、压力 | 1e-10（各自原单位） |
| 焓 | 1e-2 kJ/kg 干空气 |
| RH | 1e-4（0–1 比例） |
| 热负荷 | 1e-2 kW |

状态、分类、缺失/额外字段及数组长度同样需要一致。两侧均保留 `convergence.history`、`convergence.solver_parameters` 与 `regen_mix_rule`；严格比较不得静默过滤额外字段、丢弃真实历史或删除迭代次数检查。

脚本退出 0 表示严格通过，1 表示完整生成了未通过报告；结果写 `passed`、`strict_migration_passed`。历史出现过两侧迭代数不同，即使稳态近似也必须保留失败；本次没有沿用旧包的通过记录。

## 自足封装与产物规范

新 `package_delivery.py` 从本次 Windows 构建组装，不依赖历史 ZIP、模板或旧 Windows 证据。它核验 verify/WinForms 携带的六个模型文件与指定导出 SHA-256 全部一致，两处业务 DLL 一致；要求本次 NuGet 主包存在。它不能替代Windows运行测试和跨语言数值比较。元数据记录 `source_contract_aligned=true`，`current_python_compatible=requires_current_cross_language_validation`；严格迁移是否通过只取调用者提供的本次报告，不由源码声明推断。

```bash
.venv/bin/python deployment/dotnet/package_delivery.py \
  --build results/deployment/运行编号_windows \
  --models results/deployment/运行编号/onnx \
  --validation-report results/deployment/运行编号/reports/validation.json \
  --output results/deployment/运行编号_delivery
```

`--validation-report` 可省略，此时 `strict_migration_evaluated=false`、`strict_migration_passed=false`；有失败报告也如实保留。Windows 运行与工程验收始终保持未由打包器评估，不能仅凭打包成功改成通过。

```text
results/deployment/运行编号_delivery/
├── package_metadata.json
├── models/{xf.onnx,main.onnx,model_contract.json,preprocessing*.json}
├── verify/                       完整自包含 Win-x64 验证程序、业务 DLL 和依赖
├── examples/
│   ├── {qy1070_case,default_case}.json
│   └── WinFormsDemo/publish/      完整自包含 WinForms 程序、业务 DLL 和依赖
├── nuget/                        本次 Dehumidifier.Simulation 主包
├── source/deployment/dotnet/      可维护的 C#、导出、构建、测试源码（无模型/缓存）
├── source/docs/handoff/          交接文档；完整训练/仿真源码从 GitHub 仓库获取
└── reports/validation.{json,md}   仅在显式提供本次报告时出现
运行编号_delivery.manifest.json   外置逐文件 SHA-256
运行编号_delivery.zip             CRC 和逐文件 SHA-256 已回读校验
运行编号_delivery.zip.sha256      ZIP 自身 SHA-256
```

新组装器不自动收集离线 NuGet 全部传递依赖，元数据明确 `offline_nuget_dependencies_included=false`。如需完全离线还原，接手者应单独锁定依赖版本、收齐主包和全部传递包，配置本地源，并使用空全局 NuGet 缓存实际 restore/build；只有离线测试完成才宣称离线交付。不要从旧交付借业务 DLL 或同名旧 NuGet 主包。

旧 `legacy_repackage_delivery.py` 保留原布局复封实现，会读取历史模板、Windows 工作区和证据。它不适合干净克隆；不得把历史验证文档复制成新构建结果。

## 交接验收顺序

训练/模型精度、部署可加载、严格数值迁移、工程物理验收分别记录。恢复已核验的最终风量模型，再验证真实模型导出、Windows构建与故障行为、逐字段严格迁移；最后由工程人员结合实测检查域外输入、风量参考状态、水分/能量收支、设备性能与联锁。模型误差、测试通过或求解收敛不能代替后续验收。

任何生成的模型、构建目录、NuGet 缓存和报告留在本地 `results/deployment/` 或明确的模型暂存目录，不加入源码提交。`src/Dehumidifier.Simulation/build/*.targets` 是源码例外，需要跟踪。
