# ONNX 与 .NET DLL 源码交接

**这里的 C# 是历史迁移基线，尚未适配当前 Python 项目。** 源码取自主 checkout 的活跃 `deployment/dotnet/`，没有取生成包中的快照，也没有复制模型、二进制或历史构建结果。当前导出器会复用本仓库 `load_artifact`，只接受本仓库现行训练契约；不要为了加载历史 DLL 而把风速列改名为风量。

| 契约 | 当前 Python 训练/仿真 | 本目录历史 C# |
| --- | --- | --- |
| 处理/冷吹输入 | 风速 m/s；XF 10 列、main 9 列 | 风量 m³/h；同维数但物理含义不同 |
| 新风再生混风 | 主转轮再生出 + 全部配置再生新风 + 新风冷吹出 | 目标为新风处理量的 1/3；按需补风/漏风 |
| 低温饱和压/露点 | Murphy–Koop 过冷水；按实际压力反演 | 企业冰面式 + DptAH 表，低湿表按 101325 Pa 标定 |
| 固定点松弛 | 0.55 | 0.8 |
| 再生补风配置 | 正流量 | 允许 0，并报告缺口 |
| 结果字段 | 无 `regen_mix_rule` | 有 `regen_mix_rule` |
| 收敛记录 schema | 无 `convergence.history` / `solver_parameters` | 增加历史数组及求解参数 |

因此，当前导出的 ONNX 会被历史 DLL 的列契约检查如实拒绝；C# 编译成功也不代表数值迁移完成。接手者应先适配 `OnnxRotor.cs`、`Simulator.cs`、`Components.cs`、`Configuration.cs`、`Psychrometrics.cs` 及测试/留档，并明确同步两侧收敛记录 schema，再执行整组验证。比较器严格检查额外字段，不能静默过滤历史/参数或删比较项来获得通过。不得改 Python 当前拓扑来迎合历史 DLL。

完整操作、接口、产物规范与数值容差见 [ONNX 与 DLL 交付](../../docs/handoff/ONNX与DLL交付.md)。历史 `docs/` 仅说明旧版交付，不能作为当前项目的执行结果。

## 文件职责

- `export_onnx.py`：读取当前 `artifacts/kan_{xf,main}_single/`，导出 KAN 图和 scaler JSON，不训练、不选择模型。
- `create_reference.py`：用当前 Python 核心和真实工况生成新的 JSON/PNG 参考记录。
- `create_examples.py`、`examples/`：schema v2 工况 YAML 到 JSON；已有 JSON 时拒绝覆盖。
- `src/Dehumidifier.Simulation/`：.NET 8 托管 DLL；`build/*.targets` 是必须跟踪的 MSBuild 源码。
- `tools/Dehumidifier.Verify/`、`samples/WinFormsDemo/`、`tests/`：验证程序、WinForms 示例、历史 C# 测试及 Python 导出测试。
- `build.ps1`：Windows x64 构建、测试、publish、pack；输出到新的 `results/deployment/`。
- `validate_migration.py`：严格逐字段比较 Python/C#，保留迭代次数不一致，退出 0/1。
- `package_delivery.py`：新的自足组装器，输入本次 `--build`、`--models`，不读取旧交付模板。只组装和核验文件；不会把缺失的 Windows/工程验证写成通过。
- `legacy_repackage_delivery.py`：旧布局复封脚本，依赖旧模板和 Windows 证据，保留用于维护；干净克隆使用新组装器。

## 干净克隆可执行的源码检查

从仓库根运行，Python 推荐 3.11/3.12；实际依赖版本应以目标机可安装 wheel 为准。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r deployment/dotnet/requirements-export.txt
.venv/bin/python -m pip install -e simulation/
.venv/bin/python deployment/dotnet/export_onnx.py --help
.venv/bin/python deployment/dotnet/package_delivery.py --help
dotnet build deployment/dotnet/src/Dehumidifier.Simulation/Dehumidifier.Simulation.csproj -c Release
.venv/bin/python -m pytest deployment/dotnet/tests/test_packaging.py -q
```

DLL 项目编译不需要模型；运行推理和导出测试必须通过独立可信渠道提供与当前契约匹配的两个完整 runtime artifact，或在明确授权后训练。仓库不包含 XLS、权重、scaler pickle、ONNX、DLL、NuGet 包或结果。

`src/Dehumidifier.Simulation/models/` 为导出暂存位置，生成内容全部应被 Git 忽略。不要忽略 `src/Dehumidifier.Simulation/build/Dehumidifier.Simulation.targets`。
