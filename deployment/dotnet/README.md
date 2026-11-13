# 最终风量 KAN + C# / ONNX 部署源码

2026-11-13 v2已实际构建：`露点与能耗预测模型v2.dll`与`最优能耗工况搜索模型v2.dll`，共享核心和NuGet均2.0.0，Windows x64/.NET8托管，不需Python。先读[本次交付](../../docs/handoff/2026-11-13-前向与反向模型v2-DLL交付.md)和[API说明](docs/v2模型DLL调用说明.md)。使用`build_v2.ps1`完成原build流程及两个入口专项测试/打包，再用`package_v2_delivery.py`生成两个中文名目录/ZIP并回读校验。实际C#测试51通过，外部DLL消费通过；严格迁移4/5通过，QY18/17差异仍失败。下方1.0.1及缺SDK描述保留其历史背景，不代表本次构建状态。

本目录与当前 Python 共同采用最终企业规则：处理/冷吹风量输入、R1处理量1/3目标的再生补/漏风、DptAH露点表、零补风告警、relax=0.8。C#核心源码与主项目最终实现一致，NuGet业务版本1.0.1。完整流程见 [ONNX与DLL交付](../../docs/handoff/ONNX与DLL交付.md)。

2026-11-11末端拓扑更新：主转轮处理出口新增`AfterR2_P → CC3后表冷 → AfterCC3 → Heater_Post后加热 → Supply`。在`control_setpoints`中可填写`supply_T_target_degC`；入口温度高于目标时后表冷按需制冷（保留凝结），低于目标时后加热等湿加热，达到目标时两者负荷均0。缺省或null兼容旁通，但仍输出六项热负荷键。总冷量为CC1/CC2/CC3绝对值之和，总热量为Heater_Low/Heater_OA/Heater_Post之和，均为热负荷而非直接电耗。

本次同步的是C#源码与测试，现有DLL/NuGet不会自动更新。当前环境未提供dotnet/csc/mcs，尚未实际编译、执行C#测试或重建DLL；新增无模型测试见`tests/Dehumidifier.Tests/TerminalConditioningTests.cs`。有.NET 8工具链后应先运行这些组件/配置/记录测试，再做真实模型和跨语言严格迁移。既有集成测试中的历史迭代黄金值须以实际新包运行重新核验，不能凭Python结果修改。

`export_onnx.py` 使用正式仿真的 `SingleRotorModel.from_artifact` 核验风量包；旧 XLS 训练加载器仍保留风速契约，不能用于当前包导出。两台完整包需带外恢复，身份见 [模型与数据身份](../../docs/handoff/模型与数据身份.json)。

- `export_onnx.py`：导出两台KAN图、scaler JSON和误差验证，不训练。
- `create_reference.py`：用真实YAML生成新的Python JSON/PNG；任一未收敛退出4。
- `src/Dehumidifier.Simulation/`：.NET 8整机实现；`build/*.targets`必须跟踪。
- `tools/Dehumidifier.Verify/`、`samples/WinFormsDemo/`、`tests/`：验证器、示例和测试。
- `build.ps1`：Windows restore/build/test/publish/pack；默认输出本机results/deployment。
- `validate_migration.py`：逐字段严格比较，包括迭代次数、混风和收敛历史；差异不得隐藏。
- `package_delivery.py`：从本次构建自足组装、校验模型/DLL/ZIP哈希，不借旧二进制；不推断Windows和工程验收通过。
- `legacy_repackage_delivery.py`及`docs/`：历史布局复封工具和成品说明；依赖旧模板，历史验证不作为本次证据。

源码契约已同步，但不等于本次Windows重新构建和严格迁移全部通过。历史记录中QY1070曾有Python12次/C#13次差异，重建时必须实际比较、保留结论。另机复现必须锁定模型身份和依赖环境。

```bash
conda run -n dehumidifier python -m pip install -r deployment/dotnet/requirements-export.txt
conda run -n dehumidifier python deployment/dotnet/export_onnx.py --help
conda run -n dehumidifier python -m pytest deployment/dotnet/tests -q
```

模型、ONNX、DLL、NuGet、结果与缓存均不上传。`src/Dehumidifier.Simulation/models/`整个目录忽略；其中的JSON同样是生成产物。所有正式训练仍需用户明确授权。
