# 最终风量 KAN + C# / ONNX 部署源码

2026-11-24用户明确当前对外交付版本为v2.1，仅在实际扩充训练数据时递增数据版本（v2.1→v2.2）；字段修正、重训、修复和重新打包不递增。产物名称包含交付日期和该版本号。上一批2.0.4保留为二进制技术构建号及历史记录，不再作为对外交付版本。 详见[版本与命名规则](../../docs/handoff/2026-11-24-交付命名与数据版本规则.md)。

2026-11-23用户已单独确认并完成正式安装：新风artifacts/kan_xf_single现为本轮最佳42轮权重，旧包完整保留在results/training/20261123_xf_cold_blow_1to20/promotion_backup/kan_xf_single；主轮七文件未变。正式加载后的QY1070与已验证候选严格逐字段一致。2.0.4 DLL配套同一模型对。下述“未替换/候选”状态是安装前评估阶段记录，以本段和模型身份清单的runtime_models为准。

2026-11-23新风全量冷吹风量修正为处理风量/20，605379条及原分区/标签保留，重训最佳42/早停62轮。2.0.4候选已通过65项C#、六组严格迁移和两个QY目标独立复算；低湿等指标存在退化，非全面精度提升。原正式模型已按用户确认备份并替换，主轮不重训。 见[本轮说明](../../docs/handoff/2026-11-23-新风冷吹风量修正与DLL候选.md)。

当前**2.0.3**：[统一推理精度修复](../../docs/handoff/2026-11-15-统一推理精度与严格迁移修复.md)。正式Python/ONNX共用double内部计算及float32 I/O策略，源权重不变；C#禁算子融合/预打包并固定CPU顺序会话。新PyTorch与实际DLL原5组严格5/5、额外32组对照一致（4组共同未收敛仅诊断）。原float32历史报告保留，不改阈值、不隐藏history。65项C#与263项Python通过/35跳过；Python代码与dll/models必须同步更新。下文为此前版本的历史状态。

2026-11-15当前**2.0.2**：[封装与接口复核](../../docs/handoff/2026-11-15-DLL封装流程与接口复核.md)、[快速使用](docs/快速使用.md)。已统一导出/C#与正式Python在线缩放边界；新schema2部署契约绑定模型及预处理哈希，必须整套使用本次dll和models。两个包的dll目录均包含配套双入口，共同支持搜索→完整BestConfig→前向复算；NuGet内容与运行报告哈希均在封装时核对。65项C#、15项部署Python测试通过，两个单转轮各257样本的实际DLL数值验证通过。五组PyTorch/C#严格整机迁移仍0/5，不放宽阈值；详见报告。下文2.0.1及更早状态均为历史。

2026-11-15当前版本**2.0.1**：[QY1070一致性排查与交付](../../docs/handoff/2026-11-15-QY1070前反向一致性排查.md)。两入口示例统一`examples/qy1070_case.json`，用`SearchResult.BestConfig`/`BestConfigPath`直接前向复算，避免手工搬运四温度遗漏固定工况或交换HeaterOa/HeaterLow。`validate_v2_roundtrip.py`启动真实搜索/前向独立进程，记录模型/DLL/输入哈希并比较完整物理结果。新旧DLL均在QY1070复算一致，未复现开发端约−40℃；2.0.1补查复算T/W/露点/各负荷与迭代数，53项C#、80项相关Python测试通过。下文2.0.0为历史交付；原默认1220mm工况不作为QY1070输入。

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
