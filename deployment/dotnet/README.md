# 最终风量 KAN + C# / ONNX 部署源码

本目录与当前 Python 共同采用最终企业规则：处理/冷吹风量输入、R1处理量1/3目标的再生补/漏风、DptAH露点表、零补风告警、relax=0.8。C#核心源码与主项目最终实现一致，NuGet业务版本1.0.1。完整流程见 [ONNX与DLL交付](../../docs/handoff/ONNX与DLL交付.md)。

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
.venv/bin/python -m pip install -r deployment/dotnet/requirements-export.txt
.venv/bin/python deployment/dotnet/export_onnx.py --help
.venv/bin/python -m pytest deployment/dotnet/tests -q
```

模型、ONNX、DLL、NuGet、结果与缓存均不上传。`src/Dehumidifier.Simulation/models/`整个目录忽略；其中的JSON同样是生成产物。所有正式训练仍需用户明确授权。
