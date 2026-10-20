> 历史成品说明：这里只描述当时的交付布局和操作，不是本次构建/验证证据。当前Python/C#共同采用最终风量、企业混风与DptAH契约；最新步骤以deployment/dotnet/README.md和docs/handoff/ONNX与DLL交付.md为准。

# 低露点双转轮除湿机建模平台

本交付为 2026-09-19 企业反馈修复后的 C# 整机 DLL + ONNX Windows x64 版本，沿用 `dehumidifier_dotnet_onnx_20260910_141414` 的目录结构与说明文档形式。由本次源码重新构建，运行时不需要 Python、PyTorch 或 WSL。接入步骤见 [使用说明](使用说明.md)，逐文件目录 `文件清单.md` 仅在完整成品交付中存在，不随本次源码上传。

## 目录导航

| 位置 | 用途 |
|---|---|
| `lib/` | 整机 DLL、XML 接口说明及托管依赖 |
| `models/` | 新风/主转轮 ONNX、输入输出契约与预处理参数 |
| `runtimes/win-x64/native/` | ONNX Runtime 原生依赖 |
| `nuget/` | 本地 NuGet 源、主包、依赖包及离线还原说明 |
| `examples/` | QY1070/default 完整工况 JSON |
| `examples/WinFormsDemo/source/` | 使用包内 NuGet 的独立 WinForms 示例源码 |
| `examples/WinFormsDemo/publish/` | 自包含 Windows x64 WinForms 发布程序 |
| `verify/` | 自包含 Windows x64 自检程序 |
| `source/deployment/dotnet/` | 本次 C# 实现、模型、测试与构建脚本 |
| `source/docs/` | 企业需求原件与模型修改记录 |
| `results/simulation/` | 本次 Python 基准、Windows 配套 JSON/PNG |
| `reports/` | 本次构建日志、Windows 验证、数值对照和 ONNX 契约 |

## 本次系统模型修改

1. 再生混风：Heater_OA 入口目标风量为新风转轮处理出口风量的三分之一。主转轮再生出与新风转轮冷吹出先完全混合；超过目标漏掉多余，不足则按配置上限补风。补风仍不足时记录缺口并告警，执行记录位于 `result.regen_mix_rule`。
2. 湿空气换算：零度及以下饱和压采用企业冰面多项式；水汽比为 0.62198。低湿区露点与含湿量使用企业 DptAH 表分段线性互换，超出表上界按公式反演，极低非零湿度按末段外推；零湿度保持干燥极限。
3. `regen_outdoor_air.flow_m3h` 允许为 0，表示没有可用补风；混风不足时系统继续计算，只在结果 warnings 和 `regen_mix_rule.shortfall_*` 中记录缺口。NuGet 主包版本为 1.0.1，避免旧 1.0.0 缓存继续加载旧露点公式。

两台 KAN 的 ONNX 模型保持不变，变化在整机物性与混风逻辑。低湿区露点较旧版本上移约 3～4℃是换算契约变化，不是重新训练造成的精度变化。旧版本的验证数字不作为本次结果。

## 模型与物理契约

- 新风 10 输入、主转轮 9 输入，使用处理/冷吹风量；各一个固定六输出 KAN。露点派生计算，不是模型直接输出。
- 输入温度 ℃、含湿量 g/kg 干空气、风量 m³/h、压力 Pa；DLL 统一进行内部流量换算。
- 冷热负荷是空气热负荷，不能解释为整机电耗。四个分项为 CC1、CC2、Heater_Low、Heater_OA；自然满足设定值时零负荷正常。
- 表冷器允许发生凝结；压力随空气状态传播；零湿度露点在 JSON 中为 null，显示为“干燥极限”。
- 每次计算留存同目录 JSON/PNG，记录实际迭代输入、残差、容差和收敛状态。

## 快速运行与开发接入

首次接入按 [使用说明](使用说明.md) 操作。先执行 `verify/Dehumidifier.Verify.exe` 检查模型加载、整机计算与结果留档，再接入业务界面。已有 .NET Framework 4.x 项目不能直接引用本 .NET 8 DLL。

运行环境为 Windows x64；开发使用 .NET 8 SDK。`dependencies.json` 记录依赖版本，`THIRD-PARTY-NOTICES.md` 记录第三方来源。部署时保留整个发布目录，尤其是模型、契约及原生库。

## 源码与重新构建

在安装 .NET 8 SDK 的 Windows PowerShell 中，从包根目录执行：

```powershell
.\source\deployment\dotnet\build.ps1 -Root .\source -Output .\build\windows
```

该步骤还原依赖、编译、运行 C# 测试、发布验证程序与 WinForms，并生成 NuGet 主包。源码测试的开发依赖需要已缓存或联网还原；`nuget/` 提供的是业务接入的离线依赖，不包含完整测试 SDK。`build.ps1` 会清空指定构建输出目录，因此应使用新的专用目录。

独立示例源码的构建方法见 `nuget/离线还原说明.md`。`source/README.md` 与 `source/AGENTS.md` 为源项目快照，涉及训练及 Python 的命令在完整仓库中使用，本运行交付包不包含训练数据或 Python 环境。

## 验证与留档

`reports/validation.md` 是完整成品中的数值对照报告，比较对应 Python 基准与 Windows C# 结果，包括逐节点状态、两台转轮、冷热负荷、收敛过程及再生混风规则。`reports/windows_validation.md` 是完整成品中的 Windows 验证报告，记录对应环境、测试、自检与异常路径。这两份报告及 `reports/build.log` 均属于运行产物，不随本次源码上传。

ZIP 同目录的 `.zip.sha256` 用于校验压缩包；`.manifest.json` 保存包内每个文件的 SHA-256。原 prebuild 和旧交付均保留，不覆盖。

## 使用边界

这些验证证明当前 Windows 电脑上的部署与数值迁移一致性，不代表训练域外精度、水分守恒、设备性能或工程物理验收。现场 PLC 通信、联锁和保护由业务系统负责，未收敛或计算异常时不得显示为成功。
