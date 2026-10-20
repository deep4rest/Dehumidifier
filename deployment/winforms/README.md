# 除湿机整机模型开发交付包

交付对象：C# WinForms 开发人员。部署架构为 WinForms + 本地独立 Python/PyTorch 计算进程，使用 UTF-8 JSON Lines 通信。完整整机计算保留在引擎中，不需要在 C# 中重写空气物性、元件、拓扑或 KAN。

本包是**源码与模型 SDK**，不是已经在 Windows 验证的 EXE。当前 WSL/Linux 下验证记录随包提供。Windows 构建需要联网下载原生依赖；生成后的引擎可以离线运行。未重新训练、未修改真实工况。C# 示例目标为 .NET 8 / Windows x64；已有 .NET Framework 项目可沿用进程协议，但不能原样使用示例的 .NET 8 异步 API。

## 目录

- `engine.py`：启动器，默认从自身目录的 `artifacts/` 读取两个模型包。
- `simulation/dual_rotor_sim/`：完整仿真与推理源码；无需安装训练项目。
- `artifacts/kan_xf_single/`、`artifacts/kan_main_single/`：权重、scaler、模型元数据与训练来源。
- `examples/*.request.json`：可直接发送的两份真实工况请求。
- `examples/configs/`：对应 schema v2 YAML。
- `examples/results/simulation/`：两工况原始参考 JSON、PNG、诊断和模型来源。
- `csharp/`：异步客户端及最小 WinForms 演示工程。
- `verify_sdk.py`：文件哈希、双工况数值、协议错误恢复和未收敛验证。
- `build_windows.ps1`：Windows CPU 版目录打包脚本。
- `SHA256SUMS.json`、`package_info.json`：交付文件清单、版本和环境证据。
- `validation/`：本次交付前验证记录（如已生成）；实际仿真 JSON/PNG 保存在其配套 `results/simulation/`。

## 开发人员先做什么

1. 解压到一个新的目录；先阅读本说明与 `接口说明.md`。
2. 在 Windows x64 安装 Python 3.14 x64。直接依赖版本按当前验证环境固定，见 `requirements-runtime.txt`；PyTorch 固定 `2.13.0`，构建脚本请求官方 CPU wheel。**这些版本在 Windows 上的可获取性和 PyInstaller 兼容性尚待实际构建验证**。若对应 wheel 不可用，应记录失败，并在另一个构建环境选择兼容版本、重新执行对照；不能忽略错误或宣称构建完成。
3. 用 PowerShell 在 SDK 目录执行：

   ```powershell
   .\build_windows.ps1
   ```

   如未安装 Python Launcher，可指定 Python 3.14 的完整路径：

   ```powershell
   .\build_windows.ps1 -Python "C:\Python314\python.exe"
   ```

   脚本创建开发构建环境，先验证源码，再打包 EXE，最后调用 EXE 执行同一组对照。所有阶段失败都会停止。首次安装依赖和打包会较慢。PyInstaller 是构建工具，运行环境不需要安装它；构建后的完整依赖版本记录在 `windows-dist/dependencies-resolved.txt`。
4. 将 `windows-dist/DehumidifierEngine/` **整个目录**随 WinForms 安装，不能只复制 EXE；保留 `_internal/` 和 `artifacts/`。终端用户无需 Python、WSL 或 GPU。
5. 安装 .NET 8 SDK 后运行演示：

   ```powershell
   dotnet run --project .\csharp\DehumidifierDemo.csproj -- "C:\交付包\windows-dist\DehumidifierEngine\DehumidifierEngine.exe"
   ```

   在界面选择 `examples/qy1070_case.request.json`。生产界面由开发人员按同一 config 结构收集参数。示例用于说明接入，不包含完整工况编辑器和系统拓扑 UI。

若 PowerShell 的组织执行策略禁止脚本运行，由开发人员按组织流程处理；也可审阅并手动执行脚本中的命令。

## 源码方式调试

安装与固定版本兼容的 Python 依赖后，可直接运行（路径替换为实际位置）：

```text
python engine.py --results-dir "C:\Users\用户名\AppData\Local\Dehumidifier\results\simulation"
```

每次写入一个紧凑 JSON 请求加换行，读取一行 JSON 响应。一次启动可以连续计算多个工况，两个模型只在首次有效请求加载。结束时关闭标准输入；服务正常读到 EOF 后退出 0。可用 `--artifacts-dir` 显式指定模型父目录。源码模式和 EXE 模式完全使用相同协议。

`--results-dir` 必须指定到可写目录；不要用 `Program Files`。结果每次建新子目录，JSON 与 PNG 配套。绘图缓存默认在结果根目录的父目录 `.cache/matplotlib` 下，可通过 `MPLCONFIGDIR` 改到其他可写位置。

## 自动验证

```text
python verify_sdk.py
python verify_sdk.py --engine "C:\交付包\windows-dist\DehumidifierEngine\DehumidifierEngine.exe"
```

验证器从另一个工作目录启动引擎，不依赖原仓库 PYTHONPATH；核验所有清单文件，比较送风、全部中间空气状态、六输出及冷热负荷。浮点绝对/相对容差均为 `1e-5`（按各字段原单位），不要求跨平台迭代历史逐位相同，但要求两参考工况数值收敛。另测非法工况后的恢复与 max_iter=1 未收敛。结果写入新的 `results/simulation/<时间>_sdk_verify/`。如 SDK 所在目录不可写，使用 `--results-dir` 指向一个尚不存在的可写目录。

完成 Windows 构建后，还应在没有 Python/开发工具的 Windows 电脑上检查：中文路径、普通用户权限、冷启动、连续计算、取消/超时、关闭窗口、中文图片字体与文件导出。Linux 测试不代替这些验证。仅需 Windows 系统自带中文字体，无需分发微软字体。

## 物理模型与解释边界

完整引擎含室外参考风量换算、湿空气公式、混风/分流、带凝结的表冷器、加热器、两台 KAN 及带松弛固定点迭代。此模型是稳态模型，不模拟开机瞬态或转轮蓄热动态。

默认求解参数：relax=0.8，max_iter=60，温度容差 1e-4 ℃，含湿量容差 1e-6 g/kg。QY1070 与 default 参考工况分别 13/14 次收敛，送风含湿量约 0.175297/0.182437 g/kg，冷负荷约 105.3105/126.4849 kW，热负荷约 30.6239/36.9111 kW。它们是模型预测，不是实测验收值。

当前两工况存在训练域外输入：新风转轮再生入口含湿量约 16.2–16.6 g/kg（训练范围 0.20–9.85）；主转轮处理入口约 0.86–0.88 g/kg（训练范围 0.20–0.55）；主转轮冷吹入口约 24 ℃（训练数据固定 27 ℃）。单转轮水分收支偏差约 8%–9%。主转轮特征中没有再生入口含湿量。训练风量的测量参考状态尚需权威澄清；系统保持已有室外参考约定，不能声称已完成数据标定。详细记录见 `模型联调报告.md`。

接口 `warnings` 是固定已知限制提示，**不是逐次输入域检查器或完整物理审计**。界面应区分计算失败、迭代未收敛、数值收敛；不要显示“精度验证通过”或“工程验收通过”。不修改真实输入来回避域外问题。冷热负荷不等于整机电耗。

## 模型更新和维护

替换模型应同时交付两个完整运行时目录及匹配仿真代码，先执行双工况对照。模型字段和 scaler 均由引擎核验；缺失/不匹配直接报错，不回退历史模型。`training_run.json` 中的原训练绝对路径仅用于来源追溯，目标机无需存在该路径。

本包不含原始训练数据、训练环境和历史模型。SHA-256 清单用于交付核验，不是工程验收门禁。包包含项目源码；依赖的许可证随 Windows 打包工具收集情况由开发人员复核，项目代码的对外分发权限按项目约定处理。
