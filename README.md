# 低露点双转轮除湿机建模平台

2026-11-23新风全量冷吹风量修正为处理风量/20，605379条及原分区/标签保留，重训最佳42/早停62轮。2.0.4候选已通过65项C#、六组严格迁移和两个QY目标独立复算；低湿等指标存在退化，非全面精度提升。原正式模型未替换，主轮不重训。 见[本轮说明](docs/handoff/2026-11-23-新风冷吹风量修正与DLL候选.md)。

当前最终实现采用两台固定六输出**风量 KAN**、企业再生补/漏风规则及DptAH露点表。Python和C#采用同一套业务契约，当前v2 NuGet业务版本2.0.3。所有Agent先读 [AGENTS.md](AGENTS.md)，再读 [另一台电脑接手入口](docs/handoff/README.md)。

2026-11-15[统一推理精度修复](docs/handoff/2026-11-15-统一推理精度与严格迁移修复.md)：权重不变，Python/ONNX统一双精度内部计算、float32输入输出边界，DLL锁定CPU算子路径。新正式PyTorch与实际C#原5组严格**5/5通过**，额外32组32/32对照一致（4组共同未收敛，仅作诊断），阈值/history未改。65项C#、263项Python通过/35跳过。此结论基于升级后的运行精度；旧float32对照及2.0.2的0/5原记录保留，不能混称同一基准。请同步Python代码并整包替换dll/models。

2026-11-15[再次核查DLL封装与接口](docs/handoff/2026-11-15-DLL封装流程与接口复核.md)：修正正式在线缩放精度路径、补模型/预处理哈希绑定、补齐单包联调所需的双入口DLL，封装同时核对NuGet内容及运行报告身份。65项C#、15项部署Python测试和两台各257样本的实际DLL对照通过；QY前反向一致。**本次五组PyTorch/C#严格迁移仍为0/5**，迭代/history差异保留，不能宣称完全迁移通过。[快速使用](deployment/dotnet/docs/快速使用.md)随两个成品提供双击示例、完整DLL引用工程、单位和错误处理说明；旧段落为各次历史记录。

2026-11-15按用户要求统一以**QY1070**验证前向与搜索模型。[本次排查与2.0.1交付](docs/handoff/2026-11-15-QY1070前反向一致性排查.md)：目标−45℃/−35℃，搜索与新进程前向均为−45.001807℃/−35℃，所有物理记录一致，未复现开发端约−40℃，其确切原因仍需开发端原输入/二进制证据。修订版补强独立复算校验，示例直接传`BestConfig`或`BestConfigPath`，53项C#、80项相关Python测试通过。两模型共用核心，相对原链路只启用CC3与Heater_Post末端；其余模块一致。下文历史默认工况的9.1/16.2/98.5/91.7℃及Q_ref354.0165属于**1220mm的default_case**，不能移用QY1070。本次两原工况严格迁移1/2通过，QY18/17次差异仍保留。

2026-11-13已实际完成两个Windows x64/.NET 8托管DLL：**露点与能耗预测模型v2.dll**、**最优能耗工况搜索模型v2.dll**，无需Python运行时。[交付与验证](docs/handoff/2026-11-13-前向与反向模型v2-DLL交付.md)、[API调用](deployment/dotnet/docs/v2模型DLL调用说明.md)。51项C#测试及258项Python测试通过（Python35跳过），独立消费程序前向/反向均运行成功。严格跨语言五工况4通过，QY迭代18/17次差异保留；封装成功不等于该项严格迁移通过。用户已授权本次完成后同步GitHub，二进制、ONNX与生成结果仍本机交付。

2026-11-11已按用户要求在主转轮处理出口后串联**CC3后表冷→Heater_Post后加热→送风**，按送风温度目标自动开关，并将六个热工组件纳入总负荷。本次优化目标为送风温度**−35℃**、露点**−45℃±1℃**，并加入用户确认的归一化分区距离惩罚。[拓扑与优化交接](docs/handoff/2026-11-11-末端调温与分区加权优化.md)替代此前无末端、无送风温度目标的优化结论。当前DLL2.0.0已含末端功能，旧NuGet1.0.1仍为历史版本。

源码包含最终风量仿真实现、2026-11-09新风基线升级及2026-11-10主转轮重训的身份与验证说明；最初GitHub提交里的旧风速Python已纠正。模型、数据、仿真结果、ONNX/DLL/NuGet/ZIP和旧本机Git历史不上传。实际模型与数据身份见 [清单](docs/handoff/模型与数据身份.json)。

2026-11-10[完整系统仿真验证](docs/handoff/2026-11-10-完整系统仿真验证.md)已完成：213个合法入口工况中210收敛、3未收敛、33个末态过饱和，原始湿度均非负。拓扑和元件记账通过，但水分守恒、训练超域及严格后端差异仍待解决，不能认定工程验收通过；没有整机实测精度结论，本轮仅保留本机。

## 目录与入口

主转轮训练覆盖仍按用户要求暂缓。此前新版模型接入回归24项、三个原工况17/18/15次收敛的记录位于本机`results/simulation/20261110_新版新风接入运行复核`，属于未增加末端时的历史运行证据，不替代当前拓扑验证或物理精度结论。

| 路径 | 职责 |
| --- | --- |
| training/single_kan/train.py | 统一训练CLI；**传 --dataset-dir 才走最终风量CSV训练** |
| training/single_kan/candidate_*.py | 尺寸隔离CSV读取、MLP/KAN、训练、验证与保存 |
| training/src/ | 共用KAN和保留的兼容模块 |
| simulation/dual_rotor_sim/ | 最终风量仿真、物性、求解器、记录与JSONL bridge |
| simulation/configs/ | 真实schema v2工况，不能为迎合模型修改 |
| deployment/dotnet/ | 最终C#整机、ONNX导出、Windows构建、严格比较和封装源码 |
| deployment/winforms/ | 较早的Python JSONL桥接开发路线，区别于纯C# DLL路线 |
| docs/handoff/ | 训练、拓扑、导出/封装和恢复说明 |
| artifacts/ | 本机两台正式模型包；GitHub仅附README |
| data/ | 带外数据；GitHub仅附恢复说明 |
| results/ | 本机训练/仿真/部署/源码上传验证记录，不上传 |
| archive/ | 历史兼容源码，不作为当前模型自动回退 |

## 环境与源码回归

使用 Miniconda，在仓库根目录执行（Windows、Linux/WSL 通用）：

```bash
conda env create -f environment.yml
conda activate dehumidifier
python -m pytest -q
```

环境使用 Python 3.12，`environment.yml` 安装根依赖并以 editable 方式安装仿真包；已有环境可用 `conda env update -n dehumidifier -f environment.yml` 更新。非交互执行可用 `conda run -n dehumidifier python ...`，不再创建项目内 `.venv`。

需要 NVIDIA GPU 时，在该环境内安装 CUDA 13.0 版 PyTorch，并验证实际 GPU 计算；驱动兼容性以目标机检查结果为准，其他平台按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择安装包：

```bash
python -m pip install --upgrade "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(2, device='cuda').sum().item())"
python -m pip check
```

`environment.yml` 保留通用依赖，重建环境后需再执行 GPU 安装命令。安装或验证 CUDA 环境不代表已授权全量训练。

没有外部资料时，相应用例明确跳过，合成数据与临时模型测试照常运行。恢复后设置环境变量 `REQUIRE_EXTERNAL_FIXTURES=1`，执行 `python -m pytest -q -m 'requires_data or requires_models'` 强制外部资料检查（PowerShell 用 `$env:REQUIRE_EXTERNAL_FIXTURES='1'`）。运行环境版本没有全部锁定，不能仅凭相同源码保证跨机逐位一致。

## 当前新风基线与训练

2026-11-09用户已选定**9输入、三个湿度非负输出的KAN**为新风正式基线。主体9→24→6、grid6/order3、3600参数，处理出口含湿量保留log1p/expm1；三个湿度采用参数量为0的Softplus平滑物理下界(beta=50)，温度输出保持原形式。主转轮于2026-11-10按原结构与参数重训，用于恢复整机验证能力；整机物理契约不变。

本机已安装到`artifacts/kan_xf_single`。GitHub仅提供源码、模型身份和说明，模型/scaler、CSV、训练报告及ONNX必须带外恢复，完整SHA见[身份清单](docs/handoff/模型与数据身份.json)。修正版新风三份CSV放在`data/新风转轮基线_9输入_冷吹1比20`（旧目录保留审计），605,379条及485,041/57,000/63,338的尺寸隔离分区不变。原始转速列仍在审计数据中。

```bash
python -m training.single_kan.train \
  --profile xf --model-type kan \
  --dataset-dir data/新风转轮基线_9输入_冷吹1比20 \
  --device cpu --smoke-test --max-steps 4 --batch-size 256 --seed 20260908
```

默认`--humidity-output-constraint auto`会为两台log1p风量候选启用Softplus头。`none`仅显式重现旧无头离线对照；`--process-humidity-transform log`仅用于新风纯log实验。仅明确授权全量训练时才移除smoke参数并增加`--confirm-full --device cuda`。新候选不自动替换正式包，不省略dataset-dir进入旧XLS入口。

本次选定权重为最佳117轮、137轮早停。测试集处理出口负预测282→0，W<0.01 MAE降低62.63%，整体W MAE增加0.77%；再生出口W MAE增加28.20%的取舍如实保留。[本次修改、结果与验证边界](docs/handoff/2026-11-09-新风基线升级.md)，详细流程见[训练流程](docs/handoff/训练流程.md)。

## 仿真与部署

2026-11-10本机已按用户要求为主转轮加入与新风相同的三个湿度Softplus头(beta=50)，并重训安装。主体仍9→16→6、grid5/order3、2160参数，AdamW lr=1e-3/wd=0、batch256、seed20260908、200轮/最佳199轮；原47991/6232/6384条分区、转速输入及log1p目标变换保持。两台正式加载器均要求schema_version=4和完整的非负头契约。

主转轮全部60607条数据的原始三湿度均非负，测试处理W MAE为0.00318036g/kg；三份真实原工况分别17/18/15次满足原收敛容差。训练冷吹温度恒27℃、系统实际超域及水汽偏差仍未解决。[当前Softplus结果与验证边界](docs/handoff/2026-11-10-主转轮Softplus非负约束.md)，[此前无头对照记录](docs/handoff/2026-11-10-主转轮重训与整机验证.md)。本轮按用户决定仅保留本机，不推送GitHub。

带外恢复两个完整包 `artifacts/kan_xf_single/`、`artifacts/kan_main_single/` 并按身份清单校验后：

```bash
dual-rotor-sim simulate --config simulation/configs/qy1070_case.yaml
python simulation/visualize.py --config simulation/configs/default_case.yaml
```

每次JSON/PNG同目录留档；未收敛仍留调试结果，退出4。改图使用已有JSON重绘，不重算。模型缺失/契约错误直接报错，不自动换历史模型。拓扑、流量、露点与负荷见 [系统契约](docs/handoff/系统拓扑与物理契约.md)。

ONNX和C#构建见 [部署交接](docs/handoff/ONNX与DLL交付.md)。源码契约一致、ONNX图误差、Windows部署、严格数值迁移、工程验收分别举证。历史QY1070有Python12次/C#13次迭代差异，不能推断严格迁移全部通过。

## 反向优化（Python，含末端调温）

优化必须同时给出送风露点和温度。本次为**露点−45℃±1℃、送风温度−35℃**；按最终Supply而非主轮原出口验收。末端目标写入`control_setpoints.supply_T_target_degC`，后表冷与后加热按同一目标自动运行，不再把送风温度留作自由输出。旧前向配置未给温度目标时，两末端组件保持0负荷旁通，原始YAML不被改写。

物理总负荷`Q_sum`为CC1、CC2、CC3冷量绝对值与Heater_Low、Heater_OA、Heater_Post加热量之和。控制搜索最小化复合评分：`J=Q_sum/Q_ref+0.1*w(e)*|e|/(1℃)`，`e=预测送风露点−目标露点`。`w(e)`在`[-1,-0.5)`为0.35、`[-0.5,+0.3]`为0.1、`(+0.3,+1]`为0.5；超出±1℃仍直接不合格。`Q_ref`整次搜索固定，本次原配置在−35℃送风目标下为354.01652084694706kW。J无量纲，物理kW负荷另报，不把距离惩罚当作真实热量或电耗。

```powershell
conda run --no-capture-output -n dehumidifier python -m dual_rotor_sim.cli optimize --config simulation/configs/default_case.yaml --dew-point-target -45 --dew-point-tolerance 1 --supply-temp-target -35 --dew-point-penalty-coefficient 0.1 --output-dir results/simulation/新的搜索目录 --json
```

可用`--energy-reference`显式指定固定正基准；缺省从原配置在本次送风温度目标下计算一次，失败则明确报错，不改用其他候选或逐点重标定。候选还须满足最终送风温度误差≤1e-4℃、原闭环收敛容差及未裁剪饱和约束。无可行点或独立复算失败退出5，拒绝覆盖目录。物理裸负荷最低点与最小J候选分开记录，不能将二者混称。

末端10项专项、优化58项专项通过；最新整套253通过、35按原规则跳过。本机当前结果目录为`results/simulation/20261111_末端调温与加权优化`，具体最优控制参数和完整证据以该次报告为准，本次不推送。此前4044组、146.283443kW及871组边界对照属于**无末端、无送风温度目标的历史结果**，不能代替本次六负荷分区优化；单边约−92℃与10.67%结论仍维持失效。追溯见[双边更正](docs/handoff/2026-11-11-露点双边误差约束更正.md)、[旧拓扑诊断](docs/handoff/2026-11-11-优化拓扑与露点边界诊断.md)。

本次已评估中的最低J候选为CC1/CC2/R1再生/R2再生=**9.1/16.2/98.5/91.7℃**，最终送风**−35℃、露点−45.061384℃**，误差−0.061384℃；冷负荷305.226705kW、加热负荷25.811331kW，物理总负荷**331.038036kW**，评分**J=0.935705894**，16次迭代。完成150组试跑、4049组正式搜索和112组合并候选细化，各运行独立计数且有重复；未证明全局最优。原默认归一化基准不在露点目标带，不能把负荷差直接称同等送风服务节能率。

## 后续源码上传

每次需求完成并通过相关验证后，检查差异和暂存清单，提交本次需求相关源码及文档，再推送到当前工作分支的远程上游。无上游时使用 `git push -u origin <当前分支>`。不强推；推送失败需明确报告。提交者姓名和邮箱通过本仓库的 Git 配置设置。

远程操作优先使用已连接的 GitHub MCP：以当前远程提交为父提交创建新提交，更新分支时检查原 head 且不强推，再 fetch 并快进同步本地。MCP 使用连接账号的提交身份；本机 Git 命令行认证独立配置。

使用 `scripts/prepare_source_upload.py` 白名单生成空目录快照及独立manifest；排除全部数据、模型、结果、构建元数据与缓存。旧本机历史已跟踪模型，禁止直接push --all/--mirror。新GitHub clone可正常提交源码，仍应检查暂存文件。`deployment/dotnet/src/Dehumidifier.Simulation/build/*.targets`是必须保留的源码。
