# AGENTS.md

本文件是所有Agent的项目维护规则唯一来源；步骤见README和docs/handoff。中文优先，用户最新明确指令优先。

## 当前有效基线

用户确认最终版本为主项目中已同步的Python/C#企业实现：处理/冷吹风量输入、再生目标R1处理量/3的受控补/漏风、DptAH露点和零补风修复。NuGet业务版本1.0.1。最初源码移交误用了旧worktree，现已纠正；禁止再以旧风速/三股全量汇合/过冷水公式覆盖最终版本。

接手先读README、docs/handoff/README.md、训练流程、系统拓扑与物理契约、ONNX与DLL交付。统一使用Miniconda的dehumidifier环境（Python 3.12），不再使用项目根.venv；执行前conda activate dehumidifier，非交互可用conda run -n dehumidifier。先git status，保留已有修改，核验实际模型元数据、文件哈希及工况，不能把历史报告当当前证据。

## 训练与数据

- 最终训练通过 training.single_kan.train **显式 --dataset-dir data/转轮数据集_尺寸隔离_8比1比1**，进入candidate风量流程；CSV三分区已按厚度×直径隔离，不重划、不泄漏。
- 先相关测试和 --smoke-test --max-steps 4；本次上传/联调不授权全量训练。全量必须用户明确授权并带 --confirm-full。
- 六CSV与两模型身份见docs/handoff/模型与数据身份.json；数据只读，每次读取校验SHA-256。旧XLS及NEW数据审阅工具保留作来源追溯；无dataset-dir的旧XLS风速入口不用于最终运行时训练/安装。
- 原始输入XF10/main9，各固定六输出KAN；power与Excel露点仅审计，不入输入、scaler、loss或输出。处理出口含湿量先log1p，其余目标直变换；scaler仅train拟合。
- candidate流程不自动安装artifacts；每个候选保存model、checkpoints、reports与运行记录。恢复既有模型优先带外复制完整包，不能混权重/scaler/元数据。candidate尚无已实现的resume入口，不把旧XLS续训规则套用到它。

## 仿真与物理

- 正式仅加载artifacts/kan_xf_single、kan_main_single；核验风量列、profile、六输出、架构和scaler。缺包就报错，不读routing/candidate registry，不自动回退。
- 主转轮冷吹量=再生量；R1再生入口目标=R1处理量/3，R2再生出与R1冷吹出先完全混合，超量漏风，低于目标按配置上限补风；补风可为0，缺口继续算并warning、记regen_mix_rule。
- 所有边界风量m³/h，内部kg干空气/s，五个独立流量统一经build_runtime_inputs和SimulationConfig.m3h_to_kg_da_s，按室外参考状态换算；YAML风速仅兼容，不进入当前KAN。
- 企业饱和压T≤0℃冰面、T>0℃液态水，水汽比0.62198。DptAH表88项覆盖0至-87℃，线性插值，低端外推，表标定101325Pa；不得声称查表对任意压力物理自洽。
- 零湿度露点-inf合法，JSON保存null，重绘结合W=0恢复；NaN/+inf或非零W对应-inf报错，不用epsilon掩盖。
- 压力贯穿全部状态和元件，不同压力不得直接混合。表冷器低于露点时凝结；自然满足设定点时Q=0正常。
- 公开仅CC1/CC2/Heater_Low/Heater_OA热负荷；CC分项为负，cooling=abs(CC1)+abs(CC2)，heating=两加热器之和；无风机/电机直接电耗。
- 联立max_iter=60、relax=0.8，T容差1e-4℃、W容差1e-6g/kg；记录收敛、迭代、history、solver_parameters、末次实际输入/残差/容差。未收敛JSON/PNG可留档但退出4，缺字段标未评估。

## 保存、部署与验证

- 用户已授权本项目在每次需求完成并通过相关验证后提交并推送到当前工作分支的远程分支；推送前检查diff、暂存清单、分支与上游，仅提交本次需求相关源码及文档。无上游时以同名分支推送origin并设置上游；不强推、不推送全部分支或旧本机历史。推送失败须说明原因，不得声称已同步。
- 远程提交与分支更新优先使用已连接的GitHub MCP；基于当前远程head创建提交，以expected_sha且force=false更新分支，随后fetch并快进同步本地。MCP认证不代表本机Git已登录，不导出连接凭据。
- 训练results/training、仿真results/simulation、部署results/deployment、源码移交results/source_upload；长期产物不散落根目录/源码/系统临时目录。JSON/PNG同次同目录；展示调整用save_dashboard_from_record重绘。
- scaler共用dual_rotor_sim/scalers.py，新pickle只保存原生scaler和普通状态。保留rotor.py历史pickle兼容。training/src与simulation/dual_rotor_sim/core的kan_model.py、preprocessing.py、feature_engineering.py必须逐字节相同，修改双侧同步并md5sum。
- C#核心为最终版本，不按旧工作树回退；ONNX导出采用正式风量运行时加载器，旧训练load_artifact不适用最终风量候选包。参数/拓扑/物性变更须双侧同步。
- 源码契约、单轮ONNX误差、Windows部署、严格迁移、工程验收分别报告。历史QY1070迭代12/13差异不能抹去，不能降低比较阈值或拿旧DLL填新交付。
- 不上传原始数据、权重/scaler/checkpoint、仿真/训练结果、ONNX/DLL/NuGet/ZIP、bin/obj/egg-info或本机设置。MSBuild build/*.targets是源码例外。旧本机历史已有模型，禁止直接推送旧历史。
- 缺外部资料测试可明确skip，REQUIRE_EXTERNAL_FIXTURES=1严格失败。删除有数据历史目录需明确授权；环境、Agent目录和字体缓存不能擅自清理。
- 交付写明源码变更、实际测试、产物路径和未解决限制；入口/流程/输出变化同步维护本文件和README。
