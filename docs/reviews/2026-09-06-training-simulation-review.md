# 训练与仿真代码审查（2026-09-06）

范围：当前 `training/single_kan/`、保留的 `training/src/kan_model.py`、`simulation/dual_rotor_sim/` 与可视化入口。审查当前工作区完整代码，不把未提交的历史修改一概归于最近一次目录整理。初始审查仅增加本报告；以下问题与行号保留为修复前证据。用户随后授权修复，处理情况见末尾“修复记录”；未启动全量训练。

优先级：P1 为应优先修复的运行或结果可信性问题；P2 为有具体触发条件的正确性、维护性或效率问题。下列结论由当前源码和小型合成数据/临时产物验证，未将历史模型替换成当前正式模型。独立审查任务因额度中断，最终证据由主审直接验证。

## 1. [P1][运行效率] 全量训练实际每步保存 checkpoint，间隔参数失效

位置：`training/single_kan/trainer.py:383-384`、`368-369`、`444-459`。

`fit()` 每轮调用 `train_steps(1)`；`train_steps()` 在结尾无条件保存 checkpoint。于是 `checkpoint_interval_steps=2000` 只能限制循环内部的保存，不能限制函数结尾保存。复现使用16条合成记录、batch=8、epochs=2，真实执行4次 optimizer step，观察到4次 checkpoint_saved 事件。

每次 checkpoint 包括整个历史列表、optimizer、scaler、排列与随机状态。全量训练不仅增加磁盘I/O，随着逐步历史增长，累计历史序列化工作量还会随总step数呈二次增长。当前小样本集成测试调用多步 `train_steps`，没有覆盖 `fit` 的这一调度行为。

建议：将“单步训练、按间隔评估/保存、结束时最终保存”放到同一个循环；`fit` 与有界运行共用循环。增加保存次数测试，不以缩短训练或减少审计字段规避问题。

## 2. [P1][代码复用/包边界] 仿真模型产物反向依赖未安装的训练包

位置：`training/single_kan/model.py:215-222`、`478-479`；`simulation/dual_rotor_sim/core/system/components/rotor.py:568-569`；`simulation/pyproject.toml`。

保存 scaler 时把自定义 `training.single_kan.model.TargetScalerAdapter` 对象写入 pickle。仿真加载时必须能够 import 训练包，但仿真安装配置仅安装 `dual_rotor_sim*`。仿真文件内另有 `_ArtifactTargetScaler` 适配器也无济于事，因为 pickle.load 在选择该分支之前已经失败。

复现：用合成数据拟合 scaler 并调用实际 save_artifact；随后使用同一 `.venv/bin/python`，在 `/tmp` 中以绝对产物路径调用 `SingleRotorModel.from_artifact`，得到 `ModuleNotFoundError: No module named 'training'`。在仓库根目录的测试进程中导入训练包会掩盖此问题。

建议：运行时产物采用内建容器、sklearn对象和基础参数描述变换，或将所需类放到明确安装的共享运行时包。为仿真包增加离开仓库根目录的产物加载测试；不要依赖向 sys.path 临时补项目根来维持可安装包边界。

## 3. [P1][代码质量] 合法零含湿量被评估器当成失败

位置：`training/single_kan/evaluator.py:212-222`、`275-284`；`simulation/dual_rotor_sim/core/system/psychrometrics.py:119-143`。

评估器先把负含湿量预测投影到0，再计算露点。公共湿空气函数按既定干燥极限语义对W=0返回负无穷，评估器随后用 `isfinite` 把它当作错误。只要任何样本的处理出口含湿量预测≤0，整次评估就会中止，非负投影审计本身无法完成。

复现：16×6的正、非恒定目标，预测仅第一条处理出口含湿量设为−0.1，其他值等于目标；`evaluate_predictions` 抛出 `EvaluationError: 水面露点派生产生非有限值`。

建议：统一训练与仿真对W=0的边界语义；记录干燥极限数量，明确其露点指标不可有限比较。不要用任意正epsilon替换0来伪造有限露点，也不要把NaN与合法负无穷混为一类。

## 4. [P1][代码质量/结果可信性] 不收敛的仿真没有结构化失败状态

位置：`simulation/dual_rotor_sim/core/system/system.py:568-575`；`simulation/dual_rotor_sim/cli.py:69-87`；`simulation/dual_rotor_sim/run_records.py:44-69`。

固定点迭代耗尽后仅发出 RuntimeWarning 并返回末次结果。CLI 随后按普通成功路径生成JSON/图片并打印完成；SystemResult与结果JSON未包含converged、迭代次数或残差。终端警告离开当前进程就丢失，后续Agent仅读记录无法判断该工况是否达到稳态。

复现：以每次将温度增加10℃、含湿量增加1g/kg的确定性模拟替身驱动两次迭代；调用正常返回，只有1条warning，没有converged标记。

建议：将收敛状态、实际残差、迭代次数纳入SystemResult和留档。调试阶段可保留末次迭代，但必须标记为未收敛；不得沿无标记的成功路径展示。此建议不要求恢复promotion或工程验收门禁。

## 5. [P2][代码质量/复用] 系统压力未传递到表冷器和空气状态派生量

位置：`simulation/dual_rotor_sim/core/system/components/cooling_coil.py:55`；`simulation/dual_rotor_sim/core/system/air_state.py:38-46`；`simulation/dual_rotor_sim/config.py:309-347`。

配置接受系统压力，质量流量换算和转轮露点使用该值；表冷器却调用固定常压的humidity_ratio_sat，AirState.Tdp/RH也走固定常压函数。因此非标准压力下，同一次仿真会混用两个压力。

复现：系统压力80,000Pa、入口30℃/15g/kg、表冷器设定10℃，代码得到7.63005g/kg，按同一规范公式带实际压力计算应为9.69565g/kg。对于代码产生的7.63005g/kg，AirState报告露点10℃，80kPa下应为6.52179℃。这会影响凝结量、冷负荷和下游状态；标准大气压工况不会暴露。

建议：给状态/组件建立一致的压力上下文，统一调用已有pressure-aware公式并传递到派生展示。无需改动已有凝结建模假设。

## 6. [P2][代码质量] 恢复训练没有验证数据身份和顺序

位置：`training/single_kan/train.py:225-235`、`374-378`；`training/single_kan/trainer.py:426-433`、`444-459`。

恢复后直接注入本次数据；保存排列长度不匹配时 `_apply_resume_progress` 返回，调用方再将游标置0继续运行。checkpoint未绑定实际训练/验证行ID与顺序、样本模式/数量等信息，batch size还会被新命令参数静默覆盖。源XLS哈希不变也不能保证抽样后的实际数组相同。

复现：16条训练数据运行1步，恢复时注入8条且输入值扩大100倍的数据，仍成功训练到第2步；使用的scaler最大值仍约1.834，而新数组同列最大值约183.406。示例用于证明边界未校验，不表示当前权威XLS被修改。

建议：checkpoint绑定源哈希、profile、smoke/full模式、分区行ID及顺序、关键训练控制；注入真实数据后验证，不匹配则明确拒绝“续训”。如需改变数据继续训练，应另有明确语义，不能冒充确定性恢复。

## 7. [P2][代码质量/复用] 运行时未校验输入 scaler 列顺序

位置：`simulation/dual_rotor_sim/core/system/components/rotor.py:572-574`、`606-614`、`664-667`。

加载器校验元数据的input_cols以及输出scaler列，却未比较x_scaler.input_cols。推理直接将按元数据排列的NumPy数组送给scaler，sklearn不再拥有DataFrame列名来发现错误。归档/重打包时同维度但不同列顺序的scaler可能被接受，最终缩放错误而非立即报错。

复现：正常生成单模型产物，将pickle中的x_scaler.input_cols反转，`SingleRotorModel.from_artifact`仍成功。该检查是已有列契约的完整性，不是要求新增promotion门禁。

建议：集中实现训练保存、训练重载和仿真重载共用的scaler契约校验，核对输入/输出列名顺序与n_features_in_；避免多套局部校验逐渐分叉。

## 8. [P2][运行效率] 每次验证对同一批数据做两次完整前向

位置：`training/single_kan/trainer.py:291-310`、`348-350`。

`_validation_loss`先缩放整个验证集并前向计算MSE；随后 `_evaluate_current`再次缩放、前向、反缩放，计算物理指标。给真实模型挂forward hook，调用一次该验证组合可计数到2次forward。

建议：共享一次验证预测，同时计算scaled loss与物理尺度指标；保留统一评估核心。预处理在训练数据固定时也可缓存，而非每个epoch对相同行重复调用sklearn。计数已实测，具体节省的墙钟时间需在真实数据与目标硬件上测量，不在本审查中虚报加速倍数。

## 9. [P2][代码质量] 未达到首个验证间隔的运行无法写训练历史

位置：`training/single_kan/trainer.py:80-105`；`training/single_kan/train.py:462`。

首个验证前best_validation_loss保留为正无穷；to_dict原样输出，而 `_write_json`禁止NaN/Infinity。当max_steps小于validation_interval_steps（例如只设max_steps=4而保持默认2000），即使末尾独立评估成功，写train_history.json仍会失败。

复现：对未发生验证的TrainHistory调用实际 `_write_json`，得到 `Out of range float values are not JSON compliant: inf`。README当前的有界smoke显式设置interval=2，因而避开了这一问题，但API和CLI仍允许该组合。

建议：把“尚无验证结果”表示为null/显式状态，或在训练结束统一执行一次最终验证；不要将inf当作可序列化的正常最佳损失。

## 复用与维护建议

优先修复以上可复现行为，再做结构收敛：

1. 让fit和有界训练共享一个循环，评估共用一次预测。
2. 统一模型/scaler契约及无训练包依赖的运行时序列化，减少TargetScalerAdapter与_ArtifactTargetScaler双实现。
3. 用一致压力上下文连接配置、状态、组件和展示；共用已有湿空气公式。
4. 三对现存兼容模块受AGENTS的字节一致性约束，不把重复本身判成bug，不直接删副本。将来提取共享模块时保留旧pickle别名，先增加独立安装/跨进程重载测试。

## 验证范围

使用临时合成数据进行4步训练、checkpoint恢复、scaler序列化/加载与确定性求解替身检查；未运行全量训练、真实当前模型工况或GPU性能测试。当前正式模型缺失不影响上述单元级复现，但无法由本次审查得出工程精度或真实工况吞吐结论。

本次实际执行 `.venv/bin/python -m pytest -q`：176 passed，1 warning，59.17秒。警告为历史MinMaxScaler的sklearn版本1.6.1与当前1.9.0不一致。测试通过只能说明已有覆盖通过，以上复现说明仍存在未覆盖边界。

## 修复记录（2026-09-06）

用户授权修复后，九项问题均已落实代码与回归覆盖：

| 编号 | 修复行为 | 主要回归证据 |
| --- | --- | --- |
| 1 | `fit` 与 `train_steps` 共用循环；checkpoint 按间隔和结束保存，去除训练入口的末尾重复写入；早停原因写入最终状态 | `test_fit_saves_at_interval_and_once_at_end`、`test_checkpoint_exactly_on_final_interval_is_not_written_twice`、`test_fit_preserves_early_stopping_in_final_checkpoint`、短 smoke 集成 |
| 2 | pickle 只保存原生 scaler 与普通变换状态；目标适配器移至可安装仿真包的 `scalers.py` | 子进程禁止 `training` 导入后，重载真实生成的 artifact 并预测 |
| 3 | 接受零湿度的负无穷露点，单列干燥极限数量与有限可比样本数；JSON 输出 `null`，无可比样本的露点 MAE 为 `null` | 零湿度/负预测投影、全干预测、严格 JSON 解析及旧记录重绘；不替换为 epsilon |
| 4 | `SystemResult.convergence`、JSON、仪表盘记录成功或未收敛状态、次数、残差、容差及末次计算输入；未收敛留档后退出 4 | 确定性收敛/发散替身、真实 CLI 留档和非零退出、重绘状态保留 |
| 5 | 所有 `AirState`、转轮输入、表冷器、加热器、分流、混风统一压力；不同压力直接混风拒绝 | 80 kPa 凝结量/露点/RH、全系统每个中间节点与转轮输入压力检查 |
| 6 | checkpoint schema v2 绑定数据来源、实际数组及顺序、模型契约和控制；先核验后恢复排列/游标 | 数据长度、数值、顺序、验证集、来源、四项控制变化均拒绝；缺真实数据不能训练；同数据恢复不改旧目录 |
| 7 | 训练保存、训练重载与仿真重载共用输入/输出 scaler 列顺序和维度检查 | 两种加载器均拒绝同维度但反转输入列的 scaler |
| 8 | 周期验证一次前向同时计算 scaled loss 与物理指标，保留原模型 train/eval 状态 | 真实 forward hook 计数：一次训练前向加一次验证前向 |
| 9 | 未验证的最佳损失序列化为 `null`，重载恢复内部哨兵；短训练继续完成末尾评估、模型与报告保存 | 严格 JSON 序列化、真实源表 2-step smoke 使用默认 2000-step 验证间隔 |

新增审查回归位于 `training/single_kan/tests/test_review_regressions.py` 和 `simulation/tests/test_review_regressions.py`，入口行为扩展至训练集成测试。README 和 AGENTS.md 已同步续训、压力、干燥极限、收敛退出状态与共用 scaler 规则。

兼容限制：没有数据指纹的旧 checkpoint 不能直接当作确定性续训状态，需启动新训练记录；已有历史文件不改写。`src.preprocessing` pickle 别名与三对字节一致的兼容模块保留。性能证据为 checkpoint 写入/forward 次数，未测量或宣称全量训练加速倍数。

最终验证：`.venv/bin/python -m pytest -q` 为 **205 passed，1 warning，74.76 秒**；警告仍为上述历史 scaler 的 sklearn 版本差异。`git diff --check`、当前训练/仿真代码编译和两个入口 `--help` 检查通过；三对兼容模块 `md5sum` 各自一致。测试中的有界真实数据 smoke 输出到临时测试目录，未安装模型到 `artifacts/`。当前该目录仍只有说明文件，未运行全量训练或真实当前模型整机仿真，本次修复不构成精度或工程验收。
