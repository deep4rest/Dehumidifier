# 2026-09-05 项目结构整理

## 目标与边界

将当前训练代码收拢到 training/single_kan，根目录 results 统一保存运行记录，artifacts 仅保存当前两台转轮模型；历史代码和旧模型归档。保留原始 Excel、历史模型和结果的字节，记录迁移路径及 SHA-256。工作区已有修改不重置，不启动全量训练，不更改物理计算、模型结构或调试阶段加载契约。

## 方案

仅移动结果不能解决多个旧训练入口并存；将所有代码重新打包会扩大导入和 pickle 兼容风险。本次采用有边界的整理：当前训练包改名，旧训练工具与 XF 实验归档，保留 training/src 的三个兼容模块，统一默认结果目录和入口。

## 执行与验证

- [x] 保存迁移前清单；迁移当前训练、历史代码/产物、散落表格与文档；修正 Python 导入与根目录推导。
- [x] training/single_kan/train.py：默认 smoke/full 各自创建 results/training 下独立目录；全量默认运行完成后更新 artifacts 中对应模型；显式 output-root 保持调用方指定目录。运行记录与失败状态写 pipeline_report.json，旧记录不覆盖。
- [x] simulation/dual_rotor_sim/cli.py 与 simulation/visualize.py：统一 results/simulation，统一 save_simulation_run 留档。
- [x] 为独立训练运行、失败留档、smoke 隔离和可视化不覆盖添加回归测试；先确认旧行为失败，再实现。
- [x] 更新 AGENTS.md、README.md、CLAUDE.md 单一规则来源、结果目录说明和 .gitignore。
- [x] 运行当前训练/仿真测试、历史回归与入口 --help；逐个校验迁移文件 SHA-256、三个兼容模块一致性与 git diff --check。

## 整理前证据

- 当前基线：170 passed，1 条历史 scaler sklearn 版本警告。
- artifacts 下 16 个旧目录，当前 kan_xf_single 与 kan_main_single 不存在。
- 历史 XF 实验约 648 MB，主要为 work 下训练产物。
- 仿真两条输出路径：simulation/results 按次留档，simulation/outputs 固定文件覆盖。

完整文件迁移映射见同目录 structure-migration.json。历史结果内嵌绝对路径按原始证据保留，不批量改写。

## 实施结果（2026-09-06 完成验证）

- 43 项路径迁移、676 个文件；708,299,674 原始字节。651 个文件字节未变，25 个源码/配置/设计文档因导入或路径调整发生变更。模型权重、原始数据与历史结果均未改写。
- 两份权威 XLS 的 SHA-256 与训练契约相符；三对兼容模块 MD5 一致。
- 当前测试：176 passed，1 条历史 MinMaxScaler 版本警告（训练版本 1.6.1，当前 1.9.0）；整理前已有该警告。
- 历史 XF 测试：180 passed。已验证8个当前/历史入口 --help、Python 编译检查与 git diff --check。
- 新增回归验证训练独立留档、全量模型与记录分离、显式输出防覆盖、失败报告、恢复训练不改写原 checkpoint，以及可视化连续运行不覆盖旧图。
- 迁移后的所有676个文件均未被 .gitignore 隐藏；README/AGENTS/CLAUDE 本地链接有效。旧 experiments、doc、simulation/results、simulation/outputs、training/configs 路径已移走，不保留软链接或重复入口。
- 默认 pytest 范围固定为当前训练与仿真；生成产物默认忽略，已迁移的历史证据明确保留可跟踪。
- 当前 artifacts 两个运行时模型仍缺失；实际调用 build_system 确認缺失时报错，没有历史模型替代。本次未进行全量训练或真实工况的新模型仿真。
- 历史记录采用归档而非永久删除，因此本次以清晰分层为目标，未以释放磁盘容量为目标。工作区原有未提交改动保留，未提交 Git。

## 默认路径迁移表

| 原位置 | 当前位置 |
| --- | --- |
| experiments/dual_rotor_single_kan_retrain | training/single_kan |
| experiments/xf_rotor1_retrain | archive/xf_rotor1_retrain |
| experiments/xf_rotor1_retrain/work、reports | results/training/legacy_xf/work、reports |
| training 下旧脚本、configs、历史数据处理模块 | archive/training |
| artifacts 下旧模型与报告 | archive/artifacts |
| simulation/results | results/simulation |
| simulation/outputs | results/simulation/legacy_figures |
| doc | docs/reference |
| 根目录及 data 下企业表格 | data/templates |
| 123.xls | data/reference/123.xls |

未新增工程验收或 promotion 门禁；模型包中的 training_run.json 只作来源索引。历史 JSON 与模型快照中的旧绝对路径不改写，按逐文件迁移表定位。

## 独立审查

已针对迁移前备份审查训练存储、恢复、模型安装和文档命令，未发现本次整理引入的实质性问题。保留已说明的边界：运行时包更新并非两个目录的跨目录原子事务，文件系统写入失败时须核对两包状态。
