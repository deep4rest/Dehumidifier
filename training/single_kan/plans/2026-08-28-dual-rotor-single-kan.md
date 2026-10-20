> 路径维护说明（2026-09-05）：本文件记录原设计；训练源码现位于 `training/single_kan/`，默认记录和操作步骤以根目录 README、AGENTS.md 为准。原训练契约不因目录迁移改变。

# 双转轮单 KAN 重训与能耗语义修正 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `spml:ml-subagent-dev` to implement this plan task-by-task.

**Goal:** 用两个独立的六输出单 KAN 替换旧路由模型，并把 `power` 限定为离线加热器对照，清除公开直接电功耗字段。

**Experiment directory:** `experiments/dual_rotor_single_kan_retrain/`

**Hypothesis:** 原始物理量输入加固定多输出 KAN 足以支持双转轮状态预测；由含湿量统一派生露点可避免把派生标签和 `power` 泄漏进模型及运行时能耗语义。

**Validation scope:** 小样本 TDD/审计先行；集成阶段启用一次 L0+L1，完整数据处理/训练须在用户确认后运行。

**Evaluation design:** 每 2,000 optimizer steps 做完整验证；内存模型和 checkpoint 重载调用同一个 evaluator；日志包含阶段开始/进度/结束、MAE/RMSE/NRMSE/R²、非有限值与非负投影统计，并对缺失 checkpoint、恢复失败、空验证集、聚合失败、非有限指标和停滞给出中文硬失败。

**Architecture:** `data.py` 负责哈希锁定、双表头映射、可审计清洗和几何组切分；`model.py`/`trainer.py` 复用项目 KAN builder 完成单模型训练、评估和 checkpoint；`train.py` 组合两个 profile 并生成固定 artifact。运行时由 `SingleRotorModel` 按 artifact 元数据加载，露点和热负荷在系统层派生。

---

## Shared Scaffold

### Existing infra (don't touch unless explicitly listed)

- `training/src/kan_model.py`：唯一 KAN 构建器。
- `simulation/dual_rotor_sim/core/system/psychrometrics.py`：水面露点金标公式。
- `simulation/dual_rotor_sim/core/system/system.py`：双转轮拓扑和焓差部件。
- `experiments/xf_rotor1_retrain/` 与历史 `artifacts/`：只作对照，不覆写。

### Needs setup

- 新建实验目录下的 `config.py`、`data.py`、`model.py`、`evaluator.py`、`trainer.py`、`train.py` 与 `tests/`。
- 正式输出目录固定为 `artifacts/kan_xf_single/`、`artifacts/kan_main_single/`；smoke 输出使用临时目录，不覆盖历史产物。

## Subtask 1: 数据契约与小样本适配

**Role:** 锁定两个权威 XLS、映射双表头、排除已知异常并构造无泄漏切分。

**Implementation:** 新建 `config.py` 和 `data.py`。暴露 `read_authoritative_dataset(profile, path=None)`, `prepare_sample(profile, n=500)`, `validate_dataset(profile, frame)`, `split_dataset(profile, frame)`；哈希变化、缺列、非有限、重复 ID、非法几何组必须抛中文 `DataContractError`。`power` 与露点只能出现在 `audit_columns`，不能出现在 `input_cols`/`direct_outputs`。

**Unit Tests:** `tests/test_data_contract.py` 覆盖源哈希、双表头映射、ID 1407 排除、字段类型/缺失/重复、主转轮 70/15/15 几何组隔离、新风固定测试/验证组、power/露点张量排除。

## Subtask 2: 单 KAN、评估与 checkpoint

**Role:** 提供 10→6 XF 和 9→6 main 的固定多输出模型及共享评估器。

**Implementation:** 新建 `model.py`、`evaluator.py`、`trainer.py`；使用 `build_kan`，实现 MinMax 与可逆 `log1p`、AdamW 训练、每 2,000 steps 验证、checkpoint 保存/恢复、固定种子。artifact 元数据必须有 `direct_outputs`、`derived_outputs=["处理出口露点"]`、`audit_columns=["power", "处理出口露点"]` 等字段，且目标 scaler 不得包含它们。

**Unit Tests:** `tests/test_model_trainer.py` 覆盖形状、架构、无派生配置、log1p 往返、保存重载一致、露点金标和非有限硬失败。

## Subtask 3: 运行时双单模型加载与公开结果语义

**Role:** 将仿真前向切换为两个单 artifact，统一派生露点，清除电功耗字段。

**Implementation:** 修改 `simulation/dual_rotor_sim/core/system/components/rotor.py`、`solver.py`、`system.py`、`run_records.py`、`cli.py`、`dashboard.py`。新增 `SingleRotorModel.from_artifact()` 严格校验列契约，`build_system()` 只加载两个固定目录；删除 `RotorOutputs.power_kW`、`SystemResult.electrical_kW` 和 Fan/Motor 明细；保持 fan 配置兼容但不公开其功率。仪表盘只绘制冷/热负荷及四个热工组件，处理侧状态图占整行。

**Unit Tests:** 更新/新增 `simulation/tests`：删除 routing 文件仍可加载；缺失/篡改契约中文硬失败；两台转轮露点由含湿量派生；JSON、摘要和图片文案无 `electrical_kW`/Fan/Motor。

## Subtask 4: 双模型训练集成 `[INTEGRATION]`

**Hypothesis:** 两个固定单 KAN 可在真实权威 Excel 上完成读取、训练、保存、重载和六输出前向，并通过统一露点函数接入系统。

**Implementation:** 完善 `train.py` 的 `--profile xf|main|all`、`--smoke-test`、`--resume`、固定种子、tqdm、逐步日志和 checkpoint；smoke 仅使用每份源的代表性副本。生成 `artifacts/kan_xf_single/` 与 `artifacts/kan_main_single/` 的单模型、scaler、元数据、配置快照和报告，不生成 routing.json。

**Integration Tests:** `tests/test_integration.py` 使用 smoke profile 验证读取→缩放→反向传播→保存→重载→六输出预测→露点派生，确认 power/露点不进入 loss，删除 routing 文件不影响加载。

**Validation Pyramid:** L0 静态检查配置/日志/optimizer/维度/路由依赖；L1 约 5 分钟真实 Excel smoke 流程。数据契约、非有限输出、checkpoint 或露点失败硬停；模型精度仅报告。

## Subtask 5: 文档、审计报告与门禁

**Role:** 同步 `需求.md`、`README.md`、实验报告模板，记录小样本统计和新旧逐字段比较边界。

**Tests:** 文档/JSON grep 审计确认不再宣称直接电功耗或转轮电机功率，且明确“联调可运行≠工程验收通过”。
