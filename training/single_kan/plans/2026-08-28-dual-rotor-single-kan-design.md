> 路径维护说明（2026-09-05）：本文件记录原设计；训练源码现位于 `training/single_kan/`，默认记录和操作步骤以根目录 README、AGENTS.md 为准。原训练契约不因目录迁移改变。

# 双转轮单 KAN 重训与能耗语义修正设计

## 目标与假设

为新风转轮和主转轮各建立一个固定结构的多输出 KAN，使模型只学习六个直接空气状态；`power` 仅作为加热器功率记录的离线对照，处理出口露点统一由预测含湿量和系统压力派生。这样运行时不再依赖逐输出路由，也不会把转轮电机、风机或直接电功耗混入公开结果。

## 数据与模型契约

- 权威源固定为 `data/新风转轮数据集.xls`（工作表 `CollectRecordXF`）和 `data/主转轮数据集.xls`（工作表 `CollectRecord`）；每次读取先校验当前 SHA-256。
- XLS 第一行是技术字段，第二行是中文说明；适配器使用技术字段作为稳定键，并保存中文表头映射。源文件不修改、不插值、不做通用 IQR 删除。
- 新风转轮保留历史 10 个原始输入，删除 `Label_PowerConsumption`、露点列及全部派生量，输出六个直接状态。仅在审计清洗层排除已知异常 ID `1407`。
- 主转轮输入固定为九个原始量：厚度、直径、入口含湿量、转速、再生温度、冷吹入口温度、处理入口温度、处理风速、冷吹风速；输出六个直接状态，禁止 `derived_features`、`power` 和露点标签。
- 主转轮按厚度×直径几何组 70/15/15 划分；新风转轮使用 `300×1070` 测试组、历史 fold 1 验证组、其余开发组训练，几何组不得泄漏。
- 新风 KAN `[24]`/grid 6/order 3；主转轮 `[16]`/grid 5/order 3，AdamW、`lr=1e-3`、无 weight decay、最多 200 epochs、patience 20。MinMax 与可逆 `log1p` 可用于数值预处理，但 artifact 对外只接受原始物理量。

## 运行时与能耗契约

`build_system()` 只加载 `artifacts/kan_xf_single/` 与 `artifacts/kan_main_single/` 两个单模型 artifact，并按元数据校验输入列和六个直接输出；旧 registry、默认路由和 routing 文件不参与前向。两个模型均调用 `dew_point_water_C(process_outlet_W, pressure_Pa)` 生成运行时露点。

`RotorOutputs` 不含 `power_kW`；`SystemResult`、CLI、JSON、运行记录和仪表盘不公开 `electrical_kW`、风机功率或转轮电机功率。公开 `energy_breakdown` 只保留 `CC1_kW`、`CC2_kW`、`Heater_Low_kW`、`Heater_OA_kW`；`cooling_kW` 和 `heating_kW` 表示焓差热负荷。

## 验证范围

先对每个源文件抽取约 500 条代表性副本，TDD 验证表头映射、类型、缺失、重复 ID、`power`/露点排除及几何组隔离，并输出审计分析。完成小样本分析后，完整数据处理和正式训练必须等待用户明确确认。集成阶段使用固定种子和 `--smoke-test` 执行 L0 静态检查与 L1 真实数据流（读取→缩放→训练→保存→重载→推理→露点派生→仿真调用）；精度只记录报告，不设置 promotion 门禁。
