# 运行记录

- `training/<运行编号>/`：本次训练的 pipeline_report、checkpoint、模型副本、日志和评估。
- `simulation/<运行编号>/`：同次仿真的 result.json、工况快照和 fig1_dashboard.png。
- `training/legacy_xf/`：迁移保留的历史 XF work 与 reports，原始内容不变。
- `simulation/legacy_figures/`：旧固定输出图片，缺少一一对应的完整工况 JSON，不用于证明当前仿真结果。

按根目录 README 操作。新运行不得覆盖旧记录；仅重绘时保留源 JSON，图片在该记录下另命名。历史报告中的旧路径与旧字段按原始证据保留，位置映射见 `docs/maintenance/structure-migration.json`。
