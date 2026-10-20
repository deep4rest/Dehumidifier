# 历史资料

- `training/`：旧通用训练入口、旧配置、数据处理工具；当前唯一训练入口是 `training/single_kan/train.py`。
- `xf_rotor1_retrain/`：旧 XF 候选矩阵、路由、训练代码与测试。对应生成结果已移至根目录 `results/training/legacy_xf/`。
- `artifacts/`：旧版候选、ensemble、打包与字段审计产物，保持原始字节。

这些资料用于离线回溯与兼容测试，不参与默认前向模型选择。历史报告和模型元数据中的绝对路径可能已失效；按 `docs/maintenance/structure-migration.json` 查新位置，不改写原始报告。旧通用训练配置引用的数据文件可能已停用，不应直接作为当前训练配置使用。

相关 Python 导入已调整为 `archive.training`、`archive.xf_rotor1_retrain`；旧 `training/src` 三个兼容模块仍保留。不要从这里复制出另一套当前流水线。
