# 外部数据恢复

GitHub 仅保存源码和操作文档，不包含 XLS、企业 XLSX 模板或参考数据。另机应通过获准的带外方式恢复下列两个文件，保留原始字节，不用 Excel 打开后另存。

| 路径 | 工作表 | SHA-256 |
| --- | --- | --- |
| `data/新风转轮数据集.xls` | `CollectRecordXF` | `116ea18458bf831225765472440a21da8045cb8e9bfadd898f677b004ce026cb` |
| `data/主转轮数据集.xls` | `CollectRecord` | `a1173dee9789832e54087bd2a0f7d8865b8371fa7789d4a191449a45731f704f` |

训练读取时重新校验哈希，变化立即拒绝。数据契约、技术列映射、分区和训练步骤见 [训练流程](../docs/handoff/训练流程.md)。恢复运行时模型需要整个 `artifacts/kan_xf_single/` 与 `artifacts/kan_main_single/` 包，不能只复制 `.pt`。

没有数据/模型时，可做源码与合成数据单元测试；不能据此运行真实训练或整机仿真。
