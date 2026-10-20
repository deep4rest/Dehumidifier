"""历史集成 promotion 入口（已停用）。

正式模型不再通过逐输出路由或 promotion 门禁加载。请使用
`training/single_kan/train.py` 训练并生成
`artifacts/kan_xf_single` 与 `artifacts/kan_main_single`；历史 ensemble
artifacts 仅保留作离线对照。
"""

from __future__ import annotations


def main() -> int:
    print(
        "[提示] 历史 ensemble promotion 已停用；"
        "请运行 training/single_kan/train.py。"
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

