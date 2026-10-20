"""历史最终系统打包入口（已停用）。

当前交付边界由两个固定单 KAN artifact 和仿真侧运行记录组成，不再生成
旧集成路由、整机直接电功耗或转轮功率摘要。历史 artifacts 保留在磁盘上
供离线比较，但不会被这个入口重新包装成正式运行时产物。
"""

from __future__ import annotations


def main() -> int:
    print(
        "[提示] 历史最终系统打包已停用；"
        "请先完成双转轮单 KAN smoke/验证，再直接使用两个固定 artifact。"
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

