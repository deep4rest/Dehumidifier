"""开发交付包启动器；源码模式和 PyInstaller 目录模式共用。"""
import os
from pathlib import Path
import sys


def main():
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    # 原仓库开发时允许找到同仓库仿真包；交付包的包路径始终在自身根目录下。
    source = root / 'simulation'
    if not source.is_dir() and not getattr(sys, 'frozen', False):
        source = root.parents[1] / 'simulation'
    if not getattr(sys, 'frozen', False):
        sys.path.insert(0, str(source))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from dual_rotor_sim.bridge import main as bridge_main
    if '--artifacts-dir' not in sys.argv:
        sys.argv.extend(['--artifacts-dir', str(root / 'artifacts')])
    return bridge_main()


if __name__ == '__main__':
    raise SystemExit(main())
