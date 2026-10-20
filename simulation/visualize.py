"""为指定单工况保存独立的 JSON 与 ``fig1_dashboard.png``。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from dual_rotor_sim import build_system, forward_simulate
from dual_rotor_sim.config import build_runtime_inputs, load_simulation_config
from dual_rotor_sim.dashboard import _configure_matplotlib
from dual_rotor_sim.run_records import RESULTS_DIR, save_simulation_run


_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default_case.yaml"
OUTPUT_DIR = RESULTS_DIR


def _load_config(config_path: Path) -> dict:
    """加载 YAML 配置，并复用 CLI 的唯一流量映射。"""
    parsed = load_simulation_config(config_path)
    runtime = build_runtime_inputs(parsed)
    return {
        "oa": runtime["oa"],
        "ra": runtime["ra"],
        "regen_oa": runtime["regen_oa"],
        "params": runtime["params"],
        "base_sp": runtime["setpoints"],
        "output_dir": OUTPUT_DIR,
    }


def _setup_chinese_font() -> None:
    """保留旧入口，统一使用 dashboard 的字体配置。"""
    _configure_matplotlib()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="低露点双转轮除湿机整机仿真可视化",
    )
    parser.add_argument(
        "--config",
        metavar="文件路径",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="YAML 配置文件路径（默认使用 configs/default_case.yaml）",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"[错误] 配置文件不存在：{config_path}", file=sys.stderr)
        raise SystemExit(1)

    with config_path.open(encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    runtime = _load_config(config_path)

    print(f"加载配置：{config_path}")
    print("正在加载两台转轮的单 KAN 模型…")
    try:
        system = build_system()
        result = forward_simulate(
            system,
            runtime["oa"],
            runtime["ra"],
            runtime["regen_oa"],
            runtime["params"],
            runtime["base_sp"],
        )
        target = save_simulation_run(
            result=result,
            config=raw_config,
            config_path=config_path,
            results_dir=runtime["output_dir"],
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if result.convergence.get("converged") is False:
        print(f"[未收敛] JSON 与调试结果图已保存：{target}", file=sys.stderr)
        raise SystemExit(4)
    print(f"完成。JSON 与结果图已保存：{target}")


if __name__ == "__main__":
    main()
