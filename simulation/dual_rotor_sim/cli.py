"""CLI 入口：dual-rotor-sim simulate。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    ConfigError,
    build_runtime_inputs,
    parse_simulation_config,
)
from .run_records import RESULTS_DIR, _json_number, save_simulation_run


def _build_from_config(cfg: dict):
    """校验配置，并映射到系统内部的 kg dry-air/s 接口。"""
    from dual_rotor_sim import build_system, forward_simulate

    parsed = parse_simulation_config(cfg)
    runtime = build_runtime_inputs(parsed)

    system = build_system()
    return forward_simulate(
        system,
        runtime["oa"],
        runtime["ra"],
        runtime["regen_oa"],
        runtime["params"],
        runtime["setpoints"],
    )


def _print_result(result, as_json: bool) -> None:
    if as_json:
        out = {
            "convergence": result.convergence,
            "supply_T": round(result.supply_state.T, 3),
            "supply_W": round(result.supply_state.W, 4),
            "supply_Tdp": _json_number(round(result.supply_state.Tdp, 3)),
            "supply_RH": round(result.supply_state.RH, 4),
            "supply_h_kJ_kg": round(result.supply_state.h, 3),
            "cooling_kW": round(result.cooling_kW, 4),
            "heating_kW": round(result.heating_kW, 4),
            "energy_breakdown": {
                key: round(value, 4)
                for key, value in result.energy_breakdown.items()
            },
        }
        print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(result.summary())


def cmd_simulate(args) -> None:
    try:
        import yaml
    except ImportError:
        print("[错误] 缺少 PyYAML，请先安装：pip install pyyaml", file=sys.stderr)
        raise SystemExit(1)

    cfg_path = Path(args.config).resolve()
    try:
        with open(cfg_path, encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        if not args.quiet:
            print("加载配置，随后加载 KAN 模型…", file=sys.stderr)
        result = _build_from_config(cfg)
    except Exception as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        run_dir = save_simulation_run(
            result=result,
            config=cfg,
            config_path=cfg_path,
            results_dir=RESULTS_DIR,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[错误] 仿真结果留档失败：{exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    if result.convergence.get("converged") is False:
        _print_result(result, args.json)
        print(f"[未收敛] 仅保存调试记录：{run_dir}", file=sys.stderr)
        raise SystemExit(4)
    if not args.quiet:
        print(f"完成。结果记录：{run_dir}\n", file=sys.stderr)
    _print_result(result, args.json)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dual-rotor-sim",
        description="低露点双转轮除湿机整机仿真",
    )
    sub = parser.add_subparsers(dest="command", metavar="<命令>")
    sim = sub.add_parser(
        "simulate",
        help="运行单工况仿真",
        description="使用 schema v2 YAML 配置运行整机仿真",
    )
    sim.add_argument(
        "--config",
        required=True,
        metavar="FILE",
        help="schema v2 YAML 配置文件路径",
    )
    sim.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    sim.add_argument("--quiet", action="store_true", help="不输出进度信息")
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    if args.command == "simulate":
        cmd_simulate(args)
    else:
        parser.print_help()
        raise SystemExit(0)


if __name__ == "__main__":
    main()
