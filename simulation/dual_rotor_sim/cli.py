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


def cmd_optimize(args) -> None:
    """Search current models, then independently rerun the chosen configuration."""
    import hashlib
    import math
    from datetime import datetime
    from dataclasses import asdict
    import torch
    import yaml
    from .optimization import CONTROL_NAMES, SearchSettings, assess_result, independent_assessments_match, optimize_controls
    from .core.system.solver import build_system

    output = None
    try:
        cfg_path = Path(args.config).resolve()
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        settings = SearchSettings(
            dew_point_target_degC=args.dew_point_target,
            dew_point_tolerance_degC=args.dew_point_tolerance,
            supply_T_target_degC=args.supply_temp_target,
            energy_reference_kW=args.energy_reference,
            dew_point_penalty_coefficient=args.dew_point_penalty_coefficient,
            supply_T_min_degC=args.supply_temp_min,
            supply_T_max_degC=args.supply_temp_max,
            bounds=tuple(tuple(getattr(args, name)) for name in ("cc1_bounds", "cc2_bounds", "r1_regen_bounds", "r2_regen_bounds")),
            step_degC=args.step, cooling_weight=args.cooling_weight,
            heating_weight=args.heating_weight, seeds=tuple(args.seeds),
            maxiter=args.maxiter, popsize=args.popsize,
            initial_controls=None if args.initial_controls is None else tuple(args.initial_controls),
        )
        # Validate and load before creating output files; mismatched packages fail.
        parse_simulation_config(config)
        torch.set_num_threads(1)
        system = build_system()
        output = Path(args.output_dir).resolve() if args.output_dir else RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_optimization_{cfg_path.stem}"
        output.mkdir(parents=True, exist_ok=False)
        best_seen = None
        with (output/"evaluations.jsonl").open("w",encoding="utf-8") as stream:
            def record(item):
                nonlocal best_seen
                stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n")
                if item["feasible"]:
                    value = item["optimization_loss"]
                    best_seen = value if best_seen is None else min(best_seen,value)
                if item["evaluation"] % 100 == 0:
                    stream.flush()
                    if not args.quiet:
                        text = "尚无可行点" if best_seen is None else f"当前最低归一化loss {best_seen:.8f}"
                        print(f"已计算 {item['evaluation']} 个独立参数组合；{text}",file=sys.stderr,flush=True)
            report = optimize_controls(config,settings,system=system,evaluation_callback=record)
        root = Path(__file__).resolve().parents[2]
        report["model_file_sha256"] = {}
        for profile in ("xf","main"):
            package = root/"artifacts"/f"kan_{profile}_single"
            for name in ("model/best_model.pt","model/scalers.pkl","model_metadata.json"):
                path = package/name
                report["model_file_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["original_config_sha256"] = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        report["best_independently_verified"] = False
        if report["best"] is not None:
            best_config = report["best_config"]
            # _build_from_config constructs fresh strict artifacts and reruns
            # the original coupled solver. No cache or search state is reused.
            verified = _build_from_config(best_config)
            assessment = assess_result(verified,settings,energy_reference_kW=report['energy_reference_kW'])
            match = independent_assessments_match(assessment, report['best'])
            report["independent_verification"] = assessment
            report["best_independently_verified"] = bool(assessment["feasible"] and match)
            if report["best_independently_verified"]:
                best_yaml = output/"best_config.yaml"
                best_yaml.write_text(yaml.safe_dump(best_config,allow_unicode=True,sort_keys=False),encoding="utf-8")
                saved = save_simulation_run(verified,best_config,best_yaml,output/"best_forward")
                report["best_forward_record"] = str(saved/"result.json")
                report["best_dashboard"] = str(saved/"fig1_dashboard.png")
        report_path = output/"search.json"
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        summary = {"status":report["status"],"objective":report["objective"],
                   "actual_electricity_optimized":False,"global_minimum_proven":False,
                   "settings":asdict(settings),"baseline":report["baseline"],"best":report["best"],
                   "energy_reference_kW":report['energy_reference_kW'],"loss_contract":report['loss_contract'],
                   "best_independently_verified":report["best_independently_verified"],
                   "unique_forward_evaluations":report["unique_forward_evaluations"],
                   "feasible_evaluations":report["feasible_evaluations"],"report":str(report_path)}
        if args.json:
            print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        else:
            if report["best_independently_verified"]:
                print(f"找到并独立复算可行参数，总冷热负荷 {report['best']['thermal_load_sum_kW']:.5f} kW，归一化loss {report['best']['optimization_loss']:.8f}。")
                for name in CONTROL_NAMES:print(f"  {name}: {report['best']['controls'][name]:g} ℃")
            else:print("未找到通过独立复算的可行参数。")
            print(f"搜索记录：{report_path}")
        if not report["best_independently_verified"]:
            raise SystemExit(5)
    except (OSError,ValueError,RuntimeError,TypeError,ImportError) as exc:
        print(f"[优化错误] {exc}",file=sys.stderr)
        if output is not None:print(f"记录目录：{output}",file=sys.stderr)
        raise SystemExit(2) from exc


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
    opt = sub.add_parser("optimize",help="露点约束下搜索最低冷/热负荷控制参数")
    opt.add_argument("--config",required=True,help="固定入口、风量和几何的schema v2工况")
    opt.add_argument("--dew-point-target",type=float,required=True,help="送风露点目标（℃），必须显式给出")
    opt.add_argument("--dew-point-tolerance",type=float,default=1.,help="目标双边误差容限（℃），默认1且不得超过1")
    opt.add_argument("--supply-temp-target",type=float,required=True,help="最终送风温度目标（℃），末端冷热自动按需运行")
    opt.add_argument("--energy-reference",type=float,help="固定归一化负荷基准（kW）；缺省取原配置在目标送风温度下的六项总负荷")
    opt.add_argument("--dew-point-penalty-coefficient",type=float,default=.1,help="归一化露点距离惩罚系数，默认0.1")
    opt.add_argument("--supply-temp-min",type=float,help="可选送风干球温度下限（℃）")
    opt.add_argument("--supply-temp-max",type=float,help="可选送风干球温度上限（℃）")
    opt.add_argument("--cc1-bounds",type=float,nargs=2,default=(9.,13.),metavar=("MIN","MAX"))
    opt.add_argument("--cc2-bounds",type=float,nargs=2,default=(13.,17.),metavar=("MIN","MAX"))
    opt.add_argument("--r1-regen-bounds",type=float,nargs=2,default=(50.,140.),metavar=("MIN","MAX"))
    opt.add_argument("--r2-regen-bounds",type=float,nargs=2,default=(50.,140.),metavar=("MIN","MAX"))
    opt.add_argument("--step",type=float,default=.1,help="控制温度搜索网格步长（℃，默认0.1）")
    opt.add_argument("--cooling-weight",type=float,default=1.,help="兼容参数，按六项之和定义只能为1")
    opt.add_argument("--heating-weight",type=float,default=1.,help="兼容参数，按六项之和定义只能为1")
    opt.add_argument("--seeds",type=int,nargs="+",default=[20261110,20261111,20261112])
    opt.add_argument("--maxiter",type=int,default=40,help="每个随机种子的差分进化代数")
    opt.add_argument("--popsize",type=int,default=8,help="每个维度的种群倍数")
    opt.add_argument("--initial-controls",type=float,nargs=4,metavar=("CC1","CC2","R1_REGEN","R2_REGEN"),help="可选搜索起点；仍用真实模型重新评价")
    opt.add_argument("--output-dir",help="新建输出目录；拒绝覆盖已有目录")
    opt.add_argument("--json",action="store_true")
    opt.add_argument("--quiet",action="store_true")
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    else:
        parser.print_help()
        raise SystemExit(0)


if __name__ == "__main__":
    main()
