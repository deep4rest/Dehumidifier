"""Subtask 6 后半段：full-development 重训选中候选、外层 300×1070 验收、正式产物准备。

用法：
    .venv/bin/python archive/xf_rotor1_retrain/finalize_xf.py \
        --routing archive/xf_rotor1_retrain/work/xf_full_matrix/routing.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from archive.xf_rotor1_retrain.candidates import CANDIDATES, get_candidate
from archive.xf_rotor1_retrain.config import (
    EXPERIMENT_DIR,
    RESULTS_DIR,
    FIXED_FOLD_GROUPS,
    MODEL_INPUT_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
)
from archive.xf_rotor1_retrain.data import validate_report_snapshot
from archive.xf_rotor1_retrain.evaluator import (
    EvaluationResult,
    evaluate_in_memory,
    phase_split_total_water,
)
from archive.xf_rotor1_retrain.routing import (
    load_routing_json,
    validate_routing_artifact,
)
from archive.xf_rotor1_retrain.train import (
    PRESSURE_PA,
    build_scaled_predictor,
    load_authoritative_dataset,
)
from archive.xf_rotor1_retrain.trainer import XFTrainer
from simulation.dual_rotor_sim.core.system.psychrometrics import dew_point_water_C

WORK_DIR = RESULTS_DIR / "work"
FULL_DEV_DIR = WORK_DIR / "xf_full_dev"
OUTER_REPORT = WORK_DIR / "xf_full_dev" / "outer_validation.json"

# 计划 Subtask 6 的外层门槛
OUTER_THRESHOLDS = {
    "temperature_mae_C": 1.5,
    "humidity_phase_nrmse": 0.10,
    "power_nrmse": 0.10,
    "r2_min": 0.90,
    "process_dew_point_mae_C": 2.0,
    "projection_max_fraction": 0.01,
    "total_water_p95": 0.05,
    "total_water_max": 0.10,
}

TEMP_OUTPUTS = (
    "Label_Process_Outlet_Temp",
    "Label_Purge_Outlet_Temp",
    "Label_React_Outlet_Temp",
)
HUMIDITY_OUTPUTS = (
    "Label_Process_Outlet_AbsHum",
    "Label_Purge_Outlet_AbsHum",
    "Label_React_Outlet_AbsHum",
)
POWER_OUTPUT = "Label_PowerConsumption"


def _load_clean_and_assignments():
    dataset = load_authoritative_dataset()
    return dataset.clean, dataset.assignments


def _split_development_outer(clean: pd.DataFrame, assignments: pd.DataFrame):
    if assignments["ID"].duplicated().any() or clean["ID"].duplicated().any():
        raise RuntimeError("ID 重复")
    indexed = clean.set_index("ID", drop=False)
    dev = assignments.loc[assignments["partition"].eq("development")]
    outer = assignments.loc[assignments["partition"].eq("outer_test")]
    dev_ids = tuple(int(v) for v in dev["ID"])
    outer_ids = tuple(int(v) for v in outer["ID"])
    dev_features = indexed.loc[list(dev_ids), list(MODEL_INPUT_COLUMNS)].to_numpy(dtype=np.float64)
    dev_targets = indexed.loc[list(dev_ids), list(MODEL_OUTPUT_COLUMNS)].to_numpy(dtype=np.float64)
    outer_features = indexed.loc[list(outer_ids), list(MODEL_INPUT_COLUMNS)].to_numpy(dtype=np.float64)
    outer_targets = indexed.loc[list(outer_ids), list(MODEL_OUTPUT_COLUMNS)].to_numpy(dtype=np.float64)
    for arr in (dev_features, dev_targets, outer_features, outer_targets):
        if not np.isfinite(arr).all():
            raise RuntimeError("development/outer 数值存在非有限值")
    return dev_ids, outer_ids, dev_features, dev_targets, outer_features, outer_targets


def _best_steps_for_candidate(candidate: str) -> list[int]:
    steps = []
    for fold in FIXED_FOLD_GROUPS:
        summary_path = WORK_DIR / "xf_full_matrix" / candidate / fold / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"缺少矩阵 summary: {summary_path}")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        steps.append(int(data["best_step"]))
    return steps


def _ceil_to_2000(value: float) -> int:
    return int(math.ceil(value / 2000.0) * 2000.0)


def _planned_steps(candidate: str, dev_ids: tuple[int, ...]) -> int:
    cand = get_candidate(candidate)
    steps = _best_steps_for_candidate(candidate)
    median = float(np.median(steps))
    planned = max(2000, _ceil_to_2000(1.10 * median))
    max_steps = cand.max_epochs * math.ceil(len(dev_ids) / cand.batch_size)
    return min(max_steps, planned)


def _train_on_full_dev(
    candidate: str,
    dev_ids: tuple[int, ...],
    dev_features: np.ndarray,
    dev_targets: np.ndarray,
    device: str = "cpu",
) -> tuple[XFTrainer, int, float]:
    cand = get_candidate(candidate)
    planned = _planned_steps(candidate, dev_ids)
    run_dir = FULL_DEV_DIR / candidate
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        # 简单续训：从 last.pt 恢复；若 step 已达标则直接加载。
        last = checkpoint_dir / "last.pt"
        if last.exists():
            trainer = XFTrainer.resume(
                last,
                train_features=dev_features,
                train_targets=dev_targets,
                train_ids=dev_ids,
                checkpoint_dir=checkpoint_dir,
                device=device,
            )
            if trainer.global_step >= planned:
                return trainer, planned, 0.0
    else:
        trainer = None

    if trainer is None:
        trainer = XFTrainer(
            candidate=cand,
            train_features=dev_features,
            train_targets=dev_targets,
            train_ids=dev_ids,
            checkpoint_dir=checkpoint_dir,
            device=device,
            validation_interval_steps=planned + 1,
            patience_evaluations=9999,
        )
    start = time.perf_counter()
    # 用全量 development 训练时不切验证集；validation_interval 设在计划步数之后，
    # 因此不会触发 cadence 验证。
    trainer.train_steps(
        planned - trainer.global_step,
        validation_features=dev_features[:1],
        validation_targets=dev_targets[:1],
    )
    trainer._save_checkpoint(checkpoint_dir / "best.pt")
    return trainer, planned, time.perf_counter() - start


def _evaluate_outer(
    trainer: XFTrainer,
    outer_ids: tuple[int, ...],
    outer_features: np.ndarray,
    outer_targets: np.ndarray,
    dev_targets: np.ndarray,
) -> EvaluationResult:
    return evaluate_in_memory(
        build_scaled_predictor(trainer),
        features=trainer.preprocessor.transform_features(outer_features),
        targets_model_space=trainer.preprocessor.transform_targets(outer_targets),
        train_targets_physical=dev_targets,
        sample_ids=outer_ids,
        pressure_Pa=PRESSURE_PA,
        inverse_target_transform=trainer.preprocessor.y_scaler,
        inverse_scaler=trainer.preprocessor.target_transform,
        batch_size=1024,
    )


def _routed_outer_frame(
    artifact: Mapping[str, Any],
    results: Mapping[str, EvaluationResult],
    outer_ids: tuple[int, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame({"ID": np.asarray(outer_ids, dtype=np.int64)})
    for output in MODEL_OUTPUT_COLUMNS:
        selected = artifact["routes"][output]["selected_candidate"]
        result = results[selected]
        index = MODEL_OUTPUT_COLUMNS.index(output)
        frame[output] = result.predictions_deployed[:, index]
    return frame


def _metric_arrays(pred: np.ndarray, target: np.ndarray, span: float) -> dict[str, float]:
    error = pred - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(target - np.mean(target))))
    r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "nrmse": rmse / span if span > 0 else math.inf, "r2": r2}


def _check_outer_thresholds(
    artifact: Mapping[str, Any],
    results: Mapping[str, EvaluationResult],
    outer_ids: tuple[int, ...],
    outer_targets: np.ndarray,
    dev_targets: np.ndarray,
) -> dict[str, Any]:
    routed = _routed_outer_frame(artifact, results, outer_ids)
    spans = np.ptp(dev_targets, axis=0)
    checks: dict[str, Any] = {"passed": True, "failed": []}
    metrics: dict[str, dict[str, float]] = {}

    for idx, output in enumerate(MODEL_OUTPUT_COLUMNS):
        pred = routed[output].to_numpy(dtype=np.float64)
        target = outer_targets[:, idx]
        metrics[output] = _metric_arrays(pred, target, spans[idx])

    for output in TEMP_OUTPUTS:
        if metrics[output]["mae"] > OUTER_THRESHOLDS["temperature_mae_C"]:
            checks["failed"].append(f"{output}.MAE={metrics[output]['mae']:.4f}>1.5C")
    for output in HUMIDITY_OUTPUTS + (POWER_OUTPUT,):
        if metrics[output]["nrmse"] > OUTER_THRESHOLDS["humidity_phase_nrmse"]:
            checks["failed"].append(f"{output}.NRMSE={metrics[output]['nrmse']:.4f}>10%")
    for output in MODEL_OUTPUT_COLUMNS:
        if metrics[output]["r2"] < OUTER_THRESHOLDS["r2_min"]:
            checks["failed"].append(f"{output}.R2={metrics[output]['r2']:.4f}<0.90")

    # 处理出口水面露点 MAE（与 evaluator 一致，报告副本使用 -75C 下限）
    process_w = MODEL_OUTPUT_COLUMNS.index("Label_Process_Outlet_AbsHum")
    pred_dp = dew_point_water_C(routed["Label_Process_Outlet_AbsHum"].to_numpy(), PRESSURE_PA)
    target_dp = dew_point_water_C(outer_targets[:, process_w], PRESSURE_PA)
    floor_C = -75.0
    pred_dp_report = np.where(np.isfinite(pred_dp), np.maximum(pred_dp, floor_C), floor_C)
    target_dp_report = np.where(np.isfinite(target_dp), np.maximum(target_dp, floor_C), floor_C)
    dp_mae = float(np.mean(np.abs(pred_dp_report - target_dp_report)))
    checks["process_dew_point_mae_C"] = dp_mae
    if dp_mae > OUTER_THRESHOLDS["process_dew_point_mae_C"]:
        checks["failed"].append(f"process_dew_point.MAE={dp_mae:.4f}>2C")

    # 非负投影比例（从各候选 raw/deployed 统计）
    projection = {}
    for candidate, result in results.items():
        projection[candidate] = {}
        for idx, output in enumerate(MODEL_OUTPUT_COLUMNS):
            raw = result.predictions_raw[:, idx]
            deployed = result.predictions_deployed[:, idx]
            correction = np.abs(deployed - raw)
            count = int(np.count_nonzero(correction))
            fraction = count / len(raw)
            maximum = float(np.max(correction, initial=0.0))
            projection[candidate][output] = {"count": count, "fraction": fraction, "maximum_correction": maximum}
            max_correction_relative = maximum / spans[idx] if spans[idx] > 0 else math.inf
            if output in HUMIDITY_OUTPUTS + (POWER_OUTPUT,) and fraction > OUTER_THRESHOLDS["projection_max_fraction"]:
                checks["failed"].append(f"{candidate}.{output}.projection_fraction={fraction:.4f}>1%")
            if output in HUMIDITY_OUTPUTS + (POWER_OUTPUT,) and max_correction_relative > 0.01:
                checks["failed"].append(f"{candidate}.{output}.projection_max_rel={max_correction_relative:.4f}>1%")
            projection[candidate][output]["max_correction_relative"] = max_correction_relative
    checks["projection"] = projection

    # 含凝结水的总水量残差 P95 / max（再生出口 total W）
    regen_w_idx = MODEL_OUTPUT_COLUMNS.index("Label_React_Outlet_AbsHum")
    total_pred = routed["Label_React_Outlet_AbsHum"].to_numpy(dtype=np.float64)
    total_target = outer_targets[:, regen_w_idx]
    total_error = np.abs(total_pred - total_target)
    denom = np.maximum(np.abs(total_target), 1e-12)
    total_relative = total_error / denom
    checks["total_water_p95"] = float(np.percentile(total_relative, 95))
    checks["total_water_max"] = float(np.max(total_relative))
    if checks["total_water_p95"] > OUTER_THRESHOLDS["total_water_p95"]:
        checks["failed"].append(f"total_water_p95={checks['total_water_p95']:.4f}>5%")
    if checks["total_water_max"] > OUTER_THRESHOLDS["total_water_max"]:
        checks["failed"].append(f"total_water_max={checks['total_water_max']:.4f}>10%")

    checks["metrics"] = metrics
    checks["routed_predictions"] = routed.to_dict(orient="list")
    checks["passed"] = not checks["failed"]
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing", type=Path, default=WORK_DIR / "xf_full_matrix" / "routing.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    artifact = validate_routing_artifact(load_routing_json(args.routing))
    clean, assignments = _load_clean_and_assignments()
    dev_ids, outer_ids, dev_features, dev_targets, outer_features, outer_targets = _split_development_outer(clean, assignments)

    selected = sorted({v["selected_candidate"] for v in artifact["routes"].values()})
    print(f"selected candidates: {selected}", flush=True)

    results: dict[str, EvaluationResult] = {}
    for candidate in selected:
        print(f"training full-dev {candidate} ...", flush=True)
        trainer, planned, duration = _train_on_full_dev(candidate, dev_ids, dev_features, dev_targets, args.device)
        print(f"  {candidate} done planned={planned} elapsed={duration:.1f}s global_step={trainer.global_step}", flush=True)
        results[candidate] = _evaluate_outer(trainer, outer_ids, outer_features, outer_targets, dev_targets)
        print(f"  {candidate} outer evaluated", flush=True)

    checks = _check_outer_thresholds(artifact, results, outer_ids, outer_targets, dev_targets)
    OUTER_REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "success",
        "passed": checks["passed"],
        "failed_checks": checks["failed"],
        "selected_candidates": selected,
        "outer_row_count": len(outer_ids),
        "checks": checks,
    }
    OUTER_REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": checks["passed"], "failed": checks["failed"]}, ensure_ascii=False, indent=2))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
