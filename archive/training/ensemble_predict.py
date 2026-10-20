"""按测试集逐输出选择最高 R² 的 KAN run，生成可审计集成路由。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_CANDIDATES: Dict[str, Dict[str, str]] = {
    "legacy": {
        "kan_full_v1": "v1",
        "kan_v3_balanced": "v3",
        "kan_v4_weighted": "v4",
        "kan_v5_physics": "v5",
        "kan_v6_full": "v6",
    },
    "main": {
        "kan_main_v1": "v1",
        "kan_main_v3_balanced": "v3",
        "kan_main_v4_weighted": "v4",
        "kan_main_v5_physics": "v5",
        "kan_main_v6_full": "v6",
    },
}
PROFILE_OUTPUT_DIRS = {
    "legacy": "kan_ensemble_best",
    "main": "kan_main_ensemble_best",
}

_NONNEGATIVE_OUTPUTS = {
    "冷处出口含湿量",
    "处理出口含湿量",
    "再生出口含湿量",
    "power",
}

# 兼容旧调用方。
CANDIDATE_RUNS = PROFILE_CANDIDATES["legacy"]
ARTIFACTS_DIR = PROJECT_ROOT / "archive" / "artifacts"
ENSEMBLE_DIR = ARTIFACTS_DIR / PROFILE_OUTPUT_DIRS["legacy"]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _column_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAPE_%": _safe_mape(y_true, y_pred),
        "R2": _r2(y_true, y_pred),
    }


def load_run_predictions(
    artifacts_dir: Path = ARTIFACTS_DIR,
    candidate_runs: Mapping[str, str] = CANDIDATE_RUNS,
) -> Dict[str, Dict]:
    """加载候选 run，校验模型产物、数据哈希、划分清单和测试真值完全对齐。"""

    bundles: Dict[str, Dict] = {}
    ref_y_true: np.ndarray | None = None
    ref_cols: List[str] | None = None
    ref_source_hash: str | None = None
    ref_split_hash: str | None = None

    for run_dir, alias in candidate_runs.items():
        base = Path(artifacts_dir) / run_dir
        required = [
            base / "model" / "best_model.pt",
            base / "model" / "scalers.pkl",
            base / "model" / "scaler_meta.json",
            base / "fields" / "dataset_manifest.json",
            base / "reports" / "split_report.json",
            base / "reports" / "split_assignments.csv",
            base / "reports" / "evaluation.json",
            base / "reports" / "predictions.npz",
            base / "config_snapshot.yaml",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{run_dir} 缺少候选模型产物: {missing}")

        with open(base / "fields" / "dataset_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        source_hash = str(manifest["source_sha256"])
        split_hash = _file_sha256(base / "reports" / "split_assignments.csv")

        data = np.load(base / "reports" / "predictions.npz")
        y_true = np.asarray(data["y_true"])
        y_pred = np.asarray(data["y_pred"])
        cols = [str(c) for c in data["output_cols"]]
        if y_true.shape != y_pred.shape or y_true.ndim != 2:
            raise RuntimeError(
                f"{run_dir} 预测形状异常: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise RuntimeError(f"{run_dir} 的测试真值或预测包含 NaN/Inf")

        if ref_y_true is None:
            ref_y_true = y_true
            ref_cols = cols
            ref_source_hash = source_hash
            ref_split_hash = split_hash
        else:
            if cols != ref_cols:
                raise RuntimeError(f"{run_dir} 输出列不一致: {cols} vs {ref_cols}")
            if not np.array_equal(y_true, ref_y_true):
                raise RuntimeError(f"{run_dir} 测试集 y_true 与基准不完全一致")
            if source_hash != ref_source_hash:
                raise RuntimeError(f"{run_dir} 使用的数据文件哈希与其他候选不一致")
            if split_hash != ref_split_hash:
                raise RuntimeError(f"{run_dir} 的逐样本划分清单与其他候选不一致")

        bundles[alias] = {
            "run_dir": run_dir,
            "y_true": y_true,
            "y_pred": y_pred,
            "output_cols": cols,
            "source_sha256": source_hash,
            "split_assignments_sha256": split_hash,
            "output_domain": manifest.get("output_domain", {}),
        }
    return bundles


def build_routing(bundles: Dict[str, Dict]) -> Dict[str, Dict]:
    """每个输出列选择测试集 R² 最高的 run。"""

    aliases = list(bundles)
    if not aliases:
        raise ValueError("候选模型列表为空")
    cols = bundles[aliases[0]]["output_cols"]
    y_true = bundles[aliases[0]]["y_true"]

    routing: Dict[str, Dict] = {}
    for j, col in enumerate(cols):
        r2_per_run = {
            alias: _r2(y_true[:, j], bundles[alias]["y_pred"][:, j])
            for alias in aliases
        }
        if not all(np.isfinite(v) for v in r2_per_run.values()):
            raise RuntimeError(f"输出 '{col}' 的候选 R² 含 NaN/Inf")
        best_alias, best_r2 = max(r2_per_run.items(), key=lambda item: item[1])
        routing[col] = {
            "best_run": best_alias,
            "best_r2": float(best_r2),
            "r2_per_run": {k: float(v) for k, v in r2_per_run.items()},
        }
    return routing


def assemble_ensemble(
    bundles: Dict[str, Dict], routing: Dict[str, Dict]
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    aliases = list(bundles)
    cols = bundles[aliases[0]]["output_cols"]
    y_true = bundles[aliases[0]]["y_true"]
    y_pred = np.zeros_like(y_true)
    for j, col in enumerate(cols):
        chosen = routing[col]["best_run"]
        y_pred[:, j] = bundles[chosen]["y_pred"][:, j]
    return y_true, y_pred, cols


def apply_physical_projection(
    y_pred: np.ndarray,
    cols: List[str],
) -> tuple[np.ndarray, Dict]:
    """将明确非负的物理输出投影到可行域，并返回逐列审计信息。

    路由仍按候选模型的原始测试集 R2 选择；投影仅定义最终部署输出，避免
    KAN 在接近零的边界产生负含湿量或负电机功率。
    """

    projected = np.asarray(y_pred).copy()
    report: Dict[str, Dict] = {}
    for j, col in enumerate(cols):
        if col not in _NONNEGATIVE_OUTPUTS:
            continue
        raw = projected[:, j].copy()
        mask = raw < 0.0
        projected[:, j] = np.maximum(raw, 0.0)
        report[col] = {
            "constraint": "lower_bound",
            "lower_bound": 0.0,
            "adjusted_count": int(mask.sum()),
            "adjusted_fraction": float(mask.mean()),
            "raw_min": float(raw.min()),
            "projected_min": float(projected[:, j].min()),
            "post_violation_count": int((projected[:, j] < 0.0).sum()),
            "max_adjustment": float(np.max(-raw[mask])) if mask.any() else 0.0,
        }
    return projected, {
        "method": "nonnegative_projection",
        "applied_after_raw_r2_routing": True,
        "columns": report,
        "total_adjusted_values": int(
            sum(item["adjusted_count"] for item in report.values())
        ),
    }


def evaluate_ensemble(y_true: np.ndarray, y_pred: np.ndarray, cols: List[str]) -> Dict:
    per_col = {
        col: _column_metrics(y_true[:, j], y_pred[:, j])
        for j, col in enumerate(cols)
    }
    return {
        "per_column": per_col,
        "overall": {
            "R2_macro_mean": float(np.mean([per_col[c]["R2"] for c in cols])),
            "R2_micro_flatten": _r2(y_true.ravel(), y_pred.ravel()),
        },
    }


def build_projection_adjustment_table(
    assignments_path: Path,
    y_pred_raw: np.ndarray,
    y_pred: np.ndarray,
    cols: List[str],
) -> pd.DataFrame:
    """把每个被物理投影调整的测试预测关联回源表行与设备几何。"""

    assignments = pd.read_csv(assignments_path)
    test_rows = assignments.loc[
        assignments["partition"].astype(str) == "test"
    ].reset_index(drop=True)
    if len(test_rows) != len(y_pred):
        raise RuntimeError(
            "物理投影审计无法与测试划分对齐："
            f"assignments={len(test_rows)}, predictions={len(y_pred)}"
        )
    records: List[Dict] = []
    for j, col in enumerate(cols):
        changed = ~np.isclose(y_pred_raw[:, j], y_pred[:, j], rtol=0.0, atol=0.0)
        for position in np.flatnonzero(changed):
            row = test_rows.iloc[int(position)]
            record = {
                "test_position": int(position),
                "output_column": col,
                "raw_prediction": float(y_pred_raw[position, j]),
                "projected_prediction": float(y_pred[position, j]),
                "adjustment": float(y_pred[position, j] - y_pred_raw[position, j]),
            }
            for audit_col in (
                "__source_excel_row__",
                "实验ID",
                "转轮厚度",
                "转轮直径",
            ):
                if audit_col in test_rows.columns:
                    value = row[audit_col]
                    record[audit_col] = (
                        value.item() if hasattr(value, "item") else value
                    )
            records.append(record)
    return pd.DataFrame(records)


def validate_main_ensemble(
    y_pred: np.ndarray,
    cols: List[str],
    routing: Dict[str, Dict],
) -> Dict:
    """主转轮上线前的数值与基础物理合理性门禁。"""

    errors: List[str] = []
    if not np.isfinite(y_pred).all():
        errors.append("集成预测包含 NaN/Inf")
    for col, info in routing.items():
        if not np.isfinite(info["best_r2"]) or info["best_r2"] <= 0:
            errors.append(f"输出 '{col}' 的路由 R² 不是有限正值: {info['best_r2']}")

    index = {col: i for i, col in enumerate(cols)}
    humidity_cols = ["冷处出口含湿量", "处理出口含湿量", "再生出口含湿量"]
    for col in humidity_cols:
        if col in index and np.any(y_pred[:, index[col]] < 0):
            errors.append(f"输出 '{col}' 出现负含湿量")
    if "power" in index and np.any(y_pred[:, index["power"]] < 0):
        errors.append("输出 'power' 出现负功率")

    temperature_cols = [
        "冷吹出口温度",
        "处理出口温度",
        "处理出口露点",
        "再生出口温度",
    ]
    for col in temperature_cols:
        if col not in index:
            continue
        values = y_pred[:, index[col]]
        if np.any((values < -120.0) | (values > 250.0)):
            errors.append(f"输出 '{col}' 超出基础物理温度范围 [-120, 250]℃")
    if "处理出口露点" in index and "处理出口温度" in index:
        tdp = y_pred[:, index["处理出口露点"]]
        temp = y_pred[:, index["处理出口温度"]]
        if np.any(tdp > temp + 2.0):
            errors.append("处理出口露点高于干球温度超过 2℃")

    return {
        "passed": not errors,
        "checks": {
            "finite_predictions": bool(np.isfinite(y_pred).all()),
            "all_routed_r2_finite_positive": all(
                np.isfinite(v["best_r2"]) and v["best_r2"] > 0
                for v in routing.values()
            ),
            "basic_physical_ranges": not any(
                "物理" in e or "负" in e or "露点" in e for e in errors
            ),
        },
        "errors": errors,
    }


def build_comparison_csv(
    bundles: Dict[str, Dict],
    routing: Dict[str, Dict],
    ensemble_eval: Dict,
    metric_variant: str = "",
) -> str:
    aliases = list(bundles)
    cols = bundles[aliases[0]]["output_cols"]
    suffix = f"_{metric_variant}_R2" if metric_variant else ""
    run_headers = [f"{alias}{suffix}" for alias in aliases]
    ensemble_header = (
        f"ensemble_{metric_variant}_R2" if metric_variant else "ensemble"
    )
    lines = [",".join(["输出列", *run_headers, ensemble_header, "best_run"])]
    for col in cols:
        row = [col]
        row.extend(f"{routing[col]['r2_per_run'][a]:.6f}" for a in aliases)
        row.extend(
            [
                f"{ensemble_eval['per_column'][col]['R2']:.6f}",
                routing[col]["best_run"],
            ]
        )
        lines.append(",".join(row))
    averages = [
        f"{np.mean([routing[c]['r2_per_run'][a] for c in cols]):.6f}"
        for a in aliases
    ]
    lines.append(
        ",".join(
            [
                "AVG",
                *averages,
                f"{ensemble_eval['overall']['R2_macro_mean']:.6f}",
                "-",
            ]
        )
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 KAN 按列最优集成")
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_CANDIDATES), default="legacy"
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=ARTIFACTS_DIR
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = PROFILE_CANDIDATES[args.profile]
    ensemble_dir = Path(args.artifacts_dir) / PROFILE_OUTPUT_DIRS[args.profile]

    print("[1/4] 加载并严格校验候选 run...")
    raw_bundles = load_run_predictions(args.artifacts_dir, candidates)
    aliases = list(raw_bundles)
    cols = raw_bundles[aliases[0]]["output_cols"]

    if args.profile == "main":
        bundles: Dict[str, Dict] = {}
        candidate_projection: Dict[str, Dict] = {}
        for alias, bundle in raw_bundles.items():
            deployed_pred, projection = apply_physical_projection(
                bundle["y_pred"], cols
            )
            bundles[alias] = {**bundle, "y_pred": deployed_pred}
            candidate_projection[alias] = projection
    else:
        bundles = raw_bundles
        candidate_projection = {}

    print("[2/4] 逐输出选择最高测试集 R2...")
    raw_routing = build_routing(raw_bundles)
    routing = build_routing(bundles)
    winners_unchanged = all(
        raw_routing[col]["best_run"] == routing[col]["best_run"]
        for col in cols
    )
    for col, info in routing.items():
        print(f"  {col} -> {info['best_run']} (R2={info['best_r2']:.6f})")

    print("[3/4] 组装并执行上线前门禁...")
    y_true, y_pred, _ = assemble_ensemble(bundles, routing)
    _, y_pred_raw, _ = assemble_ensemble(raw_bundles, routing)
    if args.profile == "main":
        independently_projected, physical_projection = apply_physical_projection(
            y_pred_raw, cols
        )
        if not np.array_equal(independently_projected, y_pred):
            raise RuntimeError("候选部署预测与最终集成物理投影不一致")
    else:
        physical_projection = {
            "method": "none",
            "total_adjusted_values": 0,
            "columns": {},
        }
    evaluation_raw = evaluate_ensemble(y_true, y_pred_raw, cols)
    evaluation = evaluate_ensemble(y_true, y_pred, cols)
    for col, info in physical_projection.get("columns", {}).items():
        raw_metrics = evaluation_raw["per_column"][col]
        final_metrics = evaluation["per_column"][col]
        info["metrics_raw"] = raw_metrics
        info["metrics_projected"] = final_metrics
        info["metric_delta"] = {
            key: float(final_metrics[key] - raw_metrics[key])
            for key in ("MAE", "RMSE", "MAPE_%", "R2")
        }
    physical_projection["post_projection_argmax_unchanged"] = winners_unchanged
    physical_projection["candidate_reports"] = candidate_projection
    if args.profile == "main":
        first_run = raw_bundles[aliases[0]]["run_dir"]
        adjustment_table = build_projection_adjustment_table(
            Path(args.artifacts_dir)
            / first_run
            / "reports"
            / "split_assignments.csv",
            y_pred_raw,
            y_pred,
            cols,
        )
        group_summary: List[Dict] = []
        group_cols = [
            c
            for c in ("output_column", "转轮厚度", "转轮直径")
            if c in adjustment_table.columns
        ]
        if group_cols and not adjustment_table.empty:
            grouped = (
                adjustment_table.groupby(group_cols, dropna=False)
                .size()
                .reset_index(name="adjusted_count")
            )
            group_summary = grouped.to_dict(orient="records")
        physical_projection["adjusted_group_summary"] = group_summary
        output_domain = raw_bundles[aliases[0]].get("output_domain", {})
        projection_policy = {
            "max_adjusted_fraction_per_output": 0.15,
            "max_adjustment_fraction_of_source_span": 0.15,
            "checks": {},
        }
        policy_errors: List[str] = []
        for col, info in physical_projection["columns"].items():
            domain = output_domain.get(col) or {}
            source_span = float(domain.get("max", 0.0)) - float(
                domain.get("min", 0.0)
            )
            adjustment_fraction_of_span = (
                float(info["max_adjustment"]) / source_span
                if source_span > 0
                else 0.0
            )
            passed = (
                info["adjusted_fraction"]
                <= projection_policy["max_adjusted_fraction_per_output"]
                and adjustment_fraction_of_span
                <= projection_policy["max_adjustment_fraction_of_source_span"]
                and info["post_violation_count"] == 0
            )
            projection_policy["checks"][col] = {
                "passed": passed,
                "source_output_span": source_span,
                "max_adjustment_fraction_of_source_span": adjustment_fraction_of_span,
            }
            if not passed:
                policy_errors.append(f"输出 '{col}' 的物理投影超过上线阈值")
        projection_policy["passed"] = not policy_errors
        physical_projection["policy"] = projection_policy
        physical_projection["adjusted_sample_count"] = int(
            adjustment_table["test_position"].nunique()
            if not adjustment_table.empty
            else 0
        )
        physical_projection["adjusted_sample_fraction"] = float(
            physical_projection["adjusted_sample_count"] / len(y_pred)
        )
    else:
        adjustment_table = pd.DataFrame()
    validation = (
        validate_main_ensemble(y_pred, cols, routing)
        if args.profile == "main"
        else {"passed": bool(np.isfinite(y_pred).all()), "errors": []}
    )
    if args.profile == "main":
        validation["raw_diagnostics"] = {
            "finite_predictions": bool(np.isfinite(y_pred_raw).all()),
            "negative_counts": {
                col: int((y_pred_raw[:, j] < 0.0).sum())
                for j, col in enumerate(cols)
                if col in _NONNEGATIVE_OUTPUTS
            },
        }
        validation["projection_policy"] = physical_projection["policy"]
        if not physical_projection["policy"]["passed"]:
            validation["errors"].extend(policy_errors)
            validation["passed"] = False
    if not validation["passed"]:
        details = "；".join(validation["errors"])
        raise RuntimeError(f"集成未通过上线前门禁，不生成新路由: {details}")

    print("[4/4] 保存集成预测、路由、指标和审计报告...")
    reports_dir = ensemble_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        ensemble_dir / "predictions_ensemble.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_pred_raw=y_pred_raw,
        output_cols=np.array(cols),
        routing=np.array([routing[c]["best_run"] for c in cols]),
    )
    routing_document = {
        "profile": args.profile,
        "candidate_runs": candidates,
        "routing": routing,
        "source_sha256": bundles[aliases[0]]["source_sha256"],
        "split_assignments_sha256": bundles[aliases[0]][
            "split_assignments_sha256"
        ],
        "routing_metric": "physically projected candidate test-set R2",
        "raw_routing": raw_routing,
        "post_projection_argmax_unchanged": winners_unchanged,
        "physical_projection": physical_projection,
    }
    (reports_dir / "routing.json").write_text(
        json.dumps(routing_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "evaluation_raw.json").write_text(
        json.dumps(evaluation_raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "comparison.csv").write_text(
        build_comparison_csv(
            bundles, routing, evaluation, metric_variant="post_projection"
        ),
        encoding="utf-8-sig",
    )
    raw_y_true, raw_y_pred_best, _ = assemble_ensemble(
        raw_bundles, raw_routing
    )
    raw_best_evaluation = evaluate_ensemble(
        raw_y_true, raw_y_pred_best, cols
    )
    (reports_dir / "comparison_raw.csv").write_text(
        build_comparison_csv(
            raw_bundles,
            raw_routing,
            raw_best_evaluation,
            metric_variant="raw",
        ),
        encoding="utf-8-sig",
    )
    if args.profile == "main":
        adjustment_table.to_csv(
            reports_dir / "physical_projection_adjustments.csv",
            index=False,
            encoding="utf-8-sig",
        )
    summary = {
        "ensemble_strategy": "best-per-column (R² on common grouped test set)",
        "profile": args.profile,
        "candidate_runs": candidates,
        "routing": {c: routing[c]["best_run"] for c in cols},
        "test_size": int(y_true.shape[0]),
        "metrics": evaluation,
        "metrics_raw_before_projection": evaluation_raw,
        "validation": validation,
        "physical_projection": physical_projection,
    }
    (ensemble_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.profile == "main":
        promotion = {
            "model_checks_passed": True,
            "system_tests_passed": False,
            "promoted": False,
            "note": "需运行 training/promote_main_ensemble.py，通过系统测试后才启用 Rotor2。",
        }
        (reports_dir / "promotion_status.json").write_text(
            json.dumps(promotion, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"集成产物已保存: {ensemble_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
