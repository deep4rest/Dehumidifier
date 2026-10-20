"""候选模型的原始物理单位测试指标。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DIRECT_OUTPUT_COLUMNS

_HUMIDITY_COLUMNS = {"处理出口含湿量", "冷吹出口含湿量", "再生出口含湿量"}
_UNITS = {
    "处理出口温度": "℃", "处理出口含湿量": "g/kg", "冷吹出口温度": "℃",
    "冷吹出口含湿量": "g/kg", "再生出口温度": "℃", "再生出口含湿量": "g/kg",
}


def _matrix(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(DIRECT_OUTPUT_COLUMNS):
        raise ValueError(f"{name} 必须为 [N,6]，实际 {result.shape}")
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} 不得为空且必须全为有限值")
    return result


def _one_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    error = prediction - target
    abs_error = np.abs(error)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error * error)))
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    r2 = None if denominator == 0.0 else float(1.0 - np.sum(error * error) / denominator)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "p95_absolute_error": float(np.percentile(abs_error, 95)),
    }


def evaluate_candidate_predictions(
    targets: Any,
    predictions: Any,
    *,
    geometries: Any | None = None,
) -> dict[str, Any]:
    """保留原始预测，计算六输出及几何组指标。"""

    truth = _matrix(targets, "targets")
    predicted = _matrix(predictions, "predictions")
    if len(truth) != len(predicted):
        raise ValueError("targets 与 predictions 行数不一致")
    per_output: dict[str, dict[str, Any]] = {}
    r2_values: list[float] = []
    for index, column in enumerate(DIRECT_OUTPUT_COLUMNS):
        metrics = _one_metrics(truth[:, index], predicted[:, index])
        metrics["unit"] = _UNITS[column]
        per_output[column] = metrics
        if metrics["r2"] is not None:
            r2_values.append(float(metrics["r2"]))
    negative_ratio = {
        column: float(np.mean(predicted[:, index] < 0.0))
        for index, column in enumerate(DIRECT_OUTPUT_COLUMNS)
        if column in _HUMIDITY_COLUMNS
    }

    groups: dict[str, dict[str, Any]] = {}
    if geometries is not None:
        geometry_array = np.asarray(geometries)
        if geometry_array.ndim != 2 or geometry_array.shape != (len(truth), 2):
            raise ValueError("几何组必须为 [N,2]")
        for depth, diameter in sorted({(int(row[0]), int(row[1])) for row in geometry_array}):
            mask = (geometry_array[:, 0] == depth) & (geometry_array[:, 1] == diameter)
            group_metrics = {
                column: _one_metrics(truth[mask, index], predicted[mask, index])
                for index, column in enumerate(DIRECT_OUTPUT_COLUMNS)
            }
            group_r2 = [value["r2"] for value in group_metrics.values() if value["r2"] is not None]
            groups[f"{depth}x{diameter}"] = {
                "rows": int(mask.sum()),
                "mean_r2": None if not group_r2 else float(np.mean(group_r2)),
                "per_output": group_metrics,
            }
    return {
        "rows": int(len(truth)),
        "per_output": per_output,
        "mean_r2": None if not r2_values else float(np.mean(r2_values)),
        "negative_humidity_ratio": negative_ratio,
        "by_geometry": groups,
    }
