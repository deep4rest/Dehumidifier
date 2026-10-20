"""单 KAN 的共享评估核心与水面露点审计。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import r2_score

from simulation.dual_rotor_sim.core.system.psychrometrics import (
    dew_point_water_C,
)

from .config import DIRECT_OUTPUT_COLUMNS
from .model import Preprocessor, SingleKAN

__all__ = [
    "EvaluationBoundaryError",
    "EvaluationError",
    "EvaluationEvent",
    "EvaluationResult",
    "compare_heater_power",
    "dew_point_from_process_humidity",
    "dew_point_water_C",
    "evaluate_checkpoint",
    "evaluate_in_memory",
    "evaluate_predictions",
]


class EvaluationError(ValueError):
    """评估输入、输出或指标不满足硬边界。"""


class EvaluationBoundaryError(EvaluationError):
    """带入口 mode/stage 的评估失败，便于训练日志定位。"""

    def __init__(self, mode: str, stage: str, detail: str):
        self.mode = str(mode)
        self.stage = str(stage)
        self.detail = str(detail)
        super().__init__(f"[{self.mode}/{self.stage}] {self.detail}")


@dataclass(frozen=True)
class EvaluationEvent:
    kind: str
    mode: str
    completed: int
    total: int
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricStats:
    mae: float
    rmse: float
    nrmse: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "nrmse": float(self.nrmse),
            "r2": float(self.r2),
        }


@dataclass
class EvaluationResult:
    """真实物理尺度的六输出评估结果。"""

    metrics: dict[str, dict[str, float]]
    predictions: np.ndarray
    targets: np.ndarray
    projected_predictions: np.ndarray
    nonfinite_count: int
    nonnegative_projection_count: int
    nonnegative_projection_fraction: float
    dew_point_predictions: np.ndarray
    dew_point_targets: np.ndarray
    dew_point_excel: np.ndarray | None = None
    dew_point_mae_vs_excel: float | None = None
    mode: str = "in_memory"
    dew_point_dry_limit_prediction_count: int = 0
    dew_point_dry_limit_target_count: int = 0
    dew_point_excel_comparable_count: int = 0

    @property
    def metrics_real(self) -> dict[str, dict[str, float]]:
        return self.metrics

    @property
    def projection_count(self) -> int:
        return self.nonnegative_projection_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metrics": self.metrics,
            "nonfinite_count": self.nonfinite_count,
            "nonnegative_projection_count": self.nonnegative_projection_count,
            "nonnegative_projection_fraction": self.nonnegative_projection_fraction,
            "dew_point_mae_vs_excel": self.dew_point_mae_vs_excel,
            "dew_point_dry_limit_prediction_count": self.dew_point_dry_limit_prediction_count,
            "dew_point_dry_limit_target_count": self.dew_point_dry_limit_target_count,
            "dew_point_excel_comparable_count": self.dew_point_excel_comparable_count,
        }


def compare_heater_power(
    measured_power: Any,
    heater_load: Any,
    *,
    anomaly_z: float = 3.0,
) -> dict[str, Any]:
    """离线比较数据集 ``power`` 与加热器热负荷，不返回运行时校正系数。

    ``bias_kW`` 的定义固定为 ``heater_load - measured_power`` 的平均值；
    负功率、负热负荷、非有限值及稳健残差离群点只作审计标记，仍保留有限
    样本参与统计，避免把异常悄悄变成训练清洗规则。
    """

    measured = np.asarray(measured_power, dtype=np.float64).reshape(-1)
    reference = np.asarray(heater_load, dtype=np.float64).reshape(-1)
    if measured.shape != reference.shape:
        raise EvaluationError(
            "power 与加热器热负荷对照数组长度不一致: "
            f"{measured.shape} != {reference.shape}"
        )
    if measured.size == 0:
        raise EvaluationError("power 与加热器热负荷对照数组不得为空")
    if not np.isfinite(anomaly_z) or float(anomaly_z) <= 0.0:
        raise EvaluationError("anomaly_z 必须为正的有限值")

    finite = np.isfinite(measured) & np.isfinite(reference)
    nonfinite_count = int(np.count_nonzero(~finite))
    if not finite.any():
        raise EvaluationError("power 与加热器热负荷没有有限对照样本")

    residual = reference[finite] - measured[finite]
    bias = float(np.mean(residual))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    if not np.isfinite([bias, mae, rmse]).all():
        raise EvaluationError("power 离线对照指标存在非有限值")

    # Median/MAD is stable for the extreme main-rotor power values.  If all
    # residuals coincide, fall back to a tiny scale so only genuine deviations
    # are marked.  Negative physical quantities remain explicit anomalies.
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    scale = max(1.4826 * mad, np.finfo(np.float64).eps)
    robust_outlier = np.abs(residual - median) > float(anomaly_z) * scale

    valid_indices = np.flatnonzero(finite)
    negative_power = finite & (measured < 0.0)
    negative_load = finite & (reference < 0.0)
    anomaly_mask = np.zeros(measured.shape, dtype=bool)
    anomaly_mask[valid_indices] = robust_outlier
    anomaly_mask |= negative_power | negative_load
    anomaly_indices = np.flatnonzero(anomaly_mask).astype(int).tolist()
    return {
        "count": int(np.count_nonzero(finite)),
        "total_count": int(measured.size),
        "nonfinite_count": nonfinite_count,
        "negative_power_count": int(np.count_nonzero(negative_power)),
        "negative_heater_load_count": int(np.count_nonzero(negative_load)),
        "bias_kW": bias,
        "mae_kW": mae,
        "rmse_kW": rmse,
        "anomaly_count": int(len(anomaly_indices)),
        "anomaly_fraction": float(len(anomaly_indices) / measured.size),
        "anomaly_indices": anomaly_indices,
        "anomaly_rule": {
            "method": "median_mad_plus_negative_value",
            "z": float(anomaly_z),
        },
        "runtime_correction_applied": False,
    }


def _emit(
    sink: Callable[[EvaluationEvent], None] | None,
    event: EvaluationEvent,
) -> None:
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        # 进度显示器不是评估结果的一部分；其异常不应伪造模型评估失败。
        return


def _finite_matrix(values: Any, *, columns: int, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise EvaluationError(f"{label} 必须为 [N,{columns}]，实际 {matrix.shape}")
    if len(matrix) == 0:
        raise EvaluationError(f"{label} 不得为空")
    if not np.isfinite(matrix).all():
        raise EvaluationError(f"{label} 存在非有限值")
    return matrix


def dew_point_from_process_humidity(
    process_outlet_W: Any,
    pressure_Pa: float = 101_325.0,
) -> np.ndarray:
    """统一由处理出口含湿量和系统压力派生水面露点。"""

    humidity = np.asarray(process_outlet_W, dtype=np.float64)
    if humidity.ndim == 0:
        humidity = humidity.reshape(1)
    if not np.isfinite(humidity).all() or np.any(humidity < 0.0):
        raise EvaluationError("处理出口含湿量必须为非负有限值")
    try:
        values = np.asarray(dew_point_water_C(humidity, float(pressure_Pa)), dtype=np.float64)
    except Exception as exc:
        raise EvaluationError(f"水面露点派生失败: {exc}") from exc
    if np.isnan(values).any() or np.isposinf(values).any() or np.any(np.isneginf(values) & (humidity != 0)):
        raise EvaluationError("水面露点派生产生非有限值")
    return values


def _metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_columns: Sequence[str],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, column in enumerate(output_columns):
        truth = targets[:, index]
        predicted = predictions[:, index]
        error = predicted - truth
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        span = float(np.max(truth) - np.min(truth))
        if not np.isfinite(span) or span <= 0.0:
            raise EvaluationError(f"输出 {column} 的目标跨度为零或非有限")
        r2 = float(r2_score(truth, predicted))
        stats = MetricStats(mae=mae, rmse=rmse, nrmse=rmse / span, r2=r2)
        if not np.isfinite(np.asarray(list(stats.to_dict().values()))).all():
            raise EvaluationError(f"输出 {column} 指标存在非有限值")
        result[str(column)] = stats.to_dict()
    return result


def evaluate_predictions(
    targets: Any,
    predictions: Any,
    *,
    output_columns: Sequence[str] = DIRECT_OUTPUT_COLUMNS,
    pressure_Pa: float = 101_325.0,
    excel_dew_point: Any | None = None,
    mode: str = "predictions",
) -> EvaluationResult:
    """对内存预测和 checkpoint 重载预测共用的指标核心。"""

    columns = tuple(str(column) for column in output_columns)
    if columns != tuple(DIRECT_OUTPUT_COLUMNS):
        raise EvaluationError("评估输出必须严格为六个直接状态，露点不属于标签")
    truth = _finite_matrix(targets, columns=6, label="评估目标")
    predicted = _finite_matrix(predictions, columns=6, label="评估预测")
    if len(truth) != len(predicted):
        raise EvaluationError("评估目标与预测样本数不一致")

    # Humidity outputs are physically non-negative. Count (and report) the
    # correction, but retain raw predictions for transparent audit evidence.
    humidity_indices = (1, 3, 5)
    projected = predicted.copy()
    below_zero = projected[:, humidity_indices] < 0.0
    projection_count = int(np.count_nonzero(below_zero))
    projected[:, humidity_indices] = np.maximum(projected[:, humidity_indices], 0.0)
    metric_values = _metrics(truth, projected, columns)

    dew_pred = dew_point_from_process_humidity(projected[:, 1], pressure_Pa)
    dew_truth = dew_point_from_process_humidity(truth[:, 1], pressure_Pa)
    dew_excel = None
    dew_mae = None
    comparable_count = 0
    if excel_dew_point is not None:
        dew_excel = np.asarray(excel_dew_point, dtype=np.float64).reshape(-1)
        if len(dew_excel) != len(truth) or not np.isfinite(dew_excel).all():
            raise EvaluationError("Excel 露点审计列长度不符或存在非有限值")
        comparable = np.isfinite(dew_pred)
        comparable_count = int(comparable.sum())
        if comparable_count:
            dew_mae = float(np.mean(np.abs(dew_pred[comparable] - dew_excel[comparable])))
            if not np.isfinite(dew_mae):
                raise EvaluationError("露点审计误差非有限")
    return EvaluationResult(
        metrics=metric_values,
        predictions=predicted,
        targets=truth,
        projected_predictions=projected,
        nonfinite_count=0,
        nonnegative_projection_count=projection_count,
        nonnegative_projection_fraction=projection_count / max(predicted.size, 1),
        dew_point_predictions=dew_pred,
        dew_point_targets=dew_truth,
        dew_point_excel=dew_excel,
        dew_point_mae_vs_excel=dew_mae,
        dew_point_dry_limit_prediction_count=int(np.isneginf(dew_pred).sum()),
        dew_point_dry_limit_target_count=int(np.isneginf(dew_truth).sum()),
        dew_point_excel_comparable_count=comparable_count,
        mode=mode,
    )


def _predict_physical(
    model: Any,
    features: Any,
    preprocessor: Preprocessor,
    *,
    device: str | torch.device,
) -> np.ndarray:
    try:
        if hasattr(model, "predict"):
            try:
                result = model.predict(features, preprocessor, device=device)
            except TypeError:
                # Lightweight test predictors and older wrappers may not have
                # the optional device keyword.
                result = model.predict(features, preprocessor)
            return np.asarray(result, dtype=np.float64)
        scaled = preprocessor.transform_features(features)
        matrix = torch.from_numpy(scaled).to(torch.device(device))
        if hasattr(model, "to"):
            model = model.to(torch.device(device))
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            scaled_prediction = model(matrix)
            if isinstance(scaled_prediction, torch.Tensor):
                scaled_prediction = scaled_prediction.detach().cpu().numpy()
            else:
                scaled_prediction = np.asarray(scaled_prediction)
        return preprocessor.inverse_targets(scaled_prediction)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"模型推理失败: {exc}") from exc


def evaluate_in_memory(
    model: Any,
    features: Any,
    targets: Any,
    preprocessor: Preprocessor,
    *,
    pressure_Pa: float = 101_325.0,
    excel_dew_point: Any | None = None,
    mode: str = "in_memory",
    device: str | torch.device = "cpu",
    event_sink: Callable[[EvaluationEvent], None] | None = None,
) -> EvaluationResult:
    """以内存模型执行一次完整六输出评估。"""

    total = len(np.asarray(features))
    _emit(event_sink, EvaluationEvent("start", mode, 0, total, {"stage": "predict"}))
    try:
        predictions = _predict_physical(model, features, preprocessor, device=device)
        _emit(event_sink, EvaluationEvent("progress", mode, total, total, {"stage": "predict"}))
        result = evaluate_predictions(
            targets,
            predictions,
            pressure_Pa=pressure_Pa,
            excel_dew_point=excel_dew_point,
            mode=mode,
        )
    except Exception as exc:
        _emit(
            event_sink,
            EvaluationEvent("failure", mode, total, total, {"stage": "evaluate", "detail": str(exc)}),
        )
        _emit(event_sink, EvaluationEvent("end", mode, total, total, {"status": "failed"}))
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationBoundaryError(mode, "evaluate", str(exc)) from exc
    _emit(event_sink, EvaluationEvent("end", mode, total, total, {"status": "ok"}))
    return result


def _load_state(path: str | Path, device: str | torch.device) -> Mapping[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise EvaluationError(f"checkpoint 不存在: {checkpoint}")
    try:
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location=device)
    except Exception as exc:
        raise EvaluationError(f"checkpoint 读取失败: {checkpoint}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationError("checkpoint payload 必须为 mapping")
    return payload


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    model: Any,
    features: Any,
    targets: Any,
    preprocessor: Preprocessor,
    *,
    pressure_Pa: float = 101_325.0,
    excel_dew_point: Any | None = None,
    device: str | torch.device = "cpu",
    event_sink: Callable[[EvaluationEvent], None] | None = None,
) -> EvaluationResult:
    """重载 checkpoint 后调用同一个 ``evaluate_predictions`` 核心。"""

    mode = "checkpoint"
    total = len(np.asarray(features))
    _emit(event_sink, EvaluationEvent("start", mode, 0, total, {"stage": "load"}))
    try:
        payload = _load_state(checkpoint_path, device)
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
        if not isinstance(state, Mapping):
            raise EvaluationError("checkpoint 缺少 model_state_dict")
        if callable(model) and not hasattr(model, "load_state_dict"):
            model = model(Path(checkpoint_path))
        model.load_state_dict(state)
        result = evaluate_in_memory(
            model,
            features,
            targets,
            preprocessor,
            pressure_Pa=pressure_Pa,
            excel_dew_point=excel_dew_point,
            mode=mode,
            device=device,
            event_sink=None,
        )
        _emit(event_sink, EvaluationEvent("progress", mode, total, total, {"stage": "evaluate"}))
    except Exception as exc:
        _emit(event_sink, EvaluationEvent("failure", mode, total, total, {"stage": "load", "detail": str(exc)}))
        _emit(event_sink, EvaluationEvent("end", mode, total, total, {"status": "failed"}))
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationBoundaryError(mode, "load", str(exc)) from exc
    _emit(event_sink, EvaluationEvent("end", mode, total, total, {"status": "ok"}))
    return result
