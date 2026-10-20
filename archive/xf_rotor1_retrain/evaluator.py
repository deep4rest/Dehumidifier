"""XF 折外评估、物理部署后处理与湿空气水面公式。

该模块是训练中内存模型和 checkpoint 重载评估的唯一核心。它只消费
Subtask 1 锁定的 development/fold 划分，不接收路由决策，也不暴露 outer
标签的任何评分入口。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from simulation.dual_rotor_sim.core.system.psychrometrics import (
    dew_point_water_C,
    saturation_humidity_ratio_g_per_kg,
    saturation_vapor_pressure_water_Pa,
)

from .config import (
    FIXED_FOLD_GROUPS,
    MODEL_INPUT_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
    OUTER_GROUP,
)

__all__ = [
    "AuditedMinMaxScaler",
    "EvaluationBoundaryError",
    "DewPointReportingAudit",
    "EvaluationError",
    "EvaluationEvent",
    "EvaluationResult",
    "FoldData",
    "MetricStats",
    "OOFIntegrityError",
    "PhaseMetricStats",
    "PhaseSplit",
    "ProjectionStats",
    "assemble_oof_predictions",
    "build_fold_data",
    "dew_point_water_C",
    "evaluate_checkpoint",
    "evaluate_in_memory",
    "evaluate_predictions",
    "phase_split_total_water",
    "saturation_humidity_ratio_g_per_kg",
    "saturation_vapor_pressure_water_Pa",
]


_EXPECTED_DEVELOPMENT_ROWS = 52_435
_EXPECTED_CLEAN_ROWS = 57_185
_EXPECTED_OUTER_ROWS = 4_750
_REGEN_T_COLUMN = "Label_React_Outlet_Temp"
_REGEN_W_COLUMN = "Label_React_Outlet_AbsHum"
_PROCESS_W_COLUMN = "Label_Process_Outlet_AbsHum"
_DEW_POINT_REPORTING_FLOOR_C = -75.0


class EvaluationError(ValueError):
    """评估器内部数据或物理契约错误。"""


class OOFIntegrityError(EvaluationError):
    """OOF 的 ID 覆盖、唯一性或 outer 隔离失效。"""


class EvaluationBoundaryError(EvaluationError):
    """带入口 mode 和失败阶段的评估边界错误。"""

    def __init__(self, mode: str, stage: str, detail: str):
        self.mode = mode
        self.stage = stage
        self.detail = detail
        super().__init__(f"[{mode}/{stage}] {detail}")


class _EventSinkFailure(Exception):
    """内部标记：event sink 自身异常不转换为评估失败。"""

    def __init__(self, original: Exception):
        self.original = original
        super().__init__(str(original))


class _Predictor(Protocol):
    def __call__(self, features: np.ndarray) -> np.ndarray: ...


class _InverseTransform(Protocol):
    def inverse_transform(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EvaluationEvent:
    """长评估的模式化可观测事件。"""

    kind: str
    mode: str
    completed: int
    total: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class MetricStats:
    mae: float
    rmse: float
    nrmse: float
    r2: float


@dataclass(frozen=True)
class PhaseMetricStats:
    rmse: float
    nrmse: float


@dataclass(frozen=True)
class ProjectionStats:
    count: int
    fraction: float
    maximum_correction: float


@dataclass(frozen=True)
class DewPointReportingAudit:
    """热力学露点到指标报告地板的显式调整记录。"""

    floor_C: float
    adjusted_count: int
    fraction: float
    negative_infinity_count: int
    below_floor_finite_count: int
    maximum_finite_adjustment_C: float


@dataclass(frozen=True)
class PhaseSplit:
    """再生出口总水量的气/液相语义。"""

    total_raw: np.ndarray
    total_nonnegative: np.ndarray
    vapor: np.ndarray
    liquid: np.ndarray
    saturation: np.ndarray

    @property
    def gas_for_mixing(self) -> np.ndarray:
        """混风只允许使用气相，液水在转轮出口单独排出。"""

        return self.vapor


@dataclass(frozen=True)
class EvaluationResult:
    mode: str
    sample_ids: np.ndarray
    predictions_raw: np.ndarray
    targets_raw: np.ndarray
    predictions_deployed: np.ndarray
    targets_deployed: np.ndarray
    raw_metrics: Mapping[str, MetricStats]
    deployed_metrics: Mapping[str, MetricStats]
    projection_stats: Mapping[str, ProjectionStats]
    regen_prediction_phase: PhaseSplit
    regen_target_phase: PhaseSplit
    regen_phase_metric: PhaseMetricStats
    process_dew_point_prediction_thermodynamic_C: np.ndarray
    process_dew_point_target_thermodynamic_C: np.ndarray
    process_dew_point_prediction_reporting_C: np.ndarray
    process_dew_point_target_reporting_C: np.ndarray
    process_dew_point_prediction_reporting_audit: DewPointReportingAudit
    process_dew_point_target_reporting_audit: DewPointReportingAudit
    process_dew_point_metric: MetricStats

    @property
    def process_dew_point_prediction_C(self) -> np.ndarray:
        """兼容属性；指标消费的是显式 reporting 露点。"""

        return self.process_dew_point_prediction_reporting_C

    @property
    def process_dew_point_target_C(self) -> np.ndarray:
        """兼容属性；指标消费的是显式 reporting 露点。"""

        return self.process_dew_point_target_reporting_C


class AuditedMinMaxScaler:
    """MinMax[-1,1] 加不可丢失的拟合 ID 审计记录。"""

    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = tuple(feature_names)
        self.fit_ids: tuple[int, ...] = ()
        self._scaler = MinMaxScaler(feature_range=(-1.0, 1.0))

    def fit(
        self, values: np.ndarray, ids: Sequence[int]
    ) -> "AuditedMinMaxScaler":
        matrix = _finite_matrix(values, "scaler 拟合数据")
        if matrix.shape[1] != len(self.feature_names):
            raise EvaluationError("scaler 字段数与数组列数不一致")
        normalized_ids = tuple(int(value) for value in ids)
        if len(normalized_ids) != len(matrix):
            raise EvaluationError("scaler 拟合 ID 与样本数不一致")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise EvaluationError("scaler 拟合 ID 重复")
        self._scaler.fit(matrix)
        self.fit_ids = normalized_ids
        return self

    @property
    def data_min_(self) -> np.ndarray:
        return self._scaler.data_min_

    @property
    def data_max_(self) -> np.ndarray:
        return self._scaler.data_max_

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self._scaler.transform(values), dtype=np.float64)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self._scaler.inverse_transform(values), dtype=np.float64)


@dataclass(frozen=True)
class FoldData:
    fold_name: str
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    outer_ids: tuple[int, ...]
    input_scaler: AuditedMinMaxScaler
    target_scaler: AuditedMinMaxScaler
    target_spans: np.ndarray
    train_features_scaled: np.ndarray
    validation_features_scaled: np.ndarray
    train_targets_scaled: np.ndarray
    validation_targets_scaled: np.ndarray
    train_targets_physical: np.ndarray
    validation_targets_physical: np.ndarray


def _finite_matrix(values: Any, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise EvaluationError(f"{label}必须是二维数组")
    if not np.isfinite(matrix).all():
        raise EvaluationError(f"{label}存在非有限值")
    return matrix


def build_fold_data(
    clean: pd.DataFrame,
    assignments: pd.DataFrame,
    fold_name: str,
) -> FoldData:
    """仅在当折 development train ID 上拟合 scaler 和 NRMSE 跨度。"""

    if fold_name not in FIXED_FOLD_GROUPS:
        raise EvaluationError(f"未知固定折: {fold_name}")
    assignment_columns = {
        "ID",
        "partition",
        "validation_fold",
        "rotor_thickness_mm",
        "rotor_diameter_mm",
    }
    missing_assignments = assignment_columns - set(assignments.columns)
    if missing_assignments:
        raise EvaluationError(f"划分表缺少字段: {sorted(missing_assignments)}")
    if assignments["ID"].duplicated().any():
        raise EvaluationError("划分表 ID 重复")
    if clean["ID"].duplicated().any():
        raise EvaluationError("清洗数据 ID 重复")

    try:
        assignment_ids_numeric = pd.to_numeric(assignments["ID"], errors="raise")
        clean_ids_numeric = pd.to_numeric(clean["ID"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvaluationError("划分表或清洗数据 ID 不是数字") from exc
    if not np.equal(assignment_ids_numeric, np.floor(assignment_ids_numeric)).all():
        raise EvaluationError("划分表 ID 必须是整数")
    if not np.equal(clean_ids_numeric, np.floor(clean_ids_numeric)).all():
        raise EvaluationError("清洗数据 ID 必须是整数")
    assignment_id_set = set(assignment_ids_numeric.astype("int64"))
    clean_id_set = set(clean_ids_numeric.astype("int64"))
    if (
        len(assignments) != _EXPECTED_CLEAN_ROWS
        or len(clean) != _EXPECTED_CLEAN_ROWS
        or assignment_id_set != clean_id_set
    ):
        raise EvaluationError(
            "划分表必须与清洗数据完整覆盖同一组 57,185 个 ID"
        )

    clean_geometry = clean.set_index("ID")[[
        "TextBox_Rotor_depth",
        "TextBox_Rotor_diameter",
    ]].rename(
        columns={
            "TextBox_Rotor_depth": "authoritative_thickness_mm",
            "TextBox_Rotor_diameter": "authoritative_diameter_mm",
        }
    )
    assignment_geometry = assignments.set_index("ID")[[
        "rotor_thickness_mm",
        "rotor_diameter_mm",
    ]].join(clean_geometry, how="left", validate="one_to_one")
    geometry_values = assignment_geometry.to_numpy(dtype=np.float64)
    if not np.isfinite(geometry_values).all():
        raise EvaluationError("划分表或清洗数据 ID 几何存在非有限值")
    geometry_mismatch = (
        assignment_geometry["rotor_thickness_mm"].to_numpy(dtype=np.float64)
        != assignment_geometry["authoritative_thickness_mm"].to_numpy(
            dtype=np.float64
        )
    ) | (
        assignment_geometry["rotor_diameter_mm"].to_numpy(dtype=np.float64)
        != assignment_geometry["authoritative_diameter_mm"].to_numpy(
            dtype=np.float64
        )
    )
    if geometry_mismatch.any():
        bad_ids = assignment_geometry.index[geometry_mismatch].tolist()[:20]
        raise EvaluationError(f"划分表 ID 与权威清洗几何不一致: {bad_ids}")

    if not set(assignments["partition"]).issubset({"development", "outer_test"}):
        raise EvaluationError("划分表存在未知 partition")
    expected_fold_by_group = {
        tuple(group): expected_fold
        for expected_fold, groups in FIXED_FOLD_GROUPS.items()
        for group in groups
    }
    development_assignment = assignments.loc[
        assignments["partition"].eq("development")
    ]
    actual_folds = development_assignment["validation_fold"].astype(str)
    expected_folds = pd.Series(
        [
            expected_fold_by_group.get(
                (int(row.rotor_thickness_mm), int(row.rotor_diameter_mm))
            )
            for row in development_assignment.itertuples(index=False)
        ],
        index=development_assignment.index,
        dtype="object",
    )
    invalid_fixed_assignment = expected_folds.isna() | actual_folds.ne(expected_folds)
    outer_assignment = assignments.loc[assignments["partition"].eq("outer_test")]
    outer_groups = set(
        zip(
            outer_assignment["rotor_thickness_mm"].astype(int),
            outer_assignment["rotor_diameter_mm"].astype(int),
        )
    )
    if invalid_fixed_assignment.any() or outer_groups != {OUTER_GROUP}:
        raise EvaluationError("划分表与锁定的固定几何分组不一致")

    development = assignments.loc[assignments["partition"].eq("development")]
    outer = assignments.loc[assignments["partition"].eq("outer_test")]
    if len(development) != _EXPECTED_DEVELOPMENT_ROWS:
        raise EvaluationError(
            f"development 应为 {_EXPECTED_DEVELOPMENT_ROWS} 条，实际 {len(development)}"
        )
    authoritative_outer_ids = set(
        int(value)
        for value in clean.loc[
            clean["TextBox_Rotor_depth"].eq(OUTER_GROUP[0])
            & clean["TextBox_Rotor_diameter"].eq(OUTER_GROUP[1]),
            "ID",
        ]
    )
    assigned_outer_ids = set(int(value) for value in outer["ID"])
    if (
        len(outer) != _EXPECTED_OUTER_ROWS
        or len(authoritative_outer_ids) != _EXPECTED_OUTER_ROWS
        or assigned_outer_ids != authoritative_outer_ids
    ):
        raise EvaluationError(
            "outer 必须恰好是 4,750 个权威 300×1070 mm ID"
        )
    validation_rows = development.loc[
        development["validation_fold"].eq(fold_name)
    ]
    train_rows = development.loc[
        ~development["validation_fold"].eq(fold_name)
    ]
    if validation_rows.empty or train_rows.empty:
        raise EvaluationError(f"{fold_name} 的 train/validation 不得为空")

    train_ids = tuple(int(value) for value in train_rows["ID"])
    validation_ids = tuple(int(value) for value in validation_rows["ID"])
    outer_ids = tuple(int(value) for value in outer["ID"])
    train_set = set(train_ids)
    validation_set = set(validation_ids)
    outer_set = set(outer_ids)
    if train_set & validation_set or train_set & outer_set or validation_set & outer_set:
        raise EvaluationError(f"{fold_name} 的 train/validation/outer ID 不隔离")
    if train_set | validation_set != set(int(value) for value in development["ID"]):
        raise EvaluationError(f"{fold_name} 未完整覆盖 development ID")

    indexed = clean.set_index("ID", drop=False)
    required_ids = train_set | validation_set
    missing_ids = required_ids - set(int(value) for value in indexed.index)
    if missing_ids:
        raise EvaluationError(f"清洗数据缺少 development ID: {sorted(missing_ids)[:20]}")

    # 先以 development ID 定位，再读取建模列；outer 值不进入任何数值路径。
    train_features = _finite_matrix(
        indexed.loc[list(train_ids), list(MODEL_INPUT_COLUMNS)].to_numpy(),
        f"{fold_name} 训练输入",
    )
    validation_features = _finite_matrix(
        indexed.loc[list(validation_ids), list(MODEL_INPUT_COLUMNS)].to_numpy(),
        f"{fold_name} 验证输入",
    )
    train_targets = _finite_matrix(
        indexed.loc[list(train_ids), list(MODEL_OUTPUT_COLUMNS)].to_numpy(),
        f"{fold_name} 训练标签",
    )
    validation_targets = _finite_matrix(
        indexed.loc[list(validation_ids), list(MODEL_OUTPUT_COLUMNS)].to_numpy(),
        f"{fold_name} 验证标签",
    )

    target_spans = np.ptp(train_targets, axis=0)
    zero_columns = [
        column
        for column, span in zip(MODEL_OUTPUT_COLUMNS, target_spans)
        if not np.isfinite(span) or span == 0.0
    ]
    if zero_columns:
        raise EvaluationError(f"{fold_name} 折训练目标跨度为零: {zero_columns}")

    input_scaler = AuditedMinMaxScaler(MODEL_INPUT_COLUMNS).fit(
        train_features, train_ids
    )
    target_scaler = AuditedMinMaxScaler(MODEL_OUTPUT_COLUMNS).fit(
        train_targets, train_ids
    )
    return FoldData(
        fold_name=fold_name,
        train_ids=train_ids,
        validation_ids=validation_ids,
        outer_ids=outer_ids,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        target_spans=target_spans,
        train_features_scaled=input_scaler.transform(train_features),
        validation_features_scaled=input_scaler.transform(validation_features),
        train_targets_scaled=target_scaler.transform(train_targets),
        validation_targets_scaled=target_scaler.transform(validation_targets),
        train_targets_physical=train_targets,
        validation_targets_physical=validation_targets,
    )


def assemble_oof_predictions(
    fold_predictions: Mapping[str, pd.DataFrame],
    *,
    expected_development_ids: Sequence[int] | pd.Series,
    expected_validation_ids_by_fold: Mapping[
        str, Sequence[int] | pd.Series
    ],
    outer_ids: Sequence[int] | pd.Series,
) -> pd.DataFrame:
    """按 ID 组装单候选 OOF，并在路由前强制 52,435 唯一覆盖。"""

    if not fold_predictions:
        raise OOFIntegrityError("OOF 折预测为空")
    required_folds = set(FIXED_FOLD_GROUPS)
    if set(fold_predictions) != required_folds:
        raise OOFIntegrityError(
            "OOF fold keys 必须恰好为 fold_1..fold_5: "
            f"actual={sorted(fold_predictions)}"
        )
    if set(expected_validation_ids_by_fold) != required_folds:
        raise OOFIntegrityError(
            "OOF expected fold keys 必须恰好为 fold_1..fold_5"
        )
    expected = tuple(int(value) for value in expected_development_ids)
    if len(expected) != _EXPECTED_DEVELOPMENT_ROWS or len(set(expected)) != len(expected):
        raise OOFIntegrityError(
            f"OOF 期望 development ID 必须是 {_EXPECTED_DEVELOPMENT_ROWS} 个唯一值"
        )
    expected_set = set(expected)
    outer_set = set(int(value) for value in outer_ids)
    if expected_set & outer_set:
        raise OOFIntegrityError("OOF 期望 development ID 与 outer ID 交叉")

    expected_fold_sets: dict[str, set[int]] = {}
    seen_expected: set[int] = set()
    for fold_name in FIXED_FOLD_GROUPS:
        fold_ids = tuple(
            int(value) for value in expected_validation_ids_by_fold[fold_name]
        )
        fold_set = set(fold_ids)
        if len(fold_ids) != len(fold_set):
            raise OOFIntegrityError(f"OOF expected {fold_name} fold ID 重复")
        if fold_set & outer_set:
            raise OOFIntegrityError(f"OOF expected {fold_name} fold 含 outer ID")
        if fold_set & seen_expected:
            raise OOFIntegrityError(f"OOF expected {fold_name} fold 与其他折交叉")
        expected_fold_sets[fold_name] = fold_set
        seen_expected.update(fold_set)
    if seen_expected != expected_set:
        raise OOFIntegrityError(
            "OOF expected fold validation ID 并集与 development ID 不一致"
        )

    frames: list[pd.DataFrame] = []
    expected_columns: tuple[str, ...] | None = None
    for fold_name, frame in fold_predictions.items():
        if "ID" not in frame.columns:
            raise OOFIntegrityError(f"OOF {fold_name} 缺少 ID 字段")
        columns = tuple(str(column) for column in frame.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise OOFIntegrityError(f"OOF {fold_name} 预测字段不一致")
        if frame["ID"].duplicated().any():
            raise OOFIntegrityError(f"OOF {fold_name} 折内 ID 重复")
        try:
            frame_ids = set(
                pd.to_numeric(frame["ID"], errors="raise").astype("int64")
            )
        except (TypeError, ValueError) as exc:
            raise OOFIntegrityError(f"OOF {fold_name} fold ID 不是整数") from exc
        if frame_ids != expected_fold_sets[fold_name]:
            missing_fold = expected_fold_sets[fold_name] - frame_ids
            extra_fold = frame_ids - expected_fold_sets[fold_name]
            raise OOFIntegrityError(
                f"OOF {fold_name} fold ID 与锁定 validation 不一致: "
                f"missing={len(missing_fold)}, extra={len(extra_fold)}"
            )
        frames.append(frame.copy())

    combined = pd.concat(frames, ignore_index=True)
    combined["ID"] = pd.to_numeric(combined["ID"], errors="raise").astype("int64")
    if combined["ID"].duplicated().any():
        raise OOFIntegrityError("OOF 跨折 ID 重复")
    actual_set = set(int(value) for value in combined["ID"])
    leaked_outer = actual_set & outer_set
    if leaked_outer:
        raise OOFIntegrityError(f"OOF 含 outer ID: {sorted(leaked_outer)[:20]}")
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if len(combined) != _EXPECTED_DEVELOPMENT_ROWS or missing or extra:
        raise OOFIntegrityError(
            "OOF 未恰好覆盖 52,435 个 development ID: "
            f"rows={len(combined)}, missing={len(missing)}, extra={len(extra)}"
        )
    return combined.sort_values("ID", kind="stable").reset_index(drop=True)


def phase_split_total_water(
    total_W_g_per_kg: float | Sequence[float] | np.ndarray,
    temperature_C: float | Sequence[float] | np.ndarray,
    pressure_Pa: float,
) -> PhaseSplit:
    """先投影总水量，再按同一 T/P 做再生出口气液分相。"""

    total_raw, temperature = np.broadcast_arrays(
        np.asarray(total_W_g_per_kg, dtype=np.float64),
        np.asarray(temperature_C, dtype=np.float64),
    )
    if not np.isfinite(total_raw).all() or not np.isfinite(temperature).all():
        raise EvaluationError("气液分相输入存在非有限值")
    total_nonnegative = np.maximum(total_raw, 0.0)
    saturation = np.asarray(
        saturation_humidity_ratio_g_per_kg(temperature, pressure_Pa),
        dtype=np.float64,
    )
    vapor = np.minimum(total_nonnegative, saturation)
    liquid = np.maximum(total_nonnegative - saturation, 0.0)
    return PhaseSplit(
        total_raw=total_raw.copy(),
        total_nonnegative=total_nonnegative,
        vapor=vapor,
        liquid=liquid,
        saturation=saturation,
    )


def _apply_inverse(transform: Any, values: np.ndarray, label: str) -> np.ndarray:
    if transform is None:
        result = np.asarray(values, dtype=np.float64)
    elif hasattr(transform, "inverse_transform"):
        result = np.asarray(transform.inverse_transform(values), dtype=np.float64)
    elif callable(transform):
        result = np.asarray(transform(values), dtype=np.float64)
    else:
        raise EvaluationError(f"{label}必须可调用或实现 inverse_transform")
    if result.shape != values.shape:
        raise EvaluationError(f"{label}改变了输出形状")
    return result


def _decode_physical(
    values: np.ndarray,
    inverse_target_transform: Any,
    inverse_scaler: Any,
) -> np.ndarray:
    # 这一顺序是锁定的部署契约，不由候选模型入口自行调换。
    target_inverted = _apply_inverse(
        inverse_target_transform, values, "反目标变换"
    )
    physical = _apply_inverse(inverse_scaler, target_inverted, "反缩放")
    if not np.isfinite(physical).all():
        raise EvaluationError("反变换后物理输出存在非有限值")
    return physical


def _metric(prediction: np.ndarray, target: np.ndarray, span: float) -> MetricStats:
    if not math.isfinite(span) or span <= 0.0:
        raise EvaluationError(f"NRMSE 折训练分母必须大于零，实际 {span}")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = prediction - target
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        nrmse = float(rmse / span)
        ss_res = float(np.sum(np.square(residual)))
        centered = target - float(np.mean(target))
        ss_total = float(np.sum(np.square(centered)))
        if ss_total == 0.0:
            r2 = 1.0 if ss_res == 0.0 else 0.0
        else:
            r2 = float(1.0 - ss_res / ss_total)
    result = MetricStats(mae=mae, rmse=rmse, nrmse=nrmse, r2=r2)
    if not all(math.isfinite(value) for value in result.__dict__.values()):
        raise EvaluationError("指标存在非有限值")
    return result


def _dew_point_reporting_values(
    thermodynamic_C: np.ndarray,
    *,
    floor_C: float = _DEW_POINT_REPORTING_FLOOR_C,
) -> tuple[np.ndarray, DewPointReportingAudit]:
    """仅在指标/报告数组上施加可审计低端地板。"""

    thermodynamic = np.asarray(thermodynamic_C, dtype=np.float64)
    negative_infinity = np.isneginf(thermodynamic)
    unsupported_nonfinite = ~np.isfinite(thermodynamic) & ~negative_infinity
    if unsupported_nonfinite.any():
        raise EvaluationError("热力学露点存在非低端非有限值")
    below_floor_finite = np.isfinite(thermodynamic) & (thermodynamic < floor_C)
    adjusted = negative_infinity | below_floor_finite
    reporting = thermodynamic.copy()
    reporting[adjusted] = floor_C
    finite_adjustments = floor_C - thermodynamic[below_floor_finite]
    audit = DewPointReportingAudit(
        floor_C=floor_C,
        adjusted_count=int(np.count_nonzero(adjusted)),
        fraction=float(np.count_nonzero(adjusted) / len(thermodynamic)),
        negative_infinity_count=int(np.count_nonzero(negative_infinity)),
        below_floor_finite_count=int(np.count_nonzero(below_floor_finite)),
        maximum_finite_adjustment_C=float(
            np.max(finite_adjustments, initial=0.0)
        ),
    )
    return reporting, audit


def evaluate_predictions(
    predictions_model_space: np.ndarray,
    targets_model_space: np.ndarray,
    train_targets_physical: np.ndarray,
    *,
    sample_ids: Sequence[int],
    pressure_Pa: float,
    mode: str,
    inverse_target_transform: Any = None,
    inverse_scaler: Any = None,
) -> EvaluationResult:
    """内存/checkpoint 共用的唯一评估核心。"""

    prediction_model = _finite_matrix(predictions_model_space, "预测")
    target_model = _finite_matrix(targets_model_space, "验证标签")
    train_physical = _finite_matrix(train_targets_physical, "折训练物理标签")
    expected_shape = (len(sample_ids), len(MODEL_OUTPUT_COLUMNS))
    if prediction_model.shape != expected_shape or target_model.shape != expected_shape:
        raise EvaluationError(
            f"预测/标签形状必须为 {expected_shape}，"
            f"实际 {prediction_model.shape}/{target_model.shape}"
        )
    if train_physical.shape[1] != len(MODEL_OUTPUT_COLUMNS):
        raise EvaluationError("折训练物理标签必须是 7 列")
    normalized_ids = np.asarray([int(value) for value in sample_ids], dtype=np.int64)
    if len(set(normalized_ids.tolist())) != len(normalized_ids):
        raise EvaluationError("验证样本 ID 重复")

    predictions_raw = _decode_physical(
        prediction_model, inverse_target_transform, inverse_scaler
    )
    targets_raw = _decode_physical(
        target_model, inverse_target_transform, inverse_scaler
    )
    spans = np.ptp(train_physical, axis=0)
    zero_span = [
        column
        for column, span in zip(MODEL_OUTPUT_COLUMNS, spans)
        if not math.isfinite(float(span)) or float(span) == 0.0
    ]
    if zero_span:
        raise EvaluationError(f"折训练目标跨度为零: {zero_span}")

    predictions_deployed = np.maximum(predictions_raw, 0.0)
    targets_deployed = np.maximum(targets_raw, 0.0)
    raw_metrics: dict[str, MetricStats] = {}
    deployed_metrics: dict[str, MetricStats] = {}
    projection_stats: dict[str, ProjectionStats] = {}
    for index, column in enumerate(MODEL_OUTPUT_COLUMNS):
        raw_metrics[column] = _metric(
            predictions_raw[:, index], targets_raw[:, index], float(spans[index])
        )
        deployed_metrics[column] = _metric(
            predictions_deployed[:, index],
            targets_deployed[:, index],
            float(spans[index]),
        )
        correction = np.abs(predictions_deployed[:, index] - predictions_raw[:, index])
        projection_stats[column] = ProjectionStats(
            count=int(np.count_nonzero(correction)),
            fraction=float(np.count_nonzero(correction) / len(correction)),
            maximum_correction=float(np.max(correction, initial=0.0)),
        )

    regen_t_index = MODEL_OUTPUT_COLUMNS.index(_REGEN_T_COLUMN)
    regen_w_index = MODEL_OUTPUT_COLUMNS.index(_REGEN_W_COLUMN)
    regen_prediction_phase = phase_split_total_water(
        predictions_raw[:, regen_w_index],
        predictions_deployed[:, regen_t_index],
        pressure_Pa,
    )
    regen_target_phase = phase_split_total_water(
        targets_raw[:, regen_w_index],
        targets_deployed[:, regen_t_index],
        pressure_Pa,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        phase_squared = np.square(
            regen_prediction_phase.vapor - regen_target_phase.vapor
        ) + np.square(regen_prediction_phase.liquid - regen_target_phase.liquid)
        phase_rmse = float(np.sqrt(np.mean(phase_squared)))
        phase_nrmse = float(phase_rmse / spans[regen_w_index])
    if not math.isfinite(phase_rmse) or not math.isfinite(phase_nrmse):
        raise EvaluationError("再生气液相指标存在非有限值")
    phase_metric = PhaseMetricStats(rmse=phase_rmse, nrmse=phase_nrmse)

    process_w_index = MODEL_OUTPUT_COLUMNS.index(_PROCESS_W_COLUMN)
    prediction_dew_point_thermodynamic = np.asarray(
        dew_point_water_C(predictions_deployed[:, process_w_index], pressure_Pa),
        dtype=np.float64,
    )
    target_dew_point_thermodynamic = np.asarray(
        dew_point_water_C(targets_deployed[:, process_w_index], pressure_Pa),
        dtype=np.float64,
    )
    train_dew_point_thermodynamic = np.asarray(
        dew_point_water_C(train_physical[:, process_w_index], pressure_Pa),
        dtype=np.float64,
    )
    prediction_dew_point_reporting, prediction_dew_point_audit = (
        _dew_point_reporting_values(prediction_dew_point_thermodynamic)
    )
    target_dew_point_reporting, target_dew_point_audit = (
        _dew_point_reporting_values(target_dew_point_thermodynamic)
    )
    train_dew_point_reporting, _ = _dew_point_reporting_values(
        train_dew_point_thermodynamic
    )
    dew_span = float(np.ptp(train_dew_point_reporting))
    dew_metric = _metric(
        prediction_dew_point_reporting, target_dew_point_reporting, dew_span
    )

    return EvaluationResult(
        mode=mode,
        sample_ids=normalized_ids,
        predictions_raw=predictions_raw,
        targets_raw=targets_raw,
        predictions_deployed=predictions_deployed,
        targets_deployed=targets_deployed,
        raw_metrics=raw_metrics,
        deployed_metrics=deployed_metrics,
        projection_stats=projection_stats,
        regen_prediction_phase=regen_prediction_phase,
        regen_target_phase=regen_target_phase,
        regen_phase_metric=phase_metric,
        process_dew_point_prediction_thermodynamic_C=(
            prediction_dew_point_thermodynamic
        ),
        process_dew_point_target_thermodynamic_C=target_dew_point_thermodynamic,
        process_dew_point_prediction_reporting_C=prediction_dew_point_reporting,
        process_dew_point_target_reporting_C=target_dew_point_reporting,
        process_dew_point_prediction_reporting_audit=prediction_dew_point_audit,
        process_dew_point_target_reporting_audit=target_dew_point_audit,
        process_dew_point_metric=dew_metric,
    )


def _emit(
    sink: Callable[[EvaluationEvent], None] | None,
    *,
    kind: str,
    mode: str,
    completed: int,
    total: int,
    payload: Mapping[str, Any] | None = None,
) -> None:
    if sink is not None:
        sink(
            EvaluationEvent(
                kind=kind,
                mode=mode,
                completed=completed,
                total=total,
                payload={} if payload is None else dict(payload),
            )
        )


def _predict_in_batches(
    predictor: _Predictor,
    features: np.ndarray,
    *,
    mode: str,
    batch_size: int,
    event_sink: Callable[[EvaluationEvent], None] | None,
) -> np.ndarray:
    if not callable(predictor):
        raise EvaluationBoundaryError(mode, "load", "加载结果不是可调用 predictor")
    if batch_size <= 0:
        raise EvaluationBoundaryError(mode, "validation", "batch_size 必须大于零")
    batches: list[np.ndarray] = []
    total = len(features)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        try:
            batch = np.asarray(predictor(features[start:stop]), dtype=np.float64)
        except Exception as exc:
            raise EvaluationBoundaryError(mode, "prediction", str(exc)) from exc
        if batch.shape != (stop - start, len(MODEL_OUTPUT_COLUMNS)):
            raise EvaluationBoundaryError(
                mode,
                "prediction",
                f"批预测形状应为 {(stop - start, len(MODEL_OUTPUT_COLUMNS))}，实际 {batch.shape}",
            )
        if not np.isfinite(batch).all():
            raise EvaluationBoundaryError(mode, "prediction", "预测存在非有限值")
        batches.append(batch)
        _emit(
            event_sink,
            kind="progress",
            mode=mode,
            completed=stop,
            total=total,
            payload={"batch_rows": stop - start},
        )
    return np.concatenate(batches, axis=0)


def _validate_boundary_inputs(
    mode: str,
    features: np.ndarray,
    targets_model_space: np.ndarray,
    sample_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    feature_array = np.asarray(features, dtype=np.float64)
    target_array = np.asarray(targets_model_space, dtype=np.float64)
    if feature_array.ndim != 2 or target_array.ndim != 2:
        raise EvaluationBoundaryError(mode, "validation", "features/targets 必须是二维")
    if len(feature_array) == 0:
        raise EvaluationBoundaryError(mode, "validation", "验证集为空")
    if len(feature_array) != len(target_array) or len(feature_array) != len(sample_ids):
        raise EvaluationBoundaryError(mode, "validation", "features/targets/ID 行数不一致")
    if not np.isfinite(feature_array).all() or not np.isfinite(target_array).all():
        raise EvaluationBoundaryError(mode, "validation", "验证输入或标签存在非有限值")
    return feature_array, target_array


def _evaluate_entry(
    predictor: _Predictor,
    features: np.ndarray,
    targets_model_space: np.ndarray,
    train_targets_physical: np.ndarray,
    *,
    sample_ids: Sequence[int],
    pressure_Pa: float,
    mode: str,
    inverse_target_transform: Any,
    inverse_scaler: Any,
    batch_size: int,
    event_sink: Callable[[EvaluationEvent], None] | None,
) -> EvaluationResult:
    predictions = _predict_in_batches(
        predictor,
        features,
        mode=mode,
        batch_size=batch_size,
        event_sink=event_sink,
    )
    try:
        result = evaluate_predictions(
            predictions,
            targets_model_space,
            train_targets_physical,
            sample_ids=sample_ids,
            pressure_Pa=pressure_Pa,
            mode=mode,
            inverse_target_transform=inverse_target_transform,
            inverse_scaler=inverse_scaler,
        )
    except EvaluationBoundaryError:
        raise
    except Exception as exc:
        raise EvaluationBoundaryError(mode, "metric", str(exc)) from exc
    return result


def _run_evaluation_lifecycle(
    *,
    mode: str,
    total: int,
    event_sink: Callable[[EvaluationEvent], None] | None,
    operation: Callable[
        [Callable[[EvaluationEvent], None] | None], EvaluationResult
    ],
) -> EvaluationResult:
    """统一正常/失败评估入口的可观测生命周期。"""

    _emit(event_sink, kind="start", mode=mode, completed=0, total=total)
    started = time.perf_counter()

    progress_sink: Callable[[EvaluationEvent], None] | None = None
    if event_sink is not None:

        def progress_sink(event: EvaluationEvent) -> None:
            try:
                event_sink(event)
            except Exception as exc:
                raise _EventSinkFailure(exc) from exc

    try:
        result = operation(progress_sink)
    except _EventSinkFailure as sink_failure:
        raise sink_failure.original
    except EvaluationBoundaryError as boundary:
        _emit(
            event_sink,
            kind="failure",
            mode=mode,
            completed=0,
            total=total,
            payload={"stage": boundary.stage, "detail": boundary.detail},
        )
        _emit(
            event_sink,
            kind="end",
            mode=mode,
            completed=0,
            total=total,
            payload={"status": "failed", "stage": boundary.stage},
        )
        raise
    except Exception as exc:
        boundary = EvaluationBoundaryError(mode, "internal", str(exc))
        _emit(
            event_sink,
            kind="failure",
            mode=mode,
            completed=0,
            total=total,
            payload={"stage": boundary.stage, "detail": boundary.detail},
        )
        _emit(
            event_sink,
            kind="end",
            mode=mode,
            completed=0,
            total=total,
            payload={"status": "failed", "stage": boundary.stage},
        )
        raise boundary from exc

    elapsed = time.perf_counter() - started
    _emit(
        event_sink,
        kind="end",
        mode=mode,
        completed=total,
        total=total,
        payload={"status": "success"},
    )
    _emit(
        event_sink,
        kind="result",
        mode=mode,
        completed=total,
        total=total,
        payload={"outputs": len(result.deployed_metrics)},
    )
    _emit(
        event_sink,
        kind="efficiency",
        mode=mode,
        completed=total,
        total=total,
        payload={
            "elapsed_seconds": elapsed,
            "samples_per_second": total / elapsed,
        },
    )
    return result


def evaluate_in_memory(
    predictor: _Predictor,
    features: np.ndarray,
    targets_model_space: np.ndarray,
    train_targets_physical: np.ndarray,
    *,
    sample_ids: Sequence[int],
    pressure_Pa: float,
    inverse_target_transform: Any = None,
    inverse_scaler: Any = None,
    batch_size: int = 1024,
    event_sink: Callable[[EvaluationEvent], None] | None = None,
) -> EvaluationResult:
    """使用训练中内存模型进入共享评估核心。"""

    mode = "in_memory"
    total = len(features) if hasattr(features, "__len__") else 0

    def operation(
        lifecycle_sink: Callable[[EvaluationEvent], None] | None,
    ) -> EvaluationResult:
        feature_array, target_array = _validate_boundary_inputs(
            mode, features, targets_model_space, sample_ids
        )
        return _evaluate_entry(
            predictor,
            feature_array,
            target_array,
            train_targets_physical,
            sample_ids=sample_ids,
            pressure_Pa=pressure_Pa,
            mode=mode,
            inverse_target_transform=inverse_target_transform,
            inverse_scaler=inverse_scaler,
            batch_size=batch_size,
            event_sink=lifecycle_sink,
        )

    return _run_evaluation_lifecycle(
        mode=mode,
        total=total,
        event_sink=event_sink,
        operation=operation,
    )


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    load_predictor: Callable[[Path], _Predictor],
    features: np.ndarray,
    targets_model_space: np.ndarray,
    train_targets_physical: np.ndarray,
    *,
    sample_ids: Sequence[int],
    pressure_Pa: float,
    inverse_target_transform: Any = None,
    inverse_scaler: Any = None,
    batch_size: int = 1024,
    event_sink: Callable[[EvaluationEvent], None] | None = None,
) -> EvaluationResult:
    """从 checkpoint 严格加载 predictor 后进入同一评估核心。"""

    mode = "checkpoint"
    total = len(features) if hasattr(features, "__len__") else 0

    def operation(
        lifecycle_sink: Callable[[EvaluationEvent], None] | None,
    ) -> EvaluationResult:
        feature_array, target_array = _validate_boundary_inputs(
            mode, features, targets_model_space, sample_ids
        )
        path = Path(checkpoint_path)
        if not path.is_file():
            raise EvaluationBoundaryError(
                mode, "load", f"checkpoint 不存在: {path}"
            )
        try:
            loaded_predictor = load_predictor(path)
        except Exception as exc:
            raise EvaluationBoundaryError(mode, "load", str(exc)) from exc
        if not callable(loaded_predictor):
            raise EvaluationBoundaryError(
                mode, "load", "checkpoint 未加载出可调用 predictor"
            )
        return _evaluate_entry(
            loaded_predictor,
            feature_array,
            target_array,
            train_targets_physical,
            sample_ids=sample_ids,
            pressure_Pa=pressure_Pa,
            mode=mode,
            inverse_target_transform=inverse_target_transform,
            inverse_scaler=inverse_scaler,
            batch_size=batch_size,
            event_sink=lifecycle_sink,
        )

    return _run_evaluation_lifecycle(
        mode=mode,
        total=total,
        event_sink=event_sink,
        operation=operation,
    )
