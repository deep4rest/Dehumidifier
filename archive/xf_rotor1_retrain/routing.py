"""只使用开发集五折 OOF 的确定性逐输出路由。

路由核心只接受 Subtask 2 的 :class:`EvaluationResult` 与 :class:`FoldData`。
它不接收 outer 标签、模型 checkpoint、pickle 或可替换 artifact 路径。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import data as xf_data
from .candidates import CANDIDATES, build_candidate_model
from .config import (
    FIXED_FOLD_GROUPS,
    MODEL_OUTPUT_COLUMNS,
    OUTER_GROUP,
    REPORTS_DIR,
)
from .evaluator import (
    EvaluationResult,
    EvaluationError,
    FoldData,
    MetricStats,
    OOFIntegrityError,
    PhaseMetricStats,
    ProjectionStats,
    assemble_oof_predictions,
    build_fold_data,
    evaluate_predictions,
)

__all__ = [
    "CANDIDATE_PARAMETER_COUNTS",
    "RoutingIntegrityError",
    "apply_oof_routing",
    "build_oof_routing",
    "load_routing_json",
    "save_routing_json",
    "validate_routing_artifact",
]


_SCHEMA_VERSION = 1
_EXPECTED_DEVELOPMENT_ROWS = 52_435
_EXPECTED_OUTER_ROWS = 4_750
_TIE_THRESHOLD = 1.0e-4
_HASH_FORMAT = "sorted-int64-little-endian-v1"
_OOF_CONTENT_HASH_FORMAT = "numpy-dtype-shape-c-order-bytes-v1"
_ROUTING_METHOD = "five_fold_oof_deployed_nrmse_v1"

_PROCESS_T = "Label_Process_Outlet_Temp"
_PROCESS_W = "Label_Process_Outlet_AbsHum"
_PURGE_T = "Label_Purge_Outlet_Temp"
_PURGE_W = "Label_Purge_Outlet_AbsHum"
_REGEN_T = "Label_React_Outlet_Temp"
_REGEN_W = "Label_React_Outlet_AbsHum"
_POWER = "Label_PowerConsumption"

_DIRECT_SELECTION_ORDER = (_PROCESS_T, _PURGE_T, _REGEN_T, _POWER)
_HUMIDITY_SELECTION_ORDER = (_PROCESS_W, _PURGE_W, _REGEN_W)
_SELECTION_ORDER = _DIRECT_SELECTION_ORDER + _HUMIDITY_SELECTION_ORDER
_HUMIDITY_TEMPERATURE = {
    _PROCESS_W: _PROCESS_T,
    _PURGE_W: _PURGE_T,
    _REGEN_W: _REGEN_T,
}


class RoutingIntegrityError(ValueError):
    """OOF 路由输入或 JSON 产物违反固定契约。"""


def _actual_parameter_counts() -> dict[str, int]:
    """从唯一 KAN builder 计算可训练参数量，并恢复 torch RNG 状态。"""

    import torch

    state = torch.random.get_rng_state()
    try:
        return {
            name: int(sum(parameter.numel() for parameter in build_candidate_model(config).parameters()))
            for name, config in CANDIDATES.items()
        }
    finally:
        torch.random.set_rng_state(state)


CANDIDATE_PARAMETER_COUNTS = _actual_parameter_counts()


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutingIntegrityError(f"{label}必须是对象")
    actual = set(value)
    if actual != expected:
        raise RoutingIntegrityError(
            f"{label}字段不符: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RoutingIntegrityError(f"{label}必须是有限数值")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RoutingIntegrityError(f"{label}必须是有限数值") from exc
    if not math.isfinite(normalized):
        raise RoutingIntegrityError(f"{label}存在非有限值")
    return normalized


def _unique_integer_ids(values: Sequence[int], label: str) -> tuple[int, ...]:
    try:
        raw = np.asarray(values)
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RoutingIntegrityError(f"{label}必须是整数 ID") from exc
    if numeric.ndim != 1 or len(numeric) == 0:
        raise RoutingIntegrityError(f"{label}必须是一维非空 ID")
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise RoutingIntegrityError(f"{label}必须是有限整数 ID")
    normalized = tuple(int(value) for value in numeric)
    if len(set(normalized)) != len(normalized):
        raise RoutingIntegrityError(f"{label}存在重复 ID")
    if raw.ndim != 1:
        raise RoutingIntegrityError(f"{label}必须是一维 ID")
    return normalized


def _ids_sha256(values: Sequence[int]) -> str:
    normalized = np.asarray(sorted(int(value) for value in values), dtype="<i8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _verify_current_source_sha256() -> str:
    """每次 build 直接核验固定 XF.xls bytes，不接受路径注入。"""

    path = xf_data.SOURCE_PATH
    try:
        if not path.is_file():
            raise xf_data.DataContractError(f"XF 数据文件不存在: {path}")
        snapshot = path.read_bytes()
        actual = hashlib.sha256(snapshot).hexdigest()
        if actual != xf_data.EXPECTED_SOURCE_SHA256:
            raise xf_data.DataContractError(
                "XF.xls SHA-256 与锁定契约不一致: "
                f"expected={xf_data.EXPECTED_SOURCE_SHA256}, actual={actual}"
            )
        return actual
    except xf_data.DataContractError as exc:
        raise RoutingIntegrityError(f"XF 权威源校验失败: {exc}") from exc
    except OSError as exc:
        contract_error = xf_data.DataContractError(
            f"XF 数据文件读取失败: {path}: {exc}"
        )
        raise RoutingIntegrityError(
            f"XF 权威源校验失败: {contract_error}"
        ) from contract_error


@lru_cache(maxsize=1)
def _authoritative_clean_for_source_hash(source_sha256: str) -> pd.DataFrame:
    """只按已验证 source hash 解析/清洗一次；不缓存 bytes 校验。"""

    if source_sha256 != xf_data.EXPECTED_SOURCE_SHA256:
        raise xf_data.DataContractError("XF 权威 clean cache key 不受支持")
    raw = xf_data.read_xf_source()
    if raw.attrs.get("source_sha256") != source_sha256:
        raise xf_data.DataContractError("XF 权威解析快照哈希与 cache key 不一致")
    clean, _ = xf_data.validate_and_clean_xf(raw)
    return clean


@lru_cache(maxsize=1)
def _authoritative_folds_for_contract(
    source_sha256: str,
    snapshot_id: str,
    split_assignments_sha256: str,
) -> tuple[tuple[str, FoldData], ...]:
    """按源与已验证报告身份缓存五折 scaler/数组重建。"""

    del snapshot_id  # 键的一部分；实体已由 validate_report_snapshot 审计。
    clean = _authoritative_clean_for_source_hash(source_sha256)
    split_path = xf_data.REPORTS_DIR / "split_assignments.csv"
    try:
        content = split_path.read_bytes()
    except OSError as exc:
        raise xf_data.DataContractError(
            f"权威 split_assignments 读取失败: {split_path}: {exc}"
        ) from exc
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != split_assignments_sha256:
        raise xf_data.DataContractError(
            "权威 split_assignments 在快照验证后发生变化: "
            f"expected={split_assignments_sha256}, actual={actual_hash}"
        )
    try:
        assignments = pd.read_csv(io.BytesIO(content))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise xf_data.DataContractError(
            f"权威 split_assignments 无法解析: {exc}"
        ) from exc
    return tuple(
        (fold_name, build_fold_data(clean, assignments, fold_name))
        for fold_name in FIXED_FOLD_GROUPS
    )


def _authoritative_split_contract() -> tuple[
    tuple[int, ...],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    dict[str, Any],
    pd.DataFrame,
]:
    """固定回读已提交报告，不接受调用方注入路径或划分。"""

    try:
        marker = xf_data.validate_report_snapshot(REPORTS_DIR)
        assignments = pd.read_csv(REPORTS_DIR / "split_assignments.csv")
    except Exception as exc:
        raise RoutingIntegrityError(f"权威划分 snapshot 回读失败: {exc}") from exc
    expected_columns = [
        "ID",
        "rotor_thickness_mm",
        "rotor_diameter_mm",
        "partition",
        "validation_fold",
    ]
    if list(assignments.columns) != expected_columns:
        raise RoutingIntegrityError("权威划分字段不符")
    if len(assignments) != _EXPECTED_DEVELOPMENT_ROWS + _EXPECTED_OUTER_ROWS:
        raise RoutingIntegrityError("权威划分行数不符")
    if assignments["ID"].duplicated().any():
        raise RoutingIntegrityError("权威划分 ID 重复")
    try:
        ids = pd.to_numeric(assignments["ID"], errors="raise").to_numpy(dtype=np.float64)
        thickness = pd.to_numeric(
            assignments["rotor_thickness_mm"], errors="raise"
        ).to_numpy(dtype=np.float64)
        diameter = pd.to_numeric(
            assignments["rotor_diameter_mm"], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RoutingIntegrityError("权威划分 ID/几何不是数字") from exc
    for values, label in (
        (ids, "ID"),
        (thickness, "厚度"),
        (diameter, "直径"),
    ):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise RoutingIntegrityError(f"权威划分 {label} 必须是有限整数")
    assignments = assignments.copy()
    assignments["ID"] = ids.astype("int64")
    assignments["rotor_thickness_mm"] = thickness.astype("int64")
    assignments["rotor_diameter_mm"] = diameter.astype("int64")
    if not set(assignments["partition"]).issubset({"development", "outer_test"}):
        raise RoutingIntegrityError("权威划分含未知 partition")
    development = assignments.loc[assignments["partition"].eq("development")]
    outer = assignments.loc[assignments["partition"].eq("outer_test")]
    if len(development) != _EXPECTED_DEVELOPMENT_ROWS or len(outer) != _EXPECTED_OUTER_ROWS:
        raise RoutingIntegrityError("权威 development/outer 行数不符")
    expected_fold_by_group = {
        tuple(group): fold_name
        for fold_name, groups in FIXED_FOLD_GROUPS.items()
        for group in groups
    }
    expected_folds = [
        expected_fold_by_group.get((int(row.rotor_thickness_mm), int(row.rotor_diameter_mm)))
        for row in development.itertuples(index=False)
    ]
    if any(value is None for value in expected_folds) or not np.array_equal(
        development["validation_fold"].astype(str).to_numpy(),
        np.asarray(expected_folds, dtype=object),
    ):
        raise RoutingIntegrityError("权威划分 geometry→fold 与固定映射不一致")
    outer_groups = set(
        zip(outer["rotor_thickness_mm"], outer["rotor_diameter_mm"])
    )
    if outer_groups != {OUTER_GROUP} or outer["validation_fold"].notna().any():
        raise RoutingIntegrityError("权威 outer 必须是 300×1070 且无 validation fold")

    development_ids = tuple(int(value) for value in development["ID"])
    validation_by_fold = {
        fold_name: tuple(
            int(value)
            for value in development.loc[
                development["validation_fold"].eq(fold_name), "ID"
            ]
        )
        for fold_name in FIXED_FOLD_GROUPS
    }
    outer_ids = tuple(int(value) for value in outer["ID"])
    snapshot = marker["report_snapshot"]
    split_member = snapshot["members"]["split_assignments.csv"]
    evidence = {
        "source_sha256": marker["source_sha256"],
        "snapshot_id": snapshot["snapshot_id"],
        "split_assignments_sha256": split_member["sha256"],
        "split_assignments_size_bytes": split_member["size_bytes"],
        "split_assignments_row_count": split_member["row_count"],
    }
    return development_ids, validation_by_fold, outer_ids, evidence, assignments


def _assert_exact_array(actual: Any, expected: Any, label: str) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or not np.array_equal(left, right)
    ):
        raise RoutingIntegrityError(f"{label} 与 XF 权威重建数值不一致")


def _assert_id_aligned_matrix(
    actual: Any,
    actual_ids: Sequence[int],
    expected: Any,
    expected_ids: Sequence[int],
    label: str,
) -> None:
    normalized_actual_ids = _unique_integer_ids(actual_ids, f"{label}.ids")
    normalized_expected_ids = _unique_integer_ids(
        expected_ids, f"{label}.authoritative_ids"
    )
    if set(normalized_actual_ids) != set(normalized_expected_ids):
        raise RoutingIntegrityError(f"{label} ID 与 XF 权威重建不一致")
    position_by_id = {
        sample_id: position
        for position, sample_id in enumerate(normalized_actual_ids)
    }
    positions = np.asarray(
        [position_by_id[sample_id] for sample_id in normalized_expected_ids],
        dtype=np.int64,
    )
    matrix = np.asarray(actual)
    authoritative = np.asarray(expected)
    if matrix.ndim != 2 or matrix.shape[0] != len(normalized_actual_ids):
        raise RoutingIntegrityError(f"{label} 形状与自身 ID 不一致")
    _assert_exact_array(matrix[positions], authoritative, label)


def _assert_authoritative_scaler(
    actual: Any,
    expected: Any,
    *,
    expected_fit_ids: Sequence[int],
    label: str,
) -> None:
    if not isinstance(actual, type(expected)):
        raise RoutingIntegrityError(f"{label} 类型与 XF 权威 scaler 不一致")
    if tuple(actual.feature_names) != tuple(expected.feature_names):
        raise RoutingIntegrityError(f"{label}.feature_names 与 XF 权威重建不一致")
    actual_fit_ids = _unique_integer_ids(actual.fit_ids, f"{label}.fit_ids")
    authoritative_fit_ids = _unique_integer_ids(
        expected_fit_ids, f"{label}.authoritative_fit_ids"
    )
    if (
        len(actual_fit_ids) != len(authoritative_fit_ids)
        or set(actual_fit_ids) != set(authoritative_fit_ids)
    ):
        raise RoutingIntegrityError(f"{label}.fit_ids 与 XF 权威折训练 ID 不一致")
    actual_core = getattr(actual, "_scaler", None)
    expected_core = getattr(expected, "_scaler", None)
    if actual_core is None or expected_core is None:
        raise RoutingIntegrityError(f"{label} 缺少可审计 MinMaxScaler 状态")
    for attribute in (
        "data_min_",
        "data_max_",
        "data_range_",
        "scale_",
        "min_",
    ):
        _assert_exact_array(
            getattr(actual_core, attribute, None),
            getattr(expected_core, attribute, None),
            f"{label}.{attribute}",
        )
    for attribute in (
        "feature_range",
        "clip",
        "n_features_in_",
        "n_samples_seen_",
    ):
        if getattr(actual_core, attribute, None) != getattr(
            expected_core, attribute, None
        ):
            raise RoutingIntegrityError(
                f"{label}.{attribute} 与 XF 权威重建不一致"
            )


def _validate_fold_data(
    fold_data_by_fold: Mapping[str, FoldData],
) -> tuple[
    tuple[int, ...],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    dict[str, Any],
]:
    source_sha256 = _verify_current_source_sha256()
    (
        authoritative_development_ids,
        authoritative_validation_by_fold,
        authoritative_outer_ids,
        authoritative_evidence,
        _,
    ) = _authoritative_split_contract()
    try:
        authoritative_folds = dict(
            _authoritative_folds_for_contract(
                source_sha256,
                str(authoritative_evidence["snapshot_id"]),
                str(authoritative_evidence["split_assignments_sha256"]),
            )
        )
    except (xf_data.DataContractError, EvaluationError) as exc:
        raise RoutingIntegrityError(f"XF 权威 FoldData 重建失败: {exc}") from exc
    required_folds = set(FIXED_FOLD_GROUPS)
    if not isinstance(fold_data_by_fold, Mapping) or set(fold_data_by_fold) != required_folds:
        actual = sorted(fold_data_by_fold) if isinstance(fold_data_by_fold, Mapping) else []
        raise RoutingIntegrityError(
            f"fold context 必须恰好是 fold_1..fold_5，实际 {actual}"
        )

    validation_by_fold: dict[str, tuple[int, ...]] = {}
    train_by_fold: dict[str, tuple[int, ...]] = {}
    outer_reference: tuple[int, ...] | None = None
    seen_validation: set[int] = set()
    for fold_name in FIXED_FOLD_GROUPS:
        fold = fold_data_by_fold[fold_name]
        if not isinstance(fold, FoldData):
            raise RoutingIntegrityError(f"{fold_name} 必须来自 evaluator.FoldData")
        if fold.fold_name != fold_name:
            raise RoutingIntegrityError(f"{fold_name} 的 FoldData 名称不一致")
        train_ids = _unique_integer_ids(fold.train_ids, f"{fold_name} train")
        validation_ids = _unique_integer_ids(
            fold.validation_ids, f"{fold_name} validation"
        )
        outer_ids = _unique_integer_ids(fold.outer_ids, f"{fold_name} outer")
        if len(outer_ids) != _EXPECTED_OUTER_ROWS:
            raise RoutingIntegrityError(
                f"{fold_name} outer 应为 {_EXPECTED_OUTER_ROWS} 个 ID"
            )
        if outer_reference is None:
            outer_reference = outer_ids
        elif set(outer_ids) != set(outer_reference):
            raise RoutingIntegrityError("五折 outer ID 集合不一致")
        train_set = set(train_ids)
        validation_set = set(validation_ids)
        outer_set = set(outer_ids)
        if train_set & validation_set or train_set & outer_set or validation_set & outer_set:
            raise RoutingIntegrityError(f"{fold_name} train/validation/outer ID 不隔离")
        if seen_validation & validation_set:
            raise RoutingIntegrityError(f"{fold_name} validation 与其他折重复")
        seen_validation.update(validation_set)
        validation_by_fold[fold_name] = validation_ids
        train_by_fold[fold_name] = train_ids
        authoritative_validation = set(authoritative_validation_by_fold[fold_name])
        authoritative_train = set(authoritative_development_ids) - authoritative_validation
        if validation_set != authoritative_validation:
            raise RoutingIntegrityError(
                f"{fold_name} validation ID 与权威划分 snapshot 不一致"
            )
        if train_set != authoritative_train:
            raise RoutingIntegrityError(
                f"{fold_name} train ID 与权威划分 snapshot 不一致"
            )
        if outer_set != set(authoritative_outer_ids):
            raise RoutingIntegrityError(
                f"{fold_name} outer ID 与权威划分 snapshot 不一致"
            )
        for scaler_name in ("input_scaler", "target_scaler"):
            scaler = getattr(fold, scaler_name)
            if scaler is None or not hasattr(scaler, "fit_ids"):
                raise RoutingIntegrityError(f"{fold_name} {scaler_name}.fit_ids 缺失")
            fit_ids = _unique_integer_ids(
                scaler.fit_ids, f"{fold_name} {scaler_name}.fit_ids"
            )
            if set(fit_ids) != authoritative_train or len(fit_ids) != len(authoritative_train):
                raise RoutingIntegrityError(
                    f"{fold_name} {scaler_name}.fit_ids 不是权威折训练 ID"
                )

        train_targets = np.asarray(fold.train_targets_physical, dtype=np.float64)
        validation_targets = np.asarray(
            fold.validation_targets_physical, dtype=np.float64
        )
        if train_targets.shape != (len(train_ids), len(MODEL_OUTPUT_COLUMNS)):
            raise RoutingIntegrityError(f"{fold_name} 折训练物理标签形状不一致")
        if validation_targets.shape != (
            len(validation_ids),
            len(MODEL_OUTPUT_COLUMNS),
        ):
            raise RoutingIntegrityError(f"{fold_name} validation 物理标签形状不一致")
        if not np.isfinite(train_targets).all() or not np.isfinite(validation_targets).all():
            raise RoutingIntegrityError(f"{fold_name} development 标签存在非有限值")
        spans = np.ptp(train_targets, axis=0)
        if not np.isfinite(spans).all() or np.any(spans <= 0.0):
            raise RoutingIntegrityError(f"{fold_name} 折训练目标跨度无效")
        recorded_spans = np.asarray(fold.target_spans, dtype=np.float64)
        if recorded_spans.shape != spans.shape or not np.array_equal(recorded_spans, spans):
            raise RoutingIntegrityError(f"{fold_name} 折训练跨度与 FoldData 不一致")
        authoritative_fold = authoritative_folds[fold_name]
        _assert_exact_array(
            np.asarray(fold.target_spans),
            authoritative_fold.target_spans,
            f"{fold_name}.target_spans",
        )
        for field_name in (
            "train_targets_physical",
            "train_features_scaled",
            "train_targets_scaled",
        ):
            _assert_id_aligned_matrix(
                getattr(fold, field_name),
                train_ids,
                getattr(authoritative_fold, field_name),
                authoritative_fold.train_ids,
                f"{fold_name}.{field_name}",
            )
        for field_name in (
            "validation_targets_physical",
            "validation_features_scaled",
            "validation_targets_scaled",
        ):
            _assert_id_aligned_matrix(
                getattr(fold, field_name),
                validation_ids,
                getattr(authoritative_fold, field_name),
                authoritative_fold.validation_ids,
                f"{fold_name}.{field_name}",
            )
        _assert_authoritative_scaler(
            fold.input_scaler,
            authoritative_fold.input_scaler,
            expected_fit_ids=authoritative_fold.train_ids,
            label=f"{fold_name}.input_scaler",
        )
        _assert_authoritative_scaler(
            fold.target_scaler,
            authoritative_fold.target_scaler,
            expected_fit_ids=authoritative_fold.train_ids,
            label=f"{fold_name}.target_scaler",
        )

    if len(seen_validation) != _EXPECTED_DEVELOPMENT_ROWS:
        raise RoutingIntegrityError(
            "五折 validation 并集必须恰好覆盖 52,435 个 development ID"
        )
    development_ids = tuple(sorted(seen_validation))
    development_set = set(development_ids)
    assert outer_reference is not None
    if development_set & set(outer_reference):
        raise RoutingIntegrityError("development ID 与 outer ID 交叉")
    for fold_name in FIXED_FOLD_GROUPS:
        expected_train = development_set - set(validation_by_fold[fold_name])
        if set(train_by_fold[fold_name]) != expected_train:
            raise RoutingIntegrityError(
                f"{fold_name} train ID 必须等于 development 减去当折 validation"
            )
    if set(development_ids) != set(authoritative_development_ids):
        raise RoutingIntegrityError("development ID 与权威划分 snapshot 不一致")
    return development_ids, validation_by_fold, outer_reference, authoritative_evidence


def _validate_metric(metric: Any, label: str) -> MetricStats:
    if not isinstance(metric, MetricStats):
        raise RoutingIntegrityError(f"{label} metric 类型无效")
    for name in ("mae", "rmse", "nrmse", "r2"):
        _finite_float(getattr(metric, name), f"{label}.{name}")
    return metric


def _validate_phase_metric(metric: Any, label: str) -> PhaseMetricStats:
    if not isinstance(metric, PhaseMetricStats):
        raise RoutingIntegrityError(f"{label} 相态 metric 类型无效")
    _finite_float(metric.rmse, f"{label}.rmse")
    _finite_float(metric.nrmse, f"{label}.nrmse")
    return metric


def _validate_projection(metric: Any, label: str) -> ProjectionStats:
    if not isinstance(metric, ProjectionStats):
        raise RoutingIntegrityError(f"{label} projection 类型无效")
    if not isinstance(metric.count, int) or metric.count < 0:
        raise RoutingIntegrityError(f"{label}.count 无效")
    fraction = _finite_float(metric.fraction, f"{label}.fraction")
    maximum = _finite_float(metric.maximum_correction, f"{label}.maximum_correction")
    if not 0.0 <= fraction <= 1.0 or maximum < 0.0:
        raise RoutingIntegrityError(f"{label} projection 数值越界")
    return metric


def _validation_targets_for_ids(
    fold: FoldData, ids: Sequence[int]
) -> np.ndarray:
    """以 ID 而非行号对齐 FoldData 的权威 validation 标签。"""

    position_by_id = {
        int(sample_id): position
        for position, sample_id in enumerate(fold.validation_ids)
    }
    try:
        positions = [position_by_id[int(sample_id)] for sample_id in ids]
    except KeyError as exc:
        raise RoutingIntegrityError(
            f"{fold.fold_name} OOF ID 不属于锁定 validation fold: {exc.args[0]}"
        ) from exc
    return np.asarray(fold.validation_targets_physical, dtype=np.float64)[positions]


def _prediction_column_for_ids(
    result: EvaluationResult,
    ids: Sequence[int],
    column_index: int,
) -> np.ndarray:
    """按 ID 重排另一候选的预测列，供温度/湿量混合评估。"""

    position_by_id = {
        int(sample_id): position
        for position, sample_id in enumerate(result.sample_ids)
    }
    try:
        positions = [position_by_id[int(sample_id)] for sample_id in ids]
    except KeyError as exc:
        raise RoutingIntegrityError(
            f"温度候选缺少 OOF ID: {exc.args[0]}"
        ) from exc
    return np.asarray(result.predictions_raw, dtype=np.float64)[
        positions, column_index
    ]


def _arrays_numerically_equal(
    cached: Any,
    canonical: Any,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> bool:
    left = np.asarray(cached, dtype=np.float64)
    right = np.asarray(canonical, dtype=np.float64)
    if left.shape != right.shape:
        return False
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    if not np.array_equal(left_finite, right_finite):
        return False
    if not np.array_equal(np.signbit(left[~left_finite]), np.signbit(right[~right_finite])):
        return False
    return bool(np.allclose(left[left_finite], right[right_finite], rtol=rtol, atol=atol))


def _metric_cache_equal(cached: MetricStats, canonical: MetricStats) -> bool:
    return all(
        math.isclose(
            float(getattr(cached, name)),
            float(getattr(canonical, name)),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        for name in ("mae", "rmse", "nrmse", "r2")
    )


def _assert_evaluation_cache_matches(
    cached: EvaluationResult,
    canonical: EvaluationResult,
    *,
    label: str,
) -> None:
    """缓存只作审计；路由值必须能由共享 evaluator 唯一重算。"""

    for name in (
        "predictions_raw",
        "targets_raw",
        "predictions_deployed",
        "targets_deployed",
        "process_dew_point_prediction_thermodynamic_C",
        "process_dew_point_target_thermodynamic_C",
        "process_dew_point_prediction_reporting_C",
        "process_dew_point_target_reporting_C",
    ):
        if not _arrays_numerically_equal(getattr(cached, name), getattr(canonical, name)):
            raise RoutingIntegrityError(f"{label} evaluator 缓存与共享核心重算不一致: {name}")
    for section_name in ("raw_metrics", "deployed_metrics"):
        cached_metrics = getattr(cached, section_name)
        canonical_metrics = getattr(canonical, section_name)
        for output in MODEL_OUTPUT_COLUMNS:
            if not _metric_cache_equal(cached_metrics[output], canonical_metrics[output]):
                raise RoutingIntegrityError(
                    f"{label} evaluator 缓存与共享核心重算不一致: {section_name}.{output}"
                )
    for output in MODEL_OUTPUT_COLUMNS:
        cached_projection = cached.projection_stats[output]
        canonical_projection = canonical.projection_stats[output]
        if (
            cached_projection.count != canonical_projection.count
            or not math.isclose(
                cached_projection.fraction,
                canonical_projection.fraction,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                cached_projection.maximum_correction,
                canonical_projection.maximum_correction,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise RoutingIntegrityError(
                f"{label} evaluator 缓存与共享核心重算不一致: projection.{output}"
            )
    if not (
        math.isclose(
            cached.regen_phase_metric.rmse,
            canonical.regen_phase_metric.rmse,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            cached.regen_phase_metric.nrmse,
            canonical.regen_phase_metric.nrmse,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise RoutingIntegrityError(
            f"{label} evaluator 缓存与共享核心重算不一致: regen_phase_metric"
        )
    for phase_name in ("regen_prediction_phase", "regen_target_phase"):
        cached_phase = getattr(cached, phase_name)
        canonical_phase = getattr(canonical, phase_name)
        for field in ("total_raw", "total_nonnegative", "vapor", "liquid", "saturation"):
            if not _arrays_numerically_equal(
                getattr(cached_phase, field), getattr(canonical_phase, field)
            ):
                raise RoutingIntegrityError(
                    f"{label} evaluator 缓存与共享核心重算不一致: {phase_name}.{field}"
                )
    if not _metric_cache_equal(
        cached.process_dew_point_metric, canonical.process_dew_point_metric
    ):
        raise RoutingIntegrityError(
            f"{label} evaluator 缓存与共享核心重算不一致: process_dew_point_metric"
        )
    for audit_name in (
        "process_dew_point_prediction_reporting_audit",
        "process_dew_point_target_reporting_audit",
    ):
        cached_audit = getattr(cached, audit_name)
        canonical_audit = getattr(canonical, audit_name)
        if cached_audit != canonical_audit:
            raise RoutingIntegrityError(
                f"{label} evaluator 缓存与共享核心重算不一致: {audit_name}"
            )


def _validate_evaluation_result(
    result: Any,
    *,
    candidate_name: str,
    fold_name: str,
    fold: FoldData,
    pressure_Pa: float,
) -> EvaluationResult:
    label = f"{candidate_name}/{fold_name}"
    if not isinstance(result, EvaluationResult):
        raise RoutingIntegrityError(f"{label} 必须来自共享 evaluator.EvaluationResult")
    ids = _unique_integer_ids(result.sample_ids, f"{label} OOF")
    if set(ids) != set(fold.validation_ids) or len(ids) != len(fold.validation_ids):
        raise RoutingIntegrityError(f"{label} OOF ID 与锁定 validation fold 不一致")
    expected_shape = (len(ids), len(MODEL_OUTPUT_COLUMNS))
    for name in (
        "predictions_raw",
        "targets_raw",
        "predictions_deployed",
        "targets_deployed",
    ):
        values = np.asarray(getattr(result, name), dtype=np.float64)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise RoutingIntegrityError(f"{label} {name} 形状错误或存在非有限值")
    authoritative_targets = _validation_targets_for_ids(fold, ids)
    if not np.allclose(
        np.asarray(result.targets_raw),
        authoritative_targets,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RoutingIntegrityError(f"{label} validation 标签与 FoldData 不一致")
    if not np.array_equal(result.predictions_deployed, np.maximum(result.predictions_raw, 0.0)):
        raise RoutingIntegrityError(f"{label} deployed 预测未遵循 evaluator 非负投影")
    if not np.array_equal(result.targets_deployed, np.maximum(result.targets_raw, 0.0)):
        raise RoutingIntegrityError(f"{label} deployed 标签未遵循 evaluator 非负投影")

    expected_outputs = set(MODEL_OUTPUT_COLUMNS)
    for metric_name in ("raw_metrics", "deployed_metrics", "projection_stats"):
        metric_map = getattr(result, metric_name)
        if not isinstance(metric_map, Mapping) or set(metric_map) != expected_outputs:
            raise RoutingIntegrityError(f"{label} {metric_name} 输出集合不符")
    for output in MODEL_OUTPUT_COLUMNS:
        _validate_metric(result.raw_metrics[output], f"{label}/raw/{output}")
        _validate_metric(result.deployed_metrics[output], f"{label}/deployed/{output}")
        _validate_projection(result.projection_stats[output], f"{label}/{output}")
    _validate_phase_metric(result.regen_phase_metric, f"{label}/regen_phase")
    _validate_metric(result.process_dew_point_metric, f"{label}/process_dew_point")
    for phase_name in ("regen_prediction_phase", "regen_target_phase"):
        phase = getattr(result, phase_name)
        for field in ("total_raw", "total_nonnegative", "vapor", "liquid"):
            values = np.asarray(getattr(phase, field), dtype=np.float64)
            if values.shape != (len(ids),) or not np.isfinite(values).all():
                raise RoutingIntegrityError(f"{label} {phase_name}.{field} 无效")
        if np.any(np.asarray(phase.liquid) < 0.0):
            raise RoutingIntegrityError(f"{label} {phase_name}.liquid 为负")
    canonical = evaluate_predictions(
        np.asarray(result.predictions_raw, dtype=np.float64),
        authoritative_targets,
        np.asarray(fold.train_targets_physical, dtype=np.float64),
        sample_ids=ids,
        pressure_Pa=pressure_Pa,
        mode="routing_canonical",
    )
    _assert_evaluation_cache_matches(result, canonical, label=label)
    return canonical


def _metric_payload(metric: MetricStats) -> dict[str, float]:
    return {
        "mae": float(metric.mae),
        "rmse": float(metric.rmse),
        "nrmse": float(metric.nrmse),
        "r2": float(metric.r2),
    }


def _phase_metric_payload(metric: PhaseMetricStats) -> dict[str, float]:
    return {"rmse": float(metric.rmse), "nrmse": float(metric.nrmse)}


def _projection_payload(metric: ProjectionStats) -> dict[str, Any]:
    return {
        "count": int(metric.count),
        "fraction": float(metric.fraction),
        "maximum_correction": float(metric.maximum_correction),
    }


def _condensation_payload(values: np.ndarray) -> dict[str, Any]:
    liquid = np.asarray(values, dtype=np.float64)
    if liquid.ndim != 1 or not np.isfinite(liquid).all() or np.any(liquid < 0.0):
        raise RoutingIntegrityError("凝结水统计输入无效")
    count = int(np.count_nonzero(liquid))
    return {
        "count": count,
        "fraction": float(count / len(liquid)),
        "total_g_per_kg_da_samples": float(np.sum(liquid)),
        "mean_g_per_kg_da": float(np.mean(liquid)),
        "maximum_g_per_kg_da": float(np.max(liquid, initial=0.0)),
    }


def _evaluation_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "raw_metrics": {
            output: _metric_payload(result.raw_metrics[output])
            for output in MODEL_OUTPUT_COLUMNS
        },
        "deployed_metrics": {
            output: _metric_payload(result.deployed_metrics[output])
            for output in MODEL_OUTPUT_COLUMNS
        },
        "regen_phase_metric": _phase_metric_payload(result.regen_phase_metric),
        "projection_stats": {
            output: _projection_payload(result.projection_stats[output])
            for output in MODEL_OUTPUT_COLUMNS
        },
        "condensation_stats": {
            "prediction": _condensation_payload(result.regen_prediction_phase.liquid),
            "target": _condensation_payload(result.regen_target_phase.liquid),
        },
        "process_dew_point_metric": _metric_payload(result.process_dew_point_metric),
    }


def _regen_hybrid_payload(
    result: EvaluationResult,
    *,
    temperature_candidate: str,
) -> dict[str, Any]:
    return {
        "temperature_candidate": temperature_candidate,
        "phase_metric": _phase_metric_payload(result.regen_phase_metric),
        "condensation_stats": {
            "prediction": _condensation_payload(result.regen_prediction_phase.liquid),
            "target": _condensation_payload(result.regen_target_phase.liquid),
        },
    }


def _aggregate_candidate_scores(
    fold_values_by_candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    if set(fold_values_by_candidate) != set(CANDIDATES):
        raise RoutingIntegrityError("评分候选集合不符")
    aggregated: dict[str, dict[str, Any]] = {}
    for candidate_name in CANDIDATES:
        fold_values = fold_values_by_candidate[candidate_name]
        if set(fold_values) != set(FIXED_FOLD_GROUPS):
            raise RoutingIntegrityError(f"{candidate_name} 评分 fold 集合不符")
        ordered = np.asarray(
            [
                _finite_float(fold_values[fold], f"{candidate_name}/{fold}/nrmse")
                for fold in FIXED_FOLD_GROUPS
            ],
            dtype=np.float64,
        )
        mean = float(np.mean(ordered))
        std = float(np.std(ordered, ddof=0))
        primary = float(mean + 0.25 * std)
        if not all(math.isfinite(value) for value in (mean, std, primary)):
            raise RoutingIntegrityError(f"{candidate_name} 聚合评分存在非有限值")
        aggregated[candidate_name] = {
            "fold_nrmse": {
                fold: float(fold_values[fold]) for fold in FIXED_FOLD_GROUPS
            },
            "mean_nrmse": mean,
            "std_nrmse": std,
            "primary_score": primary,
            "parameter_count": CANDIDATE_PARAMETER_COUNTS[candidate_name],
        }
    return aggregated


def _within_strict_tie(score: float, best: float) -> bool:
    return score - best < _TIE_THRESHOLD


def _select_candidate(candidate_scores: Mapping[str, Mapping[str, Any]]) -> str:
    best_primary = min(
        _finite_float(candidate_scores[name]["primary_score"], f"{name}.primary_score")
        for name in CANDIDATES
    )
    tied = [
        name
        for name in CANDIDATES
        if _within_strict_tie(
            float(candidate_scores[name]["primary_score"]), best_primary
        )
    ]
    return min(
        tied,
        key=lambda name: (
            float(candidate_scores[name]["mean_nrmse"]),
            float(candidate_scores[name]["std_nrmse"]),
            int(candidate_scores[name]["parameter_count"]),
            name,
        ),
    )


def _route_payload(
    *,
    metric: str,
    temperature_candidate: str | None,
    fold_values_by_candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    scores = _aggregate_candidate_scores(fold_values_by_candidate)
    selected = _select_candidate(scores)
    return {
        "selected_candidate": selected,
        "metric": metric,
        "temperature_candidate": temperature_candidate,
        "primary_score": float(scores[selected]["primary_score"]),
        "mean_nrmse": float(scores[selected]["mean_nrmse"]),
        "std_nrmse": float(scores[selected]["std_nrmse"]),
        "parameter_count": int(scores[selected]["parameter_count"]),
        "candidates": scores,
    }


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RoutingIntegrityError(f"routing JSON 含不可序列化或非有限值: {exc}") from exc


def _artifact_hash(payload: Mapping[str, Any]) -> str:
    without_hash = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    return hashlib.sha256(_canonical_bytes(without_hash)).hexdigest()


def _array_content_sha256(values: Any, label: str) -> str:
    """Hash dtype、shape 与已规范数组的 C-order bytes。"""

    array = np.asarray(values)
    if array.dtype.kind not in {"i", "u", "f"}:
        raise RoutingIntegrityError(f"{label} dtype 不受支持: {array.dtype}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise RoutingIntegrityError(f"{label} 存在非有限值")
    contiguous = np.ascontiguousarray(array)
    header = _canonical_bytes(
        {
            "format": _OOF_CONTENT_HASH_FORMAT,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _oof_content_hash_payload(result: EvaluationResult, label: str) -> dict[str, str]:
    normalized_ids = np.asarray(
        _unique_integer_ids(result.sample_ids, f"{label}.sample_ids"),
        dtype=np.int64,
    )
    order = np.argsort(normalized_ids, kind="stable")

    def sorted_matrix(values: Any, field: str) -> np.ndarray:
        matrix = np.asarray(values)
        expected_shape = (len(normalized_ids), len(MODEL_OUTPUT_COLUMNS))
        if matrix.shape != expected_shape:
            raise RoutingIntegrityError(
                f"{label}.{field} 形状必须为 {expected_shape}"
            )
        return matrix[order]

    return {
        "sample_ids_sha256": _ids_sha256(normalized_ids[order]),
        "predictions_raw_sha256": _array_content_sha256(
            sorted_matrix(result.predictions_raw, "predictions_raw"),
            f"{label}.predictions_raw",
        ),
        "predictions_deployed_sha256": _array_content_sha256(
            sorted_matrix(result.predictions_deployed, "predictions_deployed"),
            f"{label}.predictions_deployed",
        ),
        "targets_raw_sha256": _array_content_sha256(
            sorted_matrix(result.targets_raw, "targets_raw"),
            f"{label}.targets_raw",
        ),
    }


def build_oof_routing(
    evaluations: Mapping[str, Mapping[str, EvaluationResult]],
    *,
    fold_data_by_fold: Mapping[str, FoldData],
    pressure_Pa: float,
) -> dict[str, Any]:
    """锁定五候选、五折、七输出的 development-only 路由。"""

    pressure = _finite_float(pressure_Pa, "pressure_Pa")
    if pressure <= 0.0:
        raise RoutingIntegrityError("pressure_Pa 必须大于零")
    if not isinstance(evaluations, Mapping) or set(evaluations) != set(CANDIDATES):
        actual = sorted(evaluations) if isinstance(evaluations, Mapping) else []
        raise RoutingIntegrityError(
            f"路由候选必须精确等于固定五候选，实际 {actual}"
        )
    (
        development_ids,
        validation_by_fold,
        outer_ids,
        authoritative_split,
    ) = _validate_fold_data(fold_data_by_fold)

    candidate_fold_metrics: dict[str, dict[str, Any]] = {}
    oof_content_hashes: dict[str, dict[str, dict[str, str]]] = {}
    validated: dict[str, dict[str, EvaluationResult]] = {}
    for candidate_name in CANDIDATES:
        fold_results = evaluations[candidate_name]
        if not isinstance(fold_results, Mapping) or set(fold_results) != set(FIXED_FOLD_GROUPS):
            actual = sorted(fold_results) if isinstance(fold_results, Mapping) else []
            raise RoutingIntegrityError(
                f"{candidate_name} fold 集合必须精确为五折，实际 {actual}"
            )
        validated[candidate_name] = {}
        candidate_fold_metrics[candidate_name] = {}
        oof_content_hashes[candidate_name] = {}
        fold_frames: dict[str, pd.DataFrame] = {}
        for fold_name in FIXED_FOLD_GROUPS:
            original_result = fold_results[fold_name]
            result = _validate_evaluation_result(
                original_result,
                candidate_name=candidate_name,
                fold_name=fold_name,
                fold=fold_data_by_fold[fold_name],
                pressure_Pa=pressure,
            )
            validated[candidate_name][fold_name] = result
            candidate_fold_metrics[candidate_name][fold_name] = _evaluation_payload(result)
            oof_content_hashes[candidate_name][fold_name] = (
                _oof_content_hash_payload(
                    original_result, f"{candidate_name}/{fold_name} OOF"
                )
            )
            fold_frames[fold_name] = pd.DataFrame(
                np.column_stack([result.sample_ids, result.predictions_deployed]),
                columns=("ID", *MODEL_OUTPUT_COLUMNS),
            )
        try:
            assemble_oof_predictions(
                fold_frames,
                expected_development_ids=development_ids,
                expected_validation_ids_by_fold=validation_by_fold,
                outer_ids=outer_ids,
            )
        except OOFIntegrityError as exc:
            raise RoutingIntegrityError(f"{candidate_name} OOF 完整性失败: {exc}") from exc

    routes: dict[str, dict[str, Any]] = {}
    for output in _DIRECT_SELECTION_ORDER:
        fold_values = {
            candidate_name: {
                fold_name: float(
                    validated[candidate_name][fold_name].deployed_metrics[output].nrmse
                )
                for fold_name in FIXED_FOLD_GROUPS
            }
            for candidate_name in CANDIDATES
        }
        routes[output] = _route_payload(
            metric="deployed_nrmse",
            temperature_candidate=None,
            fold_values_by_candidate=fold_values,
        )

    regen_hybrid_fold_audit: dict[str, dict[str, dict[str, Any]]] = {
        candidate_name: {} for candidate_name in CANDIDATES
    }
    for humidity_output in _HUMIDITY_SELECTION_ORDER:
        temperature_output = _HUMIDITY_TEMPERATURE[humidity_output]
        locked_temperature_candidate = routes[temperature_output]["selected_candidate"]
        fold_values: dict[str, dict[str, float]] = {}
        for candidate_name in CANDIDATES:
            fold_values[candidate_name] = {}
            for fold_name in FIXED_FOLD_GROUPS:
                humidity_result = validated[candidate_name][fold_name]
                if humidity_output == _REGEN_W:
                    temperature_result = validated[locked_temperature_candidate][fold_name]
                    hybrid = humidity_result.predictions_raw.copy()
                    temperature_index = MODEL_OUTPUT_COLUMNS.index(temperature_output)
                    hybrid[:, temperature_index] = _prediction_column_for_ids(
                        temperature_result,
                        humidity_result.sample_ids,
                        temperature_index,
                    )
                    rescored = evaluate_predictions(
                        hybrid,
                        _validation_targets_for_ids(
                            fold_data_by_fold[fold_name],
                            humidity_result.sample_ids,
                        ),
                        fold_data_by_fold[fold_name].train_targets_physical,
                        sample_ids=humidity_result.sample_ids,
                        pressure_Pa=pressure,
                        mode="routing_hybrid",
                    )
                    regen_hybrid_fold_audit[candidate_name][fold_name] = (
                        _regen_hybrid_payload(
                            rescored,
                            temperature_candidate=locked_temperature_candidate,
                        )
                    )
                    value = rescored.regen_phase_metric.nrmse
                else:
                    # 当前共享 evaluator 对处理/冷吹湿量的部署 NRMSE 不依赖温度；
                    # 仍显式绑定已锁定的对应温度候选，禁止携带候选自身温度。
                    value = humidity_result.deployed_metrics[humidity_output].nrmse
                fold_values[candidate_name][fold_name] = float(value)
        metric = "regen_phase_nrmse" if humidity_output == _REGEN_W else "deployed_nrmse"
        routes[humidity_output] = _route_payload(
            metric=metric,
            temperature_candidate=locked_temperature_candidate,
            fold_values_by_candidate=fold_values,
        )

    ordered_routes = {output: routes[output] for output in MODEL_OUTPUT_COLUMNS}
    artifact: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "routing_method": _ROUTING_METHOD,
        "fixed_contract": {
            "candidates": list(CANDIDATES),
            "outputs": list(MODEL_OUTPUT_COLUMNS),
            "folds": list(FIXED_FOLD_GROUPS),
            "candidate_parameter_counts": dict(CANDIDATE_PARAMETER_COUNTS),
            "expected_development_rows": _EXPECTED_DEVELOPMENT_ROWS,
            "expected_outer_rows": _EXPECTED_OUTER_ROWS,
            "primary_score_formula": "mean_5fold_deployed_nrmse + 0.25 * population_std_5fold_deployed_nrmse",
            "tie_threshold_strict_less_than": _TIE_THRESHOLD,
            "tie_break_order": ["mean_nrmse", "std_nrmse", "parameter_count", "candidate_name"],
            "selection_order": list(_SELECTION_ORDER),
        },
        "oof_integrity": {
            "id_hash_format": _HASH_FORMAT,
            "content_hash_format": _OOF_CONTENT_HASH_FORMAT,
            "authoritative_split": authoritative_split,
            "development_row_count": len(development_ids),
            "development_ids_sha256": _ids_sha256(development_ids),
            "fold_validation_ids_sha256": {
                fold: _ids_sha256(validation_by_fold[fold]) for fold in FIXED_FOLD_GROUPS
            },
            "outer_row_count": len(outer_ids),
            "outer_ids_sha256": _ids_sha256(outer_ids),
            "outer_oof_id_count": 0,
        },
        "oof_content_hashes": oof_content_hashes,
        "routes": ordered_routes,
        "candidate_fold_metrics": candidate_fold_metrics,
        "regen_hybrid_fold_audit": regen_hybrid_fold_audit,
    }
    artifact["integrity_sha256"] = _artifact_hash(artifact)
    validate_routing_artifact(artifact)
    return artifact


_METRIC_KEYS = {"mae", "rmse", "nrmse", "r2"}
_PHASE_METRIC_KEYS = {"rmse", "nrmse"}
_PROJECTION_KEYS = {"count", "fraction", "maximum_correction"}
_CONDENSATION_KEYS = {
    "count",
    "fraction",
    "total_g_per_kg_da_samples",
    "mean_g_per_kg_da",
    "maximum_g_per_kg_da",
}
_OOF_CONTENT_HASH_KEYS = {
    "sample_ids_sha256",
    "predictions_raw_sha256",
    "predictions_deployed_sha256",
    "targets_raw_sha256",
}


def _validate_serialized_metric(value: Any, label: str) -> None:
    metric = _exact_keys(value, _METRIC_KEYS, label)
    for key in _METRIC_KEYS:
        _finite_float(metric[key], f"{label}.{key}")


def _validate_serialized_condensation_stats(value: Any, label: str) -> None:
    condensation = _exact_keys(value, {"prediction", "target"}, label)
    for side in ("prediction", "target"):
        stats = _exact_keys(
            condensation[side], _CONDENSATION_KEYS, f"{label}.{side}"
        )
        if (
            not isinstance(stats["count"], int)
            or isinstance(stats["count"], bool)
            or stats["count"] < 0
        ):
            raise RoutingIntegrityError(f"{label}.{side}.count 无效")
        for key in _CONDENSATION_KEYS - {"count"}:
            number = _finite_float(stats[key], f"{label}.{side}.{key}")
            if number < 0.0:
                raise RoutingIntegrityError(f"{label}.{side}.{key} 为负")
        if float(stats["fraction"]) > 1.0:
            raise RoutingIntegrityError(f"{label}.{side}.fraction 越界")


def _validate_serialized_fold_metrics(value: Any, label: str) -> None:
    fold = _exact_keys(
        value,
        {
            "raw_metrics",
            "deployed_metrics",
            "regen_phase_metric",
            "projection_stats",
            "condensation_stats",
            "process_dew_point_metric",
        },
        label,
    )
    for section_name in ("raw_metrics", "deployed_metrics"):
        section = _exact_keys(fold[section_name], set(MODEL_OUTPUT_COLUMNS), f"{label}.{section_name}")
        for output in MODEL_OUTPUT_COLUMNS:
            _validate_serialized_metric(section[output], f"{label}.{section_name}.{output}")
    phase = _exact_keys(fold["regen_phase_metric"], _PHASE_METRIC_KEYS, f"{label}.regen_phase_metric")
    for key in _PHASE_METRIC_KEYS:
        _finite_float(phase[key], f"{label}.regen_phase_metric.{key}")
    projections = _exact_keys(
        fold["projection_stats"], set(MODEL_OUTPUT_COLUMNS), f"{label}.projection_stats"
    )
    for output in MODEL_OUTPUT_COLUMNS:
        projection = _exact_keys(projections[output], _PROJECTION_KEYS, f"{label}.projection_stats.{output}")
        count = projection["count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RoutingIntegrityError(f"{label}.{output}.projection.count 无效")
        fraction = _finite_float(projection["fraction"], f"{label}.{output}.projection.fraction")
        maximum = _finite_float(projection["maximum_correction"], f"{label}.{output}.projection.maximum")
        if not 0.0 <= fraction <= 1.0 or maximum < 0.0:
            raise RoutingIntegrityError(f"{label}.{output}.projection 数值越界")
    _validate_serialized_condensation_stats(
        fold["condensation_stats"], f"{label}.condensation_stats"
    )
    _validate_serialized_metric(fold["process_dew_point_metric"], f"{label}.process_dew_point_metric")


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RoutingIntegrityError(f"{label} 必须是 SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RoutingIntegrityError(f"{label} 必须是 SHA-256") from exc
    return value


def _same_float(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15)


def validate_routing_artifact(artifact: Any) -> dict[str, Any]:
    """严格验证 schema、固定集合、路由重算结果和内容哈希。"""

    root = _exact_keys(
        artifact,
        {
            "schema_version",
            "routing_method",
            "fixed_contract",
            "oof_integrity",
            "oof_content_hashes",
            "routes",
            "candidate_fold_metrics",
            "regen_hybrid_fold_audit",
            "integrity_sha256",
        },
        "routing root",
    )
    if root["schema_version"] != _SCHEMA_VERSION or root["routing_method"] != _ROUTING_METHOD:
        raise RoutingIntegrityError("routing schema_version/method 不受支持")
    fixed = _exact_keys(
        root["fixed_contract"],
        {
            "candidates",
            "outputs",
            "folds",
            "candidate_parameter_counts",
            "expected_development_rows",
            "expected_outer_rows",
            "primary_score_formula",
            "tie_threshold_strict_less_than",
            "tie_break_order",
            "selection_order",
        },
        "fixed_contract",
    )
    expected_fixed = {
        "candidates": list(CANDIDATES),
        "outputs": list(MODEL_OUTPUT_COLUMNS),
        "folds": list(FIXED_FOLD_GROUPS),
        "candidate_parameter_counts": dict(CANDIDATE_PARAMETER_COUNTS),
        "expected_development_rows": _EXPECTED_DEVELOPMENT_ROWS,
        "expected_outer_rows": _EXPECTED_OUTER_ROWS,
        "primary_score_formula": "mean_5fold_deployed_nrmse + 0.25 * population_std_5fold_deployed_nrmse",
        "tie_threshold_strict_less_than": _TIE_THRESHOLD,
        "tie_break_order": ["mean_nrmse", "std_nrmse", "parameter_count", "candidate_name"],
        "selection_order": list(_SELECTION_ORDER),
    }
    if dict(fixed) != expected_fixed:
        raise RoutingIntegrityError("fixed_contract 与固定候选/输出/评分契约不一致")
    oof = _exact_keys(
        root["oof_integrity"],
        {
            "id_hash_format",
            "content_hash_format",
            "authoritative_split",
            "development_row_count",
            "development_ids_sha256",
            "fold_validation_ids_sha256",
            "outer_row_count",
            "outer_ids_sha256",
            "outer_oof_id_count",
        },
        "oof_integrity",
    )
    if (
        oof["id_hash_format"] != _HASH_FORMAT
        or oof["content_hash_format"] != _OOF_CONTENT_HASH_FORMAT
        or oof["development_row_count"] != _EXPECTED_DEVELOPMENT_ROWS
        or oof["outer_row_count"] != _EXPECTED_OUTER_ROWS
        or oof["outer_oof_id_count"] != 0
    ):
        raise RoutingIntegrityError("OOF 完整性计数不符")
    (
        current_development_ids,
        current_validation_by_fold,
        current_outer_ids,
        current_authoritative_split,
        _,
    ) = _authoritative_split_contract()
    authoritative_split = _exact_keys(
        oof["authoritative_split"],
        {
            "source_sha256",
            "snapshot_id",
            "split_assignments_sha256",
            "split_assignments_size_bytes",
            "split_assignments_row_count",
        },
        "oof_integrity.authoritative_split",
    )
    if dict(authoritative_split) != current_authoritative_split:
        raise RoutingIntegrityError("routing 绑定的权威划分 snapshot 与当前提交不一致")
    development_hash = _validate_hash(
        oof["development_ids_sha256"], "development_ids_sha256"
    )
    outer_hash = _validate_hash(oof["outer_ids_sha256"], "outer_ids_sha256")
    if development_hash != _ids_sha256(current_development_ids):
        raise RoutingIntegrityError("development ID 哈希与当前权威划分不一致")
    if outer_hash != _ids_sha256(current_outer_ids):
        raise RoutingIntegrityError("outer ID 哈希与当前权威划分不一致")
    fold_hashes = _exact_keys(
        oof["fold_validation_ids_sha256"], set(FIXED_FOLD_GROUPS), "fold_validation_ids_sha256"
    )
    for fold in FIXED_FOLD_GROUPS:
        fold_hash = _validate_hash(fold_hashes[fold], f"{fold}.ids_sha256")
        if fold_hash != _ids_sha256(current_validation_by_fold[fold]):
            raise RoutingIntegrityError(
                f"{fold} validation ID 哈希与当前权威划分不一致"
            )

    content_hashes = _exact_keys(
        root["oof_content_hashes"], set(CANDIDATES), "oof_content_hashes"
    )
    for candidate_name in CANDIDATES:
        candidate_hashes = _exact_keys(
            content_hashes[candidate_name],
            set(FIXED_FOLD_GROUPS),
            f"oof_content_hashes.{candidate_name}",
        )
        for fold_name in FIXED_FOLD_GROUPS:
            fold_content_hashes = _exact_keys(
                candidate_hashes[fold_name],
                _OOF_CONTENT_HASH_KEYS,
                f"oof_content_hashes.{candidate_name}.{fold_name}",
            )
            for hash_name in _OOF_CONTENT_HASH_KEYS:
                _validate_hash(
                    fold_content_hashes[hash_name],
                    f"oof_content_hashes.{candidate_name}.{fold_name}.{hash_name}",
                )
            if (
                fold_content_hashes["sample_ids_sha256"]
                != fold_hashes[fold_name]
            ):
                raise RoutingIntegrityError(
                    f"oof_content_hashes.{candidate_name}.{fold_name}.sample_ids "
                    "与当前权威 fold ID 哈希不一致"
                )

    routes = _exact_keys(root["routes"], set(MODEL_OUTPUT_COLUMNS), "routes")
    for output in MODEL_OUTPUT_COLUMNS:
        route = _exact_keys(
            routes[output],
            {
                "selected_candidate",
                "metric",
                "temperature_candidate",
                "primary_score",
                "mean_nrmse",
                "std_nrmse",
                "parameter_count",
                "candidates",
            },
            f"routes.{output}",
        )
        candidates = _exact_keys(route["candidates"], set(CANDIDATES), f"routes.{output}.candidates")
        normalized_scores: dict[str, Mapping[str, Any]] = {}
        for candidate_name in CANDIDATES:
            score = _exact_keys(
                candidates[candidate_name],
                {"fold_nrmse", "mean_nrmse", "std_nrmse", "primary_score", "parameter_count"},
                f"routes.{output}.candidates.{candidate_name}",
            )
            folds = _exact_keys(score["fold_nrmse"], set(FIXED_FOLD_GROUPS), f"routes.{output}.{candidate_name}.fold_nrmse")
            values = np.asarray(
                [_finite_float(folds[fold], f"routes.{output}.{candidate_name}.{fold}") for fold in FIXED_FOLD_GROUPS]
            )
            expected_mean = float(np.mean(values))
            expected_std = float(np.std(values, ddof=0))
            expected_primary = expected_mean + 0.25 * expected_std
            if (
                not _same_float(_finite_float(score["mean_nrmse"], "mean_nrmse"), expected_mean)
                or not _same_float(_finite_float(score["std_nrmse"], "std_nrmse"), expected_std)
                or not _same_float(_finite_float(score["primary_score"], "primary_score"), expected_primary)
                or score["parameter_count"] != CANDIDATE_PARAMETER_COUNTS[candidate_name]
            ):
                raise RoutingIntegrityError(f"routes.{output}.{candidate_name} 聚合成绩被篡改")
            normalized_scores[candidate_name] = score
        selected = _select_candidate(normalized_scores)
        if route["selected_candidate"] != selected:
            raise RoutingIntegrityError(f"routes.{output} selected_candidate 与评分不一致")
        selected_score = normalized_scores[selected]
        for route_key, score_key in (
            ("primary_score", "primary_score"),
            ("mean_nrmse", "mean_nrmse"),
            ("std_nrmse", "std_nrmse"),
        ):
            if not _same_float(
                _finite_float(route[route_key], f"routes.{output}.{route_key}"),
                float(selected_score[score_key]),
            ):
                raise RoutingIntegrityError(f"routes.{output}.{route_key} 与选中候选不一致")
        if route["parameter_count"] != CANDIDATE_PARAMETER_COUNTS[selected]:
            raise RoutingIntegrityError(f"routes.{output}.parameter_count 不一致")
        if output in _DIRECT_SELECTION_ORDER:
            if route["metric"] != "deployed_nrmse" or route["temperature_candidate"] is not None:
                raise RoutingIntegrityError(f"routes.{output} 直接路由语义不符")
        else:
            expected_temperature = routes[_HUMIDITY_TEMPERATURE[output]]["selected_candidate"]
            expected_metric = "regen_phase_nrmse" if output == _REGEN_W else "deployed_nrmse"
            if route["temperature_candidate"] != expected_temperature or route["metric"] != expected_metric:
                raise RoutingIntegrityError(f"routes.{output} 未绑定已锁定对应温度")

    metrics = _exact_keys(root["candidate_fold_metrics"], set(CANDIDATES), "candidate_fold_metrics")
    for candidate_name in CANDIDATES:
        candidate_metrics = _exact_keys(
            metrics[candidate_name], set(FIXED_FOLD_GROUPS), f"candidate_fold_metrics.{candidate_name}"
        )
        for fold_name in FIXED_FOLD_GROUPS:
            _validate_serialized_fold_metrics(
                candidate_metrics[fold_name], f"candidate_fold_metrics.{candidate_name}.{fold_name}"
            )

    regen_hybrid = _exact_keys(
        root["regen_hybrid_fold_audit"],
        set(CANDIDATES),
        "regen_hybrid_fold_audit",
    )
    locked_regen_temperature = routes[_REGEN_T]["selected_candidate"]
    for candidate_name in CANDIDATES:
        candidate_hybrid = _exact_keys(
            regen_hybrid[candidate_name],
            set(FIXED_FOLD_GROUPS),
            f"regen_hybrid_fold_audit.{candidate_name}",
        )
        for fold_name in FIXED_FOLD_GROUPS:
            label = f"regen_hybrid_fold_audit.{candidate_name}.{fold_name}"
            hybrid = _exact_keys(
                candidate_hybrid[fold_name],
                {"temperature_candidate", "phase_metric", "condensation_stats"},
                label,
            )
            if hybrid["temperature_candidate"] != locked_regen_temperature:
                raise RoutingIntegrityError(
                    f"{label} hybrid 温度候选与已锁定 route 不一致"
                )
            phase = _exact_keys(
                hybrid["phase_metric"], _PHASE_METRIC_KEYS, f"{label}.phase_metric"
            )
            for key in _PHASE_METRIC_KEYS:
                _finite_float(phase[key], f"{label}.phase_metric.{key}")
            _validate_serialized_condensation_stats(
                hybrid["condensation_stats"], f"{label}.condensation_stats"
            )
            hybrid_target = hybrid["condensation_stats"]["target"]
            direct_target = metrics[candidate_name][fold_name][
                "condensation_stats"
            ]["target"]
            if dict(hybrid_target) != dict(direct_target):
                raise RoutingIntegrityError(
                    f"{label} hybrid 目标凝结水审计与 candidate metrics 不一致"
                )

    for output in MODEL_OUTPUT_COLUMNS:
        for candidate_name in CANDIDATES:
            for fold_name in FIXED_FOLD_GROUPS:
                route_nrmse = _finite_float(
                    routes[output]["candidates"][candidate_name]["fold_nrmse"][
                        fold_name
                    ],
                    f"routes.{output}.{candidate_name}.{fold_name}",
                )
                if output == _REGEN_W:
                    audit_nrmse = regen_hybrid[candidate_name][fold_name][
                        "phase_metric"
                    ]["nrmse"]
                else:
                    audit_nrmse = metrics[candidate_name][fold_name][
                        "deployed_metrics"
                    ][output]["nrmse"]
                if not _same_float(
                    route_nrmse,
                    _finite_float(
                        audit_nrmse,
                        f"{candidate_name}/{fold_name}/{output} 审计 NRMSE",
                    ),
                ):
                    raise RoutingIntegrityError(
                        f"routes.{output}.{candidate_name}.{fold_name} "
                        "评分与候选/hybrid 审计不一致"
                    )

    actual_hash = _validate_hash(root["integrity_sha256"], "integrity_sha256")
    expected_hash = _artifact_hash(root)
    if actual_hash != expected_hash:
        raise RoutingIntegrityError(
            f"routing JSON 完整性哈希不符: expected={expected_hash}, actual={actual_hash}"
        )
    return dict(root)


def save_routing_json(artifact: Mapping[str, Any], path: str | Path) -> None:
    """严格验证后原子写出唯一 JSON 路由格式。"""

    target = Path(path)
    if target.suffix.lower() != ".json":
        raise RoutingIntegrityError("routing 产物只允许 JSON 文件")
    validated = validate_routing_artifact(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        content = (
            json.dumps(validated, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, target)
    except RoutingIntegrityError:
        if temporary.exists():
            temporary.unlink()
        raise
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise RoutingIntegrityError(f"routing JSON 写入失败: {target}: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise RoutingIntegrityError(f"routing JSON 含非有限常量: {value}")


def load_routing_json(path: str | Path) -> dict[str, Any]:
    """只重载严格 JSON；不提供 pickle 或通用 artifact 解析入口。"""

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise RoutingIntegrityError("routing 重载只接受 JSON 文件")
    if not source.is_file():
        raise RoutingIntegrityError(f"routing JSON 不存在: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except RoutingIntegrityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoutingIntegrityError(f"routing JSON 无法解析: {source}: {exc}") from exc
    return validate_routing_artifact(payload)


def apply_oof_routing(
    artifact: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, EvaluationResult]],
) -> pd.DataFrame:
    """将已校验路由应用于同一开发集 OOF，保持 float64 逐位值。"""

    validated_artifact = validate_routing_artifact(artifact)
    if not isinstance(evaluations, Mapping) or set(evaluations) != set(CANDIDATES):
        raise RoutingIntegrityError("应用路由时候选集合不符")
    expected_fold_hashes = validated_artifact["oof_integrity"][
        "fold_validation_ids_sha256"
    ]
    expected_content_hashes = validated_artifact["oof_content_hashes"]
    frames: dict[str, pd.DataFrame] = {}
    for candidate_name in CANDIDATES:
        fold_results = evaluations[candidate_name]
        if not isinstance(fold_results, Mapping) or set(fold_results) != set(FIXED_FOLD_GROUPS):
            raise RoutingIntegrityError(f"应用路由时 {candidate_name} fold 集合不符")
        pieces: list[pd.DataFrame] = []
        for fold_name in FIXED_FOLD_GROUPS:
            result = fold_results[fold_name]
            if not isinstance(result, EvaluationResult):
                raise RoutingIntegrityError(f"应用路由时 {candidate_name}/{fold_name} 类型无效")
            actual_content_hashes = _oof_content_hash_payload(
                result, f"应用路由时 {candidate_name}/{fold_name} OOF"
            )
            if actual_content_hashes != expected_content_hashes[candidate_name][fold_name]:
                changed = sorted(
                    key
                    for key in _OOF_CONTENT_HASH_KEYS
                    if actual_content_hashes[key]
                    != expected_content_hashes[candidate_name][fold_name][key]
                )
                raise RoutingIntegrityError(
                    f"应用路由时 {candidate_name}/{fold_name} OOF 内容哈希不符: {changed}"
                )
            ids = _unique_integer_ids(result.sample_ids, f"{candidate_name}/{fold_name} OOF")
            if _ids_sha256(ids) != expected_fold_hashes[fold_name]:
                raise RoutingIntegrityError(f"应用路由时 {candidate_name}/{fold_name} ID 哈希不符")
            predictions = np.asarray(result.predictions_deployed, dtype=np.float64)
            if predictions.shape != (len(ids), len(MODEL_OUTPUT_COLUMNS)) or not np.isfinite(predictions).all():
                raise RoutingIntegrityError(f"应用路由时 {candidate_name}/{fold_name} 预测无效")
            piece = pd.DataFrame(predictions, columns=MODEL_OUTPUT_COLUMNS)
            piece.insert(0, "ID", np.asarray(ids, dtype=np.int64))
            pieces.append(piece)
        frame = pd.concat(pieces, ignore_index=True).sort_values("ID", kind="stable").reset_index(drop=True)
        if len(frame) != _EXPECTED_DEVELOPMENT_ROWS or frame["ID"].duplicated().any():
            raise RoutingIntegrityError(f"应用路由时 {candidate_name} OOF 覆盖不完整")
        if _ids_sha256(frame["ID"].to_numpy()) != validated_artifact["oof_integrity"]["development_ids_sha256"]:
            raise RoutingIntegrityError(f"应用路由时 {candidate_name} development ID 哈希不符")
        frames[candidate_name] = frame

    reference_ids = frames[next(iter(CANDIDATES))]["ID"].to_numpy(dtype=np.int64)
    routed = pd.DataFrame({"ID": reference_ids.copy()})
    for output in MODEL_OUTPUT_COLUMNS:
        selected = validated_artifact["routes"][output]["selected_candidate"]
        candidate_frame = frames[selected]
        if not np.array_equal(candidate_frame["ID"].to_numpy(dtype=np.int64), reference_ids):
            raise RoutingIntegrityError(f"应用路由时 {output} 候选 ID 顺序不一致")
        routed[output] = candidate_frame[output].to_numpy(dtype=np.float64, copy=True)
    return routed
