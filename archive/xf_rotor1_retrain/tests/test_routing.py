"""XF 开发集逐输出 OOF 路由的可执行规格。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archive.xf_rotor1_retrain.candidates import CANDIDATES
from archive.xf_rotor1_retrain.config import (
    FIXED_FOLD_GROUPS,
    MODEL_INPUT_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
    REPORTS_DIR,
)
from archive.xf_rotor1_retrain.data import (
    read_xf_source,
    validate_and_clean_xf,
    validate_report_snapshot,
)
from archive.xf_rotor1_retrain.evaluator import (
    AuditedMinMaxScaler,
    EvaluationResult,
    FoldData,
    MetricStats,
    PhaseMetricStats,
    PhaseSplit,
    ProjectionStats,
    build_fold_data,
    evaluate_predictions,
)
from archive.xf_rotor1_retrain.routing import (
    CANDIDATE_PARAMETER_COUNTS,
    RoutingIntegrityError,
    _within_strict_tie,
    apply_oof_routing,
    build_oof_routing,
    load_routing_json,
    save_routing_json,
    validate_routing_artifact,
)


_ROWS_PER_FOLD = 10_487
_DEVELOPMENT_ROWS = 52_435
_OUTER_ROWS = 4_750
_PRESSURE_PA = 101_325.0


def _physical_rows(ids: np.ndarray) -> np.ndarray:
    """生成全为非负且七列均有稳定跨度的手工物理标签。"""

    position = np.asarray(ids, dtype=np.float64)
    phase = (position % 101.0) / 100.0
    return np.column_stack(
        [
            20.0 + 5.0 * phase,
            2.0 + 2.0 * phase,
            25.0 + 4.0 * phase,
            3.0 + 1.5 * phase,
            20.0 + 3.0 * phase,
            12.0 + 6.0 * phase,
            0.5 + 1.0 * phase,
        ]
    )


def _synthetic_fold_contract() -> dict[str, FoldData]:
    development_ids = np.arange(1, _DEVELOPMENT_ROWS + 1, dtype=np.int64)
    outer_ids = tuple(range(100_000, 100_000 + _OUTER_ROWS))
    result: dict[str, FoldData] = {}
    for fold_index, fold_name in enumerate(FIXED_FOLD_GROUPS):
        start = fold_index * _ROWS_PER_FOLD
        stop = start + _ROWS_PER_FOLD
        validation_ids = development_ids[start:stop]
        train_ids = np.concatenate(
            [development_ids[:start], development_ids[stop:]]
        )
        train_targets = _physical_rows(train_ids)
        validation_targets = _physical_rows(validation_ids)
        result[fold_name] = FoldData(
            fold_name=fold_name,
            train_ids=tuple(int(value) for value in train_ids),
            validation_ids=tuple(int(value) for value in validation_ids),
            outer_ids=outer_ids,
            input_scaler=None,  # routing 只消费评估器已经锁定的 ID/物理标签。
            target_scaler=None,
            target_spans=np.ptp(train_targets, axis=0),
            train_features_scaled=np.empty((0, len(MODEL_INPUT_COLUMNS))),
            validation_features_scaled=np.empty((0, len(MODEL_INPUT_COLUMNS))),
            train_targets_scaled=np.empty((0, len(MODEL_OUTPUT_COLUMNS))),
            validation_targets_scaled=np.empty((0, len(MODEL_OUTPUT_COLUMNS))),
            train_targets_physical=train_targets,
            validation_targets_physical=validation_targets,
        )
    return result


@lru_cache(maxsize=1)
def _authoritative_fold_contract() -> tuple[tuple[str, FoldData], ...]:
    """通过公共数据/评估入口只读重建真实权威折。"""

    validate_report_snapshot(REPORTS_DIR)
    clean, _ = validate_and_clean_xf(read_xf_source())
    assignments = pd.read_csv(REPORTS_DIR / "split_assignments.csv")
    return tuple(
        (fold_name, build_fold_data(clean, assignments, fold_name))
        for fold_name in FIXED_FOLD_GROUPS
    )


def _fold_contract() -> dict[str, FoldData]:
    return dict(_authoritative_fold_contract())


def _evaluation_bundle(
    folds: dict[str, FoldData],
    *,
    error_scale_by_candidate: dict[str, float] | None = None,
) -> dict[str, dict[str, EvaluationResult]]:
    default_scales = {
        name: 0.05 - 0.01 * index
        for index, name in enumerate(CANDIDATES)
    }
    scales = default_scales if error_scale_by_candidate is None else error_scale_by_candidate
    bundle: dict[str, dict[str, EvaluationResult]] = {}
    for candidate_name in CANDIDATES:
        candidate_results: dict[str, EvaluationResult] = {}
        for fold_index, (fold_name, fold) in enumerate(folds.items(), start=1):
            targets = fold.validation_targets_physical
            column_scale = np.arange(1.0, 8.0, dtype=np.float64)
            predictions = targets + scales[candidate_name] * fold_index * column_scale
            candidate_results[fold_name] = evaluate_predictions(
                predictions,
                targets,
                fold.train_targets_physical,
                sample_ids=fold.validation_ids,
                pressure_Pa=_PRESSURE_PA,
                mode="in_memory",
            )
        bundle[candidate_name] = candidate_results
    return bundle


@pytest.fixture(scope="module")
def synthetic_oof() -> tuple[
    dict[str, FoldData], dict[str, dict[str, EvaluationResult]]
]:
    folds = _fold_contract()
    return folds, _evaluation_bundle(folds)


@pytest.fixture(scope="module")
def routing_artifact(synthetic_oof) -> dict:
    """完整 52,435 行基线只聚合一次；故障测试仍各自重建输入。"""

    folds, evaluations = synthetic_oof
    return build_oof_routing(
        evaluations,
        fold_data_by_fold=folds,
        pressure_Pa=_PRESSURE_PA,
    )


def _replace_deployed_nrmse(
    result: EvaluationResult,
    output: str,
    nrmse: float,
) -> EvaluationResult:
    metrics = dict(result.deployed_metrics)
    original = metrics[output]
    metrics[output] = MetricStats(
        mae=original.mae,
        rmse=original.rmse,
        nrmse=nrmse,
        r2=original.r2,
    )
    return replace(result, deployed_metrics=metrics)


def _set_fold_nrmse(
    bundle: dict[str, dict[str, EvaluationResult]],
    folds: dict[str, FoldData],
    candidate: str,
    output: str,
    values: list[float],
) -> None:
    output_index = MODEL_OUTPUT_COLUMNS.index(output)
    for fold_name, value in zip(FIXED_FOLD_GROUPS, values, strict=True):
        result = bundle[candidate][fold_name]
        fold = folds[fold_name]
        targets = np.asarray(result.targets_raw, dtype=np.float64)
        predictions = np.asarray(result.predictions_raw, dtype=np.float64).copy()
        train_span = float(
            np.ptp(fold.train_targets_physical[:, output_index])
        )
        predictions[:, output_index] = (
            targets[:, output_index] + float(value) * train_span
        )
        bundle[candidate][fold_name] = evaluate_predictions(
            predictions,
            targets,
            fold.train_targets_physical,
            sample_ids=result.sample_ids,
            pressure_Pa=_PRESSURE_PA,
            mode="routing_score_fixture",
        )


def _reorder_result(result: EvaluationResult, order: np.ndarray) -> EvaluationResult:
    """保持指标不变，只重排所有逐样本字段，模拟按 ID 写出的 OOF 文件。"""

    def reorder_phase(phase: PhaseSplit) -> PhaseSplit:
        return replace(
            phase,
            total_raw=phase.total_raw[order],
            total_nonnegative=phase.total_nonnegative[order],
            vapor=phase.vapor[order],
            liquid=phase.liquid[order],
            saturation=phase.saturation[order],
        )

    return replace(
        result,
        sample_ids=result.sample_ids[order],
        predictions_raw=result.predictions_raw[order],
        targets_raw=result.targets_raw[order],
        predictions_deployed=result.predictions_deployed[order],
        targets_deployed=result.targets_deployed[order],
        regen_prediction_phase=reorder_phase(result.regen_prediction_phase),
        regen_target_phase=reorder_phase(result.regen_target_phase),
        process_dew_point_prediction_thermodynamic_C=(
            result.process_dew_point_prediction_thermodynamic_C[order]
        ),
        process_dew_point_target_thermodynamic_C=(
            result.process_dew_point_target_thermodynamic_C[order]
        ),
        process_dew_point_prediction_reporting_C=(
            result.process_dew_point_prediction_reporting_C[order]
        ),
        process_dew_point_target_reporting_C=(
            result.process_dew_point_target_reporting_C[order]
        ),
    )


def _rehash_artifact(payload: dict) -> None:
    without_hash = {
        key: value for key, value in payload.items()
        if key != "integrity_sha256"
    }
    canonical = json.dumps(
        without_hash,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["integrity_sha256"] = hashlib.sha256(canonical).hexdigest()


def test_parameter_counts_are_fixed_from_the_exact_five_candidate_models() -> None:
    """捕获 tie-break 使用 hidden width、文件大小或可注入假参数量。"""

    assert CANDIDATE_PARAMETER_COUNTS == {
        "xf_v1_baseline": 2_448,
        "xf_v2_balanced": 4_080,
        "xf_v3_capacity": 5_984,
        "xf_v4_weighted": 4_080,
        "xf_v5_robust": 9_440,
    }


def test_builds_all_seven_routes_from_exact_development_oof_only(
    routing_artifact,
) -> None:
    """捕获少候选、少输出、漏行或 outer 样本进入路由成绩。"""

    artifact = routing_artifact

    assert artifact["fixed_contract"]["candidates"] == list(CANDIDATES)
    assert artifact["fixed_contract"]["outputs"] == list(MODEL_OUTPUT_COLUMNS)
    assert artifact["fixed_contract"]["folds"] == list(FIXED_FOLD_GROUPS)
    assert list(artifact["routes"]) == list(MODEL_OUTPUT_COLUMNS)
    assert artifact["oof_integrity"]["development_row_count"] == 52_435
    assert artifact["oof_integrity"]["outer_row_count"] == 4_750
    assert artifact["oof_integrity"]["outer_oof_id_count"] == 0
    assert set(artifact["candidate_fold_metrics"]) == set(CANDIDATES)
    for candidate_metrics in artifact["candidate_fold_metrics"].values():
        assert set(candidate_metrics) == set(FIXED_FOLD_GROUPS)
        for fold_metrics in candidate_metrics.values():
            assert set(fold_metrics["raw_metrics"]) == set(MODEL_OUTPUT_COLUMNS)
            assert set(fold_metrics["deployed_metrics"]) == set(MODEL_OUTPUT_COLUMNS)
            assert set(fold_metrics["projection_stats"]) == set(MODEL_OUTPUT_COLUMNS)
            assert set(fold_metrics["regen_phase_metric"]) == {"rmse", "nrmse"}
            assert set(fold_metrics["condensation_stats"]) == {
                "prediction",
                "target",
            }


def test_rejects_self_consistent_fold_data_not_bound_to_authoritative_snapshot(
) -> None:
    """自洽的 52,435 行 synthetic ID 不能替代已提交权威划分报告。"""

    folds = _synthetic_fold_contract()
    evaluations = _evaluation_bundle(folds)
    with pytest.raises(RoutingIntegrityError, match="权威|snapshot|划分"):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=folds,
            pressure_Pa=_PRESSURE_PA,
        )


def test_rejects_scaler_fit_ids_that_are_not_authoritative_fold_train_ids(
    synthetic_oof,
) -> None:
    """权威 ID 不足够；两个 scaler 也只能 fit 当折 development train。"""

    folds, evaluations = synthetic_oof
    bad_folds = dict(folds)
    fold = bad_folds["fold_1"]
    bad_scaler = copy.copy(fold.input_scaler)
    bad_scaler.fit_ids = (*fold.train_ids[1:], fold.validation_ids[0])
    bad_folds["fold_1"] = replace(fold, input_scaler=bad_scaler)

    with pytest.raises(RoutingIntegrityError, match="scaler|fit_ids|训练"):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=bad_folds,
            pressure_Pa=_PRESSURE_PA,
        )


def test_rejects_self_consistent_fold_numeric_values_not_rebuilt_from_xf(
    synthetic_oof,
) -> None:
    """ID/fit_ids 正确不能授权调用方替换折训练物理标签。"""

    folds, _ = synthetic_oof
    bad_folds = dict(folds)
    fold = bad_folds["fold_1"]
    train_targets = fold.train_targets_physical.copy()
    train_targets[:, 0] *= 1_000_000.0
    target_scaler = AuditedMinMaxScaler(MODEL_OUTPUT_COLUMNS).fit(
        train_targets, fold.train_ids
    )
    bad_folds["fold_1"] = replace(
        fold,
        target_scaler=target_scaler,
        target_spans=np.ptp(train_targets, axis=0),
        train_targets_physical=train_targets,
        train_targets_scaled=target_scaler.transform(train_targets),
        validation_targets_scaled=target_scaler.transform(
            fold.validation_targets_physical
        ),
    )
    evaluations = _evaluation_bundle(bad_folds)

    with pytest.raises(
        RoutingIntegrityError, match="权威|XF|physical|target|标签|数值"
    ):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=bad_folds,
            pressure_Pa=_PRESSURE_PA,
        )


@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_rejects_missing_or_extra_candidate(
    synthetic_oof, fault: str
) -> None:
    """固定五候选集合不允许被训练入口减少或扩充。"""

    folds, source = synthetic_oof
    evaluations = {name: dict(results) for name, results in source.items()}
    if fault == "missing":
        del evaluations["xf_v5_robust"]
    else:
        evaluations["xf_unplanned"] = dict(evaluations["xf_v1_baseline"])

    with pytest.raises(RoutingIntegrityError, match="候选"):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=folds,
            pressure_Pa=_PRESSURE_PA,
        )


@pytest.mark.parametrize("fault", ["missing", "extra", "wrong_membership"])
def test_rejects_missing_extra_or_wrong_fold(
    synthetic_oof, fault: str
) -> None:
    """全局 ID 数正确也不能掩盖候选预测落到错误 validation fold。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    candidate = evaluations["xf_v1_baseline"]
    if fault == "missing":
        del candidate["fold_5"]
    elif fault == "extra":
        candidate["fold_6"] = candidate["fold_5"]
    else:
        candidate["fold_1"], candidate["fold_2"] = (
            candidate["fold_2"],
            candidate["fold_1"],
        )

    with pytest.raises(RoutingIntegrityError, match="fold|OOF"):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=folds,
            pressure_Pa=_PRESSURE_PA,
        )


@pytest.mark.parametrize("fault", ["duplicate", "outer", "unknown"])
def test_rejects_duplicate_outer_or_unknown_oof_id(
    synthetic_oof, fault: str
) -> None:
    """任一非锁定 development ID 都必须在评分前失败。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    result = evaluations["xf_v1_baseline"]["fold_1"]
    ids = result.sample_ids.copy()
    if fault == "duplicate":
        ids[1] = ids[0]
    elif fault == "outer":
        ids[0] = folds["fold_1"].outer_ids[0]
    else:
        ids[0] = 9_999_999
    evaluations["xf_v1_baseline"]["fold_1"] = replace(result, sample_ids=ids)

    with pytest.raises(RoutingIntegrityError, match="OOF|ID|fold"):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=folds,
            pressure_Pa=_PRESSURE_PA,
        )


def test_rejects_fold_context_that_no_longer_covers_52435_ids(
    synthetic_oof,
) -> None:
    """捕获调用方把期望 development 集合本身缩短后使 OOF 假通过。"""

    folds, evaluations = synthetic_oof
    bad_folds = dict(folds)
    fold = bad_folds["fold_5"]
    bad_folds["fold_5"] = replace(
        fold,
        validation_ids=fold.validation_ids[:-1],
        validation_targets_physical=fold.validation_targets_physical[:-1],
    )

    with pytest.raises(
        RoutingIntegrityError, match="validation|52,435|development|snapshot"
    ):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=bad_folds,
            pressure_Pa=_PRESSURE_PA,
        )


def test_routes_by_id_when_one_candidate_fold_row_order_differs(synthetic_oof) -> None:
    """捕获相态组合按行号而不是 ID 对齐候选温度与湿量。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    result = evaluations["xf_v1_baseline"]["fold_1"]
    evaluations["xf_v1_baseline"]["fold_1"] = _reorder_result(
        result, np.arange(len(result.sample_ids) - 1, -1, -1)
    )

    artifact = build_oof_routing(
        evaluations,
        fold_data_by_fold=folds,
        pressure_Pa=_PRESSURE_PA,
    )
    routed = apply_oof_routing(artifact, evaluations)
    assert routed["ID"].is_monotonic_increasing
    assert len(routed) == 52_435


@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_rejects_missing_or_extra_output_metric(
    synthetic_oof, fault: str
) -> None:
    """评估结果的七输出集合必须逐折精确一致。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    result = evaluations["xf_v1_baseline"]["fold_1"]
    deployed = dict(result.deployed_metrics)
    if fault == "missing":
        del deployed[MODEL_OUTPUT_COLUMNS[-1]]
    else:
        deployed["Label_Unplanned"] = next(iter(deployed.values()))
    evaluations["xf_v1_baseline"]["fold_1"] = replace(
        result, deployed_metrics=deployed
    )

    with pytest.raises(RoutingIntegrityError, match="输出|metric"):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=folds,
            pressure_Pa=_PRESSURE_PA,
        )


def test_rejects_nonfinite_metric_before_routing(synthetic_oof) -> None:
    """NaN 成绩不得借 Python 排序行为获得或避开路由。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    result = evaluations["xf_v1_baseline"]["fold_1"]
    evaluations["xf_v1_baseline"]["fold_1"] = _replace_deployed_nrmse(
        result, MODEL_OUTPUT_COLUMNS[0], math.nan
    )

    with pytest.raises(RoutingIntegrityError, match="非有限"):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=folds,
            pressure_Pa=_PRESSURE_PA,
        )


@pytest.mark.parametrize(
    "fault",
    ["deployed", "raw", "phase", "projection", "dewpoint", "phase_arrays"],
)
def test_rejects_forged_evaluator_cache_with_unchanged_predictions_and_targets(
    synthetic_oof, fault: str
) -> None:
    """任何 evaluator 缓存都必须与共享核心从原始预测重算的结果一致。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    candidate = "xf_v1_baseline"
    fold_name = "fold_1"
    result = evaluations[candidate][fold_name]
    output = MODEL_OUTPUT_COLUMNS[0]
    if fault in {"deployed", "raw"}:
        attribute = f"{fault}_metrics"
        metrics = dict(getattr(result, attribute))
        cached = metrics[output]
        metrics[output] = replace(cached, nrmse=-100.0)
        result = replace(result, **{attribute: metrics})
    elif fault == "phase":
        result = replace(
            result,
            regen_phase_metric=PhaseMetricStats(
                rmse=result.regen_phase_metric.rmse,
                nrmse=result.regen_phase_metric.nrmse + 1.0,
            ),
        )
    elif fault == "projection":
        projections = dict(result.projection_stats)
        cached = projections[output]
        projections[output] = ProjectionStats(
            count=cached.count + 1,
            fraction=cached.fraction,
            maximum_correction=cached.maximum_correction,
        )
        result = replace(result, projection_stats=projections)
    elif fault == "dewpoint":
        result = replace(
            result,
            process_dew_point_metric=replace(
                result.process_dew_point_metric,
                nrmse=result.process_dew_point_metric.nrmse + 1.0,
            ),
        )
    else:
        phase = result.regen_prediction_phase
        result = replace(
            result,
            regen_prediction_phase=replace(
                phase,
                liquid=phase.liquid + 1.0,
            ),
        )
    evaluations[candidate][fold_name] = result

    with pytest.raises(RoutingIntegrityError, match="缓存|重算|canonical"):
        build_oof_routing(
            evaluations,
            fold_data_by_fold=folds,
            pressure_Pa=_PRESSURE_PA,
        )


def test_primary_score_uses_population_mean_plus_quarter_standard_deviation(
    synthetic_oof,
) -> None:
    """捕获按全 OOF 一次性 NRMSE 或漏掉折间波动惩罚。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    output = MODEL_OUTPUT_COLUMNS[0]
    values = [0.10, 0.20, 0.30, 0.40, 0.50]
    _set_fold_nrmse(evaluations, folds, "xf_v1_baseline", output, values)
    for candidate in list(CANDIDATES)[1:]:
        _set_fold_nrmse(evaluations, folds, candidate, output, [0.9] * 5)

    artifact = build_oof_routing(
        evaluations,
        fold_data_by_fold=folds,
        pressure_Pa=_PRESSURE_PA,
    )
    score = artifact["routes"][output]["candidates"]["xf_v1_baseline"]
    assert score["mean_nrmse"] == pytest.approx(0.30)
    assert score["std_nrmse"] == pytest.approx(np.std(values, ddof=0))
    assert score["primary_score"] == pytest.approx(
        np.mean(values) + 0.25 * np.std(values, ddof=0)
    )


def test_less_than_1e4_tie_uses_mean_before_primary_score(synthetic_oof) -> None:
    """严格小于阈值时，较低 mean 可覆盖略优的首要分数。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    output = MODEL_OUTPUT_COLUMNS[0]
    delta = 0.00039
    baseline = [0.1 + delta, 0.1 - delta, 0.1 + delta, 0.1 - delta, 0.1]
    balanced = [0.10001] * 5
    _set_fold_nrmse(evaluations, folds, "xf_v1_baseline", output, baseline)
    _set_fold_nrmse(evaluations, folds, "xf_v2_balanced", output, balanced)
    for candidate in list(CANDIDATES)[2:]:
        _set_fold_nrmse(evaluations, folds, candidate, output, [0.9] * 5)

    artifact = build_oof_routing(
        evaluations,
        fold_data_by_fold=folds,
        pressure_Pa=_PRESSURE_PA,
    )
    route = artifact["routes"][output]
    baseline_score = route["candidates"]["xf_v1_baseline"]["primary_score"]
    balanced_score = route["candidates"]["xf_v2_balanced"]["primary_score"]
    assert baseline_score > balanced_score
    assert baseline_score - balanced_score < 1.0e-4
    assert route["selected_candidate"] == "xf_v1_baseline"


def test_near_below_1e4_is_still_a_tie_without_tolerance_band() -> None:
    """只要实际 float 差值小于 1e-4，就必须执行 mean 优先的 tie-break。"""

    near_below = float(np.nextafter(1.0e-4, 0.0))
    assert near_below < 1.0e-4
    assert _within_strict_tie(near_below, 0.0)


def test_exact_1e4_score_difference_is_not_a_tie() -> None:
    """阈值边界必须是 `<1e-4`，不能误写成 `<=`。"""

    assert not _within_strict_tie(1.0e-4, 0.0)


def test_tie_break_uses_parameter_count_then_candidate_name(synthetic_oof) -> None:
    """mean/std 相同时先比较真实参数量，再按固定候选名字典序。"""

    folds, source = synthetic_oof
    evaluations = {
        name: dict(results) for name, results in source.items()
    }
    output = MODEL_OUTPUT_COLUMNS[2]
    for candidate in CANDIDATES:
        _set_fold_nrmse(evaluations, folds, candidate, output, [0.2] * 5)
    first = build_oof_routing(
        evaluations,
        fold_data_by_fold=folds,
        pressure_Pa=_PRESSURE_PA,
    )
    assert first["routes"][output]["selected_candidate"] == "xf_v1_baseline"

    _set_fold_nrmse(
        evaluations, folds, "xf_v1_baseline", output, [0.5] * 5
    )
    _set_fold_nrmse(
        evaluations, folds, "xf_v3_capacity", output, [0.5] * 5
    )
    _set_fold_nrmse(
        evaluations, folds, "xf_v5_robust", output, [0.5] * 5
    )
    second = build_oof_routing(
        evaluations,
        fold_data_by_fold=folds,
        pressure_Pa=_PRESSURE_PA,
    )
    assert CANDIDATE_PARAMETER_COUNTS["xf_v2_balanced"] == CANDIDATE_PARAMETER_COUNTS["xf_v4_weighted"]
    assert second["routes"][output]["selected_candidate"] == "xf_v2_balanced"


def test_regen_humidity_is_rescored_with_locked_regen_temperature() -> None:
    """捕获含湿量候选错误携带自己的再生温度进行相态评分。"""

    folds = _fold_contract()
    evaluations = _evaluation_bundle(folds)
    regen_t = MODEL_OUTPUT_COLUMNS.index("Label_React_Outlet_Temp")
    regen_w = MODEL_OUTPUT_COLUMNS.index("Label_React_Outlet_AbsHum")
    for fold_name, fold in folds.items():
        targets = fold.validation_targets_physical.copy()
        for candidate_index, candidate in enumerate(CANDIDATES):
            predictions = targets.copy()
            predictions[:, regen_t] += candidate_index * 2.5
            predictions[:, regen_w] += 5.0 + candidate_index
            if candidate == "xf_v4_weighted":
                predictions[:, regen_t] = 0.0
                predictions[:, regen_w] = targets[:, regen_w]
            elif candidate == "xf_v5_robust":
                predictions[:, regen_t] = targets[:, regen_t] + 0.1
                predictions[:, regen_w] = targets[:, regen_w] + 1.0
            evaluations[candidate][fold_name] = evaluate_predictions(
                predictions,
                targets,
                fold.train_targets_physical,
                sample_ids=fold.validation_ids,
                pressure_Pa=_PRESSURE_PA,
                mode="in_memory",
            )

    artifact = build_oof_routing(
        evaluations,
        fold_data_by_fold=folds,
        pressure_Pa=_PRESSURE_PA,
    )
    temperature_output = "Label_React_Outlet_Temp"
    humidity_output = "Label_React_Outlet_AbsHum"
    assert artifact["routes"][temperature_output]["selected_candidate"] == "xf_v1_baseline"
    humidity_route = artifact["routes"][humidity_output]
    assert humidity_route["temperature_candidate"] == "xf_v1_baseline"
    assert humidity_route["metric"] == "regen_phase_nrmse"
    assert humidity_route["selected_candidate"] == "xf_v4_weighted"
    own_phase = artifact["candidate_fold_metrics"]["xf_v4_weighted"]["fold_1"]
    assert own_phase["regen_phase_metric"]["nrmse"] > 0.0
    assert humidity_route["candidates"]["xf_v4_weighted"]["fold_nrmse"]["fold_1"] == 0.0


def test_routes_temperature_and_power_before_all_humidity_outputs(routing_artifact) -> None:
    """产物必须显式记录先锁定温度/功率、后组合对应温度的顺序。"""

    artifact = routing_artifact
    assert artifact["fixed_contract"]["selection_order"] == [
        "Label_Process_Outlet_Temp",
        "Label_Purge_Outlet_Temp",
        "Label_React_Outlet_Temp",
        "Label_PowerConsumption",
        "Label_Process_Outlet_AbsHum",
        "Label_Purge_Outlet_AbsHum",
        "Label_React_Outlet_AbsHum",
    ]
    assert artifact["routes"]["Label_Process_Outlet_AbsHum"]["temperature_candidate"] == artifact["routes"]["Label_Process_Outlet_Temp"]["selected_candidate"]
    assert artifact["routes"]["Label_Purge_Outlet_AbsHum"]["temperature_candidate"] == artifact["routes"]["Label_Purge_Outlet_Temp"]["selected_candidate"]
    assert artifact["routes"]["Label_React_Outlet_AbsHum"]["temperature_candidate"] == artifact["routes"]["Label_React_Outlet_Temp"]["selected_candidate"]


def test_routing_json_roundtrip_applies_bitwise_identical_oof(
    synthetic_oof, routing_artifact, tmp_path: Path
) -> None:
    """JSON 重载不得改变候选、列顺序、ID 顺序或任一浮点位。"""

    folds, evaluations = synthetic_oof
    artifact = routing_artifact
    before = apply_oof_routing(artifact, evaluations)
    path = tmp_path / "routing.json"
    save_routing_json(artifact, path)
    restored = load_routing_json(path)
    after = apply_oof_routing(restored, evaluations)

    pd.testing.assert_frame_equal(before, after, check_exact=True)
    assert path.read_bytes().endswith(b"\n")
    assert restored == artifact


def test_apply_accepts_all_candidate_fold_rows_reordered_by_sample_id(
    synthetic_oof, routing_artifact
) -> None:
    """OOF 内容绑定 ID→值，不应绑定输入文件的物理行序。"""

    _, source = synthetic_oof
    expected = apply_oof_routing(routing_artifact, source)
    reordered: dict[str, dict[str, EvaluationResult]] = {}
    for candidate_name, fold_results in source.items():
        reordered[candidate_name] = {}
        for fold_name, result in fold_results.items():
            reverse = np.arange(len(result.sample_ids) - 1, -1, -1)
            reordered[candidate_name][fold_name] = _reorder_result(result, reverse)

    actual = apply_oof_routing(routing_artifact, reordered)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_build_then_apply_accepts_real_minmax_target_ulp_roundtrip(
    synthetic_oof,
) -> None:
    """build 已验证的 scaler ULP 目标必须能被同一 artifact apply。"""

    folds, source = synthetic_oof
    evaluations = {name: dict(results) for name, results in source.items()}
    candidate = "xf_v1_baseline"
    fold_name = "fold_1"
    fold = folds[fold_name]
    original = evaluations[candidate][fold_name]
    target_scaled = fold.target_scaler.transform(
        fold.validation_targets_physical
    )
    prediction_scaled = fold.target_scaler.transform(original.predictions_raw)
    roundtripped = evaluate_predictions(
        prediction_scaled,
        target_scaled,
        fold.train_targets_physical,
        sample_ids=fold.validation_ids,
        pressure_Pa=_PRESSURE_PA,
        mode="minmax_ulp_roundtrip",
        inverse_scaler=fold.target_scaler,
    )
    assert not np.array_equal(
        roundtripped.targets_raw, fold.validation_targets_physical
    )
    maximum_ulp_difference = float(
        np.max(
            np.abs(
                roundtripped.targets_raw - fold.validation_targets_physical
            )
        )
    )
    assert 0.0 < maximum_ulp_difference < 1.0e-12
    evaluations[candidate][fold_name] = roundtripped

    artifact = build_oof_routing(
        evaluations,
        fold_data_by_fold=folds,
        pressure_Pa=_PRESSURE_PA,
    )
    routed = apply_oof_routing(artifact, evaluations)

    assert len(routed) == 52_435
    assert routed["ID"].is_monotonic_increasing


def test_apply_rejects_same_ids_with_predictions_changed_by_123(
    synthetic_oof, routing_artifact
) -> None:
    _, source = synthetic_oof
    evaluations = {name: dict(results) for name, results in source.items()}
    output = MODEL_OUTPUT_COLUMNS[0]
    selected = routing_artifact["routes"][output]["selected_candidate"]
    result = evaluations[selected]["fold_1"]
    evaluations[selected]["fold_1"] = replace(
        result,
        predictions_raw=result.predictions_raw + 123.0,
        predictions_deployed=result.predictions_deployed + 123.0,
    )

    with pytest.raises(RoutingIntegrityError, match="OOF|内容|预测|hash|哈希"):
        apply_oof_routing(routing_artifact, evaluations)


def test_rehashed_candidate_metric_cannot_contradict_direct_route(
    routing_artifact,
) -> None:
    payload = json.loads(json.dumps(routing_artifact, ensure_ascii=False))
    output = MODEL_OUTPUT_COLUMNS[0]
    candidate = "xf_v1_baseline"
    payload["candidate_fold_metrics"][candidate]["fold_1"]["deployed_metrics"][
        output
    ]["nrmse"] += 0.5
    _rehash_artifact(payload)

    with pytest.raises(RoutingIntegrityError, match="route|审计|评分|一致"):
        validate_routing_artifact(payload)


def test_rehashed_zero_id_hashes_cannot_replace_current_authoritative_ids(
    routing_artifact,
) -> None:
    payload = json.loads(json.dumps(routing_artifact, ensure_ascii=False))
    zero = "0" * 64
    integrity = payload["oof_integrity"]
    integrity["development_ids_sha256"] = zero
    integrity["outer_ids_sha256"] = zero
    for fold_name in FIXED_FOLD_GROUPS:
        integrity["fold_validation_ids_sha256"][fold_name] = zero
    for candidate_name in CANDIDATES:
        for fold_name in FIXED_FOLD_GROUPS:
            payload["oof_content_hashes"][candidate_name][fold_name][
                "sample_ids_sha256"
            ] = zero
    _rehash_artifact(payload)

    with pytest.raises(RoutingIntegrityError, match="权威|ID|哈希|hash"):
        validate_routing_artifact(payload)


def test_regen_hybrid_audit_is_saved_and_cross_checked_after_rehash(
    routing_artifact,
) -> None:
    assert "regen_hybrid_fold_audit" in routing_artifact
    payload = json.loads(json.dumps(routing_artifact, ensure_ascii=False))
    candidate = "xf_v1_baseline"
    payload["regen_hybrid_fold_audit"][candidate]["fold_1"]["phase_metric"][
        "nrmse"
    ] += 0.5
    _rehash_artifact(payload)

    with pytest.raises(RoutingIntegrityError, match="hybrid|相态|route|一致"):
        validate_routing_artifact(payload)


@pytest.mark.parametrize("tamper", ["hash", "route", "unknown_key"])
def test_routing_json_rejects_tampering_or_unknown_keys(
    routing_artifact, tmp_path: Path, tamper: str
) -> None:
    """完整性字段不能被删除、路由不能被改写、未知 schema 键不能混入。"""

    artifact = routing_artifact
    path = tmp_path / "routing.json"
    save_routing_json(artifact, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "hash":
        payload["integrity_sha256"] = "0" * 64
    elif tamper == "route":
        route = payload["routes"][MODEL_OUTPUT_COLUMNS[0]]
        route["selected_candidate"] = (
            "xf_v1_baseline"
            if route["selected_candidate"] != "xf_v1_baseline"
            else "xf_v5_robust"
        )
    else:
        payload["unexpected"] = True
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(RoutingIntegrityError):
        load_routing_json(path)


def test_rejects_pickle_or_non_json_routing_input(tmp_path: Path) -> None:
    """路由重载只接受严格 JSON，不提供 pickle 或通用 artifact 入口。"""

    path = tmp_path / "routing.pkl"
    path.write_bytes(b"not-json")
    with pytest.raises(RoutingIntegrityError, match="JSON|json"):
        load_routing_json(path)


def test_serialized_artifact_contains_no_model_or_external_artifact_path(
    routing_artifact, tmp_path: Path
) -> None:
    """路由产物只记录固定候选名，不接受可替换模型路径。"""

    artifact = routing_artifact
    path = tmp_path / "routing.json"
    save_routing_json(artifact, path)
    text = path.read_text(encoding="utf-8").lower()
    assert "artifact_path" not in text
    assert "model_path" not in text
    assert ".pt" not in text
    assert "pickle" not in text
