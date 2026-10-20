"""XF 无泄漏评估器与湿空气后处理的可执行规格。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archive.xf_rotor1_retrain.config import (
    FIXED_FOLD_GROUPS,
    MODEL_INPUT_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
    REPORTS_DIR,
)
from archive.xf_rotor1_retrain.data import (
    read_xf_source,
    validate_and_clean_xf,
)
from archive.xf_rotor1_retrain.evaluator import (
    EvaluationBoundaryError,
    EvaluationError,
    OOFIntegrityError,
    assemble_oof_predictions,
    build_fold_data,
    dew_point_water_C,
    evaluate_checkpoint,
    evaluate_in_memory,
    phase_split_total_water,
    saturation_humidity_ratio_g_per_kg,
    saturation_vapor_pressure_water_Pa,
)


@pytest.fixture(scope="session")
def clean_and_assignments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取权威数据与 Subtask 1 已回写的固定划分。"""

    clean, _ = validate_and_clean_xf(read_xf_source())
    assignments = pd.read_csv(REPORTS_DIR / "split_assignments.csv")
    return clean, assignments


def _physical_targets(n_rows: int) -> np.ndarray:
    """构造有非零训练跨度、可以独立核对的七输出样本。"""

    row = np.arange(n_rows, dtype=np.float64)
    return np.column_stack(
        [
            20.0 + row,
            2.0 + 0.2 * row,
            25.0 + 0.5 * row,
            3.0 + 0.1 * row,
            30.0 + row,
            10.0 + 2.0 * row,
            1.0 + row,
        ]
    )


def _expected_validation_ids_by_fold(
    assignments: pd.DataFrame,
) -> dict[str, pd.Series]:
    development = assignments.loc[assignments["partition"].eq("development")]
    return {
        fold_name: development.loc[
            development["validation_fold"].eq(fold_name), "ID"
        ]
        for fold_name in FIXED_FOLD_GROUPS
    }


def _assert_failed_lifecycle(
    events: list,
    *,
    mode: str,
    stage: str,
) -> None:
    """失败路径可以含 progress，但必须以唯一 failure/end 收束。"""

    assert events[0].kind == "start"
    assert [event.kind for event in events].count("failure") == 1
    assert [event.kind for event in events].count("end") == 1
    assert [event.kind for event in events][-2:] == ["failure", "end"]
    assert "result" not in [event.kind for event in events]
    assert "efficiency" not in [event.kind for event in events]
    assert {event.mode for event in events} == {mode}
    failure = events[-2]
    assert failure.payload["stage"] == stage
    assert failure.payload["detail"]
    assert events[-1].payload == {"status": "failed", "stage": stage}


def test_fold_scalers_fit_only_development_train_ids_and_never_outer(
    clean_and_assignments: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """捕获当折 validation 或 outer ID 泄漏到 scaler/span 拟合。"""

    clean, assignments = clean_and_assignments
    development = assignments.loc[assignments["partition"].eq("development")]
    outer_ids = set(assignments.loc[assignments["partition"].eq("outer_test"), "ID"])
    validation_union: set[int] = set()

    for fold_name in FIXED_FOLD_GROUPS:
        fold = build_fold_data(clean, assignments, fold_name)
        expected_validation = set(
            development.loc[development["validation_fold"].eq(fold_name), "ID"]
        )
        expected_train = set(development["ID"]) - expected_validation

        assert set(fold.train_ids) == expected_train
        assert set(fold.validation_ids) == expected_validation
        assert set(fold.outer_ids) == outer_ids
        assert set(fold.input_scaler.fit_ids) == expected_train
        assert set(fold.target_scaler.fit_ids) == expected_train
        assert set(fold.input_scaler.fit_ids).isdisjoint(expected_validation)
        assert set(fold.target_scaler.fit_ids).isdisjoint(outer_ids)
        assert np.all(fold.target_spans > 0.0)
        validation_union.update(fold.validation_ids)

    assert validation_union == set(development["ID"])
    assert len(validation_union) == 52_435
    assert validation_union.isdisjoint(outer_ids)


def test_outer_values_are_not_read_for_scalers_or_nrmse_spans(
    clean_and_assignments: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """outer 标签/连续特征为毒值时，开发折拟合结果也必须不变。"""

    clean, assignments = clean_and_assignments
    baseline = build_fold_data(clean, assignments, "fold_1")
    outer_ids = assignments.loc[assignments["partition"].eq("outer_test"), "ID"]
    poisoned = clean.copy()
    poisoned.loc[
        poisoned["ID"].isin(outer_ids),
        list(MODEL_INPUT_COLUMNS[2:]) + list(MODEL_OUTPUT_COLUMNS),
    ] = np.nan

    audited = build_fold_data(poisoned, assignments, "fold_1")
    np.testing.assert_array_equal(
        audited.input_scaler.data_min_, baseline.input_scaler.data_min_
    )
    np.testing.assert_array_equal(
        audited.input_scaler.data_max_, baseline.input_scaler.data_max_
    )
    np.testing.assert_array_equal(
        audited.target_scaler.data_min_, baseline.target_scaler.data_min_
    )
    np.testing.assert_array_equal(
        audited.target_scaler.data_max_, baseline.target_scaler.data_max_
    )
    np.testing.assert_array_equal(audited.target_spans, baseline.target_spans)


def test_fold_builder_rejects_tampered_fixed_geometry_assignment(
    clean_and_assignments: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """捕获划分报告被改写后评估器仍按错误折训练。"""

    clean, assignments = clean_and_assignments
    tampered = assignments.copy()
    victim = tampered.index[
        tampered["partition"].eq("development")
        & tampered["validation_fold"].eq("fold_1")
    ][0]
    tampered.loc[victim, "validation_fold"] = "fold_2"

    with pytest.raises(EvaluationError, match="固定几何分组"):
        build_fold_data(clean, tampered, "fold_1")


def test_fold_builder_cross_checks_every_assignment_id_against_clean_geometry(
    clean_and_assignments: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """捕获保留分区/几何文本但交换 outer 与 development 真实 ID 的泄漏。"""

    clean, assignments = clean_and_assignments
    tampered = assignments.copy()
    outer_index = tampered.index[tampered["partition"].eq("outer_test")][0]
    development_index = tampered.index[tampered["partition"].eq("development")][0]
    outer_id = int(tampered.loc[outer_index, "ID"])
    development_id = int(tampered.loc[development_index, "ID"])
    tampered.loc[outer_index, "ID"] = development_id
    tampered.loc[development_index, "ID"] = outer_id

    with pytest.raises(EvaluationError, match="ID.*几何|几何.*ID"):
        build_fold_data(clean, tampered, "fold_1")


def test_zero_fold_train_target_span_fails_hard(
    clean_and_assignments: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """捕获以 epsilon 或全局跨度代替折训练目标零跨度的绕过。"""

    clean, assignments = clean_and_assignments
    development = assignments.loc[assignments["partition"].eq("development")]
    fold_1_validation = set(
        development.loc[development["validation_fold"].eq("fold_1"), "ID"]
    )
    fold_1_train = set(development["ID"]) - fold_1_validation
    bad = clean.copy()
    bad.loc[bad["ID"].isin(fold_1_train), MODEL_OUTPUT_COLUMNS[0]] = 12

    with pytest.raises(EvaluationError, match="fold_1.*跨度为零"):
        build_fold_data(bad, assignments, "fold_1")


def test_oof_join_covers_every_development_id_exactly_once(
    clean_and_assignments: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """捕获 OOF 按行号拼接、漏 ID 或重复 ID。"""

    _, assignments = clean_and_assignments
    development = assignments.loc[assignments["partition"].eq("development")]
    outer_ids = assignments.loc[assignments["partition"].eq("outer_test"), "ID"]
    fold_predictions = {
        fold_name: pd.DataFrame(
            {
                "ID": fold_rows["ID"].to_numpy(),
                "candidate_prediction": np.full(len(fold_rows), fold_index),
            }
        )
        for fold_index, (fold_name, fold_rows) in enumerate(
            development.groupby("validation_fold", sort=True), start=1
        )
    }

    oof = assemble_oof_predictions(
        fold_predictions,
        expected_development_ids=development["ID"],
        expected_validation_ids_by_fold=_expected_validation_ids_by_fold(
            assignments
        ),
        outer_ids=outer_ids,
    )
    assert len(oof) == 52_435
    assert oof["ID"].is_unique
    assert set(oof["ID"]) == set(development["ID"])
    assert set(oof["ID"]).isdisjoint(outer_ids)
    assert oof["ID"].is_monotonic_increasing


@pytest.mark.parametrize("fault", ["missing", "duplicate", "outer"])
def test_oof_join_rejects_missing_duplicate_or_outer_ids(
    clean_and_assignments: tuple[pd.DataFrame, pd.DataFrame], fault: str
) -> None:
    """任一 OOF ID 完整性故障都必须在路由前失败。"""

    _, assignments = clean_and_assignments
    development = assignments.loc[assignments["partition"].eq("development")]
    outer_ids = assignments.loc[assignments["partition"].eq("outer_test"), "ID"]
    frames = {
        name: pd.DataFrame({"ID": rows["ID"], "prediction": 1.0})
        for name, rows in development.groupby("validation_fold", sort=True)
    }
    first_name = sorted(frames)[0]
    first = frames[first_name].copy().reset_index(drop=True)
    if fault == "missing":
        frames[first_name] = first.iloc[1:].copy()
    elif fault == "duplicate":
        first.loc[1, "ID"] = first.loc[0, "ID"]
        frames[first_name] = first
    else:
        first.loc[0, "ID"] = int(outer_ids.iloc[0])
        frames[first_name] = first

    with pytest.raises(OOFIntegrityError, match="OOF"):
        assemble_oof_predictions(
            frames,
            expected_development_ids=development["ID"],
            expected_validation_ids_by_fold=_expected_validation_ids_by_fold(
                assignments
            ),
            outer_ids=outer_ids,
        )


@pytest.mark.parametrize("fault", ["missing_fold", "wrong_fold", "all_in_fold_1"])
def test_oof_join_rejects_missing_or_wrong_fold_membership(
    clean_and_assignments: tuple[pd.DataFrame, pd.DataFrame], fault: str
) -> None:
    """全局 ID 集合正确也不足够；每折必须恰好是该折 validation ID。"""

    _, assignments = clean_and_assignments
    development = assignments.loc[assignments["partition"].eq("development")]
    outer_ids = assignments.loc[assignments["partition"].eq("outer_test"), "ID"]
    expected_by_fold = _expected_validation_ids_by_fold(assignments)
    frames = {
        name: pd.DataFrame({"ID": ids.to_numpy(), "prediction": 1.0})
        for name, ids in expected_by_fold.items()
    }
    if fault == "missing_fold":
        del frames["fold_5"]
    elif fault == "wrong_fold":
        frames["fold_1"], frames["fold_2"] = frames["fold_2"], frames["fold_1"]
    else:
        frames = {
            name: pd.DataFrame(columns=["ID", "prediction"])
            for name in FIXED_FOLD_GROUPS
        }
        frames["fold_1"] = pd.DataFrame(
            {"ID": development["ID"].to_numpy(), "prediction": 1.0}
        )

    with pytest.raises(OOFIntegrityError, match="OOF.*fold"):
        assemble_oof_predictions(
            frames,
            expected_development_ids=development["ID"],
            expected_validation_ids_by_fold=expected_by_fold,
            outer_ids=outer_ids,
        )


def test_saturation_pressure_matches_water_basis_gold_vector() -> None:
    """饱和压必须与 docs/requirement.txt 企业参考代码逐系数一致。

    2026-09-17：<=0 ℃ 分支由 Murphy–Koop 过冷水公式迁移为企业冰面基准
    多项式，金标准值重录；>0 ℃ 分支公式未变。
    """

    temperatures = np.array([-75.0, -40.0, -20.0, 0.0, 20.0, 60.0])
    expected = np.array(
        [
            0.12211935692063862,
            12.84511625118676,
            103.25929284812797,
            611.1470566150772,
            2338.8037000739814,
            19943.760621620728,
        ]
    )
    actual = saturation_vapor_pressure_water_Pa(temperatures)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-11)


def test_saturation_humidity_is_infinite_when_vapor_pressure_reaches_total() -> None:
    """捕获对 p_ws >= P 做压力裁剪后伪造有限 W_sat。"""

    assert math.isinf(saturation_humidity_ratio_g_per_kg(60.0, 10_000.0))
    finite = saturation_humidity_ratio_g_per_kg(20.0, 101_325.0)
    # 2026-09-17：水汽比对齐企业参考代码 0.62198，重录该金标准。
    assert finite == pytest.approx(14.695878614876, rel=2e-10)


def test_water_basis_dew_point_numerically_inverts_same_saturation_curve() -> None:
    """露点换算执行企业 DptAH 表契约（2026-09-17 迁移）。

    露点 > 0 ℃（W > 3.764 g/kg）按同一饱和压公式精确反演；
    露点 <= 0 ℃ 走 DptAH 表分段线性插值，允许表格固有偏差（< 0.05 ℃）。
    """

    gold_temperatures = np.array([-75.0, -40.0, -20.0, 0.0, 20.0, 60.0])
    humidity = saturation_humidity_ratio_g_per_kg(
        gold_temperatures, 101_325.0
    )
    recovered = dew_point_water_C(humidity, 101_325.0)
    np.testing.assert_allclose(recovered[3:], gold_temperatures[3:], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(recovered[:3], gold_temperatures[:3], atol=0.05, rtol=0.0)


def test_regen_total_water_phase_split_preserves_raw_and_gas_mixing_semantics() -> None:
    """再生液水必须单独排出，混风只能消费气相含湿量。"""

    temperatures = np.array([20.0, 20.0, 20.0])
    total = np.array([-2.0, 10.0, 20.0])
    phase = phase_split_total_water(total, temperatures, 101_325.0)
    saturation = 14.695878614876145  # 2026-09-17：水汽比对齐企业 0.62198 后重录

    np.testing.assert_array_equal(phase.total_raw, total)
    np.testing.assert_array_equal(phase.total_nonnegative, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(phase.vapor, [0.0, 10.0, saturation], rtol=2e-10)
    np.testing.assert_allclose(
        phase.liquid, [0.0, 0.0, 20.0 - saturation], rtol=2e-10
    )
    np.testing.assert_array_equal(phase.gas_for_mixing, phase.vapor)
    np.testing.assert_allclose(
        phase.vapor + phase.liquid, phase.total_nonnegative, atol=1e-12
    )


class _AddTenInverseTarget:
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) + 10.0


class _TimesTwoInverseScaler:
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * 2.0


def test_evaluator_applies_inverse_target_before_inverse_scaling() -> None:
    """捕获反目标变换与反缩放次序被调换。"""

    targets = _physical_targets(3)
    train_targets = _physical_targets(5)
    result = evaluate_in_memory(
        lambda _: np.ones((3, 7), dtype=np.float64),
        np.zeros((3, 10), dtype=np.float64),
        targets,
        train_targets,
        sample_ids=[1, 2, 3],
        pressure_Pa=101_325.0,
        inverse_target_transform=_AddTenInverseTarget(),
        inverse_scaler=_TimesTwoInverseScaler(),
    )
    # Required order is (1 + 10) * 2 = 22, not 1 * 2 + 10 = 12.
    np.testing.assert_array_equal(result.predictions_raw, np.full((3, 7), 22.0))


def test_raw_and_deployed_metrics_record_nonnegative_projection() -> None:
    """捕获原始指标被覆盖，或负功率没有在部署指标前投影。"""

    targets = _physical_targets(3)
    predictions = targets.copy()
    power_index = MODEL_OUTPUT_COLUMNS.index("Label_PowerConsumption")
    predictions[:, power_index] = [-1.0, 2.0, 5.0]
    targets[:, power_index] = [1.0, 2.0, 3.0]
    train_targets = _physical_targets(5)
    result = evaluate_in_memory(
        lambda _: predictions,
        np.zeros((3, 10), dtype=np.float64),
        targets,
        train_targets,
        sample_ids=[1, 2, 3],
        pressure_Pa=101_325.0,
    )

    raw = result.raw_metrics["Label_PowerConsumption"]
    deployed = result.deployed_metrics["Label_PowerConsumption"]
    assert raw.rmse == pytest.approx(math.sqrt(8.0 / 3.0))
    assert deployed.rmse == pytest.approx(math.sqrt(5.0 / 3.0))
    assert raw.nrmse == pytest.approx(math.sqrt(8.0 / 3.0) / 4.0)
    assert deployed.nrmse == pytest.approx(math.sqrt(5.0 / 3.0) / 4.0)
    assert result.projection_stats["Label_PowerConsumption"].count == 1
    assert result.projection_stats["Label_PowerConsumption"].maximum_correction == 1.0
    np.testing.assert_array_equal(result.predictions_deployed[:, power_index], [0, 2, 5])


def test_negative_process_humidity_keeps_thermodynamic_infinity_but_reports_at_floor() -> None:
    """负处理湿量的 0W 热力学露点是 -inf，只有报告指标使用 -75℃ 地板。"""

    targets = _physical_targets(3)
    predictions = targets.copy()
    process_w = MODEL_OUTPUT_COLUMNS.index("Label_Process_Outlet_AbsHum")
    predictions[0, process_w] = -0.1
    result = evaluate_in_memory(
        lambda _: predictions,
        np.zeros((3, 10), dtype=np.float64),
        targets,
        _physical_targets(5),
        sample_ids=[1, 2, 3],
        pressure_Pa=101_325.0,
    )

    assert result.predictions_raw[0, process_w] == -0.1
    assert result.predictions_deployed[0, process_w] == 0.0
    assert np.isneginf(result.process_dew_point_prediction_thermodynamic_C[0])
    assert result.process_dew_point_prediction_reporting_C[0] == -75.0
    assert np.isneginf(dew_point_water_C(0.0, 101_325.0))
    audit = result.process_dew_point_prediction_reporting_audit
    assert audit.adjusted_count == 1
    assert audit.fraction == pytest.approx(1.0 / 3.0)
    assert audit.negative_infinity_count == 1
    assert audit.below_floor_finite_count == 0
    assert audit.maximum_finite_adjustment_C == 0.0
    assert result.process_dew_point_target_reporting_audit.adjusted_count == 0
    assert all(
        math.isfinite(value)
        for value in result.process_dew_point_metric.__dict__.values()
    )
    assert result.projection_stats["Label_Process_Outlet_AbsHum"].count == 1


def test_regen_phase_metric_splits_prediction_and_label_with_fold_train_span() -> None:
    """再生含湿路由指标必须是气/液相向量 RMSE，分母只用折训练总水跨度。"""

    targets = _physical_targets(2)
    predictions = targets.copy()
    regen_t = MODEL_OUTPUT_COLUMNS.index("Label_React_Outlet_Temp")
    regen_w = MODEL_OUTPUT_COLUMNS.index("Label_React_Outlet_AbsHum")
    targets[:, regen_t] = 20.0
    predictions[:, regen_t] = 20.0
    targets[:, regen_w] = [12.0, 16.0]
    predictions[:, regen_w] = [10.0, 20.0]
    train_targets = _physical_targets(5)
    train_targets[:, regen_w] = [8.0, 10.0, 12.0, 16.0, 20.0]

    result = evaluate_in_memory(
        lambda _: predictions,
        np.zeros((2, 10), dtype=np.float64),
        targets,
        train_targets,
        sample_ids=[11, 12],
        pressure_Pa=101_325.0,
    )
    saturation = 14.695878614876145  # 2026-09-17：水汽比对齐企业 0.62198 后重录
    squared = (
        (10.0 - 12.0) ** 2
        + (0.0 - 0.0) ** 2
        + (saturation - saturation) ** 2
        + ((20.0 - saturation) - (16.0 - saturation)) ** 2
    )
    expected_rmse = math.sqrt(squared / 2.0)
    assert result.regen_phase_metric.rmse == pytest.approx(expected_rmse, rel=2e-10)
    assert result.regen_phase_metric.nrmse == pytest.approx(expected_rmse / 12.0)
    np.testing.assert_allclose(
        result.regen_prediction_phase.vapor, [10.0, saturation], rtol=2e-10
    )
    np.testing.assert_allclose(
        result.regen_target_phase.liquid, [0.0, 16.0 - saturation], rtol=2e-10
    )


def test_in_memory_and_checkpoint_entries_share_identical_prediction_core(
    tmp_path: Path,
) -> None:
    """捕获 checkpoint 入口复制一套评估逻辑导致预测或指标漂移。"""

    features = np.arange(40, dtype=np.float64).reshape(4, 10) / 10.0
    targets = _physical_targets(4)
    train_targets = _physical_targets(6)

    def predictor(values: np.ndarray) -> np.ndarray:
        result = targets[: len(values)].copy()
        result[:, 0] += values[:, 0] * 0.01
        return result

    checkpoint = tmp_path / "model.json"
    checkpoint.write_text(json.dumps({"kind": "deterministic"}), encoding="utf-8")

    def load_predictor(path: Path):
        assert json.loads(path.read_text(encoding="utf-8"))["kind"] == "deterministic"
        return predictor

    memory = evaluate_in_memory(
        predictor,
        features,
        targets,
        train_targets,
        sample_ids=[1, 2, 3, 4],
        pressure_Pa=101_325.0,
        batch_size=2,
    )
    restored = evaluate_checkpoint(
        checkpoint,
        load_predictor,
        features,
        targets,
        train_targets,
        sample_ids=[1, 2, 3, 4],
        pressure_Pa=101_325.0,
        batch_size=2,
    )

    np.testing.assert_array_equal(memory.predictions_raw, restored.predictions_raw)
    np.testing.assert_array_equal(
        memory.predictions_deployed, restored.predictions_deployed
    )
    assert memory.raw_metrics == restored.raw_metrics
    assert memory.deployed_metrics == restored.deployed_metrics
    assert memory.regen_phase_metric == restored.regen_phase_metric


@pytest.mark.parametrize("mode", ["in_memory", "checkpoint"])
def test_evaluation_reports_mode_aware_start_progress_end_result_and_efficiency(
    tmp_path: Path, mode: str
) -> None:
    """长评估必须对每种入口显式报告全部观测阶段。"""

    features = np.zeros((4, 10), dtype=np.float64)
    targets = _physical_targets(4)
    train_targets = _physical_targets(6)
    events = []
    predictor = lambda values: targets[: len(values)]
    common = dict(
        features=features,
        targets_model_space=targets,
        train_targets_physical=train_targets,
        sample_ids=[1, 2, 3, 4],
        pressure_Pa=101_325.0,
        batch_size=2,
        event_sink=events.append,
    )
    if mode == "in_memory":
        evaluate_in_memory(predictor, **common)
    else:
        checkpoint = tmp_path / "model.ckpt"
        checkpoint.write_bytes(b"valid")
        evaluate_checkpoint(checkpoint, lambda _: predictor, **common)

    assert [event.kind for event in events] == [
        "start",
        "progress",
        "progress",
        "end",
        "result",
        "efficiency",
    ]
    assert {event.mode for event in events} == {mode}
    assert events[1].completed == 2
    assert events[2].completed == events[2].total == 4
    assert events[-1].payload["samples_per_second"] > 0.0


def test_checkpoint_boundary_errors_name_mode_and_failure_stage(tmp_path: Path) -> None:
    """缺失或损坏 checkpoint 必须在 checkpoint/load 边界报错。"""

    targets = _physical_targets(2)
    common = dict(
        features=np.zeros((2, 10)),
        targets_model_space=targets,
        train_targets_physical=_physical_targets(4),
        sample_ids=[1, 2],
        pressure_Pa=101_325.0,
    )
    missing_events = []
    with pytest.raises(EvaluationBoundaryError, match="checkpoint.*load.*不存在"):
        evaluate_checkpoint(
            tmp_path / "missing.ckpt",
            lambda _: None,
            event_sink=missing_events.append,
            **common,
        )
    _assert_failed_lifecycle(missing_events, mode="checkpoint", stage="load")

    broken = tmp_path / "broken.ckpt"
    broken.write_bytes(b"broken")

    def fail_load(_: Path):
        raise ValueError("corrupted payload")

    corrupt_events = []
    with pytest.raises(EvaluationBoundaryError, match="checkpoint.*load.*corrupted") as caught:
        evaluate_checkpoint(
            broken,
            fail_load,
            event_sink=corrupt_events.append,
            **common,
        )
    assert isinstance(caught.value.__cause__, ValueError)
    _assert_failed_lifecycle(corrupt_events, mode="checkpoint", stage="load")


@pytest.mark.parametrize("mode", ["in_memory", "checkpoint"])
def test_empty_validation_set_has_mode_aware_boundary_error(
    tmp_path: Path, mode: str
) -> None:
    """空验证集不得被当作零指标或成功事件。"""

    common = dict(
        features=np.empty((0, 10)),
        targets_model_space=np.empty((0, 7)),
        train_targets_physical=_physical_targets(4),
        sample_ids=[],
        pressure_Pa=101_325.0,
    )
    events = []
    with pytest.raises(EvaluationBoundaryError, match=rf"{mode}.*validation.*空"):
        if mode == "in_memory":
            evaluate_in_memory(
                lambda _: np.empty((0, 7)), event_sink=events.append, **common
            )
        else:
            checkpoint = tmp_path / "model.ckpt"
            checkpoint.write_bytes(b"valid")
            evaluate_checkpoint(
                checkpoint,
                lambda _: (lambda _: np.empty((0, 7))),
                event_sink=events.append,
                **common,
            )
    _assert_failed_lifecycle(events, mode=mode, stage="validation")


class _FailingInverseTransform:
    def inverse_transform(self, _: np.ndarray) -> np.ndarray:
        raise RuntimeError("inverse transform exploded")


@pytest.mark.parametrize("mode", ["in_memory", "checkpoint"])
@pytest.mark.parametrize(
    ("fault", "expected_stage", "expected_detail"),
    [
        ("predictor_nan", "prediction", "非有限"),
        ("predictor_raises", "prediction", "predictor exploded"),
        ("invalid_pressure", "metric", "总压"),
        ("inverse_raises", "metric", "inverse transform exploded"),
    ],
)
def test_all_evaluation_failures_emit_mode_aware_lifecycle(
    tmp_path: Path,
    mode: str,
    fault: str,
    expected_stage: str,
    expected_detail: str,
) -> None:
    """预测和指标阶段任何异常都必须可观测且保留原始 cause。"""

    targets = _physical_targets(2)
    predictions = targets.copy()
    inverse_transform = None
    pressure = 101_325.0

    def predictor(_: np.ndarray) -> np.ndarray:
        if fault == "predictor_raises":
            raise RuntimeError("predictor exploded")
        if fault == "predictor_nan":
            bad = predictions.copy()
            bad[0, 0] = np.nan
            return bad
        return predictions

    if fault == "invalid_pressure":
        pressure = -1.0
    if fault == "inverse_raises":
        inverse_transform = _FailingInverseTransform()

    events = []
    common = dict(
        features=np.zeros((2, 10)),
        targets_model_space=targets,
        train_targets_physical=_physical_targets(4),
        sample_ids=[1, 2],
        pressure_Pa=pressure,
        inverse_target_transform=inverse_transform,
        event_sink=events.append,
    )
    with pytest.raises(
        EvaluationBoundaryError,
        match=rf"{mode}.*{expected_stage}.*{expected_detail}",
    ) as caught:
        if mode == "in_memory":
            evaluate_in_memory(predictor, **common)
        else:
            checkpoint = tmp_path / f"{fault}.ckpt"
            checkpoint.write_bytes(b"valid")
            evaluate_checkpoint(checkpoint, lambda _: predictor, **common)

    if fault in {"predictor_raises", "invalid_pressure", "inverse_raises"}:
        assert caught.value.__cause__ is not None
    _assert_failed_lifecycle(events, mode=mode, stage=expected_stage)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_predictions_fail_at_mode_aware_prediction_boundary(
    bad_value: float,
) -> None:
    """捕获 NaN/Inf 被非负投影、相分离或指标聚合掩盖。"""

    targets = _physical_targets(2)
    predictions = targets.copy()
    predictions[0, 0] = bad_value
    with pytest.raises(EvaluationBoundaryError, match="in_memory.*prediction.*非有限"):
        evaluate_in_memory(
            lambda _: predictions,
            np.zeros((2, 10)),
            targets,
            _physical_targets(4),
            sample_ids=[1, 2],
            pressure_Pa=101_325.0,
        )


def test_nonfinite_metric_fails_at_mode_aware_metric_boundary() -> None:
    """浮点溢出产生的非有限指标必须失败，不得序列化为成绩。"""

    targets = _physical_targets(2)
    predictions = targets.copy()
    predictions[:, 0] = [1.0e308, -1.0e308]
    with pytest.raises(EvaluationBoundaryError, match="in_memory.*metric.*非有限"):
        evaluate_in_memory(
            lambda _: predictions,
            np.zeros((2, 10)),
            targets,
            _physical_targets(4),
            sample_ids=[1, 2],
            pressure_Pa=101_325.0,
        )
