"""单 KAN 模型、评估器和 checkpoint 的 TDD 规格。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from training.single_kan.config import (
    DIRECT_OUTPUT_COLUMNS,
    MAIN_INPUT_COLUMNS,
    PROFILE_CONFIGS,
    XF_INPUT_COLUMNS,
)
from training.single_kan.evaluator import (
    EvaluationError,
    compare_heater_power,
    dew_point_from_process_humidity,
    evaluate_in_memory,
)
from training.single_kan.model import (
    ReversibleTargetTransform,
    build_single_kan,
    fit_preprocessor,
    load_artifact,
    save_artifact,
)
from training.single_kan.trainer import (
    SingleKANTrainer,
    load_checkpoint_payload,
    set_deterministic_seed,
)


def _arrays(n: int, in_features: int) -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(n, dtype=np.float32)
    x = np.column_stack([row + i for i in range(in_features)]).astype(np.float32)
    y = np.column_stack([2.0 + row * (0.1 + i * 0.01) for i in range(6)]).astype(
        np.float32
    )
    return x, y


@pytest.mark.parametrize(
    ("profile", "in_features", "hidden", "grid"),
    [("xf", 10, [24], 6), ("main", 9, [16], 5)],
)
def test_single_kan_has_fixed_shape_and_architecture(
    profile: str, in_features: int, hidden: list[int], grid: int
) -> None:
    model = build_single_kan(profile)
    result = model(torch.zeros(3, in_features))
    assert result.shape == (3, 6)
    assert model.config.in_features == in_features
    assert model.config.out_features == 6
    assert list(model.config.hidden_widths) == hidden
    assert model.config.grid_size == grid
    assert model.config.spline_order == 3


def test_main_metadata_has_no_derived_features_and_only_direct_outputs() -> None:
    model = build_single_kan("main")
    assert model.metadata["input_cols"] == list(MAIN_INPUT_COLUMNS)
    assert model.metadata["direct_outputs"] == list(DIRECT_OUTPUT_COLUMNS)
    assert model.metadata["derived_outputs"] == ["处理出口露点"]
    assert model.metadata["audit_columns"] == ["power", "处理出口露点"]
    assert model.metadata["derived_features"] == []
    assert "power" not in model.metadata["input_cols"]
    assert "处理出口露点" not in model.metadata["direct_outputs"]


def test_reversible_log1p_round_trip_and_negative_guard() -> None:
    transform = ReversibleTargetTransform(
        output_columns=DIRECT_OUTPUT_COLUMNS,
        log1p_targets=("处理出口含湿量",),
    )
    values = np.array([[10.0, 3.0, 20.0, 4.0, 30.0, 5.0]], dtype=np.float64)
    restored = transform.inverse_transform(transform.transform(values))
    np.testing.assert_allclose(restored, values, rtol=0, atol=1e-10)
    values[0, 1] = -0.1
    with pytest.raises(ValueError, match="负值"):
        transform.transform(values)


def test_preprocessor_target_scaler_does_not_contain_audit_columns() -> None:
    x, y = _arrays(8, len(XF_INPUT_COLUMNS))
    pre = fit_preprocessor(x, y, XF_INPUT_COLUMNS, DIRECT_OUTPUT_COLUMNS)
    assert tuple(pre.input_columns) == tuple(XF_INPUT_COLUMNS)
    assert tuple(pre.target_columns) == tuple(DIRECT_OUTPUT_COLUMNS)
    assert "power" not in pre.target_columns
    assert "处理出口露点" not in pre.target_columns
    np.testing.assert_allclose(pre.inverse_targets(pre.transform_targets(y)), y)


def test_in_memory_and_checkpoint_evaluation_share_metrics_and_reject_nonfinite(
    tmp_path: Path,
) -> None:
    set_deterministic_seed(42)
    x, y = _arrays(12, len(MAIN_INPUT_COLUMNS))
    pre = fit_preprocessor(x, y, MAIN_INPUT_COLUMNS, DIRECT_OUTPUT_COLUMNS)
    model = build_single_kan("main")
    result = evaluate_in_memory(model, x, y, pre, pressure_Pa=101325.0)
    assert set(result.metrics) == set(DIRECT_OUTPUT_COLUMNS)
    assert result.predictions.shape == (12, 6)
    assert result.nonfinite_count == 0
    bad = y.copy()
    bad[0, 0] = np.nan
    with pytest.raises(EvaluationError, match="非有限"):
        evaluate_in_memory(model, x, bad, pre)


def test_train_steps_validate_on_cadence_and_checkpoint_reload_is_consistent(
    tmp_path: Path,
) -> None:
    set_deterministic_seed(42)
    x, y = _arrays(16, len(MAIN_INPUT_COLUMNS))
    trainer = SingleKANTrainer(
        profile="main",
        train_features=x,
        train_targets=y,
        validation_features=x[:4],
        validation_targets=y[:4],
        checkpoint_dir=tmp_path,
        validation_interval_steps=2,
        checkpoint_interval_steps=2,
        device="cpu",
    )
    events: list[dict] = []
    trainer.event_sink = events.append
    history = trainer.train_steps(3)
    assert history.validation_steps == [2]
    assert all(
        {"grad_norm", "learning_rate", "step_seconds"} <= set(item)
        for item in history.step_metrics
    )
    assert any(
        item.get("kind") == "checkpoint_saved" and item.get("global_step") == 2
        for item in events
    )
    checkpoint = trainer.save_checkpoint()
    payload = load_checkpoint_payload(checkpoint)
    assert payload["batch_cursor"] >= 0
    assert "rng_state" in payload
    assert payload["rng_state"]["batch_rng"]
    reloaded = SingleKANTrainer.from_checkpoint(checkpoint, device="cpu")
    np.testing.assert_allclose(
        trainer.predict(x[:3]), reloaded.predict(x[:3]), rtol=0, atol=1e-6
    )
    assert reloaded.model.metadata["direct_outputs"] == list(DIRECT_OUTPUT_COLUMNS)


def test_checkpoint_resume_replays_the_same_batch_sequence(tmp_path: Path) -> None:
    """恢复训练必须保留 batch 游标、排列和 RNG，而不是从新 epoch 开始。"""

    x, y = _arrays(24, len(MAIN_INPUT_COLUMNS))
    common = {
        "profile": "main",
        "train_features": x,
        "train_targets": y,
        "validation_features": x[:8],
        "validation_targets": y[:8],
        "validation_interval_steps": 100,
        "checkpoint_interval_steps": 100,
        "batch_size": 4,
        "device": "cpu",
        "seed": 123,
    }

    set_deterministic_seed(123)
    uninterrupted = SingleKANTrainer(checkpoint_dir=tmp_path / "full", **common)
    uninterrupted.train_steps(5)
    expected_state = {
        key: value.detach().clone()
        for key, value in uninterrupted.model.state_dict().items()
    }

    set_deterministic_seed(123)
    partial = SingleKANTrainer(checkpoint_dir=tmp_path / "partial", **common)
    partial.train_steps(3)
    checkpoint = partial.save_checkpoint()
    resumed = SingleKANTrainer.from_checkpoint(
        checkpoint,
        device="cpu",
        train_features=x,
        train_targets=y,
        validation_features=x[:8],
        validation_targets=y[:8],
    )
    resumed.train_steps(2)

    for key, expected in expected_state.items():
        np.testing.assert_allclose(
            resumed.model.state_dict()[key].detach().cpu().numpy(),
            expected.detach().cpu().numpy(),
            rtol=0,
            atol=1e-7,
        )
    assert resumed.history.global_step == uninterrupted.history.global_step == 5


def test_dew_point_uses_humidity_and_pressure_water_basis() -> None:
    result = dew_point_from_process_humidity(np.array([3.0]), 101325.0)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(-3.1, abs=0.5)


def test_checkpoint_resume_replays_same_random_batch_sequence(tmp_path: Path) -> None:
    x, y = _arrays(16, len(MAIN_INPUT_COLUMNS))
    set_deterministic_seed(123)
    uninterrupted = SingleKANTrainer(
        profile="main",
        train_features=x,
        train_targets=y,
        validation_features=x[:4],
        validation_targets=y[:4],
        checkpoint_dir=tmp_path / "full",
        validation_interval_steps=99,
        checkpoint_interval_steps=99,
        batch_size=4,
        seed=123,
        device="cpu",
    )
    uninterrupted.train_steps(4)

    set_deterministic_seed(123)
    first_half = SingleKANTrainer(
        profile="main",
        train_features=x,
        train_targets=y,
        validation_features=x[:4],
        validation_targets=y[:4],
        checkpoint_dir=tmp_path / "split",
        validation_interval_steps=99,
        checkpoint_interval_steps=99,
        batch_size=4,
        seed=123,
        device="cpu",
    )
    first_half.train_steps(2)
    checkpoint = first_half.save_checkpoint()
    resumed = SingleKANTrainer.from_checkpoint(
        checkpoint,
        train_features=x,
        train_targets=y,
        validation_features=x[:4],
        validation_targets=y[:4],
        device="cpu",
    )
    resumed.train_steps(2)
    np.testing.assert_allclose(
        uninterrupted.predict(x), resumed.predict(x), rtol=0.0, atol=1e-6
    )


def test_power_comparison_reports_separate_bias_errors_and_anomalies() -> None:
    report = compare_heater_power(
        measured_power=np.array([10.0, 10.0, -1.0, np.nan]),
        heater_load=np.array([8.0, 12.0, 4.0, 9.0]),
        anomaly_z=2.0,
    )
    assert report["count"] == 3
    assert report["bias_kW"] == pytest.approx(5.0 / 3.0)
    assert report["mae_kW"] == pytest.approx(3.0)
    assert report["rmse_kW"] == pytest.approx(11.0**0.5)
    assert report["nonfinite_count"] == 1
    assert report["anomaly_count"] >= 1


def test_save_artifact_rejects_output_cols_mismatch(tmp_path: Path) -> None:
    x, y = _arrays(8, len(MAIN_INPUT_COLUMNS))
    pre = fit_preprocessor(x, y, MAIN_INPUT_COLUMNS, DIRECT_OUTPUT_COLUMNS)
    model = build_single_kan("main")
    metadata = model.metadata.copy()
    metadata["output_cols"] = ["power"]
    with pytest.raises(ValueError, match="output_cols|direct_outputs"):
        save_artifact(tmp_path / "bad", model, pre, metadata=metadata)


def test_artifact_reload_rejects_tampered_output_and_architecture_contract(
    tmp_path: Path,
) -> None:
    model = build_single_kan("main")
    x, y = _arrays(8, len(MAIN_INPUT_COLUMNS))
    pre = fit_preprocessor(x, y, MAIN_INPUT_COLUMNS, DIRECT_OUTPUT_COLUMNS)
    save_artifact(tmp_path, model, pre)
    metadata_path = tmp_path / "model_metadata.json"
    metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
    metadata["audit_columns"] = ["power"]
    metadata["architecture"]["grid_size"] = 99
    metadata_path.write_text(
        __import__("json").dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="direct_outputs|audit_columns|architecture"):
        load_artifact(tmp_path)


def test_artifact_reload_rejects_output_cols_not_equal_to_direct_outputs(
    tmp_path: Path,
) -> None:
    model = build_single_kan("main")
    x, y = _arrays(8, len(MAIN_INPUT_COLUMNS))
    pre = fit_preprocessor(x, y, MAIN_INPUT_COLUMNS, DIRECT_OUTPUT_COLUMNS)
    save_artifact(tmp_path, model, pre)
    metadata_path = tmp_path / "model_metadata.json"
    metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_cols"] = ["power"]
    metadata_path.write_text(
        __import__("json").dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="output_cols|direct_outputs"):
        load_artifact(tmp_path)
