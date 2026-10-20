"""XF 五候选 KAN 训练组件的可执行规格。"""

from __future__ import annotations

import ast
import copy
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from archive.xf_rotor1_retrain import trainer as trainer_module
from archive.xf_rotor1_retrain.candidates import (
    CANDIDATES,
    OUTPUT_WEIGHTS,
    build_candidate_model,
    get_candidate,
)
from archive.xf_rotor1_retrain.config import (
    MODEL_INPUT_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
)
from archive.xf_rotor1_retrain.trainer import (
    CheckpointError,
    TargetTransform,
    TrainingDataError,
    ValidationSnapshot,
    XFTrainer,
    build_candidate_loss,
    fit_training_preprocessor,
    load_checkpoint_payload,
    set_deterministic_seed,
)


def _toy_data(n_rows: int = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成手工可核对且七个目标均有非零跨度的小数据。"""

    row = np.arange(n_rows, dtype=np.float64)
    features = np.column_stack(
        [row + 0.25 * column for column in range(len(MODEL_INPUT_COLUMNS))]
    )
    targets = np.column_stack(
        [1.0 + row * (0.1 + 0.02 * column) for column in range(len(MODEL_OUTPUT_COLUMNS))]
    )
    return features, targets, np.arange(10_000, 10_000 + n_rows)


def _new_trainer(
    checkpoint_dir: Path,
    *,
    candidate_name: str = "xf_v1_baseline",
    n_rows: int = 13,
    validation_interval_steps: int = 2,
    event_sink=None,
) -> tuple[XFTrainer, np.ndarray, np.ndarray]:
    features, targets, ids = _toy_data(n_rows)
    trainer = XFTrainer(
        candidate=get_candidate(candidate_name),
        train_features=features,
        train_targets=targets,
        train_ids=ids,
        checkpoint_dir=checkpoint_dir,
        device="cpu",
        validation_interval_steps=validation_interval_steps,
        event_sink=event_sink,
    )
    return trainer, features[:5] + 0.05, targets[:5] + 0.01


def _with_nan_feature(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    corrupted = features.copy()
    corrupted[0, 0] = np.nan
    return corrupted, targets


def test_candidate_matrix_is_exact_and_has_no_derived_inputs() -> None:
    """捕获候选参数漂移或把派生特征混入 10 输入。"""

    compact = {
        name: {
            "hidden_widths": list(config.hidden_widths),
            "grid_size": config.grid_size,
            "loss": config.loss,
            "log1p_targets": list(config.log1p_targets),
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
        for name, config in CANDIDATES.items()
    }
    assert compact == {
        "xf_v1_baseline": {
            "hidden_widths": [16],
            "grid_size": 5,
            "loss": "mse",
            "log1p_targets": [],
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        "xf_v2_balanced": {
            "hidden_widths": [24],
            "grid_size": 6,
            "loss": "mse",
            "log1p_targets": [],
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
        "xf_v3_capacity": {
            "hidden_widths": [32],
            "grid_size": 7,
            "loss": "mse",
            "log1p_targets": [],
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
        "xf_v4_weighted": {
            "hidden_widths": [24],
            "grid_size": 6,
            "loss": "weighted_mse",
            "log1p_targets": ["Label_Process_Outlet_AbsHum"],
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
        "xf_v5_robust": {
            "hidden_widths": [32, 16],
            "grid_size": 6,
            "loss": "weighted_huber",
            "log1p_targets": ["Label_Process_Outlet_AbsHum"],
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
        },
    }
    expected_inputs = (
        "TextBox_Rotor_depth",
        "TextBox_Rotor_diameter",
        "TextBox_Process_inlet_Temp",
        "TextBox_Purge_Inlet_AbsHum",
        "Label_Process_FV",
        "Label_Purge_FV",
        "TextBox_Rotation_Speed",
        "TextBox_React_Heater_Temp",
        "Label_React_CMH",
        "Label_React_Inlet_AbsHum",
    )
    expected_outputs = (
        "Label_Process_Outlet_Temp",
        "Label_Process_Outlet_AbsHum",
        "Label_Purge_Outlet_Temp",
        "Label_Purge_Outlet_AbsHum",
        "Label_React_Outlet_Temp",
        "Label_React_Outlet_AbsHum",
        "Label_PowerConsumption",
    )
    assert tuple(MODEL_INPUT_COLUMNS) == expected_inputs
    assert tuple(MODEL_OUTPUT_COLUMNS) == expected_outputs
    for candidate in CANDIDATES.values():
        assert candidate.input_columns == expected_inputs
        assert candidate.output_columns == expected_outputs
        assert candidate.spline_order == 3
        assert candidate.batch_size == 64
        assert candidate.max_epochs == 300
        assert candidate.grad_clip == 1.0
        assert candidate.seed == 42
        assert candidate.validation_interval_steps == 2_000
        assert candidate.patience_evaluations == 10
        assert candidate.huber_delta == 1.0
        assert candidate.feature_range == (-1.0, 1.0)
    assert OUTPUT_WEIGHTS == (1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0)


@pytest.mark.parametrize("candidate_name", list(CANDIDATES))
def test_every_candidate_reuses_builder_for_exact_10_to_7_forward(
    candidate_name: str,
) -> None:
    """捕获模型使用旧 8 输入、8 输出或自行复制架构。"""

    model = build_candidate_model(get_candidate(candidate_name))
    output = model(torch.zeros(4, 10))
    assert output.shape == (4, 7)
    assert model.config.in_features == 10
    assert model.config.out_features == 7


def test_minmax_scalers_are_fit_only_on_training_rows() -> None:
    """捕获 validation 极值泄漏进 scaler 的训练污染。"""

    train_x, train_y, train_ids = _toy_data(8)
    preprocessor = fit_training_preprocessor(
        train_x,
        train_y,
        train_ids,
        get_candidate("xf_v1_baseline"),
    )

    assert preprocessor.x_scaler.feature_range == (-1.0, 1.0)
    assert preprocessor.y_scaler.feature_range == (-1.0, 1.0)
    assert preprocessor.x_scaler.fit_ids == tuple(train_ids)
    assert preprocessor.y_scaler.fit_ids == tuple(train_ids)
    np.testing.assert_allclose(preprocessor.transform_features(train_x).min(axis=0), -1.0)
    np.testing.assert_allclose(preprocessor.transform_features(train_x).max(axis=0), 1.0)

    validation_x = train_x.copy()
    validation_x[0, 0] = 10_000.0
    assert preprocessor.transform_features(validation_x)[0, 0] > 1.0
    assert preprocessor.x_scaler.data_max_[0] == train_x[:, 0].max()


def test_log1p_applies_only_to_process_outlet_humidity_and_is_reversible() -> None:
    """捕获 log 目标列错位或反变换丢失。"""

    values = np.array([[10.0, 3.0, 20.0, 4.0, 30.0, 5.0, 2.0]])
    transform = TargetTransform(
        output_columns=MODEL_OUTPUT_COLUMNS,
        log1p_targets=("Label_Process_Outlet_AbsHum",),
    )
    transformed = transform.transform(values)

    expected = values.copy()
    expected[0, 1] = math.log1p(3.0)
    np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_allclose(transform.inverse_transform(transformed), values, rtol=0, atol=1e-12)
    assert transform.log_indices == (1,)


def test_log1p_rejects_negative_process_humidity() -> None:
    """捕获用 log1p 静默接受物理负含湿量。"""

    values = np.ones((2, 7))
    values[1, 1] = -0.01
    transform = TargetTransform(
        output_columns=MODEL_OUTPUT_COLUMNS,
        log1p_targets=("Label_Process_Outlet_AbsHum",),
    )
    with pytest.raises(TrainingDataError, match="负值"):
        transform.transform(values)


def test_unweighted_and_weighted_mse_match_hand_calculation() -> None:
    """捕获列权重顺序错误或未归一化的量纲漂移。"""

    prediction = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    target = torch.zeros_like(prediction)
    mse = build_candidate_loss(get_candidate("xf_v1_baseline"))(prediction, target)
    weighted = build_candidate_loss(get_candidate("xf_v4_weighted"))(prediction, target)

    squared = np.square(np.arange(1.0, 8.0))
    assert float(mse) == pytest.approx(float(squared.mean()))
    assert float(weighted) == pytest.approx(
        float(np.dot(squared, np.asarray(OUTPUT_WEIGHTS)) / sum(OUTPUT_WEIGHTS))
    )


def test_weighted_huber_uses_delta_one_in_scaled_space() -> None:
    """捕获 Huber delta 错误、在物理空间计算或列权重错位。"""

    errors = np.array([0.5, 2.0, -0.5, -2.0, 1.0, -1.0, 3.0])
    prediction = torch.from_numpy(errors.astype(np.float32)).unsqueeze(0)
    target = torch.zeros_like(prediction)
    actual = build_candidate_loss(get_candidate("xf_v5_robust"))(prediction, target)
    elementwise = np.where(np.abs(errors) <= 1.0, 0.5 * errors**2, np.abs(errors) - 0.5)
    expected = np.dot(elementwise, np.asarray(OUTPUT_WEIGHTS)) / sum(OUTPUT_WEIGHTS)
    assert float(actual) == pytest.approx(float(expected), rel=1e-7)


def test_fixed_seed_reproduces_parameters_and_batch_order(tmp_path: Path) -> None:
    """捕获只固定模型初始化而 DataLoader 乱序仍漂移。"""

    set_deterministic_seed(42)
    first_model = build_candidate_model(get_candidate("xf_v1_baseline"))
    first_state = {name: value.detach().clone() for name, value in first_model.state_dict().items()}
    first, _, _ = _new_trainer(tmp_path / "first")

    set_deterministic_seed(42)
    second_model = build_candidate_model(get_candidate("xf_v1_baseline"))
    second, _, _ = _new_trainer(tmp_path / "second")

    for name, value in second_model.state_dict().items():
        assert torch.equal(first_state[name], value)
    np.testing.assert_array_equal(first.peek_next_batch_indices(), second.peek_next_batch_indices())


def test_validation_and_plateau_scheduler_fire_only_on_step_cadence(tmp_path: Path) -> None:
    """捕获按 epoch/小批随意验证或每步调度 scheduler。"""

    events: list[dict] = []
    trainer, val_x, val_y = _new_trainer(
        tmp_path,
        validation_interval_steps=2,
        event_sink=events.append,
    )
    trainer.train_steps(5, validation_features=val_x, validation_targets=val_y)

    validation_end_steps = [
        event["global_step"]
        for event in events
        if event["kind"] == "validation_end"
    ]
    assert validation_end_steps == [2, 4]
    assert trainer.validation_evaluations == 2
    assert trainer.scheduler.last_epoch == 2
    assert trainer.global_step == 5


def test_shared_validation_callback_runs_only_on_trainer_cadence(
    tmp_path: Path,
) -> None:
    """捕获集成层绕过 trainer cadence 或 trainer 复制 evaluator 逻辑。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=2,
    )
    callback_steps: list[int] = []
    callback_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def shared_evaluator(
        snapshot: ValidationSnapshot,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        assert isinstance(snapshot, ValidationSnapshot)
        assert not isinstance(snapshot, XFTrainer)
        callback_steps.append(snapshot.global_step)
        callback_shapes.append((np.asarray(features).shape, np.asarray(targets).shape))

    trainer.train_steps(
        5,
        validation_features=validation_x,
        validation_targets=validation_y,
        validation_callback=shared_evaluator,
    )

    assert callback_steps == [2, 4]
    assert callback_shapes == [((5, 10), (5, 7)), ((5, 10), (5, 7))]
    assert trainer.validation_evaluations == 2
    assert trainer.scheduler.last_epoch == 2
    assert math.isfinite(trainer.best_validation_loss)
    assert load_checkpoint_payload(tmp_path / "best.pt")["global_step"] in {2, 4}


@pytest.mark.parametrize("invalid", [0.0, "metric"])
def test_shared_validation_callback_rejects_control_value(
    tmp_path: Path,
    invalid: object,
) -> None:
    """共享 evaluator hook 不得改写 scaled-loss 训练控制状态。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path / str(invalid),
        validation_interval_steps=1,
    )

    with pytest.raises(TrainingDataError, match="共享验证.*None|返回值"):
        trainer.train_steps(
            1,
            validation_features=validation_x,
            validation_targets=validation_y,
            validation_callback=lambda *_: invalid,
        )


def test_completed_cadence_persists_resumable_last_before_later_failure(
    tmp_path: Path,
) -> None:
    """捕获训练中断时只有 best、没有固定 last 可续训状态。"""

    def fail_after_first_cadence(event: dict) -> None:
        if event["kind"] == "train_step" and event["global_step"] == 3:
            raise RuntimeError("injected interruption")

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=2,
        event_sink=fail_after_first_cadence,
    )
    with pytest.raises(RuntimeError, match="injected interruption"):
        trainer.train_steps(
            4,
            validation_features=validation_x,
            validation_targets=validation_y,
        )

    payload = load_checkpoint_payload(tmp_path / "last.pt")
    assert payload["global_step"] == 2
    assert payload["validation_evaluations"] == 1


def _assert_nested_state_equal(left, right) -> None:
    """比较 optimizer/scheduler/RNG 的嵌套状态，不做容差放宽。"""

    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_state_equal(left_item, right_item)
    else:
        assert left == right


def _trainer_observable_state(trainer: XFTrainer) -> dict:
    return {
        "model": {
            name: value.detach().cpu().clone()
            for name, value in trainer.model.state_dict().items()
        },
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer.scheduler.state_dict()),
        "progress": (
            trainer.global_step,
            trainer.epoch,
            trainer.batch_cursor,
            trainer.validation_evaluations,
            trainer.evaluations_without_improvement,
            trainer.best_validation_loss,
            trainer.stopped_early,
        ),
        "x_scaler": copy.deepcopy(trainer.preprocessor.x_scaler.state_dict()),
        "y_scaler": copy.deepcopy(trainer.preprocessor.y_scaler.state_dict()),
        "target_transform": copy.deepcopy(
            trainer.preprocessor.target_transform.state_dict()
        ),
        "model_training": trainer.model.training,
        "loss_training": trainer.loss_fn.training,
        "training_controls": (
            trainer.validation_interval_steps,
            trainer.patience_evaluations,
            trainer.scheduler_patience,
        ),
        "train_features": trainer._train_features.copy(),
        "train_targets": trainer._train_targets.copy(),
        "train_ids": trainer._train_ids,
        "train_x_scaled": trainer._train_x_scaled.detach().cpu().clone(),
        "train_y_scaled": trainer._train_y_scaled.detach().cpu().clone(),
        "training_data_fingerprint": trainer.training_data_fingerprint,
        "batch_order": trainer._batch_order.detach().cpu().clone(),
        "batch_index_history": [value.copy() for value in trainer.batch_index_history],
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state().clone(),
        "generator_rng": trainer.generator.get_state().clone(),
    }


def test_validation_snapshot_isolates_model_scaler_mode_rng_and_live_controls(
    tmp_path: Path,
) -> None:
    """callback 对快照的任意修改不得污染 live trainer 的任一状态。"""

    control, validation_x, validation_y = _new_trainer(
        tmp_path / "control",
        validation_interval_steps=1,
    )
    control.train_steps(
        1,
        validation_features=validation_x,
        validation_targets=validation_y,
        validation_callback=lambda *_: None,
    )
    expected = _trainer_observable_state(control)

    attacked, attacked_x, attacked_y = _new_trainer(
        tmp_path / "attacked",
        validation_interval_steps=1,
    )

    def mutate_every_exposed_copy(
        snapshot: ValidationSnapshot,
        _features: np.ndarray,
        _targets: np.ndarray,
    ) -> None:
        assert not hasattr(snapshot, "optimizer")
        assert not hasattr(snapshot, "scheduler")
        assert not hasattr(snapshot, "generator")
        with torch.no_grad():
            for parameter in list(snapshot.model.parameters())[:4]:
                parameter.zero_()
        snapshot.model.train(not snapshot.model.training)
        snapshot.preprocessor.x_scaler._scaler.scale_[0] = 123456.0
        snapshot.preprocessor.y_scaler._scaler.min_[0] = -123456.0
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        with pytest.raises(AttributeError):
            snapshot.global_step = 999

    attacked.train_steps(
        1,
        validation_features=attacked_x,
        validation_targets=attacked_y,
        validation_callback=mutate_every_exposed_copy,
    )
    actual = _trainer_observable_state(attacked)

    _assert_nested_state_equal(actual, expected)


def test_callback_closure_attack_is_detected_and_fully_restored_atomically(
    tmp_path: Path,
) -> None:
    """callback 闭包捕获 live trainer 时不得绕过 facade 改状态。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )
    trainer.train_steps(
        1,
        validation_features=validation_x,
        validation_targets=validation_y,
        validation_callback=lambda *_: None,
    )
    last_path = tmp_path / "last.pt"
    committed_bytes = last_path.read_bytes()
    expected_before_attack: dict = {}

    def mutate_live_trainer_through_closure(
        _snapshot: ValidationSnapshot,
        _features: np.ndarray,
        _targets: np.ndarray,
    ) -> None:
        expected_before_attack.update(_trainer_observable_state(trainer))
        original_model = trainer.model
        trainer.model = copy.deepcopy(original_model)
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(123456.0)
        trainer.model.train(not trainer.model.training)
        trainer.loss_fn.train(not trainer.loss_fn.training)
        trainer.optimizer.param_groups[0]["lr"] = 9.9
        trainer.optimizer.state.clear()
        trainer.scheduler.best = -123.0
        trainer.scheduler.last_epoch = 999
        trainer.preprocessor.x_scaler._scaler.scale_[0] = 123456.0
        trainer.preprocessor.y_scaler._scaler.min_[0] = -123456.0
        trainer.preprocessor.target_transform.log_indices = ()
        trainer.validation_interval_steps = 17
        trainer.patience_evaluations = 19
        trainer.scheduler_patience = 23
        trainer.generator.manual_seed(987654)
        trainer._batch_order = torch.flip(trainer._batch_order, dims=(0,))
        trainer.batch_index_history[0][0] = -1
        trainer.global_step = 999
        trainer.epoch = 888
        trainer.batch_cursor = 7
        trainer.validation_evaluations = 777
        trainer.evaluations_without_improvement = 666
        trainer.best_validation_loss = 555.0
        trainer.stopped_early = True
        trainer._train_features[0, 0] = 111111.0
        trainer._train_targets[0, 0] = 222222.0
        trainer._train_x_scaled[0, 0] = 333333.0
        trainer._train_y_scaled[0, 0] = 444444.0
        trainer._train_ids = tuple(reversed(trainer._train_ids))
        trainer.training_data_fingerprint = "0" * 64
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

    with pytest.raises(
        TrainingDataError,
        match=r"\[in_memory/callback\].*修改训练状态",
    ):
        trainer.train_steps(
            1,
            validation_features=validation_x,
            validation_targets=validation_y,
            validation_callback=mutate_live_trainer_through_closure,
        )

    assert expected_before_attack
    _assert_nested_state_equal(
        _trainer_observable_state(trainer),
        expected_before_attack,
    )
    assert trainer.model is not None
    assert last_path.read_bytes() == committed_bytes


def test_callback_transaction_restores_complete_trainer_object_graph(
    tmp_path: Path,
) -> None:
    """捕获字段白名单遗漏配置、普通属性、hook 和引用关系。"""

    observed_events: list[dict] = []

    def event_sink(event: dict) -> None:
        observed_events.append(event)

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
        event_sink=event_sink,
    )

    def forward_hook(_module, _inputs, _output) -> None:
        return None

    trainer.model.register_forward_hook(forward_hook)
    trainer.train_steps(
        1,
        validation_features=validation_x,
        validation_targets=validation_y,
        validation_callback=lambda *_: None,
    )
    committed_last = (tmp_path / "last.pt").read_bytes()
    original_candidate = trainer.candidate
    original_device = trainer.device
    original_checkpoint_dir = trainer.checkpoint_dir
    original_reduction = trainer.loss_fn.reduction
    original_log_indices = trainer.preprocessor.target_transform.log_indices
    original_hook_items = tuple(trainer.model._forward_hooks.items())

    def attack_unlisted_state(
        _snapshot: ValidationSnapshot,
        _features: np.ndarray,
        _targets: np.ndarray,
    ) -> None:
        trainer.candidate = replace(trainer.candidate, seed=trainer.candidate.seed + 1)
        trainer.device = torch.device("meta")
        trainer.checkpoint_dir = tmp_path / "hijacked"
        trainer.event_sink = lambda _event: None
        trainer.loss_fn.reduction = "sum"
        trainer.preprocessor.target_transform.log_indices = (0,)
        trainer.model.callback_injected_attribute = {"mutated": True}
        trainer.model._forward_hooks.clear()
        trainer.scheduler.optimizer = torch.optim.SGD(
            trainer.model.parameters(),
            lr=0.5,
        )

    with pytest.raises(
        TrainingDataError,
        match=r"\[in_memory/callback\].*修改训练状态",
    ):
        trainer.train_steps(
            1,
            validation_features=validation_x,
            validation_targets=validation_y,
            validation_callback=attack_unlisted_state,
        )

    assert trainer.candidate == original_candidate
    assert trainer.device == original_device
    assert trainer.checkpoint_dir == original_checkpoint_dir
    assert trainer.event_sink is event_sink
    assert trainer.loss_fn.reduction == original_reduction
    assert trainer.preprocessor.target_transform.log_indices == original_log_indices
    assert not hasattr(trainer.model, "callback_injected_attribute")
    assert tuple(trainer.model._forward_hooks.items()) == original_hook_items
    assert trainer.scheduler.optimizer is trainer.optimizer
    assert (tmp_path / "last.pt").read_bytes() == committed_last


def test_callback_restores_torch_global_execution_state(tmp_path: Path) -> None:
    """callback 不得改变 grad/deterministic 和训练相关 backend flags。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )
    original = {
        "grad_enabled": torch.is_grad_enabled(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }

    def mutate_torch_globals(
        _snapshot: ValidationSnapshot,
        _features: np.ndarray,
        _targets: np.ndarray,
    ) -> None:
        torch.set_grad_enabled(not original["grad_enabled"])
        torch.use_deterministic_algorithms(not original["deterministic"])
        torch.backends.cudnn.enabled = not original["cudnn_enabled"]
        torch.backends.cudnn.benchmark = not original["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = not original["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = not original["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = not original["matmul_allow_tf32"]

    try:
        trainer.train_steps(
            1,
            validation_features=validation_x,
            validation_targets=validation_y,
            validation_callback=mutate_torch_globals,
        )
        assert torch.is_grad_enabled() == original["grad_enabled"]
        assert (
            torch.are_deterministic_algorithms_enabled()
            == original["deterministic"]
        )
        assert (
            torch.is_deterministic_algorithms_warn_only_enabled()
            == original["warn_only"]
        )
        assert torch.backends.cudnn.enabled == original["cudnn_enabled"]
        assert torch.backends.cudnn.benchmark == original["cudnn_benchmark"]
        assert (
            torch.backends.cudnn.deterministic
            == original["cudnn_deterministic"]
        )
        assert torch.backends.cudnn.allow_tf32 == original["cudnn_allow_tf32"]
        assert (
            torch.backends.cuda.matmul.allow_tf32
            == original["matmul_allow_tf32"]
        )
    finally:
        torch.set_grad_enabled(original["grad_enabled"])
        torch.use_deterministic_algorithms(
            original["deterministic"],
            warn_only=original["warn_only"],
        )
        torch.backends.cudnn.enabled = original["cudnn_enabled"]
        torch.backends.cudnn.benchmark = original["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = original["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = original["matmul_allow_tf32"]


def test_callback_argument_mutation_is_isolated_from_validation_authority(
    tmp_path: Path,
) -> None:
    """callback 只能改独立参数副本，不能污染本次后续 cadence。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )
    original_x = validation_x.copy()
    original_y = validation_y.copy()

    def mutate_callback_arguments(
        _snapshot: ValidationSnapshot,
        callback_x: np.ndarray,
        callback_y: np.ndarray,
    ) -> None:
        callback_x.fill(123456.0)
        callback_y.fill(-123456.0)

    trainer.train_steps(
        2,
        validation_features=validation_x,
        validation_targets=validation_y,
        validation_callback=mutate_callback_arguments,
    )

    np.testing.assert_array_equal(validation_x, original_x)
    np.testing.assert_array_equal(validation_y, original_y)
    assert trainer.validation_evaluations == 2


def test_callback_closure_validation_mutation_is_detected_and_restored(
    tmp_path: Path,
) -> None:
    """闭包直接改调用方 validation 原数组时必须拒绝并还原。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )
    original_x = validation_x.copy()
    original_y = validation_y.copy()

    def mutate_caller_arrays(
        _snapshot: ValidationSnapshot,
        _callback_x: np.ndarray,
        _callback_y: np.ndarray,
    ) -> None:
        validation_x[0, 0] = 999999.0
        validation_y[0, 0] = -999999.0

    with pytest.raises(
        TrainingDataError,
        match=r"\[in_memory/callback\].*修改验证输入",
    ):
        trainer.train_steps(
            1,
            validation_features=validation_x,
            validation_targets=validation_y,
            validation_callback=mutate_caller_arrays,
        )

    np.testing.assert_array_equal(validation_x, original_x)
    np.testing.assert_array_equal(validation_y, original_y)


def test_callback_closure_validation_dtype_layout_attack_is_fully_restored(
    tmp_path: Path,
) -> None:
    """C-contiguous NumPy 的 dtype/shape/strides/flags/value 攻击必须无损还原。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )
    original_x = validation_x.copy()
    original_y = validation_y.copy()
    original_x_layout = (
        validation_x.dtype,
        validation_x.shape,
        validation_x.strides,
        validation_x.flags.writeable,
        validation_x.flags.aligned,
    )
    original_y_layout = (
        validation_y.dtype,
        validation_y.shape,
        validation_y.strides,
        validation_y.flags.writeable,
        validation_y.flags.aligned,
    )

    def mutate_dtype_layout_and_values(
        _snapshot: ValidationSnapshot,
        _callback_x: np.ndarray,
        _callback_y: np.ndarray,
    ) -> None:
        validation_x.dtype = np.float32
        validation_x.shape = (validation_x.shape[0], -1)
        validation_x[0, 0] = np.float32(987654.0)
        validation_x.setflags(write=False)
        validation_y.strides = (0, validation_y.strides[1])
        validation_y[0, 0] = -987654.0

    with pytest.raises(
        TrainingDataError,
        match=r"\[in_memory/callback\].*修改验证输入.*还原",
    ):
        trainer.train_steps(
            1,
            validation_features=validation_x,
            validation_targets=validation_y,
            validation_callback=mutate_dtype_layout_and_values,
        )

    assert (
        validation_x.dtype,
        validation_x.shape,
        validation_x.strides,
        validation_x.flags.writeable,
        validation_x.flags.aligned,
    ) == original_x_layout
    assert (
        validation_y.dtype,
        validation_y.shape,
        validation_y.strides,
        validation_y.flags.writeable,
        validation_y.flags.aligned,
    ) == original_y_layout
    np.testing.assert_array_equal(validation_x, original_x)
    np.testing.assert_array_equal(validation_y, original_y)


def test_snapshot_construction_failure_has_distinct_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """snapshot deepcopy/构造失败不得误报成 callback 执行失败。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )

    def explode_snapshot(**_kwargs):
        raise RuntimeError("snapshot deepcopy exploded")

    monkeypatch.setattr(trainer_module, "ValidationSnapshot", explode_snapshot)
    with pytest.raises(
        TrainingDataError,
        match=r"\[in_memory/snapshot\].*snapshot deepcopy exploded",
    ):
        trainer.train_steps(
            1,
            validation_features=validation_x,
            validation_targets=validation_y,
            validation_callback=lambda *_: None,
        )


def test_callback_exception_preserves_previous_cadence_last_bytes_and_wraps_mode(
    tmp_path: Path,
) -> None:
    """callback 失败不得覆盖上一个已完成 cadence 的 last。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )
    trainer.train_steps(
        1,
        validation_features=validation_x,
        validation_targets=validation_y,
        validation_callback=lambda *_: None,
    )
    last_path = tmp_path / "last.pt"
    committed_bytes = last_path.read_bytes()
    committed_payload = load_checkpoint_payload(last_path)

    def fail_after_mutating_snapshot(
        snapshot: ValidationSnapshot,
        _features: np.ndarray,
        _targets: np.ndarray,
    ) -> None:
        with torch.no_grad():
            next(snapshot.model.parameters()).zero_()
        torch.manual_seed(123456)
        raise RuntimeError("snapshot evaluation exploded")

    with pytest.raises(
        TrainingDataError,
        match="in_memory.*callback.*snapshot evaluation exploded",
    ) as caught:
        trainer.train_steps(
            1,
            validation_features=validation_x,
            validation_targets=validation_y,
            validation_callback=fail_after_mutating_snapshot,
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert last_path.read_bytes() == committed_bytes
    after = load_checkpoint_payload(last_path)
    assert after["global_step"] == committed_payload["global_step"] == 1
    assert after["validation_evaluations"] == 1


def test_validation_callback_cannot_reenter_live_train_steps(tmp_path: Path) -> None:
    """callback 不得通过闭包重入 live trainer 并改变步进状态。"""

    trainer, validation_x, validation_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )
    rejected: list[str] = []

    def attempt_reentry(
        _snapshot: ValidationSnapshot,
        _features: np.ndarray,
        _targets: np.ndarray,
    ) -> None:
        with pytest.raises(TrainingDataError, match="重入|reentr") as caught:
            trainer.train_steps(
                0,
                validation_features=validation_x,
                validation_targets=validation_y,
            )
        rejected.append(str(caught.value))

    trainer.train_steps(
        1,
        validation_features=validation_x,
        validation_targets=validation_y,
        validation_callback=attempt_reentry,
    )

    assert len(rejected) == 1
    assert trainer.global_step == 1
    assert load_checkpoint_payload(tmp_path / "last.pt")["global_step"] == 1


def test_tiny_validation_improvement_keeps_trainer_and_scheduler_best_aligned(
    tmp_path: Path,
) -> None:
    """捕获 trainer 严格小于与 Plateau 默认相对 threshold 的语义分裂。"""

    trainer, validation_x, _ = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
    )
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
    trainer.optimizer.param_groups[0]["lr"] = 0.0

    first_scaled_target = np.ones((len(validation_x), 7), dtype=np.float64)
    tiny_improvement_scaled_target = np.full(
        (len(validation_x), 7),
        math.sqrt(0.99995),
        dtype=np.float64,
    )
    first_physical_target = trainer.preprocessor.inverse_targets(first_scaled_target)
    tiny_improvement_physical_target = trainer.preprocessor.inverse_targets(
        tiny_improvement_scaled_target
    )

    trainer.train_steps(
        1,
        validation_features=validation_x,
        validation_targets=first_physical_target,
    )
    trainer.train_steps(
        1,
        validation_features=validation_x,
        validation_targets=tiny_improvement_physical_target,
    )

    assert trainer.best_validation_loss == pytest.approx(0.99995, rel=1e-6)
    assert trainer.scheduler.best == trainer.best_validation_loss
    assert trainer.scheduler.mode == "min"
    assert trainer.scheduler.threshold_mode == "rel"
    assert trainer.scheduler.threshold == 0.0


def test_best_and_last_checkpoints_are_complete_and_roundtrip(tmp_path: Path) -> None:
    """捕获只保存模型权重、缺 scaler/优化器/游标的不可恢复产物。"""

    trainer, val_x, val_y = _new_trainer(tmp_path, validation_interval_steps=1)
    trainer.train_steps(1, validation_features=val_x, validation_targets=val_y)
    required = {
        "format_version",
        "candidate",
        "model",
        "optimizer",
        "scheduler",
        "x_scaler",
        "y_scaler",
        "target_transform",
        "global_step",
        "epoch",
        "batch_cursor",
        "seed",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "generator_state",
        "batch_order",
        "best_validation_loss",
        "validation_evaluations",
        "evaluations_without_improvement",
        "training_data_fingerprint",
        "validation_interval_steps",
        "patience_evaluations",
        "scheduler_patience",
        "stopped_early",
    }

    for checkpoint_name in ("best.pt", "last.pt"):
        path = tmp_path / checkpoint_name
        assert path.is_file()
        payload = load_checkpoint_payload(path)
        assert required <= payload.keys()
        assert payload["global_step"] == 1
        assert payload["candidate"]["name"] == "xf_v1_baseline"
        assert payload["x_scaler"]["fit_ids"] == list(range(10_000, 10_013))
        assert payload["target_transform"]["log1p_targets"] == []
        assert payload["validation_interval_steps"] == 1
        assert payload["patience_evaluations"] == 10
        assert payload["scheduler_patience"] == 5
        assert payload["stopped_early"] is False
        assert len(payload["training_data_fingerprint"]) == 64


def test_resume_matches_continuous_next_batches_lr_and_predictions(tmp_path: Path) -> None:
    """捕获 resume 后重洗牌、LR 断档或模型状态偏移。"""

    continuous, val_x, val_y = _new_trainer(
        tmp_path / "continuous",
        n_rows=70,
        validation_interval_steps=2,
    )
    continuous.train_steps(4, validation_features=val_x, validation_targets=val_y)

    split, split_val_x, split_val_y = _new_trainer(
        tmp_path / "split",
        n_rows=70,
        validation_interval_steps=2,
    )
    split.train_steps(2, validation_features=split_val_x, validation_targets=split_val_y)
    first_history = [indices.copy() for indices in split.batch_index_history]
    resumed = XFTrainer.resume(
        tmp_path / "split" / "last.pt",
        train_features=_toy_data(70)[0],
        train_targets=_toy_data(70)[1],
        train_ids=_toy_data(70)[2],
        checkpoint_dir=tmp_path / "split",
        device="cpu",
        validation_interval_steps=2,
    )
    resumed.train_steps(2, validation_features=split_val_x, validation_targets=split_val_y)

    combined_history = first_history + resumed.batch_index_history
    assert len(combined_history) == len(continuous.batch_index_history) == 4
    for expected, actual in zip(continuous.batch_index_history, combined_history):
        np.testing.assert_array_equal(actual, expected)
    assert resumed.current_lr == continuous.current_lr
    np.testing.assert_allclose(
        resumed.predict_physical(val_x),
        continuous.predict_physical(val_x),
        rtol=0,
        atol=0,
    )
    assert resumed.global_step == continuous.global_step == 4
    assert resumed.epoch == continuous.epoch
    assert resumed.batch_cursor == continuous.batch_cursor
    np.testing.assert_array_equal(
        resumed.peek_next_batch_indices(), continuous.peek_next_batch_indices()
    )


def test_training_data_fingerprint_is_deterministic_and_covers_order_and_fields() -> None:
    """捕获只哈希数值、却漏掉 ID 顺序或字段契约的弱指纹。"""

    features, targets, ids = _toy_data(8)
    candidate = get_candidate("xf_v1_baseline")
    fingerprint = trainer_module.compute_training_data_fingerprint(
        features,
        targets,
        ids,
        input_columns=candidate.input_columns,
        output_columns=candidate.output_columns,
    )
    repeated = trainer_module.compute_training_data_fingerprint(
        features.copy(),
        targets.copy(),
        ids.copy(),
        input_columns=tuple(candidate.input_columns),
        output_columns=tuple(candidate.output_columns),
    )
    reordered = trainer_module.compute_training_data_fingerprint(
        features[::-1],
        targets[::-1],
        ids[::-1],
        input_columns=candidate.input_columns,
        output_columns=candidate.output_columns,
    )
    changed_fields = replace(
        candidate,
        input_columns=(*candidate.input_columns[:-1], "derived_enthalpy"),
    )
    changed_contract = trainer_module.compute_training_data_fingerprint(
        features,
        targets,
        ids,
        input_columns=changed_fields.input_columns,
        output_columns=changed_fields.output_columns,
    )

    assert fingerprint == repeated
    assert len(fingerprint) == 64
    assert fingerprint != reordered
    assert fingerprint != changed_contract


@pytest.mark.parametrize("changed_part", ["features", "targets"])
def test_resume_rejects_changed_training_values_with_same_ids(
    tmp_path: Path, changed_part: str
) -> None:
    """捕获只比较 fit_ids，放过同 ID 数值被换的旧 scaler。"""

    features, targets, ids = _toy_data(13)
    trainer, val_x, val_y = _new_trainer(tmp_path)
    trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)
    changed_features = features.copy()
    changed_targets = targets.copy()
    if changed_part == "features":
        changed_features[0, 0] += 0.125
    else:
        changed_targets[0, 0] += 0.125

    with pytest.raises(CheckpointError, match="指纹"):
        XFTrainer.resume(
            tmp_path / "last.pt",
            train_features=changed_features,
            train_targets=changed_targets,
            train_ids=ids,
            checkpoint_dir=tmp_path,
            device="cpu",
        )


def test_stopped_early_roundtrip_preserves_terminal_state(tmp_path: Path) -> None:
    """捕获 resume 丢失早停状态并继续走新批次。"""

    features, targets, ids = _toy_data(13)
    trainer = XFTrainer(
        candidate=get_candidate("xf_v1_baseline"),
        train_features=features,
        train_targets=targets,
        train_ids=ids,
        checkpoint_dir=tmp_path,
        device="cpu",
        validation_interval_steps=1,
        patience_evaluations=1,
    )
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
    trainer.optimizer.param_groups[0]["lr"] = 0.0
    validation_x = features[:3]
    validation_y = trainer.preprocessor.inverse_targets(
        np.ones((len(validation_x), 7), dtype=np.float64)
    )
    trainer.train_steps(
        2,
        validation_features=validation_x,
        validation_targets=validation_y,
    )
    assert trainer.stopped_early is True
    assert trainer.global_step == 2

    resumed = XFTrainer.resume(
        tmp_path / "last.pt",
        train_features=features,
        train_targets=targets,
        train_ids=ids,
        checkpoint_dir=tmp_path,
        device="cpu",
    )
    assert resumed.stopped_early is True
    assert resumed.patience_evaluations == 1
    step_before = resumed.global_step
    resumed.train_steps(
        1,
        validation_features=validation_x,
        validation_targets=validation_y,
    )
    assert resumed.global_step == step_before


@pytest.mark.parametrize(
    "override",
    [
        {"validation_interval_steps": 3},
        {"patience_evaluations": 9},
        {"scheduler_patience": 4},
    ],
)
def test_resume_rejects_training_control_override_drift(
    tmp_path: Path, override: dict[str, int]
) -> None:
    """捕获 resume 调用方改写 cadence/耐心后改变后续轨迹。"""

    features, targets, ids = _toy_data(13)
    trainer, val_x, val_y = _new_trainer(tmp_path, validation_interval_steps=2)
    trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)

    with pytest.raises(CheckpointError, match="不一致"):
        XFTrainer.resume(
            tmp_path / "last.pt",
            train_features=features,
            train_targets=targets,
            train_ids=ids,
            checkpoint_dir=tmp_path,
            device="cpu",
            **override,
        )


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("seed", 7),
        ("global_step", 1),
        ("epoch", 1),
        ("batch_cursor", 1),
    ],
)
def test_resume_rejects_seed_or_batch_cursor_state_tampering(
    tmp_path: Path, field: str, new_value: int
) -> None:
    """捕获种子或 step/epoch/cursor 不可能组合被直接恢复。"""

    features, targets, ids = _toy_data(70)
    trainer, val_x, val_y = _new_trainer(
        tmp_path / "source", n_rows=70, validation_interval_steps=2
    )
    trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "source" / "last.pt")
    payload[field] = new_value
    path = tmp_path / f"tampered_{field}.pt"
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="seed|游标|步|不一致"):
        XFTrainer.resume(
            path,
            train_features=features,
            train_targets=targets,
            train_ids=ids,
            checkpoint_dir=tmp_path,
            device="cpu",
        )


def test_resume_rejects_validation_count_inconsistent_with_cadence(
    tmp_path: Path,
) -> None:
    """捕获伪造自洽 scheduler 计数，但在 cadence 上尚不可能验证的状态。"""

    features, targets, ids = _toy_data(70)
    trainer, val_x, val_y = _new_trainer(
        tmp_path / "source", n_rows=70, validation_interval_steps=2
    )
    trainer.train_steps(1, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "source" / "last.pt")
    payload["validation_evaluations"] = 1
    payload["scheduler"]["last_epoch"] = 1
    payload["scheduler"]["best"] = 1.0
    payload["best_validation_loss"] = 1.0
    path = tmp_path / "impossible_validation_count.pt"
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="cadence|验证次数"):
        XFTrainer.resume(
            path,
            train_features=features,
            train_targets=targets,
            train_ids=ids,
            checkpoint_dir=tmp_path,
            device="cpu",
        )


def test_resume_rejects_inconsistent_stopped_early_flag(tmp_path: Path) -> None:
    """捕获无任何无改善验证却标记早停的伪造状态。"""

    features, targets, ids = _toy_data(13)
    trainer, val_x, val_y = _new_trainer(tmp_path / "source")
    trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "source" / "last.pt")
    payload["stopped_early"] = True
    path = tmp_path / "impossible_early_stop.pt"
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="早停"):
        XFTrainer.resume(
            path,
            train_features=features,
            train_targets=targets,
            train_ids=ids,
            checkpoint_dir=tmp_path,
            device="cpu",
        )


@pytest.mark.parametrize("bad_kind", ["corrupt", "missing_key"])
def test_checkpoint_load_rejects_corruption_or_missing_state(
    tmp_path: Path, bad_kind: str
) -> None:
    """捕获损坏或半截 checkpoint 被当作可恢复产物。"""

    bad_path = tmp_path / "bad.pt"
    if bad_kind == "corrupt":
        bad_path.write_bytes(b"not a torch checkpoint")
    else:
        torch.save({"format_version": 1, "model": {}}, bad_path)
    with pytest.raises(CheckpointError):
        load_checkpoint_payload(bad_path)


def test_checkpoint_load_rejects_nonfinite_model_tensor(tmp_path: Path) -> None:
    """捕获结构完整但权重已变为 NaN 的损坏产物。"""

    trainer, val_x, val_y = _new_trainer(tmp_path / "source", validation_interval_steps=1)
    trainer.train_steps(1, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "source" / "last.pt")
    first_name = next(iter(payload["model"]))
    corrupted = payload["model"][first_name].clone()
    corrupted.reshape(-1)[0] = torch.nan
    payload["model"][first_name] = corrupted
    bad_path = tmp_path / "nonfinite.pt"
    torch.save(payload, bad_path)

    with pytest.raises(CheckpointError, match="非有限"):
        load_checkpoint_payload(bad_path)


def test_checkpoint_save_rejects_nonfinite_model_tensor(tmp_path: Path) -> None:
    """捕获训练器把非有限权重写成看似可恢复的 last。"""

    trainer, val_x, val_y = _new_trainer(tmp_path, validation_interval_steps=1)
    with torch.no_grad():
        next(trainer.model.parameters()).reshape(-1)[0] = torch.nan
    with pytest.raises(CheckpointError, match="非有限"):
        trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)
    assert not (tmp_path / "last.pt").exists()


def test_checkpoint_load_rejects_nan_optimizer_python_float(tmp_path: Path) -> None:
    """捕获递归校验只检查 tensor，却放过 param_groups 的 NaN LR。"""

    trainer, val_x, val_y = _new_trainer(tmp_path / "source")
    trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "source" / "last.pt")
    payload["optimizer"]["param_groups"][0]["lr"] = float("nan")
    path = tmp_path / "nan_optimizer.pt"
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="optimizer.*非有限"):
        load_checkpoint_payload(path)


def test_checkpoint_load_rejects_nan_scheduler_state(tmp_path: Path) -> None:
    """捕获 ReduceLROnPlateau.best=NaN 绕过 checkpoint 边界。"""

    trainer, val_x, val_y = _new_trainer(tmp_path / "source")
    trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "source" / "last.pt")
    payload["scheduler"]["best"] = float("nan")
    path = tmp_path / "nan_scheduler.pt"
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="scheduler.*非有限"):
        load_checkpoint_payload(path)


def test_checkpoint_load_rejects_negative_infinite_best_loss(tmp_path: Path) -> None:
    """捕获用 isinf 笼统接受不可能的 -inf 最佳损失。"""

    trainer, val_x, val_y = _new_trainer(tmp_path / "source")
    trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "source" / "last.pt")
    payload["best_validation_loss"] = float("-inf")
    path = tmp_path / "negative_infinite_best.pt"
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="best_validation_loss"):
        load_checkpoint_payload(path)


def test_checkpoint_load_rejects_positive_infinite_best_after_validation(
    tmp_path: Path,
) -> None:
    """捕获已完成验证的 checkpoint 伪装成尚无 best 的初始状态。"""

    trainer, val_x, val_y = _new_trainer(
        tmp_path / "source", validation_interval_steps=1
    )
    trainer.train_steps(1, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "source" / "last.pt")
    payload["best_validation_loss"] = float("inf")
    path = tmp_path / "post_validation_infinite_best.pt"
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="best_validation_loss"):
        load_checkpoint_payload(path)


def test_checkpoint_allows_consistent_prevalidation_positive_infinity_sentinel(
    tmp_path: Path,
) -> None:
    """保护零次验证时 best/scheduler.best 的合法 +inf 初始哨兵。"""

    trainer, val_x, val_y = _new_trainer(tmp_path)
    trainer.train_steps(0, validation_features=val_x, validation_targets=val_y)
    payload = load_checkpoint_payload(tmp_path / "last.pt")

    assert payload["validation_evaluations"] == 0
    assert payload["evaluations_without_improvement"] == 0
    assert payload["best_validation_loss"] == float("inf")
    assert payload["scheduler"]["best"] == float("inf")


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda x, y: (x[:, :9], y), "10"),
        (lambda x, y: (x, y[:, :6]), "7"),
        (_with_nan_feature, "非有限"),
    ],
)
def test_training_data_shape_and_finite_contract_fail_hard(mutator, match: str) -> None:
    """捕获错形状或 NaN 被静默删行后继续训练。"""

    features, targets, ids = _toy_data(8)
    bad_x, bad_y = mutator(features, targets)
    with pytest.raises(TrainingDataError, match=match):
        fit_training_preprocessor(
            bad_x,
            bad_y,
            ids,
            get_candidate("xf_v1_baseline"),
        )


def test_resume_rejects_dataset_shape_mismatch(tmp_path: Path) -> None:
    """捕获 checkpoint 被恢复到不同 10→7 数据契约。"""

    trainer, val_x, val_y = _new_trainer(tmp_path, validation_interval_steps=1)
    trainer.train_steps(1, validation_features=val_x, validation_targets=val_y)
    features, targets, ids = _toy_data(13)
    with pytest.raises((CheckpointError, TrainingDataError), match="10"):
        XFTrainer.resume(
            tmp_path / "last.pt",
            train_features=np.column_stack([features, np.ones(len(features))]),
            train_targets=targets,
            train_ids=ids,
            checkpoint_dir=tmp_path,
            device="cpu",
        )


def test_events_include_training_health_and_cpu_mfu_fields(tmp_path: Path) -> None:
    """捕获训练无损失/梯度/速度/内存/验证进度可观测性。"""

    events: list[dict] = []
    trainer, val_x, val_y = _new_trainer(
        tmp_path,
        validation_interval_steps=1,
        event_sink=events.append,
    )
    trainer.train_steps(1, validation_features=val_x, validation_targets=val_y)
    required = {
        "kind",
        "loss",
        "grad_norm",
        "lr",
        "global_step",
        "step_time",
        "samples_s",
        "memory",
        "validation_progress",
        "mfu",
    }

    assert {"train_step", "validation_progress", "validation_end"} <= {
        event["kind"] for event in events
    }
    for event in events:
        assert required <= event.keys()
        assert event["mfu"] == "not_applicable"
        assert event["memory"] >= 0
        assert event["validation_progress"]["completed"] <= event["validation_progress"]["total"]


def test_core_modules_do_not_import_tests_or_validation_toolkit() -> None:
    """捕获生产训练代码反向依赖 tests/校验工具。"""

    root = Path(__file__).resolve().parents[1]
    for filename in ("candidates.py", "trainer.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert all("tests" not in name and "validation" not in name for name in imported)
