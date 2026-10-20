"""XF KAN 的确定性、可续训步进式训练组件。

训练器只消费锁定的 10 个原始输入和 7 个直接目标。scaler
仅在当前训练分区拟合；验证只在优化步 cadence 命中时完整执行，
ReduceLROnPlateau 和早停计数也只在该边界更新。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from .candidates import (
    OUTPUT_WEIGHTS,
    CandidateConfig,
    build_candidate_model,
    get_candidate,
)

__all__ = [
    "CheckpointError",
    "FittedTrainingScaler",
    "TargetTransform",
    "TrainingDataError",
    "TrainingPreprocessor",
    "ValidationSnapshot",
    "XFTrainer",
    "build_candidate_loss",
    "compute_training_data_fingerprint",
    "fit_training_preprocessor",
    "load_checkpoint_payload",
    "set_deterministic_seed",
]


_CHECKPOINT_FORMAT_VERSION = 2
_REQUIRED_CHECKPOINT_KEYS = {
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


class TrainingDataError(ValueError):
    """训练数据、缩放或目标变换违反契约。"""


class CheckpointError(RuntimeError):
    """checkpoint 不完整、损坏或与当前训练不一致。"""


def set_deterministic_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 的可见随机状态。"""

    normalized = int(seed)
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)
    torch.use_deterministic_algorithms(True)


def _finite_matrix(
    values: Any,
    *,
    label: str,
    expected_columns: int,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
        raise TrainingDataError(
            f"{label}必须是 [N, {expected_columns}] 二维数组，实际 {matrix.shape}"
        )
    if matrix.shape[0] == 0:
        raise TrainingDataError(f"{label}不得为空")
    if not np.isfinite(matrix).all():
        raise TrainingDataError(f"{label}存在非有限值")
    return matrix


def compute_training_data_fingerprint(
    train_features: Any,
    train_targets: Any,
    train_ids: Sequence[int],
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
) -> str:
    """对训练数值、ID 顺序和字段契约生成确定性 SHA-256。"""

    normalized_inputs = tuple(str(value) for value in input_columns)
    normalized_outputs = tuple(str(value) for value in output_columns)
    features = _finite_matrix(
        train_features,
        label="指纹训练输入",
        expected_columns=len(normalized_inputs),
    )
    targets = _finite_matrix(
        train_targets,
        label="指纹训练目标",
        expected_columns=len(normalized_outputs),
    )
    ids = tuple(int(value) for value in train_ids)
    if len(features) != len(targets) or len(features) != len(ids):
        raise TrainingDataError("指纹的输入、目标与 ID 样本数不一致")
    if len(set(ids)) != len(ids):
        raise TrainingDataError("指纹的训练 ID 重复")

    digest = hashlib.sha256()
    contract = json.dumps(
        {
            "format": "xf-training-data-v1",
            "input_columns": normalized_inputs,
            "output_columns": normalized_outputs,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(contract).to_bytes(8, byteorder="little", signed=False))
    digest.update(contract)
    for label, values, dtype in (
        (b"ids", ids, "<i8"),
        (b"features", features, "<f8"),
        (b"targets", targets, "<f8"),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(len(label).to_bytes(2, byteorder="little", signed=False))
        digest.update(label)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class TargetTransform:
    """按目标名称定位、可序列化的列级 log1p 变换。"""

    def __init__(
        self,
        *,
        output_columns: Sequence[str],
        log1p_targets: Sequence[str],
    ) -> None:
        self.output_columns = tuple(str(value) for value in output_columns)
        self.log1p_targets = tuple(str(value) for value in log1p_targets)
        if len(set(self.output_columns)) != len(self.output_columns):
            raise TrainingDataError("目标字段名重复")
        unknown = set(self.log1p_targets) - set(self.output_columns)
        if unknown:
            raise TrainingDataError(f"log1p 目标不在输出契约中: {sorted(unknown)}")
        self.log_indices = tuple(
            self.output_columns.index(name) for name in self.log1p_targets
        )

    def transform(self, values: Any) -> np.ndarray:
        matrix = _finite_matrix(
            values,
            label="目标变换输入",
            expected_columns=len(self.output_columns),
        ).copy()
        if self.log_indices:
            selected = matrix[:, self.log_indices]
            if np.any(selected < 0.0):
                raise TrainingDataError("log1p 目标含有物理负值")
            matrix[:, self.log_indices] = np.log1p(selected)
        if not np.isfinite(matrix).all():
            raise TrainingDataError("log1p 目标变换产生非有限值")
        return matrix

    def inverse_transform(self, values: Any) -> np.ndarray:
        matrix = _finite_matrix(
            values,
            label="目标反变换输入",
            expected_columns=len(self.output_columns),
        ).copy()
        if self.log_indices:
            with np.errstate(over="ignore", invalid="ignore"):
                matrix[:, self.log_indices] = np.expm1(matrix[:, self.log_indices])
        if not np.isfinite(matrix).all():
            raise TrainingDataError("目标反变换产生非有限值")
        return matrix

    def state_dict(self) -> dict[str, Any]:
        return {
            "output_columns": list(self.output_columns),
            "log1p_targets": list(self.log1p_targets),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TargetTransform":
        try:
            return cls(
                output_columns=state["output_columns"],
                log1p_targets=state["log1p_targets"],
            )
        except (KeyError, TypeError, TrainingDataError) as exc:
            raise CheckpointError("checkpoint 的 target transform 无效") from exc


class FittedTrainingScaler:
    """MinMax[-1,1] scaler 及其唯一拟合分区 ID 证据。"""

    def __init__(self, feature_names: Sequence[str], feature_range=(-1.0, 1.0)):
        self.feature_names = tuple(str(value) for value in feature_names)
        self.feature_range = tuple(float(value) for value in feature_range)
        if self.feature_range != (-1.0, 1.0):
            raise TrainingDataError("XF scaler 的 feature_range 必须为 (-1, 1)")
        self.fit_ids: tuple[int, ...] = ()
        self._scaler = MinMaxScaler(feature_range=self.feature_range)

    def fit(self, values: Any, fit_ids: Sequence[int]) -> "FittedTrainingScaler":
        matrix = _finite_matrix(
            values,
            label="scaler 拟合数据",
            expected_columns=len(self.feature_names),
        )
        normalized_ids = tuple(int(value) for value in fit_ids)
        if len(normalized_ids) != len(matrix):
            raise TrainingDataError("scaler 拟合 ID 数与样本数不一致")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise TrainingDataError("scaler 拟合 ID 重复")
        self._scaler.fit(matrix)
        self.fit_ids = normalized_ids
        return self

    @property
    def data_min_(self) -> np.ndarray:
        return self._scaler.data_min_

    @property
    def data_max_(self) -> np.ndarray:
        return self._scaler.data_max_

    def transform(self, values: Any) -> np.ndarray:
        matrix = _finite_matrix(
            values,
            label="scaler 转换输入",
            expected_columns=len(self.feature_names),
        )
        transformed = np.asarray(self._scaler.transform(matrix), dtype=np.float64)
        if not np.isfinite(transformed).all():
            raise TrainingDataError("scaler 转换产生非有限值")
        return transformed

    def inverse_transform(self, values: Any) -> np.ndarray:
        matrix = _finite_matrix(
            values,
            label="scaler 反变换输入",
            expected_columns=len(self.feature_names),
        )
        restored = np.asarray(self._scaler.inverse_transform(matrix), dtype=np.float64)
        if not np.isfinite(restored).all():
            raise TrainingDataError("scaler 反变换产生非有限值")
        return restored

    def state_dict(self) -> dict[str, Any]:
        if not self.fit_ids:
            raise TrainingDataError("不得序列化未拟合 scaler")
        attributes = {
            name: copy.deepcopy(getattr(self._scaler, name))
            for name in (
                "min_",
                "scale_",
                "data_min_",
                "data_max_",
                "data_range_",
                "n_features_in_",
                "n_samples_seen_",
            )
        }
        return {
            "feature_names": list(self.feature_names),
            "feature_range": list(self.feature_range),
            "fit_ids": list(self.fit_ids),
            "attributes": attributes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FittedTrainingScaler":
        try:
            scaler = cls(state["feature_names"], state["feature_range"])
            fit_ids = tuple(int(value) for value in state["fit_ids"])
            attributes = state["attributes"]
            if not fit_ids:
                raise ValueError("fit_ids 为空")
            for name in (
                "min_",
                "scale_",
                "data_min_",
                "data_max_",
                "data_range_",
                "n_features_in_",
                "n_samples_seen_",
            ):
                setattr(scaler._scaler, name, copy.deepcopy(attributes[name]))
            scaler.fit_ids = fit_ids
            if int(scaler._scaler.n_features_in_) != len(scaler.feature_names):
                raise ValueError("scaler 特征数不一致")
            for values in (
                scaler._scaler.min_,
                scaler._scaler.scale_,
                scaler._scaler.data_min_,
                scaler._scaler.data_max_,
                scaler._scaler.data_range_,
            ):
                if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                    raise ValueError("scaler 状态非有限")
            return scaler
        except (KeyError, TypeError, ValueError, TrainingDataError) as exc:
            raise CheckpointError("checkpoint 的 scaler 状态无效") from exc


class TrainingPreprocessor:
    """两个训练分区 scaler 与目标变换的固定组合。"""

    def __init__(
        self,
        *,
        x_scaler: FittedTrainingScaler,
        y_scaler: FittedTrainingScaler,
        target_transform: TargetTransform,
    ) -> None:
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.target_transform = target_transform

    def transform_features(self, values: Any) -> np.ndarray:
        return self.x_scaler.transform(values)

    def transform_targets(self, values: Any) -> np.ndarray:
        return self.y_scaler.transform(self.target_transform.transform(values))

    def inverse_targets(self, values: Any) -> np.ndarray:
        transformed = self.y_scaler.inverse_transform(values)
        return self.target_transform.inverse_transform(transformed)


class ValidationSnapshot:
    """callback 只读 facade；其可变成员全部是 live trainer 的独立副本。"""

    __slots__ = ("_model", "_preprocessor", "_device", "_global_step")

    def __init__(
        self,
        *,
        model: nn.Module,
        preprocessor: TrainingPreprocessor,
        device: torch.device,
        global_step: int,
    ) -> None:
        self._model = copy.deepcopy(model)
        self._preprocessor = copy.deepcopy(preprocessor)
        self._device = torch.device(device)
        self._global_step = int(global_step)

    @property
    def model(self) -> nn.Module:
        """独立模型副本；修改它不影响 live trainer。"""

        return self._model

    @property
    def preprocessor(self) -> TrainingPreprocessor:
        """独立预处理副本；修改 scaler 不影响 live trainer。"""

        return self._preprocessor

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def global_step(self) -> int:
        return self._global_step

    def predict_model_space(self, features: Any) -> np.ndarray:
        """只在快照 clone 上执行有限 10→7 model-space 推理。"""

        values = _finite_matrix(
            features,
            label="validation snapshot 输入",
            expected_columns=10,
        )
        self._model.eval()
        with torch.no_grad():
            output = self._model(
                torch.from_numpy(values.astype(np.float32)).to(self._device)
            )
        result = output.detach().cpu().numpy().astype(np.float64)
        if result.shape != (len(values), 7) or not np.isfinite(result).all():
            raise TrainingDataError("validation snapshot 预测形状错误或非有限")
        return result


def fit_training_preprocessor(
    train_features: Any,
    train_targets: Any,
    train_ids: Sequence[int],
    candidate: CandidateConfig,
) -> TrainingPreprocessor:
    """仅使用传入的训练 ID 拟合输入/目标 scaler。"""

    features = _finite_matrix(
        train_features,
        label="训练输入",
        expected_columns=len(candidate.input_columns),
    )
    targets = _finite_matrix(
        train_targets,
        label="训练目标",
        expected_columns=len(candidate.output_columns),
    )
    if len(features) != len(targets):
        raise TrainingDataError("训练输入与目标样本数不一致")
    ids = tuple(int(value) for value in train_ids)
    if len(ids) != len(features):
        raise TrainingDataError("训练 ID 数与样本数不一致")
    if len(set(ids)) != len(ids):
        raise TrainingDataError("训练 ID 重复")
    target_transform = TargetTransform(
        output_columns=candidate.output_columns,
        log1p_targets=candidate.log1p_targets,
    )
    transformed_targets = target_transform.transform(targets)
    return TrainingPreprocessor(
        x_scaler=FittedTrainingScaler(
            candidate.input_columns, candidate.feature_range
        ).fit(features, ids),
        y_scaler=FittedTrainingScaler(
            candidate.output_columns, candidate.feature_range
        ).fit(transformed_targets, ids),
        target_transform=target_transform,
    )


class _WeightedLoss(nn.Module):
    def __init__(self, *, huber_delta: float | None = None) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer("weights", torch.tensor(OUTPUT_WEIGHTS, dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise TrainingDataError(
                f"损失输入必须是同形状二维数组，实际 {tuple(prediction.shape)}/{tuple(target.shape)}"
            )
        if prediction.shape[1] != len(OUTPUT_WEIGHTS):
            raise TrainingDataError("损失输出维度必须为 7")
        error = prediction - target
        if self.huber_delta is None:
            elementwise = error.square()
        else:
            delta = float(self.huber_delta)
            absolute = error.abs()
            elementwise = torch.where(
                absolute <= delta,
                0.5 * error.square(),
                delta * (absolute - 0.5 * delta),
            )
        weights = self.weights.to(device=elementwise.device, dtype=elementwise.dtype)
        return (elementwise * weights.view(1, -1)).sum(dim=1).div(weights.sum()).mean()


def build_candidate_loss(candidate: CandidateConfig) -> nn.Module:
    """在缩放后目标空间构建候选指定损失。"""

    if candidate.loss == "mse":
        return nn.MSELoss()
    if candidate.loss == "weighted_mse":
        return _WeightedLoss()
    if candidate.loss == "weighted_huber":
        if candidate.huber_delta != 1.0:
            raise ValueError("XF robust 候选的 Huber delta 必须为 1.0")
        return _WeightedLoss(huber_delta=candidate.huber_delta)
    raise ValueError(f"未知 XF 候选损失: {candidate.loss}")


def _memory_bytes() -> int:
    """返回 Python 已跟踪的峰值内存，仅作训练健康日志。"""

    if not tracemalloc.is_tracing():
        tracemalloc.start()
    _, peak = tracemalloc.get_traced_memory()
    return max(0, int(peak))


def _capture_global_rng_state() -> dict[str, Any]:
    """callback 边界不得改变后续训练的全局 RNG。"""

    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_global_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _object_graph_equal(
    left: Any,
    right: Any,
    left_to_right: dict[int, int] | None = None,
    right_to_left: dict[int, int] | None = None,
) -> bool:
    """递归比较任意 Python/Torch 对象图，同时校验别名关系。"""

    if type(left) is not type(right):
        return False
    if left is right:
        return True
    if left is None or isinstance(left, (str, bytes, int, float, bool, complex)):
        return bool(left == right)

    left_to_right = {} if left_to_right is None else left_to_right
    right_to_left = {} if right_to_left is None else right_to_left
    left_id = id(left)
    right_id = id(right)
    if left_id in left_to_right:
        return left_to_right[left_id] == right_id
    if right_id in right_to_left:
        return right_to_left[right_id] == left_id
    left_to_right[left_id] = right_id
    right_to_left[right_id] = left_id

    if isinstance(left, torch.Tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.layout == right.layout
            and left.requires_grad == right.requires_grad
            and left.stride() == right.stride()
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.strides == right.strides
            and left.flags.writeable == right.flags.writeable
            and left.flags.aligned == right.flags.aligned
            and np.array_equal(left, right)
        )
    if isinstance(left, torch.Generator):
        return left.device == right.device and torch.equal(
            left.get_state(),
            right.get_state(),
        )

    if isinstance(left, Mapping):
        if len(left) != len(right):
            # PyTorch 内部一些仅运行时存在、deepcopy 不保留的标记不应触发
            # callback 事务误报。目前已知的是 Adam 的
            # `_warned_capturable_if_run_uncaptured`；如后续出现同类
            # 非训练状态标记，可继续加入此集合。
            _IGNORABLE_EXTRA_RIGHT_KEYS = {
                "_warned_capturable_if_run_uncaptured",
            }
            if len(left) > len(right) or not {
                key for key in right if key not in left
            } <= _IGNORABLE_EXTRA_RIGHT_KEYS:
                return False
        right_items = list(right.items())
        if len(left) != len(right):
            right_items = [
                item for item in right_items
                if item[0] not in _IGNORABLE_EXTRA_RIGHT_KEYS or item[0] in left
            ]
        used_indices: set[int] = set()
        for left_key, left_value in left.items():
            matched = False
            for index, (right_key, right_value) in enumerate(right_items):
                if index in used_indices:
                    continue
                candidate_left_map = dict(left_to_right)
                candidate_right_map = dict(right_to_left)
                if not _object_graph_equal(
                    left_key,
                    right_key,
                    candidate_left_map,
                    candidate_right_map,
                ):
                    continue
                if not _object_graph_equal(
                    left_value,
                    right_value,
                    candidate_left_map,
                    candidate_right_map,
                ):
                    continue
                left_to_right.clear()
                left_to_right.update(candidate_left_map)
                right_to_left.clear()
                right_to_left.update(candidate_right_map)
                used_indices.add(index)
                matched = True
                break
            if not matched:
                return False
        return True
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _object_graph_equal(
                left_item,
                right_item,
                left_to_right,
                right_to_left,
            )
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, (set, frozenset)):
        try:
            return left == right
        except Exception:
            return sorted(map(repr, left)) == sorted(map(repr, right))
    if hasattr(left, "__dict__"):
        return _object_graph_equal(
            vars(left),
            vars(right),
            left_to_right,
            right_to_left,
        )
    try:
        equal = left == right
        return bool(equal) if not isinstance(equal, torch.Tensor) else False
    except Exception:
        return False


_TORCH_BACKEND_FLAGS = (
    ("cudnn.enabled", torch.backends.cudnn, "enabled"),
    ("cudnn.benchmark", torch.backends.cudnn, "benchmark"),
    ("cudnn.deterministic", torch.backends.cudnn, "deterministic"),
    ("cudnn.allow_tf32", torch.backends.cudnn, "allow_tf32"),
    ("cuda.matmul.allow_tf32", torch.backends.cuda.matmul, "allow_tf32"),
    (
        "cuda.matmul.allow_fp16_reduced_precision_reduction",
        torch.backends.cuda.matmul,
        "allow_fp16_reduced_precision_reduction",
    ),
    (
        "cuda.matmul.allow_bf16_reduced_precision_reduction",
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
    ),
)


@dataclass(frozen=True)
class _TorchExecutionState:
    grad_enabled: bool
    deterministic_enabled: bool
    deterministic_warn_only: bool
    default_dtype: torch.dtype
    float32_matmul_precision: str
    backend_flags: dict[str, Any]

    @classmethod
    def capture(cls) -> "_TorchExecutionState":
        return cls(
            grad_enabled=torch.is_grad_enabled(),
            deterministic_enabled=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            default_dtype=torch.get_default_dtype(),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            backend_flags={
                name: copy.deepcopy(getattr(owner, attribute))
                for name, owner, attribute in _TORCH_BACKEND_FLAGS
                if hasattr(owner, attribute)
            },
        )

    def restore(self) -> None:
        torch.set_grad_enabled(self.grad_enabled)
        torch.use_deterministic_algorithms(
            self.deterministic_enabled,
            warn_only=self.deterministic_warn_only,
        )
        torch.set_default_dtype(self.default_dtype)
        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        for name, owner, attribute in _TORCH_BACKEND_FLAGS:
            if name in self.backend_flags:
                setattr(owner, attribute, self.backend_flags[name])

    def matches_current(self) -> bool:
        return self == _TorchExecutionState.capture()


@dataclass(frozen=True)
class _ExternalValidationArray:
    """记录调用方 NumPy validation 数组，用于检测闭包改写。"""

    label: str
    reference: np.ndarray
    values: np.ndarray
    raw_bytes: bytes
    dtype: np.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    writeable: bool
    aligned: bool
    c_contiguous: bool
    f_contiguous: bool
    owndata: bool

    @classmethod
    def capture(cls, value: Any, *, label: str) -> "_ExternalValidationArray | None":
        if not isinstance(value, np.ndarray):
            return None
        return cls(
            label=label,
            reference=value,
            values=value.copy(order="K"),
            raw_bytes=value.tobytes(order="K"),
            dtype=value.dtype,
            shape=value.shape,
            strides=value.strides,
            writeable=bool(value.flags.writeable),
            aligned=bool(value.flags.aligned),
            c_contiguous=bool(value.flags.c_contiguous),
            f_contiguous=bool(value.flags.f_contiguous),
            owndata=bool(value.flags.owndata),
        )

    def matches_current(self) -> bool:
        reference = self.reference
        return (
            reference.dtype == self.dtype
            and reference.shape == self.shape
            and reference.strides == self.strides
            and bool(reference.flags.writeable) == self.writeable
            and bool(reference.flags.aligned) == self.aligned
            and bool(reference.flags.c_contiguous) == self.c_contiguous
            and bool(reference.flags.f_contiguous) == self.f_contiguous
            and bool(reference.flags.owndata) == self.owndata
            and np.array_equal(reference, self.values)
        )

    def restore_if_changed(self) -> bool:
        reference = self.reference
        if self.matches_current():
            return False
        try:
            if (
                reference.dtype != self.dtype
                or reference.shape != self.shape
                or reference.strides != self.strides
            ):
                # NumPy 2.x 对直接修改 dtype/shape/strides 后无法再用属性赋值
                # 恢复布局；ndarray.__setstate__ 可以原位重建相同 dtype/shape/
                # strides/data，从而继续满足 callback 事务的“完整还原”保证。
                reference.__setstate__(
                    (
                        1,
                        self.shape,
                        self.dtype,
                        False,
                        self.raw_bytes,
                    )
                )
            else:
                if not reference.flags.writeable:
                    reference.setflags(write=True)
                reference[...] = self.values
            reference.setflags(align=self.aligned)
            reference.setflags(write=self.writeable)
        except Exception as exc:
            raise TrainingDataError(
                f"{self.label}被修改且无法完整恢复"
            ) from exc
        if not self.matches_current():
            raise TrainingDataError(
                f"{self.label}被修改，恢复后 dtype/layout/value 仍不一致"
            )
        return True


def _load_raw_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"checkpoint 无法读取: {path}") from exc


def _assert_finite_state(value: Any, *, label: str) -> None:
    """递归检查模型/优化器/scaler 中的浮点数值。"""

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise CheckpointError(f"checkpoint {label} 存在非有限张量")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise CheckpointError(f"checkpoint {label} 存在非有限数组")
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise CheckpointError(f"checkpoint {label} 存在非有限数值")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_state(child, label=label)
        return
    if isinstance(value, (tuple, list)):
        for child in value:
            _assert_finite_state(child, label=label)


def _validate_checkpoint_numerics(payload: Mapping[str, Any]) -> None:
    _assert_finite_state(payload["model"], label="model")
    _assert_finite_state(payload["optimizer"], label="optimizer")
    _assert_finite_state(payload["x_scaler"].get("attributes", {}), label="x_scaler")
    _assert_finite_state(payload["y_scaler"].get("attributes", {}), label="y_scaler")

    scheduler = payload["scheduler"]
    scheduler_regular_state = {
        key: value
        for key, value in scheduler.items()
        if key not in {"best", "mode_worse"}
    }
    _assert_finite_state(scheduler_regular_state, label="scheduler")
    try:
        scheduler_best = float(scheduler["best"])
        mode_worse = float(scheduler["mode_worse"])
        scheduler_last_epoch = int(scheduler["last_epoch"])
        best_validation_loss = float(payload["best_validation_loss"])
        validation_evaluations = int(payload["validation_evaluations"])
        evaluations_without_improvement = int(
            payload["evaluations_without_improvement"]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CheckpointError("checkpoint scheduler/验证状态无效") from exc

    if math.isnan(mode_worse) or mode_worse != math.inf:
        raise CheckpointError("checkpoint scheduler 存在非有限或错误 mode_worse")
    if validation_evaluations == 0:
        if best_validation_loss != math.inf:
            raise CheckpointError(
                "checkpoint best_validation_loss 在零次验证时必须为 +inf"
            )
        if scheduler_best != math.inf:
            if not math.isfinite(scheduler_best):
                raise CheckpointError("checkpoint scheduler 存在非有限 best")
            raise CheckpointError("checkpoint scheduler.best 与零次验证状态不一致")
        if evaluations_without_improvement != 0 or scheduler_last_epoch != 0:
            raise CheckpointError("checkpoint 零次验证的计数器状态不一致")
        return

    if not math.isfinite(best_validation_loss) or best_validation_loss < 0.0:
        raise CheckpointError(
            "checkpoint best_validation_loss 在已验证状态必须为非负有限值"
        )
    if not math.isfinite(scheduler_best) or scheduler_best < 0.0:
        raise CheckpointError("checkpoint scheduler 存在非有限或负 best")
    if scheduler_best != best_validation_loss:
        raise CheckpointError("checkpoint scheduler.best 与 best_validation_loss 不一致")
    if scheduler_last_epoch != validation_evaluations:
        raise CheckpointError("checkpoint scheduler.last_epoch 与验证次数不一致")
    if evaluations_without_improvement > validation_evaluations:
        raise CheckpointError("checkpoint 无改善计数超过验证次数")


def _validate_checkpoint_progress(payload: Mapping[str, Any]) -> None:
    """校验批次游标、cadence、耐心和早停的联合状态。"""

    candidate = payload["candidate"]
    try:
        candidate_seed = int(candidate["seed"])
        batch_size = int(candidate["batch_size"])
        x_fit_ids = tuple(int(value) for value in payload["x_scaler"]["fit_ids"])
        y_fit_ids = tuple(int(value) for value in payload["y_scaler"]["fit_ids"])
        global_step = int(payload["global_step"])
        epoch = int(payload["epoch"])
        batch_cursor = int(payload["batch_cursor"])
        seed = int(payload["seed"])
        validation_interval = int(payload["validation_interval_steps"])
        validation_evaluations = int(payload["validation_evaluations"])
        patience_evaluations = int(payload["patience_evaluations"])
        scheduler_patience = int(payload["scheduler_patience"])
        no_improvement = int(payload["evaluations_without_improvement"])
        scheduler_state_patience = int(payload["scheduler"]["patience"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CheckpointError("checkpoint 训练进度状态无效") from exc

    if seed != candidate_seed:
        raise CheckpointError("checkpoint seed 与候选固定种子不一致")
    if not x_fit_ids or x_fit_ids != y_fit_ids:
        raise CheckpointError("checkpoint x/y scaler 拟合 ID 不一致")
    if batch_size <= 0:
        raise CheckpointError("checkpoint 候选 batch_size 必须 > 0")
    batches_per_epoch = math.ceil(len(x_fit_ids) / batch_size)
    if batch_cursor >= batches_per_epoch:
        raise CheckpointError("checkpoint batch_cursor 游标越界")
    if global_step != epoch * batches_per_epoch + batch_cursor:
        raise CheckpointError("checkpoint global_step/epoch/batch_cursor 步进状态不一致")

    if validation_interval <= 0 or patience_evaluations <= 0 or scheduler_patience < 0:
        raise CheckpointError("checkpoint cadence/耐心参数无效")
    expected_evaluations = global_step // validation_interval
    if validation_evaluations != expected_evaluations:
        raise CheckpointError("checkpoint 验证次数与 global_step cadence 不一致")
    if scheduler_state_patience != scheduler_patience:
        raise CheckpointError("checkpoint scheduler patience 状态不一致")
    scheduler = payload["scheduler"]
    if (
        scheduler.get("mode") != "min"
        or scheduler.get("threshold_mode") != "rel"
        or float(scheduler.get("threshold", math.nan)) != 0.0
    ):
        raise CheckpointError("checkpoint scheduler improvement 语义不一致")

    stopped_early = payload["stopped_early"]
    if type(stopped_early) is not bool:
        raise CheckpointError("checkpoint stopped_early 必须是布尔值")
    if stopped_early:
        if validation_evaluations == 0 or no_improvement < patience_evaluations:
            raise CheckpointError("checkpoint 早停标志与验证耐心状态不一致")
    elif no_improvement >= patience_evaluations:
        raise CheckpointError("checkpoint 未早停标志与验证耐心状态不一致")


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """读取并完整校验 XF trainer checkpoint 外壳。"""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointError(f"checkpoint 不存在: {checkpoint_path}")
    payload = _load_raw_checkpoint(checkpoint_path)
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint 顶层必须是字典")
    missing = _REQUIRED_CHECKPOINT_KEYS - set(payload)
    if missing:
        raise CheckpointError(f"checkpoint 缺少必需状态: {sorted(missing)}")
    if payload["format_version"] != _CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError("checkpoint format_version 不受支持")
    if not isinstance(payload["candidate"], dict) or "name" not in payload["candidate"]:
        raise CheckpointError("checkpoint 候选配置无效")
    for key in ("model", "optimizer", "scheduler", "x_scaler", "y_scaler", "target_transform"):
        if not isinstance(payload[key], dict):
            raise CheckpointError(f"checkpoint {key} 状态无效")
    for key in (
        "global_step",
        "epoch",
        "batch_cursor",
        "seed",
        "validation_evaluations",
        "evaluations_without_improvement",
        "validation_interval_steps",
        "patience_evaluations",
        "scheduler_patience",
    ):
        value = payload[key]
        if not isinstance(value, int) or value < 0:
            raise CheckpointError(f"checkpoint {key} 必须是非负整数")
    fingerprint = payload["training_data_fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise CheckpointError("checkpoint 训练数据指纹无效")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise CheckpointError("checkpoint 训练数据指纹无效") from exc
    if not isinstance(payload["batch_order"], torch.Tensor):
        raise CheckpointError("checkpoint batch_order 无效")
    _validate_checkpoint_numerics(payload)
    _validate_checkpoint_progress(payload)
    return payload


class XFTrainer:
    """以优化步为单位、可严格续训的 CPU/GPU 训练器。"""

    def __init__(
        self,
        *,
        candidate: CandidateConfig,
        train_features: Any,
        train_targets: Any,
        train_ids: Sequence[int],
        checkpoint_dir: str | Path,
        device: str = "cpu",
        validation_interval_steps: int | None = None,
        patience_evaluations: int | None = None,
        scheduler_patience: int = 5,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        set_deterministic_seed(candidate.seed)
        self.candidate = candidate
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingDataError("CUDA 设备不可用")
        self.validation_interval_steps = int(
            validation_interval_steps
            if validation_interval_steps is not None
            else candidate.validation_interval_steps
        )
        if self.validation_interval_steps <= 0:
            raise TrainingDataError("validation_interval_steps 必须 > 0")
        self.patience_evaluations = int(
            patience_evaluations
            if patience_evaluations is not None
            else candidate.patience_evaluations
        )
        if self.patience_evaluations <= 0:
            raise TrainingDataError("patience_evaluations 必须 > 0")
        if int(scheduler_patience) < 0:
            raise TrainingDataError("scheduler_patience 不得为负")
        self.scheduler_patience = int(scheduler_patience)
        self.event_sink = event_sink
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._train_features = _finite_matrix(
            train_features,
            label="训练输入",
            expected_columns=len(candidate.input_columns),
        )
        self._train_targets = _finite_matrix(
            train_targets,
            label="训练目标",
            expected_columns=len(candidate.output_columns),
        )
        if len(self._train_features) != len(self._train_targets):
            raise TrainingDataError("训练输入与目标样本数不一致")
        self._train_ids = tuple(int(value) for value in train_ids)
        self.preprocessor = fit_training_preprocessor(
            self._train_features,
            self._train_targets,
            self._train_ids,
            candidate,
        )
        self.training_data_fingerprint = compute_training_data_fingerprint(
            self._train_features,
            self._train_targets,
            self._train_ids,
            input_columns=candidate.input_columns,
            output_columns=candidate.output_columns,
        )
        self._train_x_scaled = torch.from_numpy(
            self.preprocessor.transform_features(self._train_features).astype(np.float32)
        )
        self._train_y_scaled = torch.from_numpy(
            self.preprocessor.transform_targets(self._train_targets).astype(np.float32)
        )

        self.model = build_candidate_model(candidate).to(self.device)
        self.loss_fn = build_candidate_loss(candidate).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=candidate.learning_rate,
            weight_decay=candidate.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=self.scheduler_patience,
            threshold=0.0,
            threshold_mode="rel",
        )
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(candidate.seed)
        self.epoch = 0
        self.batch_cursor = 0
        self.global_step = 0
        self.validation_evaluations = 0
        self.evaluations_without_improvement = 0
        self.best_validation_loss = math.inf
        self.stopped_early = False
        self.batch_index_history: list[np.ndarray] = []
        self._batch_order = self._new_batch_order()
        self._train_steps_active = False

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _new_batch_order(self) -> torch.Tensor:
        return torch.randperm(len(self._train_features), generator=self.generator)

    def peek_next_batch_indices(self) -> np.ndarray:
        start = self.batch_cursor * self.candidate.batch_size
        stop = min(start + self.candidate.batch_size, len(self._batch_order))
        return self._batch_order[start:stop].cpu().numpy().copy()

    def _advance_batch(self) -> np.ndarray:
        indices = self.peek_next_batch_indices()
        if len(indices) == 0:
            raise TrainingDataError("确定性批次游标越界")
        self.batch_cursor += 1
        batches_per_epoch = math.ceil(len(self._train_features) / self.candidate.batch_size)
        if self.batch_cursor >= batches_per_epoch:
            self.epoch += 1
            self.batch_cursor = 0
            self._batch_order = self._new_batch_order()
        return indices

    def _event(
        self,
        *,
        kind: str,
        loss: float,
        grad_norm: float,
        step_time: float,
        samples_s: float,
        validation_completed: int,
        validation_total: int,
    ) -> None:
        if self.event_sink is None:
            return
        event = {
            "kind": kind,
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "lr": self.current_lr,
            "global_step": self.global_step,
            "step_time": float(step_time),
            "samples_s": float(samples_s),
            "memory": _memory_bytes(),
            "validation_progress": {
                "completed": int(validation_completed),
                "total": int(validation_total),
            },
            "mfu": "not_applicable" if self.device.type == "cpu" else "unavailable",
        }
        self.event_sink(event)

    def _train_one_step(self) -> None:
        indices = self._advance_batch()
        self.batch_index_history.append(indices.copy())
        index_tensor = torch.from_numpy(indices).long()
        features = self._train_x_scaled[index_tensor].to(self.device)
        targets = self._train_y_scaled[index_tensor].to(self.device)
        self.model.train()
        start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(features)
        if prediction.shape != targets.shape:
            raise TrainingDataError("模型输出形状与目标不一致")
        loss = self.loss_fn(prediction, targets)
        if not torch.isfinite(loss):
            raise TrainingDataError("训练损失出现非有限值")
        loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.candidate.grad_clip
        )
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if not math.isfinite(grad_norm):
            raise TrainingDataError("梯度范数出现非有限值")
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(time.perf_counter() - start, 1e-12)
        self.global_step += 1
        self._event(
            kind="train_step",
            loss=float(loss.detach().cpu()),
            grad_norm=grad_norm,
            step_time=elapsed,
            samples_s=len(indices) / elapsed,
            validation_completed=0,
            validation_total=0,
        )

    def _capture_callback_training_state(self) -> dict[str, Any]:
        """一次深拷贝完整 trainer 对象图，不维护字段白名单。"""

        return copy.deepcopy(self.__dict__)

    def _callback_training_state_matches(
        self,
        state: Mapping[str, Any],
    ) -> bool:
        try:
            return _object_graph_equal(state, self.__dict__)
        except Exception:
            return False

    def _restore_callback_training_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        """从完整图快照恢复，并用第二份拷贝验证图值与别名。"""

        restored = copy.deepcopy(dict(state))
        self.__dict__.clear()
        self.__dict__.update(restored)
        if not _object_graph_equal(state, self.__dict__):
            raise TrainingDataError("callback 完整对象图恢复后复核失败")

    def _full_validation(
        self,
        features: Any,
        targets: Any,
        validation_callback: Callable[[ValidationSnapshot, Any, Any], None] | None = None,
        validation_sources: Sequence[_ExternalValidationArray | None] = (),
    ) -> float:
        validation_x = self.preprocessor.transform_features(features).astype(np.float32)
        validation_y = self.preprocessor.transform_targets(targets).astype(np.float32)
        if len(validation_x) != len(validation_y):
            raise TrainingDataError("验证输入与目标样本数不一致")
        total_loss = 0.0
        completed = 0
        self.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            for offset in range(0, len(validation_x), self.candidate.batch_size):
                end = min(offset + self.candidate.batch_size, len(validation_x))
                batch_x = torch.from_numpy(validation_x[offset:end]).to(self.device)
                batch_y = torch.from_numpy(validation_y[offset:end]).to(self.device)
                prediction = self.model(batch_x)
                loss = self.loss_fn(prediction, batch_y)
                if not torch.isfinite(loss):
                    raise TrainingDataError("完整验证损失出现非有限值")
                batch_size = end - offset
                total_loss += float(loss.detach().cpu()) * batch_size
                completed = end
                elapsed = max(time.perf_counter() - start, 1e-12)
                self._event(
                    kind="validation_progress",
                    loss=total_loss / completed,
                    grad_norm=0.0,
                    step_time=elapsed,
                    samples_s=completed / elapsed,
                    validation_completed=completed,
                    validation_total=len(validation_x),
                )
        validation_loss = total_loss / len(validation_x)
        if validation_callback is not None:
            rng_state = _capture_global_rng_state()
            torch_execution_state = _TorchExecutionState.capture()
            callback_state: dict[str, Any] | None = None
            snapshot_error: Exception | None = None
            callback_error: Exception | None = None
            global_restore_error: Exception | None = None
            callback_result: Any = None
            try:
                try:
                    callback_state = self._capture_callback_training_state()
                except Exception as exc:
                    snapshot_error = RuntimeError(
                        f"trainer 完整事务快照失败: {exc}"
                    )
                if snapshot_error is None:
                    try:
                        snapshot = ValidationSnapshot(
                            model=self.model,
                            preprocessor=self.preprocessor,
                            device=self.device,
                            global_step=self.global_step,
                        )
                    except Exception as exc:
                        snapshot_error = exc
                if snapshot_error is None:
                    try:
                        callback_result = validation_callback(
                            snapshot,
                            np.asarray(features, dtype=np.float64).copy(),
                            np.asarray(targets, dtype=np.float64).copy(),
                        )
                    except Exception as exc:
                        callback_error = exc
            finally:
                try:
                    _restore_global_rng_state(rng_state)
                    torch_execution_state.restore()
                    if not torch_execution_state.matches_current():
                        raise TrainingDataError("Torch 全局执行状态恢复后不一致")
                except Exception as exc:
                    global_restore_error = exc

            if callback_state is None:
                if global_restore_error is not None:
                    raise TrainingDataError(
                        f"[in_memory/snapshot] [restore] 全局状态恢复失败: "
                        f"{global_restore_error}"
                    ) from global_restore_error
                assert snapshot_error is not None
                raise TrainingDataError(
                    f"[in_memory/snapshot] {snapshot_error}"
                ) from snapshot_error

            if global_restore_error is not None:
                if not self._callback_training_state_matches(callback_state):
                    try:
                        self._restore_callback_training_state(callback_state)
                    except Exception as restore_exc:
                        raise TrainingDataError(
                            "[in_memory/callback] [restore] trainer 与全局状态"
                            f"均恢复失败: {restore_exc}; {global_restore_error}"
                        ) from restore_exc
                raise TrainingDataError(
                    "[in_memory/callback] [restore] 全局 RNG/Torch 执行状态"
                    f"恢复失败: {global_restore_error}"
                ) from global_restore_error

            source_changed = False
            try:
                for source in validation_sources:
                    if source is not None:
                        source_changed = source.restore_if_changed() or source_changed
            except Exception as exc:
                if not self._callback_training_state_matches(callback_state):
                    self._restore_callback_training_state(callback_state)
                raise TrainingDataError(
                    f"[in_memory/callback] [restore] 验证输入无法完整恢复: {exc}"
                ) from exc

            state_changed = not self._callback_training_state_matches(callback_state)
            if state_changed:
                try:
                    self._restore_callback_training_state(callback_state)
                except Exception as exc:
                    raise TrainingDataError(
                        f"[in_memory/callback] [restore] 训练对象图恢复失败: {exc}"
                    ) from exc
            if snapshot_error is not None:
                raise TrainingDataError(
                    f"[in_memory/snapshot] {snapshot_error}"
                ) from snapshot_error
            if state_changed:
                raise TrainingDataError(
                    "[in_memory/callback] [mutation] callback 修改训练状态，"
                    "已完整恢复并复核"
                ) from callback_error
            if source_changed:
                raise TrainingDataError(
                    "[in_memory/callback] [mutation] callback 通过闭包修改"
                    "验证输入，已完整还原"
                ) from callback_error
            if callback_error is not None:
                raise TrainingDataError(
                    f"[in_memory/callback] [execution] {callback_error}"
                ) from callback_error
            if callback_result is not None:
                raise TrainingDataError(
                    "[in_memory/callback] [contract] 共享验证 callback 必须返回 None，"
                    "不得改写 scaled validation loss 训练控制状态"
                )
        self.validation_evaluations += 1
        self.scheduler.step(validation_loss)
        improved = validation_loss < self.best_validation_loss
        if improved:
            self.best_validation_loss = validation_loss
            self.evaluations_without_improvement = 0
        else:
            self.evaluations_without_improvement += 1
        elapsed = max(time.perf_counter() - start, 1e-12)
        self._event(
            kind="validation_end",
            loss=validation_loss,
            grad_norm=0.0,
            step_time=elapsed,
            samples_s=len(validation_x) / elapsed,
            validation_completed=len(validation_x),
            validation_total=len(validation_x),
        )
        if improved:
            self._save_checkpoint(self.checkpoint_dir / "best.pt")
        if self.evaluations_without_improvement >= self.patience_evaluations:
            self.stopped_early = True
        self._save_checkpoint(self.checkpoint_dir / "last.pt")
        return validation_loss

    def train_steps(
        self,
        optimizer_steps: int,
        *,
        validation_features: Any,
        validation_targets: Any,
        validation_callback: Callable[[ValidationSnapshot, Any, Any], None] | None = None,
    ) -> None:
        """最多执行指定优化步，命中 cadence 时才完整验证。"""

        if self._train_steps_active:
            raise TrainingDataError("train_steps 禁止重入/reentrant 调用")
        requested_steps = int(optimizer_steps)
        if requested_steps < 0:
            raise TrainingDataError("optimizer_steps 不得为负")
        feature_source = _ExternalValidationArray.capture(
            validation_features,
            label="validation features",
        )
        target_source = _ExternalValidationArray.capture(
            validation_targets,
            label="validation targets",
        )
        validation_features_internal = _finite_matrix(
            validation_features,
            label="验证输入",
            expected_columns=len(self.candidate.input_columns),
        ).copy()
        validation_targets_internal = _finite_matrix(
            validation_targets,
            label="验证目标",
            expected_columns=len(self.candidate.output_columns),
        ).copy()
        if len(validation_features_internal) != len(validation_targets_internal):
            raise TrainingDataError("验证输入与目标样本数不一致")
        self._train_steps_active = True
        try:
            max_total_steps = self.candidate.max_epochs * math.ceil(
                len(self._train_features) / self.candidate.batch_size
            )
            for _ in range(requested_steps):
                if self.stopped_early or self.global_step >= max_total_steps:
                    break
                self._train_one_step()
                if self.global_step % self.validation_interval_steps == 0:
                    self._full_validation(
                        validation_features_internal,
                        validation_targets_internal,
                        validation_callback,
                        (feature_source, target_source),
                    )
                    if self.stopped_early:
                        break
            self._save_checkpoint(self.checkpoint_dir / "last.pt")
        finally:
            self._train_steps_active = False

    def predict_physical(self, features: Any) -> np.ndarray:
        scaled = self.preprocessor.transform_features(features).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(scaled).to(self.device))
        values = prediction.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise TrainingDataError("模型预测出现非有限值")
        return self.preprocessor.inverse_targets(values)

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "candidate": asdict(self.candidate),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "x_scaler": self.preprocessor.x_scaler.state_dict(),
            "y_scaler": self.preprocessor.y_scaler.state_dict(),
            "target_transform": self.preprocessor.target_transform.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
            "seed": self.candidate.seed,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "generator_state": self.generator.get_state(),
            "batch_order": self._batch_order.clone(),
            "best_validation_loss": self.best_validation_loss,
            "validation_evaluations": self.validation_evaluations,
            "evaluations_without_improvement": self.evaluations_without_improvement,
            "training_data_fingerprint": self.training_data_fingerprint,
            "validation_interval_steps": self.validation_interval_steps,
            "patience_evaluations": self.patience_evaluations,
            "scheduler_patience": self.scheduler_patience,
            "stopped_early": self.stopped_early,
        }

    def _save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            payload = self._checkpoint_payload()
            _validate_checkpoint_numerics(payload)
            _validate_checkpoint_progress(payload)
            torch.save(payload, temporary)
            temporary.replace(path)
        except CheckpointError:
            if temporary.exists():
                temporary.unlink()
            raise
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            raise CheckpointError(f"checkpoint 保存失败: {path}") from exc

    @classmethod
    def resume(
        cls,
        checkpoint_path: str | Path,
        *,
        train_features: Any,
        train_targets: Any,
        train_ids: Sequence[int],
        checkpoint_dir: str | Path,
        device: str = "cpu",
        validation_interval_steps: int | None = None,
        patience_evaluations: int | None = None,
        scheduler_patience: int | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> "XFTrainer":
        payload = load_checkpoint_payload(checkpoint_path)
        candidate_name = str(payload["candidate"]["name"])
        try:
            candidate = get_candidate(candidate_name)
        except ValueError as exc:
            raise CheckpointError("checkpoint 候选名称不受支持") from exc
        expected_candidate = asdict(candidate)
        checkpoint_candidate = payload["candidate"]
        # torch 会原样保存 tuple；若外部改写字段则必须拒绝。
        if checkpoint_candidate != expected_candidate:
            raise CheckpointError("checkpoint 候选配置与固定矩阵不一致")
        saved_controls = {
            "validation_interval_steps": int(payload["validation_interval_steps"]),
            "patience_evaluations": int(payload["patience_evaluations"]),
            "scheduler_patience": int(payload["scheduler_patience"]),
        }
        requested_controls = {
            "validation_interval_steps": validation_interval_steps,
            "patience_evaluations": patience_evaluations,
            "scheduler_patience": scheduler_patience,
        }
        for name, requested in requested_controls.items():
            if requested is not None and int(requested) != saved_controls[name]:
                raise CheckpointError(f"resume {name} 与 checkpoint 不一致")
        trainer = cls(
            candidate=candidate,
            train_features=train_features,
            train_targets=train_targets,
            train_ids=train_ids,
            checkpoint_dir=checkpoint_dir,
            device=device,
            validation_interval_steps=saved_controls["validation_interval_steps"],
            patience_evaluations=saved_controls["patience_evaluations"],
            scheduler_patience=saved_controls["scheduler_patience"],
            event_sink=event_sink,
        )
        try:
            if trainer.training_data_fingerprint != payload["training_data_fingerprint"]:
                raise CheckpointError("checkpoint 训练数据指纹与续训数据不一致")
            x_scaler = FittedTrainingScaler.from_state_dict(payload["x_scaler"])
            y_scaler = FittedTrainingScaler.from_state_dict(payload["y_scaler"])
            target_transform = TargetTransform.from_state_dict(payload["target_transform"])
            if x_scaler.feature_names != candidate.input_columns:
                raise CheckpointError("checkpoint x_scaler 输入字段不一致")
            if y_scaler.feature_names != candidate.output_columns:
                raise CheckpointError("checkpoint y_scaler 输出字段不一致")
            if x_scaler.fit_ids != trainer._train_ids or y_scaler.fit_ids != trainer._train_ids:
                raise CheckpointError("checkpoint scaler 拟合 ID 与续训分区不一致")
            if target_transform.state_dict() != trainer.preprocessor.target_transform.state_dict():
                raise CheckpointError("checkpoint target transform 与候选不一致")
            trainer.preprocessor = TrainingPreprocessor(
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                target_transform=target_transform,
            )
            trainer._train_x_scaled = torch.from_numpy(
                trainer.preprocessor.transform_features(trainer._train_features).astype(np.float32)
            )
            trainer._train_y_scaled = torch.from_numpy(
                trainer.preprocessor.transform_targets(trainer._train_targets).astype(np.float32)
            )
            trainer.model.load_state_dict(payload["model"], strict=True)
            trainer.optimizer.load_state_dict(payload["optimizer"])
            trainer.scheduler.load_state_dict(payload["scheduler"])
            trainer.global_step = int(payload["global_step"])
            trainer.epoch = int(payload["epoch"])
            trainer.batch_cursor = int(payload["batch_cursor"])
            trainer.validation_evaluations = int(payload["validation_evaluations"])
            trainer.evaluations_without_improvement = int(
                payload["evaluations_without_improvement"]
            )
            trainer.best_validation_loss = float(payload["best_validation_loss"])
            trainer.stopped_early = bool(payload["stopped_early"])
            trainer.generator.set_state(payload["generator_state"])
            trainer._batch_order = payload["batch_order"].detach().cpu().clone()
            if trainer._batch_order.ndim != 1 or len(trainer._batch_order) != len(
                trainer._train_features
            ):
                raise CheckpointError("checkpoint batch_order 形状无效")
            if not torch.equal(
                torch.sort(trainer._batch_order).values,
                torch.arange(len(trainer._train_features)),
            ):
                raise CheckpointError("checkpoint batch_order 不是完整排列")
            batches_per_epoch = math.ceil(
                len(trainer._train_features) / trainer.candidate.batch_size
            )
            if trainer.batch_cursor >= batches_per_epoch:
                raise CheckpointError("checkpoint batch_cursor 越界")
            random.setstate(payload["python_rng_state"])
            np.random.set_state(payload["numpy_rng_state"])
            torch.random.set_rng_state(payload["torch_rng_state"])
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError("checkpoint 内部状态无法恢复") from exc
        return trainer
