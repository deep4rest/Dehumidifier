"""固定单 KAN 的小步训练、按 step 评估与 checkpoint。"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import DIRECT_OUTPUT_COLUMNS, get_profile
from .evaluator import EvaluationEvent, EvaluationResult, evaluate_in_memory, evaluate_predictions
from .model import (
    Preprocessor,
    SingleKAN,
    build_metadata,
    build_single_kan,
    fit_preprocessor,
)

__all__ = [
    "CheckpointError",
    "SingleKANTrainer",
    "Trainer",
    "TrainHistory",
    "TrainingDataError",
    "load_checkpoint_payload",
    "set_deterministic_seed",
    "set_seed",
]


class TrainingDataError(ValueError):
    """训练数组、契约或数值状态无效。"""


class CheckpointError(RuntimeError):
    """checkpoint 缺失、损坏或与模型契约不匹配。"""


def set_deterministic_seed(seed: int = 42) -> None:
    """固定 Python/NumPy/PyTorch 的随机源。"""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CI 通常无 CUDA
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _matrix(values: Any, columns: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != columns:
        raise TrainingDataError(f"{label} 必须为 [N,{columns}]，实际 {result.shape}")
    if len(result) == 0:
        raise TrainingDataError(f"{label} 不得为空")
    if not np.isfinite(result).all():
        raise TrainingDataError(f"{label} 存在非有限值")
    return result


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_steps: list[int] = field(default_factory=list)
    validation_metrics: list[dict[str, dict[str, float]]] = field(default_factory=list)
    step_metrics: list[dict[str, float]] = field(default_factory=list)
    global_step: int = 0
    best_step: int = 0
    best_validation_loss: float = math.inf
    no_improvement_count: int = 0
    stop_reason: str = "running"

    @property
    def val_loss(self) -> list[float]:
        return self.validation_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": list(self.train_loss),
            "validation_loss": list(self.validation_loss),
            "validation_steps": list(self.validation_steps),
            "validation_metrics": list(self.validation_metrics),
            "step_metrics": list(self.step_metrics),
            "global_step": int(self.global_step),
            "best_step": int(self.best_step),
            "best_validation_loss": float(self.best_validation_loss) if math.isfinite(self.best_validation_loss) else None,
            "no_improvement_count": int(self.no_improvement_count),
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "TrainHistory":
        return cls(
            train_loss=[float(value) for value in state.get("train_loss", ())],
            validation_loss=[float(value) for value in state.get("validation_loss", ())],
            validation_steps=[int(value) for value in state.get("validation_steps", ())],
            validation_metrics=list(state.get("validation_metrics", ())),
            step_metrics=[
                {str(key): float(value) for key, value in dict(item).items()}
                for item in state.get("step_metrics", ())
            ],
            global_step=int(state.get("global_step", 0)),
            best_step=int(state.get("best_step", 0)),
            best_validation_loss=(math.inf if state.get("best_validation_loss") is None else float(state["best_validation_loss"])),
            no_improvement_count=int(state.get("no_improvement_count", 0)),
            stop_reason=str(state.get("stop_reason", "resumed")),
        )


def load_checkpoint_payload(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointError(f"checkpoint 不存在: {checkpoint}")
    try:
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location=device)
    except Exception as exc:
        raise CheckpointError(f"checkpoint 读取失败: {checkpoint}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload 必须是字典")
    if "model_state_dict" not in payload:
        raise CheckpointError("checkpoint 缺少 model_state_dict")
    return payload


class SingleKANTrainer:
    """两个 profile 共用的单 KAN trainer。

    Trainer 接收真实物理量数组，内部只把输入和六个直接目标送进 scaler、
    loss 和模型。power/Excel 露点由调用方另行保存在 audit 表中。
    """

    def __init__(
        self,
        *,
        profile: str,
        train_features: Any | None = None,
        train_targets: Any | None = None,
        validation_features: Any | None = None,
        validation_targets: Any | None = None,
        checkpoint_dir: str | Path = ".",
        validation_interval_steps: int | None = None,
        checkpoint_interval_steps: int | None = None,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        seed: int = 42,
        preprocessor: Preprocessor | None = None,
        model: SingleKAN | None = None,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
        data_context: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = get_profile(profile)
        self.profile = cfg.name
        self.contract = cfg
        self.device = torch.device(device)
        self.seed = int(seed)
        self.batch_size = max(1, int(batch_size))
        self.validation_interval_steps = int(
            validation_interval_steps or cfg.validation_interval_steps
        )
        self.checkpoint_interval_steps = int(
            checkpoint_interval_steps or cfg.checkpoint_interval_steps
        )
        if self.validation_interval_steps <= 0:
            raise TrainingDataError("验证间隔必须为正整数 step")
        if self.checkpoint_interval_steps <= 0:
            raise TrainingDataError("checkpoint 间隔必须为正整数 step")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.event_sink = event_sink
        self.data_context = json.loads(json.dumps(dict(data_context or {}), sort_keys=True))

        if train_features is None:
            train_features = np.zeros((1, cfg.in_features), dtype=np.float32)
        if train_targets is None:
            train_targets = np.ones((1, cfg.out_features), dtype=np.float32)
        self.train_features = _matrix(train_features, cfg.in_features, "训练输入")
        self.train_targets = _matrix(train_targets, cfg.out_features, "训练目标")
        if len(self.train_features) != len(self.train_targets):
            raise TrainingDataError("训练输入与目标样本数不一致")
        if validation_features is None:
            validation_features = self.train_features[: min(1, len(self.train_features))]
        if validation_targets is None:
            validation_targets = self.train_targets[: min(1, len(self.train_targets))]
        self.validation_features = _matrix(validation_features, cfg.in_features, "验证输入")
        self.validation_targets = _matrix(validation_targets, cfg.out_features, "验证目标")
        if len(self.validation_features) != len(self.validation_targets):
            raise TrainingDataError("验证输入与目标样本数不一致")

        self.preprocessor = preprocessor or fit_preprocessor(
            self.train_features,
            self.train_targets,
            cfg.input_columns,
            cfg.direct_outputs,
        )
        if tuple(self.preprocessor.input_columns) != tuple(cfg.input_columns):
            raise TrainingDataError("输入 scaler 与 profile 契约不匹配")
        if tuple(self.preprocessor.target_columns) != tuple(DIRECT_OUTPUT_COLUMNS):
            raise TrainingDataError("目标 scaler 必须严格为六个直接输出")
        self.model = model or build_single_kan(cfg.name)
        if self.model.profile != cfg.name:
            raise TrainingDataError("模型 profile 与 trainer 不匹配")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        self.history = TrainHistory()
        self._batch_cursor = 0
        self._rng = np.random.default_rng(self.seed)
        self._indices: np.ndarray | None = None
        self._resume_indices: np.ndarray | None = None
        self._resume_batch_cursor: int | None = None
        self._best_checkpoint = self.checkpoint_dir / "best_model.pt"
        self._checkpoint = self.checkpoint_dir / "checkpoint.pt"
        self._resume_needs_data = False
        self._expected_data_fingerprint = self._data_fingerprint()

    def _data_fingerprint(self) -> str:
        """绑定实际数组顺序、数据来源上下文与影响恢复语义的控制参数。"""
        digest = hashlib.sha256()
        controls = {"profile": self.profile, "seed": self.seed, "batch_size": self.batch_size,
                    "validation_interval_steps": self.validation_interval_steps,
                    "checkpoint_interval_steps": self.checkpoint_interval_steps,
                    "data_context": self.data_context, "contract": build_metadata(self.profile)}
        digest.update(json.dumps(controls, sort_keys=True, ensure_ascii=False).encode())
        for array in (self.train_features, self.train_targets, self.validation_features, self.validation_targets):
            values = np.ascontiguousarray(array, dtype=np.float32)
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes())
        return digest.hexdigest()

    def _validate_data_binding(self) -> None:
        if self._resume_needs_data:
            raise CheckpointError("恢复训练必须注入并校验真实训练及验证数据")
        if self._data_fingerprint() != self._expected_data_fingerprint:
            raise CheckpointError("checkpoint 数据/顺序/训练控制指纹不匹配，拒绝续训")

    def _event(self, payload: Mapping[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(dict(payload))

    def _next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        if self._indices is not None and len(self._indices) != len(self.train_features):
            self._indices = None
            self._batch_cursor = 0
        if self._batch_cursor >= len(self.train_features):
            self._batch_cursor = 0
        if self._batch_cursor == 0:
            # A deterministic permutation per epoch avoids accidental fixed-row
            # training while preserving exact replay for a fixed seed.
            self._indices = self._rng.permutation(len(self.train_features))
        stop = min(self._batch_cursor + self.batch_size, len(self.train_features))
        indices = self._indices[self._batch_cursor : stop]
        self._batch_cursor = stop
        return self.train_features[indices], self.train_targets[indices]

    def _scaled_batch(self, features: np.ndarray, targets: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            x = self.preprocessor.transform_features(features)
            y = self.preprocessor.transform_targets(targets)
        except Exception as exc:
            raise TrainingDataError(f"训练缩放失败: {exc}") from exc
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

    def _train_one_step(self) -> float:
        started = time.perf_counter()
        features, targets = self._next_batch()
        x, y = self._scaled_batch(features, targets)
        self.model.train(True)
        prediction = self.model(x)
        if prediction.shape != y.shape:
            raise TrainingDataError(
                f"模型输出形状 {tuple(prediction.shape)} 与目标 {tuple(y.shape)} 不匹配"
            )
        loss = torch.mean((prediction - y) ** 2)
        if not torch.isfinite(loss):
            raise TrainingDataError("训练损失出现 NaN/Inf")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu())
        if not math.isfinite(grad_norm_value):
            raise TrainingDataError("梯度范数出现 NaN/Inf")
        self.optimizer.step()
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise TrainingDataError("训练损失转换后出现 NaN/Inf")
        step_seconds = float(time.perf_counter() - started)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        if not math.isfinite(step_seconds) or not math.isfinite(learning_rate):
            raise TrainingDataError("训练 step 日志出现非有限值")
        self._last_step_metrics = {
            "loss": value,
            "grad_norm": grad_norm_value,
            "learning_rate": learning_rate,
            "step_seconds": step_seconds,
        }
        return value

    def _validate_current(self) -> tuple[float, EvaluationResult]:
        """一次前向同时产出缩放损失和物理指标，保持模型原先的train/eval状态。"""
        x = self.preprocessor.transform_features(self.validation_features)
        y = self.preprocessor.transform_targets(self.validation_targets)
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(torch.from_numpy(x).to(self.device))
                loss = torch.mean((prediction - torch.from_numpy(y).to(self.device)) ** 2)
                values = prediction.detach().cpu().numpy()
        finally:
            self.model.train(was_training)
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise TrainingDataError("验证损失出现 NaN/Inf")
        physical = self.preprocessor.inverse_targets(values)
        evaluation = evaluate_predictions(self.validation_targets, physical, mode="in_memory")
        return value, evaluation

    def _run_steps(self, steps: int, *, early_stopping: bool) -> TrainHistory:
        """训练精确 ``steps`` 个 optimizer step，并按固定 cadence 验证。"""

        self._validate_data_binding()
        steps = int(steps)
        if steps <= 0:
            raise TrainingDataError("训练 step 数必须为正")
        self.history.stop_reason = "running"
        for _ in range(steps):
            loss = self._train_one_step()
            self.history.global_step += 1
            self.history.train_loss.append(loss)
            step_metrics = {
                "global_step": float(self.history.global_step),
                **getattr(
                    self,
                    "_last_step_metrics",
                    {
                        "loss": loss,
                        "grad_norm": 0.0,
                        "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                        "step_seconds": 0.0,
                    },
                ),
            }
            self.history.step_metrics.append(step_metrics)
            self._event(
                {
                    "kind": "train_step",
                    "global_step": self.history.global_step,
                    "loss": loss,
                    "grad_norm": step_metrics["grad_norm"],
                    "learning_rate": step_metrics["learning_rate"],
                    "step_seconds": step_metrics["step_seconds"],
                }
            )
            if self.history.global_step % self.validation_interval_steps == 0:
                val_loss, evaluation = self._validate_current()
                self.history.validation_steps.append(self.history.global_step)
                self.history.validation_loss.append(val_loss)
                self.history.validation_metrics.append(evaluation.metrics)
                if val_loss < self.history.best_validation_loss:
                    self.history.best_validation_loss = val_loss
                    self.history.best_step = self.history.global_step
                    self.history.no_improvement_count = 0
                    self.save_checkpoint(self._best_checkpoint)
                else:
                    self.history.no_improvement_count += 1
                self._event(
                    {
                        "kind": "validation_end",
                        "global_step": self.history.global_step,
                        "val_loss": val_loss,
                        "metrics": evaluation.metrics,
                    }
                )
            if early_stopping and self.history.no_improvement_count >= self.contract.patience:
                self.history.stop_reason = "early_stopping"
                break
            if self.history.global_step % self.checkpoint_interval_steps == 0 and _ != steps - 1:
                self.save_checkpoint()
        if self.history.stop_reason == "running":
            self.history.stop_reason = "completed"
        self.save_checkpoint(self._checkpoint)
        return self.history

    def train_steps(self, steps: int) -> TrainHistory:
        """有界训练共用同一循环，按间隔及结束时保存。"""
        return self._run_steps(steps, early_stopping=False)

    def fit(self, *, epochs: int | None = None) -> TrainHistory:
        """按 profile epoch 上限训练，验证与checkpoint只按step触发。"""
        total_epochs = self.contract.max_epochs if epochs is None else int(epochs)
        if total_epochs <= 0:
            raise TrainingDataError("epochs 必须为正")
        steps_per_epoch = max(1, math.ceil(len(self.train_features) / self.batch_size))
        remaining_steps = max(0, total_epochs * steps_per_epoch - self.history.global_step)
        if remaining_steps:
            return self._run_steps(remaining_steps, early_stopping=True)
        return self.history

    def predict(self, features: Any) -> np.ndarray:
        return self.model.predict(features, self.preprocessor, device=self.device)

    def _capture_rng_state(self) -> dict[str, Any]:
        cuda_state = None
        if torch.cuda.is_available():  # pragma: no cover - CI 通常无 CUDA
            cuda_state = torch.cuda.get_rng_state_all()
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": cuda_state,
            "batch_rng": self._rng.bit_generator.state,
        }

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        try:
            random.setstate(state["python"])
            np.random.set_state(tuple(state["numpy"]))
            torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
            if torch.cuda.is_available() and state.get("torch_cuda") is not None:  # pragma: no cover
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = state["batch_rng"]
        except Exception as exc:
            raise CheckpointError("checkpoint 随机状态无效") from exc

    def _apply_resume_progress(self) -> None:
        if self._resume_indices is None:
            return
        if len(self._resume_indices) != len(self.train_features):
            raise CheckpointError("checkpoint 排列与训练数据长度不匹配")
        if sorted(self._resume_indices.tolist()) != list(range(len(self.train_features))):
            raise CheckpointError("checkpoint batch permutation 与训练数据长度不匹配")
        cursor = int(self._resume_batch_cursor or 0)
        if cursor < 0 or cursor > len(self.train_features):
            raise CheckpointError("checkpoint batch cursor 越界")
        self._indices = self._resume_indices.copy()
        self._batch_cursor = cursor

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        target = self._checkpoint if path is None else Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._validate_data_binding()
        payload = {
            "schema_version": 2,
            "data_fingerprint": self._expected_data_fingerprint,
            "data_context": self.data_context,
            "profile": self.profile,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "preprocessor": self.preprocessor.state_dict(),
            "metadata": build_metadata(self.profile),
            "history": self.history.to_dict(),
            "global_step": self.history.global_step,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "validation_interval_steps": self.validation_interval_steps,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "batch_cursor": int(self._batch_cursor),
            "indices": None if self._indices is None else self._indices.tolist(),
            "rng_state": self._capture_rng_state(),
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(target)
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            raise CheckpointError(f"checkpoint 保存失败: {target}: {exc}") from exc
        self._event(
            {
                "kind": "checkpoint_saved",
                "global_step": int(self.history.global_step),
                "path": str(target),
            }
        )
        return target

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        train_features: Any | None = None,
        train_targets: Any | None = None,
        validation_features: Any | None = None,
        validation_targets: Any | None = None,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
        data_context: Mapping[str, Any] | None = None,
        expected_controls: Mapping[str, Any] | None = None,
    ) -> "SingleKANTrainer":
        payload = load_checkpoint_payload(path, device=device)
        if payload.get("schema_version") != 2 or not payload.get("data_fingerprint"):
            raise CheckpointError("checkpoint 缺少可验证的数据身份；请重新启动训练记录")
        supplied = (train_features, train_targets, validation_features, validation_targets)
        has_data = all(item is not None for item in supplied)
        if any(item is not None for item in supplied) and not has_data:
            raise CheckpointError("恢复时需完整提供训练和验证数据")
        if expected_controls is not None:
            for name, expected in expected_controls.items():
                if payload.get(name) != expected:
                    raise CheckpointError(f"checkpoint 训练控制 {name} 不匹配")
        saved_context = payload.get("data_context", {})
        if data_context is not None and dict(data_context) != saved_context:
            raise CheckpointError("checkpoint 数据来源/模式/分区身份不匹配")
        profile = str(payload.get("profile", ""))
        try:
            cfg = get_profile(profile)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"checkpoint profile 无效: {profile!r}") from exc
        try:
            preprocessor = Preprocessor.from_state_dict(payload["preprocessor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"checkpoint preprocessor 无效: {path}") from exc
        trainer = cls(
            profile=cfg.name,
            checkpoint_dir=Path(path).parent,
            device=device,
            preprocessor=preprocessor,
            train_features=train_features,
            train_targets=train_targets,
            validation_features=validation_features,
            validation_targets=validation_targets,
            validation_interval_steps=int(
                payload.get("validation_interval_steps", cfg.validation_interval_steps)
            ),
            checkpoint_interval_steps=int(
                payload.get("checkpoint_interval_steps", cfg.checkpoint_interval_steps)
            ),
            batch_size=int(payload.get("batch_size", 64)),
            seed=int(payload.get("seed", 42)),
            event_sink=event_sink,
            data_context=saved_context,
        )
        trainer._expected_data_fingerprint = str(payload["data_fingerprint"])
        trainer._resume_needs_data = not has_data
        if has_data:
            trainer._validate_data_binding()
        try:
            trainer.model.load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
            trainer.history = TrainHistory.from_dict(payload.get("history", {}))
            trainer.history.global_step = int(payload.get("global_step", trainer.history.global_step))
            trainer._resume_batch_cursor = int(payload.get("batch_cursor", 0))
            raw_indices = payload.get("indices")
            if raw_indices is not None:
                trainer._resume_indices = np.asarray(raw_indices, dtype=np.int64)
            rng_state = payload.get("rng_state")
            if not isinstance(rng_state, Mapping):
                raise CheckpointError("checkpoint 缺少 rng_state")
            trainer._restore_rng_state(rng_state)
            if has_data:
                trainer._apply_resume_progress()
        except Exception as exc:
            if isinstance(exc, CheckpointError):
                raise
            raise CheckpointError(f"checkpoint 模型/优化器状态无效: {path}") from exc
        return trainer


Trainer = SingleKANTrainer
set_seed = set_deterministic_seed
