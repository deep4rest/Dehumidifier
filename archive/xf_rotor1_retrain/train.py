"""XF 新风转轮的固定、可续训最终训练流水线。

本入口不接受可替换数据源、通用 artifact/checkpoint 路径或未晋升
绕过参数。所有中间输出只能写入本实验的 ``work/`` 目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:  # 支持计划中直接执行 train.py 的命令。
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from archive.xf_rotor1_retrain.candidates import (  # noqa: E402
    CANDIDATES,
    CandidateConfig,
    get_candidate,
)
from archive.xf_rotor1_retrain.config import (  # noqa: E402
    EXPERIMENT_DIR,
    RESULTS_DIR,
    FIXED_FOLD_GROUPS,
    MODEL_INPUT_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
    REPORTS_DIR,
)
from archive.xf_rotor1_retrain.data import (  # noqa: E402
    EXPECTED_SOURCE_SHA256,
    read_xf_source,
    validate_and_clean_xf,
    validate_report_snapshot,
)
from archive.xf_rotor1_retrain.evaluator import (  # noqa: E402
    EvaluationEvent,
    EvaluationResult,
    FoldData,
    build_fold_data,
    evaluate_checkpoint,
    evaluate_in_memory,
)
from archive.xf_rotor1_retrain.routing import (  # noqa: E402
    build_oof_routing,
    save_routing_json,
)
from archive.xf_rotor1_retrain.trainer import (  # noqa: E402
    CheckpointError,
    TrainingDataError,
    ValidationSnapshot,
    XFTrainer,
    load_checkpoint_payload,
)

__all__ = [
    "WORK_DIR",
    "AuthoritativeDataset",
    "ExecutionGuard",
    "InvocationPlan",
    "PipelineError",
    "RunArrays",
    "RunLogger",
    "RunOutcome",
    "RunSpec",
    "SharedValidationCallback",
    "build_checkpoint_predictor_loader",
    "build_invocation_plan",
    "build_scaled_predictor",
    "evaluation_to_oof_frame",
    "execute_run",
    "extract_run_arrays",
    "load_authoritative_dataset",
    "resolve_run_dir",
    "run_invocation",
    "validate_complete_matrix",
]


WORK_DIR = RESULTS_DIR / "work"
PRESSURE_PA = 101_325.0
SMOKE_CANDIDATE = "xf_v1_baseline"
SMOKE_FOLD = "fold_1"
SMOKE_STEPS = 2_000
SMOKE_TRAIN_ROWS = 512
SMOKE_VALIDATION_ROWS = 128
_NO_PROGRESS_SECONDS = 30.0
_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MATRIX_KEYS = {
    (candidate_name, fold_name)
    for candidate_name in CANDIDATES
    for fold_name in FIXED_FOLD_GROUPS
}


class PipelineError(RuntimeError):
    """训练流水线任一硬边界失败。"""


@dataclass(frozen=True)
class RunSpec:
    candidate_name: str
    fold_name: str
    planned_steps: int | None
    train_row_limit: int | None
    validation_row_limit: int | None


@dataclass(frozen=True)
class InvocationPlan:
    mode: str
    run_name: str
    runs: tuple[RunSpec, ...]
    max_runtime_seconds: float | None
    resume: bool
    device: str


@dataclass(frozen=True)
class AuthoritativeDataset:
    clean: pd.DataFrame
    assignments: pd.DataFrame
    source_sha256: str
    snapshot_id: str
    load_duration_seconds: float


@dataclass(frozen=True)
class RunArrays:
    train_features: np.ndarray
    train_targets: np.ndarray
    train_ids: tuple[int, ...]
    validation_features: np.ndarray
    validation_targets: np.ndarray
    validation_ids: tuple[int, ...]


@dataclass(frozen=True)
class RunOutcome:
    spec: RunSpec
    fold_data: FoldData
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_evaluation: EvaluationResult
    last_evaluation: EvaluationResult
    oof_frame: pd.DataFrame
    global_step: int


class ExecutionGuard:
    """显式 max-runtime 与无进度硬失败边界。"""

    def __init__(
        self,
        *,
        max_runtime_seconds: float | None,
        no_progress_seconds: float = _NO_PROGRESS_SECONDS,
    ) -> None:
        if max_runtime_seconds is not None and (
            not math.isfinite(float(max_runtime_seconds))
            or float(max_runtime_seconds) <= 0.0
        ):
            raise PipelineError("max-runtime-seconds 必须是正的有限数")
        if not math.isfinite(float(no_progress_seconds)) or no_progress_seconds <= 0.0:
            raise PipelineError("无进度超时阈值必须是正的有限数")
        self.max_runtime_seconds = (
            None if max_runtime_seconds is None else float(max_runtime_seconds)
        )
        self.no_progress_seconds = float(no_progress_seconds)
        self.started = time.monotonic()
        self.last_progress = self.started

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    def progress(self, label: str) -> None:
        now = time.monotonic()
        elapsed = now - self.started
        gap = now - self.last_progress
        if self.max_runtime_seconds is not None and elapsed > self.max_runtime_seconds:
            raise PipelineError(
                f"max-runtime 超时: elapsed={elapsed:.3f}s, "
                f"limit={self.max_runtime_seconds:.3f}s, stage={label}"
            )
        if gap > self.no_progress_seconds:
            raise PipelineError(
                f"长时间无进度: gap={gap:.3f}s, "
                f"limit={self.no_progress_seconds:.3f}s, stage={label}"
            )
        self.last_progress = now


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                _jsonable(payload),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


class RunLogger:
    """每个 candidate/fold 的 JSONL 事件与 CSV 健康日志。"""

    _LOSS_COLUMNS = (
        "timestamp_utc",
        "candidate",
        "fold",
        "step",
        "loss",
        "grad_norm",
        "lr",
        "step_time_seconds",
        "samples_per_second",
        "memory_bytes",
        "mfu",
    )
    _SPEED_COLUMNS = (
        "timestamp_utc",
        "candidate",
        "fold",
        "event",
        "mode",
        "completed",
        "total",
        "elapsed_seconds",
        "samples_per_second",
        "memory_bytes",
    )

    def __init__(
        self,
        run_dir: Path,
        *,
        candidate_name: str,
        fold_name: str,
        resume: bool,
    ) -> None:
        self.run_dir = Path(run_dir)
        if self.run_dir.resolve().parent.parent.parent != WORK_DIR.resolve():
            raise PipelineError("日志路径必须是 work/run/candidate/fold")
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not resume:
            raise PipelineError(
                f"运行目录已非空，拒绝覆盖: {self.run_dir}; 请使用 --resume"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_name = candidate_name
        self.fold_name = fold_name
        self.events_path = self.run_dir / "events.jsonl"
        self.loss_path = self.run_dir / "loss.csv"
        self.speed_path = self.run_dir / "speed.csv"
        self._ensure_csv_header(self.loss_path, self._LOSS_COLUMNS)
        self._ensure_csv_header(self.speed_path, self._SPEED_COLUMNS)
        self.loss_values: list[float] = []

    @staticmethod
    def _ensure_csv_header(path: Path, columns: Sequence[str]) -> None:
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=list(columns)).writeheader()

    @staticmethod
    def _append_csv(path: Path, columns: Sequence[str], row: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns))
            writer.writerow({column: row.get(column, "") for column in columns})

    def event(self, payload: Mapping[str, Any]) -> None:
        record = {
            "timestamp_utc": _utc_now(),
            "candidate": self.candidate_name,
            "fold": self.fold_name,
            **dict(payload),
        }
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    _jsonable(record),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )

    def stage(
        self,
        stage: int,
        status: str,
        *,
        duration_seconds: float,
        **payload: Any,
    ) -> None:
        if stage not in range(1, 7):
            raise PipelineError(f"未知六阶段编号: {stage}")
        if status not in {"start", "end", "failure"}:
            raise PipelineError(f"未知阶段状态: {status}")
        self.event(
            {
                "kind": "stage",
                "stage": stage,
                "status": status,
                "duration_seconds": float(duration_seconds),
                **payload,
            }
        )

    def training_event(self, event: Mapping[str, Any]) -> None:
        self.event({"kind": "trainer", "trainer_event": dict(event)})
        kind = str(event.get("kind", ""))
        if kind == "train_step":
            loss = float(event["loss"])
            self.loss_values.append(loss)
            self._append_csv(
                self.loss_path,
                self._LOSS_COLUMNS,
                {
                    "timestamp_utc": _utc_now(),
                    "candidate": self.candidate_name,
                    "fold": self.fold_name,
                    "step": int(event["global_step"]),
                    "loss": loss,
                    "grad_norm": float(event["grad_norm"]),
                    "lr": float(event["lr"]),
                    "step_time_seconds": float(event["step_time"]),
                    "samples_per_second": float(event["samples_s"]),
                    "memory_bytes": int(event["memory"]),
                    "mfu": str(event["mfu"]),
                },
            )
        if kind in {"validation_progress", "validation_end"}:
            progress = event["validation_progress"]
            self._append_csv(
                self.speed_path,
                self._SPEED_COLUMNS,
                {
                    "timestamp_utc": _utc_now(),
                    "candidate": self.candidate_name,
                    "fold": self.fold_name,
                    "event": kind,
                    "mode": "scaled_validation_loss",
                    "completed": int(progress["completed"]),
                    "total": int(progress["total"]),
                    "elapsed_seconds": float(event["step_time"]),
                    "samples_per_second": float(event["samples_s"]),
                    "memory_bytes": int(event["memory"]),
                },
            )

    def evaluation_event(
        self,
        event: EvaluationEvent,
        *,
        artifact: str = "cadence",
    ) -> None:
        payload = dict(event.payload)
        self.event(
            {
                "kind": "evaluator",
                "artifact": artifact,
                "evaluator_event": {
                    "kind": event.kind,
                    "mode": event.mode,
                    "completed": event.completed,
                    "total": event.total,
                    "payload": payload,
                },
            }
        )
        self._append_csv(
            self.speed_path,
            self._SPEED_COLUMNS,
            {
                "timestamp_utc": _utc_now(),
                "candidate": self.candidate_name,
                "fold": self.fold_name,
                "event": event.kind,
                "mode": event.mode,
                "completed": int(event.completed),
                "total": int(event.total),
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
                "samples_per_second": float(payload.get("samples_per_second", 0.0)),
                "memory_bytes": 0,
            },
        )


def _positive_runtime(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是数字") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("必须是正的有限数")
    return number


def _validate_run_name(value: str) -> str:
    if not _SAFE_RUN_NAME.fullmatch(value):
        raise PipelineError(
            "run-name/运行名只允许 1–64 位英文字母、数字、_ 和 -"
        )
    return value


def build_invocation_plan(argv: Sequence[str] | None = None) -> InvocationPlan:
    """解析唯一受控 CLI；无 candidate/fold 时组装完整 5×5。"""

    parser = argparse.ArgumentParser(description="XF 新风转轮 KAN 最终训练")
    parser.add_argument("--candidate", choices=list(CANDIDATES))
    parser.add_argument("--fold", choices=list(FIXED_FOLD_GROUPS))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=_positive_runtime)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    if (args.candidate is None) != (args.fold is None):
        parser.error("--candidate 与 --fold 必须同时提供")
    if args.smoke_test and args.candidate is not None:
        parser.error("--smoke-test 不得与 --candidate/--fold 并用")

    if args.smoke_test:
        mode = "smoke"
        default_name = "xf_smoke"
        runs = (
            RunSpec(
                SMOKE_CANDIDATE,
                SMOKE_FOLD,
                SMOKE_STEPS,
                SMOKE_TRAIN_ROWS,
                SMOKE_VALIDATION_ROWS,
            ),
        )
        max_runtime = (
            300.0 if args.max_runtime_seconds is None else args.max_runtime_seconds
        )
    elif args.candidate is not None:
        mode = "single"
        default_name = f"{args.candidate}_{args.fold}"
        runs = (RunSpec(args.candidate, args.fold, None, None, None),)
        max_runtime = args.max_runtime_seconds
    else:
        mode = "matrix"
        default_name = "xf_full_matrix"
        runs = tuple(
            RunSpec(candidate_name, fold_name, None, None, None)
            for candidate_name in CANDIDATES
            for fold_name in FIXED_FOLD_GROUPS
        )
        max_runtime = args.max_runtime_seconds

    try:
        run_name = _validate_run_name(args.run_name or default_name)
    except PipelineError as exc:
        parser.error(str(exc))
    return InvocationPlan(
        mode=mode,
        run_name=run_name,
        runs=runs,
        max_runtime_seconds=max_runtime,
        resume=bool(args.resume),
        device=str(args.device),
    )


def resolve_run_dir(run_name: str, candidate_name: str, fold_name: str) -> Path:
    """只生成 work/run/candidate/fold 下的受控目录。"""

    normalized_run = _validate_run_name(run_name)
    if candidate_name not in CANDIDATES:
        raise PipelineError(f"未知固定候选: {candidate_name}")
    if fold_name not in FIXED_FOLD_GROUPS:
        raise PipelineError(f"未知固定 fold: {fold_name}")
    path = WORK_DIR / normalized_run / candidate_name / fold_name
    if not path.resolve().is_relative_to(WORK_DIR.resolve()):
        raise PipelineError("运行路径逃逸实验 work 目录")
    return path


def load_authoritative_dataset(
    guard: ExecutionGuard | None = None,
) -> AuthoritativeDataset:
    """回读固定 XF bytes 与已提交报告快照，不重发布报告。"""

    started = time.perf_counter()
    if guard is not None:
        guard.progress("data_load:start")
    marker = validate_report_snapshot(REPORTS_DIR)
    raw = read_xf_source()
    clean, _ = validate_and_clean_xf(raw)
    try:
        assignments = pd.read_csv(REPORTS_DIR / "split_assignments.csv")
    except Exception as exc:
        raise PipelineError(f"权威 split_assignments 回读失败: {exc}") from exc
    if marker.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise PipelineError("报告快照源哈希与固定 XF 源不一致")
    duration = time.perf_counter() - started
    if guard is not None:
        guard.progress("data_load:end")
    return AuthoritativeDataset(
        clean=clean,
        assignments=assignments,
        source_sha256=EXPECTED_SOURCE_SHA256,
        snapshot_id=str(marker["report_snapshot"]["snapshot_id"]),
        load_duration_seconds=duration,
    )


def _take_prefix(ids: Sequence[int], limit: int | None, label: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in ids)
    if limit is None:
        selected = normalized
    else:
        if int(limit) <= 0:
            raise PipelineError(f"{label} row limit 必须 > 0")
        selected = normalized[: int(limit)]
    if not selected:
        raise PipelineError(f"{label} 分区为空")
    return selected


def extract_run_arrays(
    clean: pd.DataFrame,
    fold: FoldData,
    spec: RunSpec,
) -> RunArrays:
    """只按当折 development ID 读取建模数值，outer 只作 ID 隔离审计。"""

    if spec.candidate_name not in CANDIDATES or spec.fold_name != fold.fold_name:
        raise PipelineError("运行规格与固定 candidate/fold 不一致")
    train_ids = _take_prefix(fold.train_ids, spec.train_row_limit, "train")
    validation_ids = _take_prefix(
        fold.validation_ids,
        spec.validation_row_limit,
        "validation",
    )
    outer = set(int(value) for value in fold.outer_ids)
    if set(train_ids) & set(validation_ids) or set(train_ids) & outer or set(validation_ids) & outer:
        raise PipelineError("train/validation/outer ID 不隔离")
    indexed = clean.set_index("ID", drop=False)

    def matrix(ids: tuple[int, ...], columns: Sequence[str], label: str) -> np.ndarray:
        try:
            values = indexed.loc[list(ids), list(columns)].to_numpy(dtype=np.float64)
        except Exception as exc:
            raise PipelineError(f"{label} 按权威 ID 取值失败: {exc}") from exc
        if values.shape != (len(ids), len(columns)) or not np.isfinite(values).all():
            raise PipelineError(f"{label} 形状错误或存在非有限值")
        return values

    train_features = matrix(train_ids, MODEL_INPUT_COLUMNS, "train features")
    train_targets = matrix(train_ids, MODEL_OUTPUT_COLUMNS, "train targets")
    validation_features = matrix(
        validation_ids, MODEL_INPUT_COLUMNS, "validation features"
    )
    validation_targets = matrix(
        validation_ids, MODEL_OUTPUT_COLUMNS, "validation targets"
    )
    spans = np.ptp(train_targets, axis=0)
    if not np.isfinite(spans).all() or np.any(spans <= 0.0):
        bad = [
            column
            for column, span in zip(MODEL_OUTPUT_COLUMNS, spans)
            if not math.isfinite(float(span)) or float(span) <= 0.0
        ]
        raise PipelineError(f"训练子集目标跨度无效: {bad}")
    return RunArrays(
        train_features=train_features,
        train_targets=train_targets,
        train_ids=train_ids,
        validation_features=validation_features,
        validation_targets=validation_targets,
        validation_ids=validation_ids,
    )


def build_scaled_predictor(
    trainer: XFTrainer | ValidationSnapshot,
) -> Callable[[np.ndarray], np.ndarray]:
    """把内存 KAN 暴露为共享 evaluator 唯一的 model-space predictor。"""

    if isinstance(trainer, ValidationSnapshot):
        return trainer.predict_model_space

    def predictor(features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(MODEL_INPUT_COLUMNS)
            or not np.isfinite(values).all()
        ):
            raise TrainingDataError("evaluator model-space 输入必须是有限 10 列")
        trainer.model.eval()
        with torch.no_grad():
            output = trainer.model(
                torch.from_numpy(values.astype(np.float32)).to(trainer.device)
            )
        result = output.detach().cpu().numpy().astype(np.float64)
        if result.shape != (len(values), len(MODEL_OUTPUT_COLUMNS)):
            raise TrainingDataError("evaluator model-space 预测必须是 7 列")
        if not np.isfinite(result).all():
            raise TrainingDataError("evaluator model-space 预测存在非有限值")
        return result

    return predictor


def build_checkpoint_predictor_loader(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_ids: Sequence[int],
    checkpoint_dir: Path,
    device: str,
) -> Callable[[Path], Callable[[np.ndarray], np.ndarray]]:
    """只恢复 evaluator 传入的固定 run 内 checkpoint。"""

    fixed_dir = Path(checkpoint_dir).resolve()

    def loader(path: Path) -> Callable[[np.ndarray], np.ndarray]:
        resolved = Path(path).resolve()
        if resolved.parent != fixed_dir or resolved.name not in {"best.pt", "last.pt"}:
            raise CheckpointError("评估只允许当前受控运行的 best/last checkpoint")
        restored = XFTrainer.resume(
            resolved,
            train_features=train_features,
            train_targets=train_targets,
            train_ids=train_ids,
            checkpoint_dir=fixed_dir,
            device=device,
        )
        return build_scaled_predictor(restored)

    return loader


class SharedValidationCallback:
    """在 trainer cadence 上调用共享 in-memory evaluator 的纯观测 hook。"""

    def __init__(
        self,
        *,
        validation_ids: Sequence[int],
        train_targets_physical: np.ndarray,
        pressure_Pa: float,
        batch_size: int = 1024,
        event_sink: Callable[[EvaluationEvent], None] | None = None,
    ) -> None:
        self.validation_ids = tuple(int(value) for value in validation_ids)
        self.train_targets_physical = np.asarray(
            train_targets_physical, dtype=np.float64
        )
        self.pressure_Pa = float(pressure_Pa)
        self.batch_size = int(batch_size)
        self.event_sink = event_sink
        self.results: list[EvaluationResult] = []

    @property
    def latest_result(self) -> EvaluationResult | None:
        return self.results[-1] if self.results else None

    def __call__(
        self,
        snapshot: ValidationSnapshot,
        validation_features: Any,
        validation_targets: Any,
    ) -> None:
        features = np.asarray(validation_features, dtype=np.float64)
        targets = np.asarray(validation_targets, dtype=np.float64)
        if len(features) == 0:
            # 空集必须由共享 evaluator 生成 mode-aware failure 生命周期。
            feature_model_space = features
            target_model_space = targets
        else:
            feature_model_space = snapshot.preprocessor.transform_features(features)
            target_model_space = snapshot.preprocessor.transform_targets(targets)
        result = evaluate_in_memory(
            build_scaled_predictor(snapshot),
            feature_model_space,
            target_model_space,
            self.train_targets_physical,
            sample_ids=self.validation_ids,
            pressure_Pa=self.pressure_Pa,
            inverse_target_transform=snapshot.preprocessor.y_scaler,
            inverse_scaler=snapshot.preprocessor.target_transform,
            batch_size=self.batch_size,
            event_sink=self.event_sink,
        )
        _assert_finite_evaluation(result)
        self.results.append(result)
        return None


def _assert_finite_evaluation(result: EvaluationResult) -> None:
    for label, values in (
        ("predictions_raw", result.predictions_raw),
        ("predictions_deployed", result.predictions_deployed),
        ("targets_raw", result.targets_raw),
    ):
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise PipelineError(f"评估 {label} 存在非有限值")
    metric_values: list[float] = []
    for metrics in (result.raw_metrics, result.deployed_metrics):
        for metric in metrics.values():
            metric_values.extend(
                [float(metric.mae), float(metric.rmse), float(metric.nrmse), float(metric.r2)]
            )
    metric_values.extend(
        [
            float(result.regen_phase_metric.rmse),
            float(result.regen_phase_metric.nrmse),
            float(result.process_dew_point_metric.mae),
            float(result.process_dew_point_metric.rmse),
            float(result.process_dew_point_metric.nrmse),
            float(result.process_dew_point_metric.r2),
        ]
    )
    if not all(math.isfinite(value) for value in metric_values):
        raise PipelineError("评估聚合指标存在非有限值")


def evaluation_to_oof_frame(result: EvaluationResult) -> pd.DataFrame:
    """写出可回读的 ID 对齐 raw/deployed/target 路由输入。"""

    _assert_finite_evaluation(result)
    ids = np.asarray(result.sample_ids, dtype=np.int64)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise PipelineError("OOF sample ID 必须唯一")
    frame = pd.DataFrame({"ID": ids})
    for prefix, matrix in (
        ("prediction_raw", result.predictions_raw),
        ("prediction_deployed", result.predictions_deployed),
        ("target_raw", result.targets_raw),
    ):
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (len(ids), len(MODEL_OUTPUT_COLUMNS)):
            raise PipelineError(f"OOF {prefix} 形状不符")
        for index, column in enumerate(MODEL_OUTPUT_COLUMNS):
            frame[f"{prefix}__{column}"] = values[:, index]
    return frame.sort_values("ID", kind="stable").reset_index(drop=True)


def _planned_steps(spec: RunSpec, arrays: RunArrays, candidate: CandidateConfig) -> int:
    if spec.planned_steps is not None:
        steps = int(spec.planned_steps)
    else:
        batches_per_epoch = math.ceil(len(arrays.train_ids) / candidate.batch_size)
        steps = candidate.max_epochs * batches_per_epoch
    if steps <= 0:
        raise PipelineError("计划 optimizer steps 必须 > 0")
    if steps < candidate.validation_interval_steps:
        raise PipelineError(
            f"生产运行步数 {steps} 未命中固定验证 cadence "
            f"{candidate.validation_interval_steps}"
        )
    return steps


def _evaluation_common(
    trainer: XFTrainer,
    arrays: RunArrays,
) -> dict[str, Any]:
    return {
        "features": trainer.preprocessor.transform_features(
            arrays.validation_features
        ),
        "targets_model_space": trainer.preprocessor.transform_targets(
            arrays.validation_targets
        ),
        "train_targets_physical": arrays.train_targets,
        "sample_ids": arrays.validation_ids,
        "pressure_Pa": PRESSURE_PA,
        "inverse_target_transform": trainer.preprocessor.y_scaler,
        "inverse_scaler": trainer.preprocessor.target_transform,
        "batch_size": 1024,
    }


def _stage_failure(
    logger: RunLogger,
    stage: int,
    started: float,
    exc: Exception,
) -> None:
    logger.stage(
        stage,
        "failure",
        duration_seconds=time.perf_counter() - started,
        error_type=type(exc).__name__,
        detail=str(exc),
    )


def execute_run(
    *,
    dataset: AuthoritativeDataset,
    spec: RunSpec,
    run_name: str,
    resume: bool,
    device: str,
    guard: ExecutionGuard,
) -> RunOutcome:
    """执行一个固定 candidate/fold 的六阶段训练与回读。"""

    run_dir = resolve_run_dir(run_name, spec.candidate_name, spec.fold_name)
    logger = RunLogger(
        run_dir,
        candidate_name=spec.candidate_name,
        fold_name=spec.fold_name,
        resume=resume,
    )
    guard.progress("stage1:data_snapshot")
    logger.stage(1, "start", duration_seconds=0.0)
    logger.stage(
        1,
        "end",
        duration_seconds=dataset.load_duration_seconds,
        source_sha256=dataset.source_sha256,
        report_snapshot_id=dataset.snapshot_id,
        clean_rows=len(dataset.clean),
    )

    stage_started = time.perf_counter()
    logger.stage(2, "start", duration_seconds=0.0)
    try:
        guard.progress("stage2:fold_prepare:start")
        fold = build_fold_data(dataset.clean, dataset.assignments, spec.fold_name)
        arrays = extract_run_arrays(dataset.clean, fold, spec)
        candidate = get_candidate(spec.candidate_name)
        planned_steps = _planned_steps(spec, arrays, candidate)
        logger.event(
            {
                "kind": "run_plan",
                "limited_optimizer_steps": planned_steps,
                "validation_interval_steps": candidate.validation_interval_steps,
                "train_rows": len(arrays.train_ids),
                "validation_rows": len(arrays.validation_ids),
                "resume": resume,
            }
        )
        guard.progress("stage2:fold_prepare:end")
        logger.stage(
            2,
            "end",
            duration_seconds=time.perf_counter() - stage_started,
            train_rows=len(arrays.train_ids),
            validation_rows=len(arrays.validation_ids),
        )
    except Exception as exc:
        _stage_failure(logger, 2, stage_started, exc)
        raise

    stage_started = time.perf_counter()
    logger.stage(3, "start", duration_seconds=0.0)
    try:
        def trainer_sink(event: dict[str, Any]) -> None:
            guard.progress(f"training:{event.get('kind', 'event')}")
            logger.training_event(event)

        def cadence_evaluator_sink(event: EvaluationEvent) -> None:
            guard.progress(f"evaluator:{event.mode}:{event.kind}")
            logger.evaluation_event(event, artifact="cadence")

        checkpoint_dir = run_dir / "checkpoints"
        last_checkpoint = checkpoint_dir / "last.pt"
        best_checkpoint = checkpoint_dir / "best.pt"
        if resume:
            if not last_checkpoint.is_file():
                raise PipelineError(f"--resume 的固定 last checkpoint 不存在: {last_checkpoint}")
            trainer = XFTrainer.resume(
                last_checkpoint,
                train_features=arrays.train_features,
                train_targets=arrays.train_targets,
                train_ids=arrays.train_ids,
                checkpoint_dir=checkpoint_dir,
                device=device,
                event_sink=trainer_sink,
            )
        else:
            trainer = XFTrainer(
                candidate=candidate,
                train_features=arrays.train_features,
                train_targets=arrays.train_targets,
                train_ids=arrays.train_ids,
                checkpoint_dir=checkpoint_dir,
                device=device,
                event_sink=trainer_sink,
            )
        if trainer.global_step > planned_steps:
            raise PipelineError(
                f"resume checkpoint step={trainer.global_step} 超过固定计划 {planned_steps}"
            )
        shared_callback = SharedValidationCallback(
            validation_ids=arrays.validation_ids,
            train_targets_physical=arrays.train_targets,
            pressure_Pa=PRESSURE_PA,
            event_sink=cadence_evaluator_sink,
        )
        trainer.train_steps(
            planned_steps - trainer.global_step,
            validation_features=arrays.validation_features,
            validation_targets=arrays.validation_targets,
            validation_callback=shared_callback,
        )
        if trainer.validation_evaluations == 0:
            raise PipelineError("固定 2000-step cadence 未触发共享 in-memory evaluator")
        if trainer.global_step < candidate.validation_interval_steps:
            raise PipelineError("训练未达到第一个完整验证 cadence")
        guard.progress("stage3:training:end")
        logger.stage(
            3,
            "end",
            duration_seconds=time.perf_counter() - stage_started,
            global_step=trainer.global_step,
            validation_evaluations=trainer.validation_evaluations,
            stopped_early=trainer.stopped_early,
        )
    except Exception as exc:
        _stage_failure(logger, 3, stage_started, exc)
        raise

    stage_started = time.perf_counter()
    logger.stage(4, "start", duration_seconds=0.0)
    try:
        best_payload = load_checkpoint_payload(best_checkpoint)
        last_payload = load_checkpoint_payload(last_checkpoint)
        if best_payload["candidate"]["name"] != spec.candidate_name:
            raise PipelineError("best checkpoint 候选名不一致")
        if last_payload["global_step"] != trainer.global_step:
            raise PipelineError("last checkpoint global_step 与内存训练器不一致")
        guard.progress("stage4:checkpoint_reload:end")
        logger.stage(
            4,
            "end",
            duration_seconds=time.perf_counter() - stage_started,
            best_checkpoint=best_checkpoint,
            best_step=best_payload["global_step"],
            last_checkpoint=last_checkpoint,
            last_step=last_payload["global_step"],
        )
    except Exception as exc:
        _stage_failure(logger, 4, stage_started, exc)
        raise

    stage_started = time.perf_counter()
    logger.stage(5, "start", duration_seconds=0.0)
    try:
        common = _evaluation_common(trainer, arrays)

        def sink_for(artifact: str) -> Callable[[EvaluationEvent], None]:
            def sink(event: EvaluationEvent) -> None:
                guard.progress(f"stage5:{artifact}:{event.mode}:{event.kind}")
                logger.evaluation_event(event, artifact=artifact)

            return sink

        in_memory_evaluation = evaluate_in_memory(
            build_scaled_predictor(trainer),
            event_sink=sink_for("final_in_memory"),
            **common,
        )
        loader = build_checkpoint_predictor_loader(
            train_features=arrays.train_features,
            train_targets=arrays.train_targets,
            train_ids=arrays.train_ids,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        last_evaluation = evaluate_checkpoint(
            last_checkpoint,
            loader,
            event_sink=sink_for("last_checkpoint"),
            **common,
        )
        best_evaluation = evaluate_checkpoint(
            best_checkpoint,
            loader,
            event_sink=sink_for("best_checkpoint"),
            **common,
        )
        for result in (in_memory_evaluation, last_evaluation, best_evaluation):
            _assert_finite_evaluation(result)
        if not np.array_equal(
            in_memory_evaluation.predictions_raw,
            last_evaluation.predictions_raw,
        ):
            raise PipelineError("last checkpoint 恢复前后预测不一致")
        guard.progress("stage5:evaluation:end")
        logger.stage(
            5,
            "end",
            duration_seconds=time.perf_counter() - stage_started,
            modes=["in_memory", "checkpoint"],
            validation_rows=len(arrays.validation_ids),
        )
    except Exception as exc:
        _stage_failure(logger, 5, stage_started, exc)
        raise

    stage_started = time.perf_counter()
    logger.stage(6, "start", duration_seconds=0.0)
    try:
        oof = evaluation_to_oof_frame(best_evaluation)
        oof_path = run_dir / "oof_predictions.csv"
        temporary_oof = oof_path.with_suffix(".csv.tmp")
        oof.to_csv(temporary_oof, index=False, lineterminator="\n")
        os.replace(temporary_oof, oof_path)
        oof_hash = hashlib.sha256(oof_path.read_bytes()).hexdigest()
        route_input = {
            "candidate": spec.candidate_name,
            "fold": spec.fold_name,
            "source_sha256": dataset.source_sha256,
            "report_snapshot_id": dataset.snapshot_id,
            "validation_rows": len(oof),
            "validation_ids_sha256": hashlib.sha256(
                np.asarray(sorted(oof["ID"].astype(int)), dtype="<i8").tobytes()
            ).hexdigest(),
            "oof_predictions_sha256": oof_hash,
            "evaluation_modes": ["in_memory", "checkpoint"],
            "best_step": int(best_payload["global_step"]),
            "last_step": int(last_payload["global_step"]),
        }
        _write_json_atomic(run_dir / "route_input.json", route_input)
        summary = {
            **route_input,
            "status": "success",
            "planned_steps": planned_steps,
            "global_step": trainer.global_step,
            "validation_evaluations": trainer.validation_evaluations,
            "loss_first": logger.loss_values[0] if logger.loss_values else None,
            "loss_last": logger.loss_values[-1] if logger.loss_values else None,
            "samples_per_second_last": (
                None
                if not logger.loss_values
                else pd.read_csv(logger.loss_path).iloc[-1]["samples_per_second"]
            ),
            "memory_bytes_max": (
                0
                if not logger.loss_values
                else int(pd.read_csv(logger.loss_path)["memory_bytes"].max())
            ),
            "mfu": "not_applicable" if device == "cpu" else "unavailable",
            "best_deployed_metrics": {
                output: {
                    "mae": metric.mae,
                    "rmse": metric.rmse,
                    "nrmse": metric.nrmse,
                    "r2": metric.r2,
                }
                for output, metric in best_evaluation.deployed_metrics.items()
            },
        }
        _write_json_atomic(run_dir / "summary.json", summary)
        guard.progress("stage6:oof:end")
        logger.stage(
            6,
            "end",
            duration_seconds=time.perf_counter() - stage_started,
            oof_path=oof_path,
            oof_sha256=oof_hash,
        )
    except Exception as exc:
        _stage_failure(logger, 6, stage_started, exc)
        raise

    return RunOutcome(
        spec=spec,
        fold_data=fold,
        run_dir=run_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        best_evaluation=best_evaluation,
        last_evaluation=last_evaluation,
        oof_frame=oof,
        global_step=trainer.global_step,
    )


def validate_complete_matrix(
    outcomes: Mapping[tuple[str, str], RunOutcome],
) -> None:
    """路由聚合前确保恰好完成固定 25 个有限运行。"""

    actual = set(outcomes)
    if actual != _MATRIX_KEYS:
        raise PipelineError(
            f"完整 5×5 矩阵必须恰好 25 项: "
            f"missing={len(_MATRIX_KEYS - actual)}, extra={len(actual - _MATRIX_KEYS)}"
        )
    for key, outcome in outcomes.items():
        if not isinstance(outcome, RunOutcome):
            raise PipelineError(f"矩阵运行结果类型无效: {key}")
        _assert_finite_evaluation(outcome.best_evaluation)
        if not outcome.best_checkpoint.is_file() or not outcome.last_checkpoint.is_file():
            raise PipelineError(f"矩阵运行 checkpoint 不完整: {key}")


def _append_invocation_event(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                {"timestamp_utc": _utc_now(), **_jsonable(payload)},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


def run_invocation(plan: InvocationPlan) -> dict[str, Any]:
    """执行单运行、smoke 或完整矩阵的固定入口。"""

    root = WORK_DIR / plan.run_name
    if root.exists() and any(root.iterdir()) and not plan.resume:
        raise PipelineError(f"调用目录已非空，拒绝覆盖: {root}")
    root.mkdir(parents=True, exist_ok=True)
    invocation_log = root / "invocation.jsonl"
    _append_invocation_event(
        invocation_log,
        {
            "kind": "plan",
            "mode": plan.mode,
            "run_name": plan.run_name,
            "run_count": len(plan.runs),
            "max_runtime_seconds": plan.max_runtime_seconds,
            "runs": [
                {
                    "candidate": run.candidate_name,
                    "fold": run.fold_name,
                    "limited_optimizer_steps": run.planned_steps,
                    "train_row_limit": run.train_row_limit,
                    "validation_row_limit": run.validation_row_limit,
                }
                for run in plan.runs
            ],
        },
    )
    guard = ExecutionGuard(max_runtime_seconds=plan.max_runtime_seconds)
    try:
        dataset = load_authoritative_dataset(guard)
        _append_invocation_event(
            invocation_log,
            {
                "kind": "data_load",
                "status": "success",
                "duration_seconds": dataset.load_duration_seconds,
                "source_sha256": dataset.source_sha256,
                "report_snapshot_id": dataset.snapshot_id,
            },
        )
        outcomes: dict[tuple[str, str], RunOutcome] = {}
        for spec in plan.runs:
            outcome = execute_run(
                dataset=dataset,
                spec=spec,
                run_name=plan.run_name,
                resume=plan.resume,
                device=plan.device,
                guard=guard,
            )
            outcomes[(spec.candidate_name, spec.fold_name)] = outcome

        routing_path: Path | None = None
        if plan.mode == "matrix":
            validate_complete_matrix(outcomes)
            evaluations = {
                candidate_name: {
                    fold_name: outcomes[(candidate_name, fold_name)].best_evaluation
                    for fold_name in FIXED_FOLD_GROUPS
                }
                for candidate_name in CANDIDATES
            }
            fold_data_by_fold = {
                fold_name: outcomes[(next(iter(CANDIDATES)), fold_name)].fold_data
                for fold_name in FIXED_FOLD_GROUPS
            }
            artifact = build_oof_routing(
                evaluations,
                fold_data_by_fold=fold_data_by_fold,
                pressure_Pa=PRESSURE_PA,
            )
            routing_path = root / "routing.json"
            save_routing_json(artifact, routing_path)
        result = {
            "status": "success",
            "mode": plan.mode,
            "run_name": plan.run_name,
            "completed_runs": len(outcomes),
            "elapsed_seconds": guard.elapsed_seconds,
            "routing_path": routing_path,
        }
        _append_invocation_event(invocation_log, {"kind": "end", **result})
        return result
    except Exception as exc:
        _append_invocation_event(
            invocation_log,
            {
                "kind": "failure",
                "status": "failed",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "elapsed_seconds": guard.elapsed_seconds,
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        plan = build_invocation_plan(argv)
        result = run_invocation(plan)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"XF 训练流水线失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_jsonable(result), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
