"""尺寸隔离风量数据上的独立 MLP/KAN 候选训练入口。"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .candidate_data import CandidateDataset, load_candidate_dataset
from .candidate_metrics import evaluate_candidate_predictions
from .candidate_models import CandidateContract, CandidateModel, build_candidate_model
from .config import DIRECT_OUTPUT_COLUMNS, TRAINING_RESULTS_DIR
from .trainer import set_deterministic_seed


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不可序列化对象: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n")


def _write_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml

        text = yaml.safe_dump(json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default)), allow_unicode=True, sort_keys=False)
    except ImportError:  # pragma: no cover
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(text, encoding="utf-8")


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        value = state[name]
        if isinstance(value, torch.Tensor):
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _preprocessor_hash(dataset: CandidateDataset) -> str:
    return hashlib.sha256(pickle.dumps(dataset.preprocessor.state_dict(), protocol=4)).hexdigest()


def _scaled_forward(model: CandidateModel, features: torch.Tensor, targets: torch.Tensor, device: torch.device) -> tuple[float, np.ndarray]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(features.to(device))
            loss = torch.mean((prediction - targets.to(device)) ** 2)
            values = prediction.detach().cpu().numpy()
    finally:
        model.train(was_training)
    value = float(loss.detach().cpu())
    if not np.isfinite(value) or not np.isfinite(values).all():
        raise ValueError("验证输出或损失存在 NaN/Inf")
    return value, values


def _save_checkpoint(path: Path, *, model: CandidateModel, optimizer: torch.optim.Optimizer, contract: CandidateContract, dataset: CandidateDataset, epoch: int, global_step: int, best_validation_loss: float | None, no_improvement_count: int) -> None:
    payload = {
        "schema_version": 2,
        "model_state_dict": _state_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "profile": contract.profile,
        "model_type": contract.model_type,
        "data_fingerprint": dataset.data_fingerprint,
        "preprocessor_hash": _preprocessor_hash(dataset),
        "input_columns": list(contract.input_columns),
        "output_columns": list(contract.output_columns),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": best_validation_loss,
        "no_improvement_count": int(no_improvement_count),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _prediction_frame(dataset: CandidateDataset, predictions: np.ndarray) -> pd.DataFrame:
    # Local import keeps pandas out of the hot training path.
    import pandas as pd

    frame = dataset.frames["test"][["ID", "转轮厚度", "转轮直径"]].copy()
    for index, column in enumerate(DIRECT_OUTPUT_COLUMNS):
        frame[f"真实_{column}"] = dataset.raw["test"][1][:, index]
        frame[f"预测_{column}"] = predictions[:, index]
    return frame


def train_candidate(
    dataset: CandidateDataset,
    model_type: str,
    output_dir: str | Path,
    *,
    batch_size: int = 256,
    seed: int = 20260908,
    device: str | torch.device = "cuda",
    smoke_test: bool = False,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """训练一个独立候选并将全部证据写入 ``output_dir``。"""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = output / "reports"
    checkpoints = output / "checkpoints"
    reports.mkdir()
    checkpoints.mkdir()
    started = time.perf_counter()
    contract = __import__("training.single_kan.candidate_models", fromlist=["candidate_contract"]).candidate_contract(dataset.profile, model_type)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但项目虚拟环境中 CUDA 不可用")
    set_deterministic_seed(seed)
    model = build_candidate_model(dataset.profile, model_type).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract.learning_rate, weight_decay=contract.weight_decay)
    train_x = torch.from_numpy(dataset.scaled["train"][0]).to(target_device)
    train_y = torch.from_numpy(dataset.scaled["train"][1]).to(target_device)
    validation_x = torch.from_numpy(dataset.scaled["validation"][0]).to(target_device)
    validation_y = torch.from_numpy(dataset.scaled["validation"][1]).to(target_device)
    steps_per_epoch = max(1, (len(train_x) + max(1, int(batch_size)) - 1) // max(1, int(batch_size)))
    requested_steps = int(max_steps) if max_steps is not None else contract.max_epochs * steps_per_epoch
    if requested_steps <= 0:
        raise ValueError("max_steps 必须为正整数")
    validation_every = 2 if smoke_test else steps_per_epoch
    best_validation_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    epoch = 0
    stop_reason = "running"
    while global_step < requested_steps:
        epoch += 1
        permutation = torch.randperm(len(train_x), device=target_device)
        epoch_start = time.perf_counter()
        model.train(True)
        epoch_losses: list[float] = []
        for start in range(0, len(train_x), max(1, int(batch_size))):
            if global_step >= requested_steps:
                break
            indices = permutation[start:start + max(1, int(batch_size))]
            prediction = model(train_x[indices])
            loss = torch.mean((prediction - train_y[indices]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失出现 NaN/Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise RuntimeError("梯度范数出现 NaN/Inf")
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if global_step % validation_every == 0 or global_step == requested_steps:
                validation_loss_scaled, validation_scaled = _scaled_forward(model, validation_x, validation_y, target_device)
                validation_physical = dataset.preprocessor.inverse_targets(validation_scaled)
                validation_metrics = evaluate_candidate_predictions(
                    dataset.raw["validation"][1],
                    validation_physical,
                    geometries=dataset.frames["validation"][["转轮厚度", "转轮直径"]].to_numpy(),
                )
                improved = validation_loss_scaled < best_validation_loss
                if improved:
                    best_validation_loss = validation_loss_scaled
                    best_epoch = epoch
                    no_improvement = 0
                    best_state = _state_cpu(model)
                    _save_checkpoint(
                        checkpoints / "best_model.pt",
                        model=model,
                        optimizer=optimizer,
                        contract=contract,
                        dataset=dataset,
                        epoch=epoch,
                        global_step=global_step,
                        best_validation_loss=best_validation_loss,
                        no_improvement_count=no_improvement,
                    )
                else:
                    no_improvement += 1
                event = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": float(np.mean(epoch_losses)),
                    "validation_loss_scaled": validation_loss_scaled,
                    "validation_metrics": validation_metrics,
                    "improved": improved,
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
                history.append(event)
                _write_json(output / "train_history.json", history)
                if not smoke_test and no_improvement >= 20:
                    stop_reason = "early_stopping"
                    break
        _save_checkpoint(
            checkpoints / "checkpoint.pt",
            model=model,
            optimizer=optimizer,
            contract=contract,
            dataset=dataset,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=None if not np.isfinite(best_validation_loss) else best_validation_loss,
            no_improvement_count=no_improvement,
        )
        _append_jsonl(output / "train.log.jsonl", {"epoch": epoch, "global_step": global_step, "seconds": time.perf_counter() - epoch_start})
        if stop_reason == "early_stopping":
            break
    if stop_reason == "running":
        stop_reason = "smoke_steps_completed" if smoke_test else "completed"
    if best_state is None:
        _, validation_scaled = _scaled_forward(model, validation_x, validation_y, target_device)
        best_state = _state_cpu(model)
        best_validation_loss = float(np.mean((validation_scaled - dataset.scaled["validation"][1]) ** 2))
        best_epoch = epoch
    model.load_state_dict(best_state)
    model.to(target_device)
    test_predictions = model.predict(dataset.raw["test"][0], dataset.preprocessor, device=target_device)
    test_metrics = evaluate_candidate_predictions(
        dataset.raw["test"][1],
        test_predictions,
        geometries=dataset.frames["test"][["转轮厚度", "转轮直径"]].to_numpy(),
    )
    # Reload the best checkpoint into a fresh model and verify identical test output.
    reloaded = build_candidate_model(dataset.profile, model_type).to(target_device)
    checkpoint_payload = torch.load(checkpoints / "best_model.pt", map_location=target_device, weights_only=False)
    reloaded.load_state_dict(checkpoint_payload["model_state_dict"])
    reloaded_predictions = reloaded.predict(dataset.raw["test"][0], dataset.preprocessor, device=target_device)
    if not np.allclose(test_predictions, reloaded_predictions, rtol=1e-5, atol=1e-6):
        raise RuntimeError("最佳 checkpoint 重载后测试预测不一致")

    model_dir = output / "model"
    model_dir.mkdir()
    torch.save(best_state, model_dir / "best_model.pt")
    scaler_state = dataset.preprocessor.state_dict()
    with (model_dir / "scalers.pkl").open("wb") as stream:
        pickle.dump(scaler_state, stream, protocol=4)
    _write_json(model_dir / "scaler_meta.json", {
        "input_cols": list(dataset.input_columns),
        "output_cols": list(dataset.output_columns),
        "log1p_targets": ["处理出口含湿量"],
        "feature_range": [-1.0, 1.0],
        "preprocessor_hash": _preprocessor_hash(dataset),
    })
    metadata = {
        "schema_version": 2,
        "profile": dataset.profile,
        "model_type": model_type,
        "input_cols": list(contract.input_columns),
        "output_cols": list(contract.output_columns),
        "direct_outputs": list(DIRECT_OUTPUT_COLUMNS),
        "derived_outputs": ["处理出口露点"],
        "target_transform": {"log1p_targets": ["处理出口含湿量"]},
        "architecture": {
            "name": "MLP" if model_type == "mlp" else "KAN",
            "hidden_widths": list(contract.hidden_widths),
            "grid_size": contract.grid_size,
            "spline_order": contract.spline_order,
            "in_features": contract.in_features,
            "out_features": contract.out_features,
        },
        "optimizer": {"name": "AdamW", "learning_rate": contract.learning_rate, "weight_decay": contract.weight_decay, "batch_size": int(batch_size), "seed": int(seed)},
        "data_fingerprint": dataset.data_fingerprint,
        "file_hashes": dataset.file_hashes,
        "preprocessor_hash": _preprocessor_hash(dataset),
    }
    _write_json(model_dir / "metadata.json", metadata)
    _write_json(output / "model_metadata.json", metadata)
    _write_snapshot(output / "config_snapshot.yaml", metadata)
    prediction_frame = _prediction_frame(dataset, test_predictions)
    prediction_frame.to_csv(reports / "test_predictions.csv", index=False, encoding="utf-8-sig")
    _write_json(reports / "evaluation.json", {"test": test_metrics})
    report = {
        "status": "completed",
        "profile": dataset.profile,
        "model_type": model_type,
        "output_dir": str(output),
        "rows": {split: len(dataset.frames[split]) for split in ("train", "validation", "test")},
        "geometry_groups": {split: [list(group) for group in groups] for split, groups in dataset.geometry_groups.items()},
        "data_fingerprint": dataset.data_fingerprint,
        "file_hashes": dataset.file_hashes,
        "preprocessor_hash": _preprocessor_hash(dataset),
        "parameter_count": model.parameter_count,
        "best_epoch": best_epoch,
        "global_steps": global_step,
        "best_validation_loss_scaled": best_validation_loss,
        "stop_reason": stop_reason,
        "elapsed_seconds": float(time.perf_counter() - started),
        "device": str(target_device),
        "test_metrics": test_metrics,
        "checkpoint_reload_max_abs_difference": float(np.max(np.abs(test_predictions - reloaded_predictions))),
        "checkpoint_reload_verified": True,
    }
    _write_json(output / "run_report.json", report)
    _write_json(output / "pipeline_report.json", report)
    return report


def run_candidate_suite(
    *,
    profile: str = "all",
    model_type: str = "all",
    dataset_dir: str | Path,
    output_root: str | Path | None = None,
    batch_size: int = 256,
    seed: int = 20260908,
    device: str = "cuda",
    smoke_test: bool = False,
    max_steps: int | None = None,
    confirm_full: bool = False,
) -> dict[str, Any]:
    if not smoke_test and not confirm_full:
        raise ValueError("全量候选训练需要 --confirm-full")
    profiles = ("xf", "main") if profile == "all" else (str(profile),)
    model_types = ("mlp", "kan") if model_type == "all" else (str(model_type),)
    parent = Path(output_root).resolve() if output_root is not None else TRAINING_RESULTS_DIR.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for current_profile in profiles:
        dataset = load_candidate_dataset(dataset_dir, current_profile)
        for current_model_type in model_types:
            label = "新风" if current_profile == "xf" else "主转轮"
            name = f"{datetime.now().astimezone():%Y%m%d_%H%M%S}_{label}_{current_model_type.upper()}_{uuid.uuid4().hex[:8]}"
            output = parent / name
            key = f"{current_profile}_{current_model_type}"
            try:
                results[key] = train_candidate(
                    dataset,
                    current_model_type,
                    output,
                    batch_size=batch_size,
                    seed=seed,
                    device=device,
                    smoke_test=smoke_test,
                    max_steps=max_steps,
                )
            except Exception as exc:
                failures[key] = str(exc)
                output.mkdir(parents=True, exist_ok=True)
                _write_json(output / "pipeline_report.json", {"status": "failed", "profile": current_profile, "model_type": current_model_type, "error": str(exc)})
    if failures:
        raise RuntimeError("候选训练失败: " + "; ".join(f"{key}: {value}" for key, value in failures.items()))
    return {"status": "completed", "results": results, "failures": failures}
