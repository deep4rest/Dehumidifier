"""双转轮单 KAN 训练集成入口。

该入口只负责把数据契约、单模型、评估器和训练器串成一条可复现流水线。
``--smoke-test`` 仅使用每份权威 Excel 的约 500 行代表性副本；非 smoke
运行必须显式传入 ``--confirm-full``，避免在小样本审计尚未确认前误启动全量
训练。正式 artifact 不包含 routing.json，也不把 ``power`` 或 Excel 露点列
送入模型张量。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Support both ``python -m training.single_kan.train`` and
# direct execution of this file.  The latter is useful for operators invoking a
# checked-out experiment from its directory, but Python does not establish a
# package context for direct scripts, so the repository root must be added first.
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    __package__ = "training.single_kan"

from .config import ARTIFACTS_DIR, TRAINING_RESULTS_DIR, PROFILE_CONFIGS, get_profile
from .data import (
    DataContractError,
    PreparedDataset,
    clean_dataset,
    make_model_arrays,
    prepare_profile_dataset,
    prepare_sample,
    split_dataset,
)
from .evaluator import (
    EvaluationError,
    compare_heater_power,
    evaluate_checkpoint,
    evaluate_in_memory,
)
from .model import build_metadata, load_artifact, save_artifact
from .trainer import (
    CheckpointError,
    SingleKANTrainer,
    TrainingDataError,
    load_checkpoint_payload,
    set_deterministic_seed,
)

from simulation.dual_rotor_sim.core.system.psychrometrics import enthalpy_kJ

try:  # pragma: no cover - tqdm 本身在 requirements 中，但保留轻量 fallback
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """集成训练流水线的中文边界错误。"""


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
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _write_yaml_or_json(path: Path, payload: Mapping[str, Any]) -> None:
    """优先写 YAML 快照；没有 PyYAML 时仍写合法 JSON 内容。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    # ``asdict(RotorProfileConfig)`` contains Path objects.  Normalize all
    # scalar/container values once so both YAML and JSON fallbacks have the
    # same deterministic representation.
    normalized = json.loads(
        json.dumps(payload, ensure_ascii=False, default=_json_default)
    )
    try:
        import yaml

        text = yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False)
    except ImportError:  # pragma: no cover - 项目 CLI 已依赖 PyYAML
        text = json.dumps(normalized, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


class _StepRecorder:
    """把逐步事件同时写入 JSONL 日志和终端进度条。"""

    def __init__(self, path: Path, *, profile: str, total_steps: int, show_progress: bool) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8")
        self.profile = profile
        self.progress = None
        if show_progress and tqdm is not None:
            self.progress = tqdm(
                total=max(1, int(total_steps)),
                desc=f"{profile} KAN",
                unit="step",
                dynamic_ncols=True,
            )

    def __call__(self, event: Mapping[str, Any]) -> None:
        payload = {"timestamp": time.time(), "profile": self.profile, **dict(event)}
        self.stream.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        self.stream.flush()
        if self.progress is None:
            return
        if payload.get("kind") == "train_step":
            self.progress.update(1)
            self.progress.set_postfix(loss=f"{float(payload['loss']):.4g}")
        elif payload.get("kind") == "validation_end":
            self.progress.set_postfix(
                loss=f"{float(payload['val_loss']):.4g}",
                val=f"step {int(payload['global_step'])}",
            )

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()
        self.stream.close()


def _prepare_dataset(profile: str, *, smoke_test: bool, sample_size: int) -> PreparedDataset:
    """读取一个 profile；smoke 只把代表性副本送入后续处理。"""

    if smoke_test:
        sample = prepare_sample(profile, n=sample_size)
        clean, report = clean_dataset(profile, sample)
        assignments = split_dataset(profile, clean)
        return PreparedDataset(
            profile=profile,
            raw=sample,
            clean=clean,
            assignments=assignments,
            source_sha256=str(sample.attrs.get("source_sha256")),
            header_mapping=dict(sample.attrs.get("header_mapping", {})),
            cleaning_report=report,
        )
    return prepare_profile_dataset(profile)


def _split_counts(assignments) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": {}, "geometry_groups": {}}
    for partition, group in assignments.groupby("partition", sort=True):
        result["rows"][str(partition)] = int(len(group))
        result["geometry_groups"][str(partition)] = sorted(
            {
                (int(depth), int(diameter))
                for depth, diameter in zip(
                    group["rotor_thickness_mm"], group["rotor_diameter_mm"]
                )
            }
        )
        result["geometry_groups"][str(partition)] = [
            list(pair) for pair in result["geometry_groups"][str(partition)]
        ]
    return result


def _extract_arrays(dataset: PreparedDataset, profile: str) -> dict[str, tuple[np.ndarray, np.ndarray, Any]]:
    assignments = dataset.assignments
    arrays: dict[str, tuple[np.ndarray, np.ndarray, Any]] = {}
    for split in ("train", "validation", "test"):
        try:
            arrays[split] = make_model_arrays(profile, dataset.clean, assignments, split)
        except DataContractError as exc:
            raise PipelineError(f"{profile} {split} 数据契约失败: {exc}") from exc
    return arrays


def _resolve_resume(path: str | Path | None, profile: str) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    candidates = (
        candidate / "checkpoints" / profile / "checkpoint.pt",
        candidate / ".checkpoints" / profile / "checkpoint.pt",
        candidate / profile / "checkpoint.pt",
        candidate / f"{profile}_checkpoint.pt",
        candidate / "checkpoint.pt",
    )
    for item in candidates:
        if item.is_file():
            return item
    raise PipelineError(f"{profile} 找不到可恢复 checkpoint: {candidate}")


def _power_comparison(frame, profile: str) -> dict[str, Any]:
    """构造单转轮样本级加热器热负荷代理并做离线 power 对照。

    样本没有整机三股混风状态，因此这里仅用 ``冷吹出口`` 到 ``再生温度``
    的显热焓差作为可审计参考：主转轮对应低湿加热支路，新风转轮对应其
    再生风支路。该代理只写入离线报告，绝不作为运行时校正或训练字段。
    """

    required = {
        "power",
        "冷吹出口温度",
        "冷吹出口含湿量",
        "再生温度(℃)",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PipelineError(f"{profile} 无法生成 power 离线对照，缺少字段: {missing}")
    flow_column = "Label_React_CMH" if profile == "xf" else "TextBox_Purge_CMH"
    if flow_column not in frame.columns:
        raise PipelineError(
            f"{profile} 无法生成 power 离线对照，缺少原始再生/冷吹风量字段: {flow_column}"
        )

    power = frame["power"].to_numpy(dtype=np.float64)
    inlet_t = frame["冷吹出口温度"].to_numpy(dtype=np.float64)
    inlet_w = frame["冷吹出口含湿量"].to_numpy(dtype=np.float64)
    outlet_t = frame["再生温度(℃)"].to_numpy(dtype=np.float64)
    flow_m3h = frame[flow_column].to_numpy(dtype=np.float64)
    finite = np.isfinite(inlet_t) & np.isfinite(inlet_w) & np.isfinite(outlet_t) & np.isfinite(flow_m3h)
    if not finite.all() or np.any(flow_m3h[finite] <= 0.0):
        raise PipelineError(f"{profile} power 离线对照的热负荷输入存在非有限值或非正风量")

    humidity_ratio = inlet_w / 1000.0
    dry_air_pressure = 101325.0 * 0.622 / (0.622 + humidity_ratio)
    density = dry_air_pressure / (287.05 * (inlet_t + 273.15))
    mdot = flow_m3h * density / 3600.0
    h_in = np.asarray(enthalpy_kJ(inlet_t, inlet_w), dtype=np.float64)
    h_out = np.asarray(enthalpy_kJ(outlet_t, inlet_w), dtype=np.float64)
    heater_load = mdot * np.maximum(h_out - h_in, 0.0)
    if not np.isfinite(heater_load).all():
        raise PipelineError(f"{profile} power 离线对照热负荷产生非有限值")
    report = compare_heater_power(power, heater_load)
    report.update(
        {
            "profile": profile,
            "measured_column": "power",
            "reference": "冷吹出口→再生温度显热焓差代理",
            "reference_flow_column": flow_column,
            "pressure_Pa": 101325.0,
            "unit_assumption": "power 按 kW；体积流量按 m3/h",
        }
    )
    return report


def _train_one_profile(
    profile: str,
    *,
    output_root: Path,
    smoke_test: bool,
    sample_size: int,
    max_steps: int | None,
    validation_interval_steps: int | None,
    checkpoint_interval_steps: int | None,
    batch_size: int,
    seed: int,
    device: str,
    resume: str | Path | None,
    show_progress: bool,
) -> dict[str, Any]:
    cfg = get_profile(profile)
    dataset = _prepare_dataset(profile, smoke_test=smoke_test, sample_size=sample_size)
    arrays = _extract_arrays(dataset, profile)
    data_context = {
        "source_sha256": dataset.source_sha256,
        "smoke_test": bool(smoke_test),
        "sample_size": int(sample_size) if smoke_test else None,
        "profile": profile,
        "partition_rows": dataset.assignments[["ID", "partition"]].astype({"ID": int}).to_dict("records"),
        "source_row_order": dataset.clean["ID"].astype(int).tolist(),
    }

    artifact_dir = output_root / ("kan_xf_single" if profile == "xf" else "kan_main_single")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints" / profile
    checkpoint_path = _resolve_resume(resume, profile)
    if checkpoint_path is not None:
        # 恢复从本次目录的副本读取，后续 best/checkpoint 写入不能污染旧运行。
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        source_best = checkpoint_path.parent / "best_model.pt"
        shutil.copy2(checkpoint_path, checkpoint_dir / "checkpoint.pt")
        if source_best.is_file():
            shutil.copy2(source_best, checkpoint_dir / "best_model.pt")
        checkpoint_path = checkpoint_dir / "checkpoint.pt"

    if max_steps is None:
        # Full training uses the profile's fixed epoch budget.  Integration
        # validation always supplies --max-steps explicitly.
        steps_per_epoch = max(1, math.ceil(len(arrays["train"][0]) / max(1, int(batch_size))))
        total_steps = cfg.max_epochs * steps_per_epoch
    else:
        total_steps = int(max_steps)
        if total_steps <= 0:
            raise PipelineError("max_steps 必须为正整数")
    interval = (
        int(validation_interval_steps)
        if validation_interval_steps is not None
        else int(cfg.validation_interval_steps)
    )
    if interval <= 0:
        raise PipelineError("validation_interval_steps 必须为正整数")
    checkpoint_interval = (
        int(checkpoint_interval_steps)
        if checkpoint_interval_steps is not None
        else int(cfg.checkpoint_interval_steps)
    )
    if checkpoint_interval <= 0:
        raise PipelineError("checkpoint_interval_steps 必须为正整数")

    recorder = _StepRecorder(
        artifact_dir / "train.log.jsonl",
        profile=profile,
        total_steps=total_steps,
        show_progress=show_progress,
    )
    started = time.perf_counter()
    try:
        set_deterministic_seed(seed)
        if checkpoint_path is None:
            trainer = SingleKANTrainer(
                profile=profile,
                train_features=arrays["train"][0],
                train_targets=arrays["train"][1],
                validation_features=arrays["validation"][0],
                validation_targets=arrays["validation"][1],
                checkpoint_dir=checkpoint_dir,
                validation_interval_steps=interval,
                checkpoint_interval_steps=checkpoint_interval,
                batch_size=batch_size,
                device=device,
                seed=seed,
                event_sink=recorder,
                data_context=data_context,
            )
        else:
            trainer = SingleKANTrainer.from_checkpoint(
                checkpoint_path, device=device,
                train_features=arrays["train"][0], train_targets=arrays["train"][1],
                validation_features=arrays["validation"][0], validation_targets=arrays["validation"][1],
                data_context=data_context, event_sink=recorder,
                expected_controls={"profile": profile, "seed": seed, "batch_size": batch_size,
                    "validation_interval_steps": interval, "checkpoint_interval_steps": checkpoint_interval},
            )

        if max_steps is None:
            history = trainer.fit()
        else:
            remaining_steps = max(0, total_steps - int(trainer.history.global_step))
            history = (
                trainer.train_steps(remaining_steps)
                if remaining_steps
                else trainer.history
            )
        # 训练循环已保存最终状态；无新增 step 时沿用本次目录中的恢复副本。
        checkpoint = checkpoint_dir / "checkpoint.pt"
        best_checkpoint = trainer._best_checkpoint  # noqa: SLF001 - artifact 需要最佳状态
        if not best_checkpoint.is_file():
            best_checkpoint = checkpoint

        # The same evaluator core is used for in-memory and checkpoint modes.
        val_x, val_y, val_audit = arrays["validation"]
        test_x, test_y, test_audit = arrays["test"]
        in_memory_eval = evaluate_in_memory(
            trainer.model,
            val_x,
            val_y,
            trainer.preprocessor,
            pressure_Pa=101325.0,
            excel_dew_point=val_audit["处理出口露点"].to_numpy(dtype=np.float64),
            device=device,
            event_sink=lambda event: recorder({
                "kind": f"evaluation_{event.kind}",
                "mode": event.mode,
                "completed": event.completed,
                "total": event.total,
                **dict(event.payload),
            }),
        )
        checkpoint_eval = evaluate_checkpoint(
            best_checkpoint,
            trainer.model,
            val_x,
            val_y,
            trainer.preprocessor,
            pressure_Pa=101325.0,
            excel_dew_point=val_audit["处理出口露点"].to_numpy(dtype=np.float64),
            device=device,
            event_sink=lambda event: recorder({
                "kind": f"evaluation_{event.kind}",
                "mode": event.mode,
                "completed": event.completed,
                "total": event.total,
                **dict(event.payload),
            }),
        )
        test_eval = evaluate_in_memory(
            trainer.model,
            test_x,
            test_y,
            trainer.preprocessor,
            pressure_Pa=101325.0,
            excel_dew_point=test_audit["处理出口露点"].to_numpy(dtype=np.float64),
            device=device,
        )

        metadata = build_metadata(profile, source_sha256=dataset.source_sha256)
        metadata.update(
            {
                "sample_size": int(len(dataset.clean)) if smoke_test else None,
                "smoke_test": bool(smoke_test),
                "split_policy": cfg.split_policy,
                "checkpoint_best_step": int(history.best_step),
                "checkpoint_interval_steps": checkpoint_interval,
            }
        )
        best_payload = load_checkpoint_payload(best_checkpoint, device=device)
        save_artifact(
            artifact_dir,
            trainer.model,
            trainer.preprocessor,
            metadata=metadata,
            checkpoint_payload=best_payload,
        )
        reports_dir = artifact_dir / "reports"
        _write_json(reports_dir / "cleaning_report.json", dataset.cleaning_report)
        _write_json(reports_dir / "split_report.json", _split_counts(dataset.assignments))
        _write_json(reports_dir / "train_history.json", history.to_dict())
        _write_json(
            reports_dir / "evaluation.json",
            {
                "in_memory": in_memory_eval.to_dict(),
                "checkpoint": checkpoint_eval.to_dict(),
                "test": test_eval.to_dict(),
            },
        )
        power_comparison = _power_comparison(dataset.clean, profile)
        _write_json(reports_dir / "power_comparison.json", power_comparison)
        _write_yaml_or_json(
            artifact_dir / "config_snapshot.yaml",
            {
                "profile": profile,
                "contract": asdict(cfg),
                "source_sha256": dataset.source_sha256,
                "smoke_test": smoke_test,
                "sample_size": sample_size if smoke_test else None,
                "seed": seed,
                "validation_interval_steps": interval,
                "checkpoint_interval_steps": checkpoint_interval,
                "batch_size": batch_size,
                "max_steps": max_steps,
            },
        )
        elapsed = time.perf_counter() - started
        result = {
            "profile": profile,
            "artifact_dir": str(artifact_dir),
            "source_sha256": dataset.source_sha256,
            "rows": int(len(dataset.clean)),
            "split_counts": _split_counts(dataset.assignments),
            "global_step": int(history.global_step),
            "best_step": int(history.best_step),
            "elapsed_seconds": float(elapsed),
            "evaluation": {
                "in_memory": in_memory_eval.to_dict(),
                "checkpoint": checkpoint_eval.to_dict(),
                "test": test_eval.to_dict(),
            },
            "power_comparison": power_comparison,
        }
        _write_json(artifact_dir / "run_report.json", result)
        return result
    except (DataContractError, TrainingDataError, CheckpointError, EvaluationError, OSError, ValueError) as exc:
        raise PipelineError(f"{profile} 集成流水线失败: {exc}") from exc
    finally:
        recorder.close()


def _install_runtime_artifacts(root: Path, profiles: tuple[str, ...]) -> None:
    """从完成的全量运行提取运行时包；旧包留在该次运行内作历史记录。"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install_", dir=ARTIFACTS_DIR) as staging:
        stage = Path(staging)
        # 先准备所有包，复制失败时不触碰当前模型。
        for profile in profiles:
            name = f"kan_{profile}_single"
            source, target = root / name, stage / name
            shutil.copytree(source / "model", target / "model")
            for filename in ("model_metadata.json", "config_snapshot.yaml"):
                shutil.copy2(source / filename, target / filename)
            _write_json(target / "training_run.json", {"output_root": str(root), "profile": profile})
        for profile in profiles:
            name = f"kan_{profile}_single"
            target = ARTIFACTS_DIR / name
            if target.exists():
                previous = root / "previous_artifacts" / name
                previous.parent.mkdir(parents=True, exist_ok=True)
                target.rename(previous)
            (stage / name).rename(target)


def run_pipeline(
    *,
    profile: str = "all",
    smoke_test: bool = False,
    output_root: str | Path | None = None,
    sample_size: int = 500,
    max_steps: int | None = None,
    validation_interval_steps: int | None = None,
    checkpoint_interval_steps: int | None = None,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
    resume: str | Path | None = None,
    show_progress: bool = False,
    confirm_full: bool = False,
) -> dict[str, Any]:
    """运行一个或两个 profile，返回可序列化的汇总。

    ``confirm_full`` 是全量训练的显式安全门；它不会影响 smoke 流程，也不会
    修改任何原始 XLS。默认按次写入 results/training；只有默认全量运行完成
    后才更新 artifacts。显式 ``output_root`` 必须为空，且不会更新运行时模型。
    """

    key = str(profile).strip().lower()
    if key not in {"xf", "main", "all"}:
        raise PipelineError(f"未知 profile: {profile!r}，仅支持 xf/main/all")
    if not smoke_test and not confirm_full:
        raise PipelineError("全量处理/训练尚未获确认；请先完成小样本审计，再传入 --confirm-full")
    if int(sample_size) <= 0:
        raise PipelineError("sample_size 必须为正整数")
    install_runtime = output_root is None and not smoke_test
    if output_root is None:
        TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        mode = "smoke" if smoke_test else "full"
        prefix = f"{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}_{mode}_{key}_"
        root = Path(tempfile.mkdtemp(prefix=prefix, dir=TRAINING_RESULTS_DIR)).resolve()
    else:
        root = Path(output_root).resolve()
        if root.exists() and any(root.iterdir()):
            raise PipelineError(f"输出目录已有文件，请指定新的空目录；恢复训练用 --resume 指向旧记录：{root}")
        root.mkdir(parents=True, exist_ok=True)
    profiles = ("xf", "main") if key == "all" else (key,)
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    report = {
        "profiles": results,
        "failures": failures,
        "output_root": str(root),
        "routing_generated": False,
        "smoke_test": bool(smoke_test),
        "status": "running",
        "runtime_artifacts_updated": False,
    }
    _write_json(root / "pipeline_report.json", report)
    for item in profiles:
        try:
            results[item] = _train_one_profile(
                item,
                output_root=root,
                smoke_test=smoke_test,
                sample_size=int(sample_size),
                max_steps=max_steps,
                validation_interval_steps=validation_interval_steps,
                checkpoint_interval_steps=checkpoint_interval_steps,
                batch_size=int(batch_size),
                seed=int(seed),
                device=device,
                resume=resume,
                show_progress=show_progress,
            )
        except Exception as exc:
            failures[item] = str(exc)
        _write_json(root / "pipeline_report.json", report)
    if failures:
        report["status"] = "failed"
        _write_json(root / "pipeline_report.json", report)
        detail = "；".join(f"{name}: {message}" for name, message in failures.items())
        raise PipelineError(f"部分 profile 失败: {detail}")
    if install_runtime:
        try:
            _install_runtime_artifacts(root, profiles)
        except OSError as exc:
            report["status"] = "failed"
            failures["runtime_install"] = str(exc)
            _write_json(root / "pipeline_report.json", report)
            raise PipelineError(f"运行时模型写入失败，训练记录已保留：{root}：{exc}") from exc
        report["runtime_artifacts_updated"] = True
    report["status"] = "completed"
    report["routing_generated"] = any(root.rglob("routing.json"))
    _write_json(root / "pipeline_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="双转轮单 KAN 重训与露点契约流水线")
    parser.add_argument("--profile", choices=("xf", "main", "all"), default="all")
    parser.add_argument("--smoke-test", action="store_true", help="只训练每份权威源的约 500 行代表性副本")
    parser.add_argument("--confirm-full", action="store_true", help="确认允许读取并训练全量数据")
    parser.add_argument("--output-root", type=Path, default=None, help="指定空运行目录；默认按次写入 results/training，默认全量完成后更新 artifacts")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validation-interval-steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path, default=None, help="旧运行目录或 checkpoint；恢复结果写入新目录")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_pipeline(
            profile=args.profile,
            smoke_test=args.smoke_test,
            output_root=args.output_root,
            sample_size=args.sample_size,
            max_steps=args.max_steps,
            validation_interval_steps=args.validation_interval_steps,
            checkpoint_interval_steps=args.checkpoint_interval_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
            show_progress=not args.no_progress,
            confirm_full=args.confirm_full,
        )
    except PipelineError as exc:
        print(f"[失败] {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
