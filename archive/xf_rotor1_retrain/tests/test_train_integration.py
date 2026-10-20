"""XF 最终训练流水线的真实数据集成规格。"""

from __future__ import annotations

import ast
import json
import math
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archive.xf_rotor1_retrain.candidates import CANDIDATES, get_candidate
from archive.xf_rotor1_retrain.config import (
    FIXED_FOLD_GROUPS,
    MODEL_OUTPUT_COLUMNS,
)
from archive.xf_rotor1_retrain.evaluator import (
    EvaluationBoundaryError,
    build_fold_data,
    evaluate_checkpoint,
    evaluate_in_memory,
)
from archive.xf_rotor1_retrain.train import (
    WORK_DIR,
    ExecutionGuard,
    PipelineError,
    RunLogger,
    RunSpec,
    SharedValidationCallback,
    build_checkpoint_predictor_loader,
    build_invocation_plan,
    build_scaled_predictor,
    evaluation_to_oof_frame,
    extract_run_arrays,
    load_authoritative_dataset,
    resolve_run_dir,
    validate_complete_matrix,
)
from archive.xf_rotor1_retrain.trainer import (
    ValidationSnapshot,
    XFTrainer,
    load_checkpoint_payload,
)


@pytest.fixture(scope="session")
def authoritative_dataset():
    """每会话只回读一次固定 XF 源和已提交报告快照。"""

    return load_authoritative_dataset()


def _unique_run_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def test_smoke_plan_is_one_real_run_with_fixed_2000_step_cadence() -> None:
    """捕获 smoke 退化为不触发完整验证的几步玩具路径。"""

    plan = build_invocation_plan(
        ["--smoke-test", "--max-runtime-seconds", "300"]
    )

    assert plan.mode == "smoke"
    assert plan.max_runtime_seconds == 300.0
    assert len(plan.runs) == 1
    run = plan.runs[0]
    assert run.candidate_name == "xf_v1_baseline"
    assert run.fold_name == "fold_1"
    assert run.planned_steps == 2_000
    assert run.train_row_limit == 512
    assert run.validation_row_limit == 128
    assert get_candidate(run.candidate_name).validation_interval_steps == 2_000


def test_full_matrix_plan_contains_exactly_five_candidates_by_five_folds() -> None:
    """捕获完整入口减少候选或 fold。"""

    plan = build_invocation_plan([])
    actual = {(run.candidate_name, run.fold_name) for run in plan.runs}
    expected = {
        (candidate_name, fold_name)
        for candidate_name in CANDIDATES
        for fold_name in FIXED_FOLD_GROUPS
    }

    assert plan.mode == "matrix"
    assert actual == expected
    assert len(plan.runs) == 25
    assert all(run.planned_steps is None for run in plan.runs)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--candidate", "xf_v1_baseline"],
        ["--fold", "fold_1"],
        ["--candidate", "xf_v1_baseline", "--fold", "fold_1", "--smoke-test"],
        ["--max-runtime-seconds", "0"],
    ],
)
def test_cli_rejects_partial_selection_or_invalid_runtime(arguments: list[str]) -> None:
    """单运行必须同时锁定候选/fold，限时必须为正。"""

    with pytest.raises(SystemExit):
        build_invocation_plan(arguments)


@pytest.mark.parametrize(
    "forbidden",
    [
        ["--data-source", "other.xls"],
        ["--artifact-dir", "elsewhere"],
        ["--allow-unpromoted"],
        ["--checkpoint", "arbitrary.pt"],
    ],
)
def test_cli_has_no_data_artifact_or_bypass_override(forbidden: list[str]) -> None:
    """捕获通用数据源、artifact、checkpoint 或未晋升绕过入口。"""

    with pytest.raises(SystemExit):
        build_invocation_plan(forbidden)


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", "a\\b", "", "."])
def test_run_name_cannot_escape_fixed_experiment_work_dir(unsafe: str) -> None:
    """运行名只能组成实验 work 目录下的受控相对路径。"""

    with pytest.raises(PipelineError, match="run-name|运行名|路径"):
        resolve_run_dir(unsafe, "xf_v1_baseline", "fold_1")


def test_resolved_run_dir_is_fixed_under_experiment_work_dir() -> None:
    """捕获输出根目录变成调用方可控路径。"""

    path = resolve_run_dir("audited_run", "xf_v2_balanced", "fold_3")
    assert path == WORK_DIR / "audited_run" / "xf_v2_balanced" / "fold_3"
    assert path.resolve().is_relative_to(WORK_DIR.resolve())


def test_real_xf_subset_runs_train_checkpoint_and_both_shared_evaluator_modes(
    authoritative_dataset,
    tmp_path: Path,
) -> None:
    """真实 XF 小流必须贯通 scale→forward/backward→checkpoint→OOF。"""

    fold = build_fold_data(
        authoritative_dataset.clean,
        authoritative_dataset.assignments,
        "fold_1",
    )
    spec = RunSpec(
        candidate_name="xf_v1_baseline",
        fold_name="fold_1",
        planned_steps=1,
        train_row_limit=128,
        validation_row_limit=32,
    )
    arrays = extract_run_arrays(authoritative_dataset.clean, fold, spec)
    evaluator_events = []
    trainer = XFTrainer(
        candidate=get_candidate(spec.candidate_name),
        train_features=arrays.train_features,
        train_targets=arrays.train_targets,
        train_ids=arrays.train_ids,
        checkpoint_dir=tmp_path,
        device="cpu",
        validation_interval_steps=1,
    )
    shared_callback = SharedValidationCallback(
        validation_ids=arrays.validation_ids,
        train_targets_physical=arrays.train_targets,
        pressure_Pa=101_325.0,
        event_sink=evaluator_events.append,
    )
    trainer.train_steps(
        1,
        validation_features=arrays.validation_features,
        validation_targets=arrays.validation_targets,
        validation_callback=shared_callback,
    )

    assert trainer.preprocessor.x_scaler.fit_ids == arrays.train_ids
    assert set(arrays.train_ids).isdisjoint(arrays.validation_ids)
    assert set(arrays.train_ids).isdisjoint(fold.outer_ids)
    assert {event.mode for event in evaluator_events} == {"in_memory"}
    assert {event.kind for event in evaluator_events} >= {
        "start",
        "progress",
        "end",
        "result",
        "efficiency",
    }
    best = tmp_path / "best.pt"
    last = tmp_path / "last.pt"
    assert load_checkpoint_payload(best)["global_step"] == 1
    assert load_checkpoint_payload(last)["global_step"] == 1

    validation_scaled = trainer.preprocessor.transform_features(
        arrays.validation_features
    )
    targets_scaled = trainer.preprocessor.transform_targets(
        arrays.validation_targets
    )
    common = dict(
        features=validation_scaled,
        targets_model_space=targets_scaled,
        train_targets_physical=arrays.train_targets,
        sample_ids=arrays.validation_ids,
        pressure_Pa=101_325.0,
        inverse_target_transform=trainer.preprocessor.y_scaler,
        inverse_scaler=trainer.preprocessor.target_transform,
        batch_size=16,
    )
    in_memory = evaluate_in_memory(build_scaled_predictor(trainer), **common)
    restored = evaluate_checkpoint(
        last,
        build_checkpoint_predictor_loader(
            train_features=arrays.train_features,
            train_targets=arrays.train_targets,
            train_ids=arrays.train_ids,
            checkpoint_dir=tmp_path,
            device="cpu",
        ),
        **common,
    )

    np.testing.assert_array_equal(
        in_memory.predictions_raw,
        restored.predictions_raw,
    )
    np.testing.assert_array_equal(
        in_memory.predictions_deployed,
        restored.predictions_deployed,
    )
    assert all(
        math.isfinite(metric.nrmse)
        for metric in restored.deployed_metrics.values()
    )
    oof = evaluation_to_oof_frame(restored)
    assert list(oof.columns) == [
        "ID",
        *[f"prediction_raw__{name}" for name in MODEL_OUTPUT_COLUMNS],
        *[f"prediction_deployed__{name}" for name in MODEL_OUTPUT_COLUMNS],
        *[f"target_raw__{name}" for name in MODEL_OUTPUT_COLUMNS],
    ]
    assert len(oof) == len(arrays.validation_ids) == 32
    assert oof["ID"].is_unique


def test_shared_evaluator_rejects_empty_validation_with_mode_aware_failure(
    authoritative_dataset,
    tmp_path: Path,
) -> None:
    """空 validation 不得跳过或产生零指标。"""

    fold = build_fold_data(
        authoritative_dataset.clean,
        authoritative_dataset.assignments,
        "fold_1",
    )
    spec = RunSpec("xf_v1_baseline", "fold_1", 1, 64, 16)
    arrays = extract_run_arrays(authoritative_dataset.clean, fold, spec)
    trainer = XFTrainer(
        candidate=get_candidate(spec.candidate_name),
        train_features=arrays.train_features,
        train_targets=arrays.train_targets,
        train_ids=arrays.train_ids,
        checkpoint_dir=tmp_path,
        validation_interval_steps=1,
    )
    callback = SharedValidationCallback(
        validation_ids=(),
        train_targets_physical=arrays.train_targets,
        pressure_Pa=101_325.0,
    )
    with pytest.raises(EvaluationBoundaryError, match="in_memory.*validation.*空"):
        callback(
            ValidationSnapshot(
                model=trainer.model,
                preprocessor=trainer.preprocessor,
                device=trainer.device,
                global_step=trainer.global_step,
            ),
            np.empty((0, 10), dtype=np.float64),
            np.empty((0, 7), dtype=np.float64),
        )


def test_execution_guard_hard_fails_explicit_runtime_expiry() -> None:
    """max-runtime 是生产运行控制，超时时必须非成功返回。"""

    guard = ExecutionGuard(max_runtime_seconds=0.001)
    wait_deadline = time.perf_counter() + 1.0
    while guard.elapsed_seconds <= 0.001:
        if time.perf_counter() > wait_deadline:
            pytest.fail("系统 monotonic clock 在 1s 内未向前进")
    with pytest.raises(PipelineError, match="超时|max-runtime"):
        guard.progress("test")


def test_jsonl_and_csv_logs_round_trip_required_health_and_six_stages() -> None:
    """捕获日志仅打印到控制台或缺少健康/阶段字段。"""

    run_name = _unique_run_name("logger")
    run_dir = resolve_run_dir(run_name, "xf_v1_baseline", "fold_1")
    try:
        logger = RunLogger(
            run_dir,
            candidate_name="xf_v1_baseline",
            fold_name="fold_1",
            resume=False,
        )
        for stage in range(1, 7):
            logger.stage(stage, "start", duration_seconds=0.0)
            logger.stage(stage, "end", duration_seconds=0.01)
        logger.training_event(
            {
                "kind": "train_step",
                "loss": 0.5,
                "grad_norm": 0.25,
                "lr": 1.0e-3,
                "global_step": 1,
                "step_time": 0.02,
                "samples_s": 3200.0,
                "memory": 1024,
                "validation_progress": {"completed": 0, "total": 0},
                "mfu": "not_applicable",
            }
        )
        logger.evaluation_event(
            type(
                "Event",
                (),
                {
                    "kind": "progress",
                    "mode": "in_memory",
                    "completed": 16,
                    "total": 32,
                    "payload": {"elapsed_seconds": 0.1, "samples_per_second": 160.0},
                },
            )()
        )

        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        losses = pd.read_csv(run_dir / "loss.csv")
        speed = pd.read_csv(run_dir / "speed.csv")
        assert {
            event["stage"]
            for event in events
            if event["kind"] == "stage" and event["status"] == "end"
        } == {1, 2, 3, 4, 5, 6}
        assert losses.loc[0, "mfu"] == "not_applicable"
        assert losses.loc[0, "grad_norm"] == pytest.approx(0.25)
        assert speed.loc[0, "mode"] == "in_memory"
        assert speed.loc[0, "completed"] == 16
    finally:
        shutil.rmtree(WORK_DIR / run_name, ignore_errors=True)


def test_complete_matrix_validation_rejects_missing_or_nonfinite_results() -> None:
    """路由聚合前必须恰好拥有 25 个有限运行结果。"""

    with pytest.raises(PipelineError, match="25|矩阵|缺失"):
        validate_complete_matrix({})


def test_train_core_does_not_import_tests_or_validation_toolkit() -> None:
    """生产流水线不得反向依赖 tests 或 Validation Pyramid 工具。"""

    path = Path(__file__).resolve().parents[1] / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert all("tests" not in name and "validation_pyramid" not in name for name in imported)
