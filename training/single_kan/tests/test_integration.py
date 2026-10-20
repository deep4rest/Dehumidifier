"""双模型单 KAN 集成流水线的红绿测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dual_rotor_sim import AirState, build_system, forward_simulate
from training.single_kan.config import SOURCE_PATHS
from training.single_kan.model import load_artifact
from training.single_kan.train import run_pipeline


pytestmark = pytest.mark.requires_data


def _requires_source_data(*profiles: str):
    """纯源码交接可跳过真实 XLS 集成；严格检查仍暴露缺文件错误。"""
    missing = [str(SOURCE_PATHS[profile]) for profile in profiles
               if not SOURCE_PATHS[profile].is_file()]
    return pytest.mark.skipif(
        bool(missing) and os.environ.get("REQUIRE_EXTERNAL_FIXTURES") != "1",
        reason="缺少单独提供的权威 XLS：" + "、".join(missing),
    )


@_requires_source_data("xf", "main")
def test_smoke_pipeline_trains_reloads_derives_dew_point_and_simulates(
    tmp_path: Path,
) -> None:
    """smoke 必须串起真实 Excel → 模型 → artifact → 仿真六个阶段。"""

    report = run_pipeline(
        profile="all",
        smoke_test=True,
        output_root=tmp_path,
        sample_size=500,
        max_steps=2,
        validation_interval_steps=1,
        batch_size=128,
        seed=17,
    )

    assert set(report["profiles"]) == {"xf", "main"}
    assert report["routing_generated"] is False
    for profile in ("xf", "main"):
        assert report["profiles"][profile]["power_comparison"][
            "runtime_correction_applied"
        ] is False
        assert {"bias_kW", "mae_kW", "rmse_kW", "anomaly_count"} <= set(
            report["profiles"][profile]["power_comparison"]
        )
        artifact_dir = tmp_path / ("kan_xf_single" if profile == "xf" else "kan_main_single")
        assert (artifact_dir / "model" / "best_model.pt").is_file()
        assert (artifact_dir / "model" / "scalers.pkl").is_file()
        assert not (artifact_dir / "reports" / "routing.json").exists()
        metadata = json.loads((artifact_dir / "model_metadata.json").read_text(encoding="utf-8"))
        assert len(metadata["input_cols"]) == (10 if profile == "xf" else 9)
        assert len(metadata["direct_outputs"]) == 6
        assert metadata["derived_features"] == []
        assert set(metadata["audit_columns"]) == {"power", "处理出口露点"}
        model, preprocessor, _ = load_artifact(artifact_dir)
        sample = np.zeros((2, len(metadata["input_cols"])), dtype=np.float32)
        prediction = model.predict(sample, preprocessor)
        assert prediction.shape == (2, 6)
        assert np.isfinite(prediction).all()

    # The runtime loader must use only the two fixed artifacts; no routing.json
    # is present in the temporary artifact root.
    system = build_system(artifacts_dir=tmp_path, device="cpu")
    result = forward_simulate(
        system,
        AirState(T=35.0, W=18.0, mdot=1.0),
        AirState(T=25.0, W=11.0, mdot=0.5),
        AirState(T=35.0, W=18.0, mdot=0.3),
    )
    assert np.isfinite(result.supply_state.T)
    # Zero humidity is a valid dry-limit boundary in the water-basis formula
    # and is represented as -inf; NaN would indicate a failed derivation.
    assert not np.isnan(result.rotor1_outputs.process_outlet_Tdp)
    assert not np.isnan(result.rotor2_outputs.process_outlet_Tdp)
    assert not hasattr(result, "electrical_kW")
    assert not hasattr(result.rotor1_outputs, "power_kW")


@_requires_source_data("main")
def test_pipeline_accepts_configurable_checkpoint_interval(tmp_path: Path) -> None:
    report = run_pipeline(
        profile="main",
        smoke_test=True,
        output_root=tmp_path,
        sample_size=500,
        max_steps=2,
        validation_interval_steps=1,
        checkpoint_interval_steps=1,
        batch_size=128,
        seed=19,
    )
    history = json.loads(
        (tmp_path / "kan_main_single" / "reports" / "train_history.json").read_text(
            encoding="utf-8"
        )
    )
    assert history["global_step"] == report["profiles"]["main"]["global_step"]
    log_lines = [
        json.loads(line)
        for line in (tmp_path / "kan_main_single" / "train.log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(item["kind"] == "checkpoint_saved" for item in log_lines)


@_requires_source_data("main")
def test_resume_writes_new_run_without_mutating_previous_checkpoints(tmp_path):
    import hashlib

    old = tmp_path / 'first'
    new = tmp_path / 'resumed'
    options = dict(profile='main', smoke_test=True, sample_size=500,
                   validation_interval_steps=1, checkpoint_interval_steps=1,
                   batch_size=128, seed=19)
    run_pipeline(output_root=old, max_steps=2, **options)
    before = {p.relative_to(old): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in old.rglob('*') if p.is_file()}
    report = run_pipeline(output_root=new, resume=old, max_steps=4, **options)
    after = {p.relative_to(old): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in old.rglob('*') if p.is_file()}
    assert before == after
    assert report['profiles']['main']['global_step'] == 4
    assert (new / 'checkpoints/main/checkpoint.pt').is_file()


@_requires_source_data("main")
def test_short_smoke_before_validation_interval_saves_valid_report(tmp_path):
    report = run_pipeline(profile='main', smoke_test=True, output_root=tmp_path,
                          max_steps=2, sample_size=500, seed=19)
    assert report['status'] == 'completed'
    history = json.loads((tmp_path / 'kan_main_single/reports/train_history.json').read_text())
    assert history['validation_steps'] == []
    assert history['best_validation_loss'] is None
    assert (tmp_path / 'kan_main_single/model/best_model.pt').is_file()
    events = [json.loads(line) for line in
              (tmp_path / 'kan_main_single/train.log.jsonl').read_text().splitlines()]
    assert [event['global_step'] for event in events
            if event['kind'] == 'checkpoint_saved'] == [2]
