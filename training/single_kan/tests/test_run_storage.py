"""训练记录必须隔离运行、保留失败信息，并保护运行时模型。"""
import json
from pathlib import Path

import pytest

from training.single_kan import train


@pytest.fixture
def storage(monkeypatch, tmp_path):
    results = tmp_path / 'results' / 'training'
    artifacts = tmp_path / 'artifacts'
    monkeypatch.setattr(train, 'TRAINING_RESULTS_DIR', results, raising=False)
    monkeypatch.setattr(train, 'ARTIFACTS_DIR', artifacts)

    # Only the costly optimizer/data stage is replaced; storage uses real files.
    def finish(profile, *, output_root, **kwargs):
        folder = output_root / f'kan_{profile}_single'
        (folder / 'model').mkdir(parents=True)
        (folder / 'model' / 'best_model.pt').write_bytes(b'model-test')
        (folder / 'model_metadata.json').write_text('{}')
        (folder / 'config_snapshot.yaml').write_text('profile: ' + profile)
        return {'profile': profile, 'artifact_dir': str(folder)}

    monkeypatch.setattr(train, '_train_one_profile', finish)
    return results, artifacts


def test_default_smoke_runs_are_persistent_separate_and_never_install(storage):
    results, artifacts = storage
    first = train.run_pipeline(smoke_test=True)
    second = train.run_pipeline(smoke_test=True)
    a, b = Path(first['output_root']), Path(second['output_root'])
    assert a.parent == b.parent == results
    assert a != b
    assert (a / 'kan_xf_single/model/best_model.pt').read_bytes() == b'model-test'
    assert json.loads((a / 'pipeline_report.json').read_text())['status'] == 'completed'
    assert not artifacts.exists()


def test_default_full_keeps_history_and_installs_only_runtime_bundle(storage):
    results, artifacts = storage
    report = train.run_pipeline(confirm_full=True)
    run = Path(report['output_root'])
    assert run.parent == results
    assert (run / 'kan_main_single/model/best_model.pt').is_file()
    for profile in ('xf', 'main'):
        bundle = artifacts / f'kan_{profile}_single'
        assert (bundle / 'model/best_model.pt').read_bytes() == b'model-test'
        assert json.loads((bundle / 'training_run.json').read_text())['output_root'] == str(run)
        assert not (bundle / 'reports').exists()
    second = train.run_pipeline(confirm_full=True)
    assert Path(second['output_root']) != run
    assert (run / 'kan_main_single/model/best_model.pt').is_file()


def test_explicit_output_does_not_install_and_cannot_overwrite_existing_run(storage, tmp_path):
    _, artifacts = storage
    target = tmp_path / 'custom'
    train.run_pipeline(smoke_test=True, output_root=target)
    before = (target / 'kan_main_single/model/best_model.pt').read_bytes()
    with pytest.raises(train.PipelineError, match='已有|非空'):
        train.run_pipeline(smoke_test=True, output_root=target)
    assert (target / 'kan_main_single/model/best_model.pt').read_bytes() == before
    assert not artifacts.exists()


def test_failure_is_recorded_and_does_not_install_partial_models(storage, monkeypatch):
    results, artifacts = storage
    finish = train._train_one_profile

    def fail_main(profile, **kwargs):
        if profile == 'main':
            raise train.PipelineError('测试数据契约失败')
        return finish(profile, **kwargs)

    monkeypatch.setattr(train, '_train_one_profile', fail_main)
    with pytest.raises(train.PipelineError, match='main'):
        train.run_pipeline(confirm_full=True)
    reports = list(results.glob('*/pipeline_report.json'))
    assert len(reports) == 1
    record = json.loads(reports[0].read_text())
    assert record['status'] == 'failed'
    assert 'main' in record['failures']
    assert 'xf' in record['profiles']
    assert not artifacts.exists()
