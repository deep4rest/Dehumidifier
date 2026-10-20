"""审查问题的行为回归：存储、恢复、评估和独立运行时加载。"""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from training.single_kan.config import get_profile
from training.single_kan.model import fit_preprocessor, build_single_kan, save_artifact, load_artifact
from training.single_kan.trainer import SingleKANTrainer, TrainHistory, CheckpointError
from training.single_kan.evaluator import evaluate_predictions
from dual_rotor_sim.core.system.components.rotor import SingleRotorModel


def arrays():
    rng = np.random.default_rng(71)
    return rng.normal(size=(16, 9)), rng.uniform(10, 20, size=(16, 6))


def trainer(path, **kwargs):
    x, y = arrays()
    return SingleKANTrainer(profile='main', train_features=x, train_targets=y,
        validation_features=x[:8], validation_targets=y[:8], checkpoint_dir=path,
        batch_size=4, validation_interval_steps=kwargs.pop('validation_interval_steps', 2000),
        checkpoint_interval_steps=kwargs.pop('checkpoint_interval_steps', 2000), **kwargs)


def test_fit_saves_at_interval_and_once_at_end(tmp_path):
    events = []
    t = trainer(tmp_path, checkpoint_interval_steps=3, event_sink=events.append)
    t.fit(epochs=1)
    saved_steps = [e['global_step'] for e in events if e['kind'] == 'checkpoint_saved']
    assert saved_steps == [3, 4]


def test_validation_reuses_one_forward_for_loss_and_metrics(tmp_path):
    t = trainer(tmp_path, validation_interval_steps=1)
    calls = []
    hook = t.model.register_forward_hook(lambda *args: calls.append(1))
    t.train_steps(1)
    hook.remove()
    assert len(calls) == 2  # 训练一次，验证一次。
    assert len(t.history.validation_metrics) == 1


def test_unvalidated_history_has_json_null_best_loss():
    h = TrainHistory(global_step=4)
    document = json.loads(json.dumps(h.to_dict(), allow_nan=False))
    assert document['best_validation_loss'] is None
    assert np.isinf(TrainHistory.from_dict(document).best_validation_loss)


def test_zero_humidity_is_reported_without_inventing_finite_dewpoint():
    _, y = arrays()
    predicted = y.copy()
    predicted[0, 1] = -0.1
    evaluation = evaluate_predictions(y, predicted, excel_dew_point=np.zeros(len(y)))
    assert np.isneginf(evaluation.dew_point_predictions[0])
    report = evaluation.to_dict()
    assert report['dew_point_dry_limit_prediction_count'] == 1
    assert report['dew_point_excel_comparable_count'] == 15
    assert report['nonnegative_projection_count'] == 1
    json.dumps(report, allow_nan=False)


def test_all_dry_predictions_have_unavailable_dewpoint_error():
    _, y = arrays()
    predicted = y.copy(); predicted[:, 1] = 0
    evaluation = evaluate_predictions(y, predicted, excel_dew_point=np.zeros(len(y)))
    assert evaluation.dew_point_mae_vs_excel is None
    assert evaluation.to_dict()['dew_point_excel_comparable_count'] == 0


@pytest.fixture
def artifact(tmp_path):
    x, y = arrays(); cfg = get_profile('main')
    pre = fit_preprocessor(x, y, cfg.input_columns, cfg.direct_outputs)
    save_artifact(tmp_path / 'artifact', build_single_kan('main'), pre)
    return tmp_path / 'artifact'


def test_runtime_artifact_loads_without_training_imports(artifact, tmp_path):
    code = '''
import importlib.abc, sys
from pathlib import Path
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'training' or fullname.startswith('training.'):
            raise ImportError('运行时不得依赖 training')
sys.meta_path.insert(0, BlockTraining())
from dual_rotor_sim.core.system.components.rotor import SingleRotorModel
model = SingleRotorModel.from_artifact(Path(sys.argv[1]), 'main')
import pandas as pd
assert len(model.predict_frame(pd.DataFrame([[1]*9], columns=model.input_cols))) == 6
'''
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, '-c', code, str(artifact)], cwd=tmp_path,
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize('loader', [load_artifact, lambda p: SingleRotorModel.from_artifact(p, 'main')])
def test_both_loaders_reject_reversed_input_scaler_columns(artifact, loader):
    path = artifact / 'model/scalers.pkl'
    with path.open('rb') as f: state = pickle.load(f)
    scaler = state.get('input_scaler', state.get('x_scaler'))
    scaler.input_cols = list(reversed(scaler.input_cols))
    with path.open('wb') as f: pickle.dump(state, f)
    with pytest.raises((RuntimeError, ValueError), match='输入|input'):
        loader(artifact)


@pytest.mark.parametrize('change', ['length', 'values', 'order', 'validation'])
def test_resume_rejects_changed_arrays(tmp_path, change):
    t = trainer(tmp_path); t.train_steps(1); checkpoint = t.save_checkpoint()
    x, y = arrays(); vx, vy = x[:8].copy(), y[:8].copy()
    if change == 'length': x, y = x[:8], y[:8]
    if change == 'values': x[0, 0] += 100
    if change == 'order': x, y = x[::-1], y[::-1]
    if change == 'validation': vy[0, 0] += 1
    with pytest.raises(CheckpointError, match='数据|指纹'):
        SingleKANTrainer.from_checkpoint(checkpoint, train_features=x, train_targets=y,
                                         validation_features=vx, validation_targets=vy)


def test_resume_cannot_train_on_placeholder_arrays(tmp_path):
    t = trainer(tmp_path); t.train_steps(1)
    resumed = SingleKANTrainer.from_checkpoint(t.save_checkpoint())
    with pytest.raises(CheckpointError, match='数据|注入'):
        resumed.train_steps(1)


@pytest.mark.parametrize('control', ['seed', 'batch_size', 'validation_interval_steps', 'checkpoint_interval_steps'])
def test_resume_rejects_changed_command_controls(tmp_path, control):
    t = trainer(tmp_path); t.train_steps(1)
    expected = getattr(t, control) + 1
    with pytest.raises(CheckpointError, match='训练控制'):
        SingleKANTrainer.from_checkpoint(t.save_checkpoint(), expected_controls={control: expected})


def test_resume_rejects_changed_source_context(tmp_path):
    t = trainer(tmp_path, data_context={'smoke_test': True, 'source_sha256': 'abc', 'rows': [3, 7]})
    t.train_steps(1)
    with pytest.raises(CheckpointError, match='来源|身份'):
        SingleKANTrainer.from_checkpoint(t.save_checkpoint(), data_context={'smoke_test': False, 'source_sha256': 'abc', 'rows': [3, 7]})


def test_fit_preserves_early_stopping_in_final_checkpoint(tmp_path):
    from training.single_kan.trainer import load_checkpoint_payload
    t = trainer(tmp_path, validation_interval_steps=1)
    # 已连续19次未改善；最佳损失0使这次真实验证不会刷新最佳值。
    t.history.best_validation_loss = 0.0
    t.history.no_improvement_count = t.contract.patience - 1
    t.fit(epochs=2)
    assert t.history.global_step == 1
    assert load_checkpoint_payload(t._checkpoint)['history']['stop_reason'] == 'early_stopping'


def test_checkpoint_exactly_on_final_interval_is_not_written_twice(tmp_path):
    events = []
    t = trainer(tmp_path, checkpoint_interval_steps=2, event_sink=events.append)
    t.fit(epochs=1)
    assert [e['global_step'] for e in events if e['kind'] == 'checkpoint_saved'] == [2, 4]
