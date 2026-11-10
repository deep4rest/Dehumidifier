"""The XF log experiment changes one target transform and preserves old packages."""
import pickle

import numpy as np
import pytest

from training.single_kan.candidate_models import candidate_contract
from training.single_kan.model import Preprocessor, ReversibleTargetTransform, fit_preprocessor


def sample():
    contract = candidate_contract("xf", "kan")
    x = np.array([[200, 550, 9, 3.9, 1000, 50, 50, 333, .2],
                  [300, 650, 10, 4.17, 2000, 100, 55, 666, .3],
                  [400, 850, 12, 5.1, 4000, 200, 140, 1200, .4]], dtype=np.float32)
    y = np.array([[15, .00371, 40, 2, 25, 10], [18, .01, 45, 3, 30, 12],
                  [25, 7.0461, 100, 8, 50, 20]], dtype=np.float32)
    return x, y, contract


def test_log_changes_only_process_W_scaling_and_survives_plain_state_pickle():
    x, y, contract = sample()
    baseline = fit_preprocessor(x, y, contract.input_columns, contract.output_columns, fit_ids=[1, 2, 3])
    log = fit_preprocessor(x, y, contract.input_columns, contract.output_columns,
                           log1p_targets=(), log_targets=("处理出口含湿量",), fit_ids=[1, 2, 3])
    np.testing.assert_array_equal(log.transform_features(x), baseline.transform_features(x))
    other = [0, 2, 3, 4, 5]
    np.testing.assert_array_equal(log.transform_targets(y)[:, other], baseline.transform_targets(y)[:, other])
    np.testing.assert_allclose(log.target_scaler.data_min_[1], np.log(y[:, 1].astype(float)).min())
    restored = Preprocessor.from_state_dict(pickle.loads(pickle.dumps(log.state_dict(), protocol=4)))
    assert restored.target_transform.log_targets == ("处理出口含湿量",)
    assert restored.target_transform.log1p_targets == ()
    assert restored.fit_ids == (1, 2, 3)
    np.testing.assert_allclose(restored.inverse_targets(restored.transform_targets(y)), y, rtol=1e-6, atol=1e-7)
    # A scaled prediction below the training minimum still represents positive W.
    z = restored.transform_targets(y)
    z[:, 1] = [-1.05, -1.2, -2.]
    assert (restored.inverse_targets(z)[:, 1] > 0).all()
    old = Preprocessor.from_state_dict(baseline.state_dict())
    assert "log_targets" not in old.target_transform.state_dict()
    np.testing.assert_array_equal(old.transform_targets(y), baseline.transform_targets(y))


@pytest.mark.parametrize("value", [0., -0.001])
def test_pure_log_rejects_zero_and_negative_targets_without_epsilon_or_filter(value):
    x, y, contract = sample()
    y[0, 1] = value
    with pytest.raises(ValueError, match="严格大于0"):
        fit_preprocessor(x, y, contract.input_columns, contract.output_columns,
                         log1p_targets=(), log_targets=("处理出口含湿量",))


def test_transform_rejects_mixed_definitions_and_exp_overflow():
    _, _, contract = sample()
    with pytest.raises(ValueError, match="同时使用"):
        ReversibleTargetTransform(output_columns=contract.output_columns,
                                  log1p_targets=("处理出口含湿量",), log_targets=("处理出口含湿量",))
    log = ReversibleTargetTransform(output_columns=contract.output_columns, log_targets=("处理出口含湿量",))
    values = np.zeros((1, 6))
    values[0, 1] = 1000
    with pytest.raises(ValueError, match="exp 反变换"):
        log.inverse_transform(values)


def test_log_cli_cannot_silently_enter_legacy_or_main_training(tmp_path):
    from training.single_kan.candidate_train import run_candidate_suite
    from training.single_kan.train import main

    assert main(["--profile", "xf", "--process-humidity-transform", "log", "--smoke-test"]) == 2
    with pytest.raises(ValueError, match="仅支持 --profile xf"):
        run_candidate_suite(profile="all", dataset_dir=tmp_path, output_root=tmp_path / "output",
                            smoke_test=True, process_humidity_transform="log")
    assert not (tmp_path / "output").exists()
