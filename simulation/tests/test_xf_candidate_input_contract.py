"""新风仅移除转速输入，固定其余训练契约。"""
import numpy as np

from training.single_kan.candidate_models import build_candidate_model, candidate_contract
from training.single_kan.config import DIRECT_OUTPUT_COLUMNS, VOLUME_MAIN_INPUT_COLUMNS
from training.single_kan.model import fit_preprocessor


def test_xf_removes_only_rotation_without_changing_training_configuration():
    contract = candidate_contract("xf", "kan")
    assert contract.input_columns == (
        "转轮厚度", "转轮直径", "处理入口温度(℃)", "入口含湿量(g/kg)",
        "处理风量", "冷吹风量", "再生温度(℃)", "再生风量", "再生入口含湿量",
    )
    assert contract.output_columns == tuple(DIRECT_OUTPUT_COLUMNS)
    assert (contract.hidden_widths, contract.grid_size, contract.spline_order) == ((24,), 6, 3)
    assert (contract.learning_rate, contract.weight_decay, contract.max_epochs) == (5e-4, 1e-4, 300)
    model = build_candidate_model("xf", "kan")
    assert model.network.config.in_features == 9
    assert model.network.config.hidden_widths == [24]
    assert model.network.config.out_features == 6
    assert model.parameter_count == 3600
    # The other profile and the six target transformations remain unchanged.
    assert "转轮转速(rpm)" in VOLUME_MAIN_INPUT_COLUMNS
    x = np.array([[200, 550, 9, 3.9, 1000, 50, 50, 333, .2],
                  [300, 650, 10, 4.17, 2000, 100, 55, 666, .3]], dtype=np.float32)
    y = np.array([[15, .01, 40, 2, 25, 10], [18, .2, 45, 3, 30, 12]], dtype=np.float32)
    pre = fit_preprocessor(x, y, contract.input_columns, contract.output_columns, fit_ids=[1, 2])
    assert pre.input_scaler.n_features_in_ == 9
    assert pre.target_scaler.n_features_in_ == 6
    assert pre.target_transform.log1p_targets == ("处理出口含湿量",)
    assert pre.input_scaler.feature_range == (-1., 1.)
    np.testing.assert_allclose(pre.inverse_targets(pre.transform_targets(y)), y, rtol=1e-6, atol=1e-6)
