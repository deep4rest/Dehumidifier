"""风量 KAN 运行时与训练产物的一致性回归。"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dual_rotor_sim.core.system.components.rotor import SingleRotorModel, RotorInputs, _raw_row

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('profile,label', [('xf','新风'),('main','主转轮')])
def test_completed_candidate_loads_and_matches_saved_prediction(profile, label):
    runs = sorted((ROOT/'results/training').glob(f'*_{label}_KAN_*'))
    if not runs:
        if os.environ.get('REQUIRE_EXTERNAL_FIXTURES') == '1':
            pytest.fail('缺少带外提供的已训练候选及测试CSV')
        pytest.skip('本地没有已训练产物')
    run = runs[-1]
    model = SingleRotorModel.from_artifact(run, profile)
    frame = pd.read_csv(ROOT/f'data/转轮数据集_尺寸隔离_8比1比1/{label if profile == 'main' else '新风转轮'}_测试集.csv').iloc[[0]]
    saved = pd.read_csv(run/'reports/test_predictions.csv').iloc[0]
    actual = model.predict_frame(frame)
    for name, value in actual.items():
        assert value == pytest.approx(saved[f'预测_{name}'], abs=2e-4)


def test_flow_mapping_is_independent_of_airspeed():
    row = _raw_row(RotorInputs(20, 5, 120, 20, process_flow_m3h=3333, cool_blow_flow_m3h=167, airspeed=99))
    assert row['处理风量'] == 3333
    assert row['冷吹风量'] == 167


def test_qy1070_branch_flows_reach_models(monkeypatch):
    import yaml
    from dual_rotor_sim.config import parse_simulation_config, build_runtime_inputs
    from dual_rotor_sim.core.system.solver import build_system
    if not (ROOT/'artifacts/kan_xf_single').exists() and os.environ.get('REQUIRE_EXTERNAL_FIXTURES') != '1':
        pytest.skip('本地没有运行时模型')
    system = build_system()
    rt = build_runtime_inputs(parse_simulation_config(yaml.safe_load((ROOT/'simulation/configs/qy1070_case.yaml').read_text())))
    frames = {}
    for profile, rotor in [('xf', system.rotor1), ('main', system.rotor2)]:
        original = rotor.model.predict_frame
        def capture(frame, profile=profile, original=original):
            frames[profile] = frame.iloc[0]
            return original(frame)
        monkeypatch.setattr(rotor.model, 'predict_frame', capture)
    system.simulate(*[rt[k] for k in ('oa','ra','regen_oa','params','setpoints')])
    assert frames['xf']['处理风量'] == pytest.approx(3333)
    assert frames['xf']['冷吹风量'] == pytest.approx(167)
    assert frames['xf']['再生风量'] == pytest.approx(1111)
    assert frames['main']['处理风量'] == pytest.approx(10000)
    assert frames['main']['冷吹风量'] == pytest.approx(667)


def test_solver_retains_parameters_and_each_raw_residual():
    import yaml
    from dual_rotor_sim.config import parse_simulation_config, build_runtime_inputs
    from dual_rotor_sim.core.system.solver import build_system
    if not (ROOT/'artifacts/kan_xf_single').exists() and os.environ.get('REQUIRE_EXTERNAL_FIXTURES') != '1':
        pytest.skip('本地没有运行时模型')
    rt = build_runtime_inputs(parse_simulation_config(yaml.safe_load((ROOT/'simulation/configs/qy1070_case.yaml').read_text())))
    result = build_system().simulate_coupled_xf(*[rt[k] for k in ('oa','ra','regen_oa','params','setpoints')])
    c = result.convergence
    assert c['solver_parameters']['relax'] == 0.8
    assert len(c['history']) == c['iterations']
    assert c['history'][-1]['residuals'] == c['residuals']
    assert c['history'][-1]['inputs'] == c['last_evaluated_inputs']


pytestmark = pytest.mark.requires_models
