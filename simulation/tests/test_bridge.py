"""真实进程边界：协议恢复、路径独立、完整结果及未收敛留档。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_jsonl_process_recovers_and_preserves_results(tmp_path):
    if not all((ROOT / f'artifacts/kan_{p}_single/model/best_model.pt').exists() for p in ('xf','main')) and os.environ.get('REQUIRE_EXTERNAL_FIXTURES') != '1':
        pytest.skip('本地没有当前 KAN 包')
    config = yaml.safe_load((ROOT / 'simulation/configs/qy1070_case.yaml').read_text())
    from dual_rotor_sim.config import build_runtime_inputs, parse_simulation_config
    from dual_rotor_sim.core.system.solver import build_system, forward_simulate

    runtime = build_runtime_inputs(parse_simulation_config(config))
    expected = forward_simulate(
        build_system(ROOT / 'artifacts'), runtime['oa'], runtime['ra'],
        runtime['regen_oa'], runtime['params'], runtime['setpoints'],
    )
    requests = [
        '{bad json',
        json.dumps({'protocol_version': 1, 'id': 'ping', 'command': 'ping'}),
        json.dumps({'protocol_version': 1, 'id': 'bad', 'command': 'simulate', 'config': {}}),
        json.dumps({'protocol_version': 1, 'id': '正常中文', 'command': 'simulate', 'config': config}),
        json.dumps({'protocol_version': 1, 'id': 'short', 'command': 'simulate', 'config': config, 'solver': {'max_iter': 1}}),
    ]
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'simulation'), 'MPLCONFIGDIR': str(tmp_path / 'cache')}
    proc = subprocess.run([
        sys.executable, '-m', 'dual_rotor_sim.bridge',
        '--artifacts-dir', str(ROOT / 'artifacts'), '--results-dir', str(tmp_path / '结果'),
    ], input='\n'.join(requests) + '\n', text=True, encoding='utf-8', capture_output=True,
        cwd=tmp_path, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(replies) == 5
    assert replies[0]['status'] == 'error'
    assert replies[1]['status'] == 'ok'
    assert replies[2]['code'] == 'INVALID_REQUEST'
    result = replies[3]
    assert result['id'] == '正常中文'
    assert result['status'] == 'ok'
    # 协议回归比较同一已安装模型的直接调用，避免新权重误撞旧模型黄金值。
    # 保留原绝对容差；训练精度及模型身份由各自报告/核验覆盖。
    assert result['result']['supply_state']['W_g_per_kg'] == pytest.approx(expected.supply_state.W, abs=2e-6, rel=0)
    assert result['result']['energy_summary_kW']['cooling_load'] == pytest.approx(expected.cooling_kW, abs=0.01, rel=0)
    assert result['result']['convergence']['converged'] == expected.convergence['converged']
    assert result['result']['intermediate_states']
    assert result['warnings']
    assert replies[4]['status'] == 'unconverged'
    assert replies[4]['result']['convergence']['iterations'] == 1
    assert replies[3]['files']['result_json'] != replies[4]['files']['result_json']
    for reply in replies[3:]:
        record = Path(reply['files']['result_json'])
        png = Path(reply['files']['dashboard_png'])
        assert record.is_file() and png.is_file()
        assert record.parent == png.parent
        assert json.loads(record.read_text())['result'] == reply['result']


def test_dashboard_honors_writable_cache(tmp_path, monkeypatch):
    from dual_rotor_sim import dashboard
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(dashboard, '__file__', '/proc/read_only_install/simulation/dual_rotor_sim/dashboard.py')
    dashboard._configure_matplotlib()
    assert (tmp_path / 'cache').is_dir()


pytestmark = pytest.mark.requires_models
