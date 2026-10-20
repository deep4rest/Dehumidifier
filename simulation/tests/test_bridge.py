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
    if not (ROOT / 'artifacts/kan_xf_single/model/best_model.pt').exists() and os.environ.get('REQUIRE_EXTERNAL_FIXTURES') != '1':
        pytest.skip('本地没有当前 KAN 包')
    config = yaml.safe_load((ROOT / 'simulation/configs/qy1070_case.yaml').read_text())
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
    # 2026-09-17：湿空气公式对齐企业参考代码（0.62198）后重录该基准值。
    assert result['result']['supply_state']['W_g_per_kg'] == pytest.approx(0.1753739726, abs=2e-6)
    assert result['result']['energy_summary_kW']['cooling_load'] == pytest.approx(105.31047178, abs=0.01)
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
