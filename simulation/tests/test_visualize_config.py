"""可视化入口必须复用前向仿真配置和统一流量换算。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pytest

from simulation.visualize import _load_config, _setup_chinese_font


ROOT = Path(__file__).resolve().parents[2]


def test_visualizer_registers_available_windows_chinese_font():
    """WSL 可读取微软雅黑时，仪表板不得退回缺少中文字形的字体。"""
    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if not font_path.is_file():
        pytest.skip("当前环境没有可复用的 Windows 中文字体")

    _setup_chinese_font()

    assert "Microsoft YaHei" in plt.rcParams["font.sans-serif"]


def test_visualizer_registers_available_windows_chinese_bold_font():
    """存在微软雅黑粗体文件时，加粗标题不应回退到常规字重。"""
    font_path = Path("/mnt/c/Windows/Fonts/msyhbd.ttc")
    if not font_path.is_file():
        pytest.skip("当前环境没有可复用的微软雅黑粗体")

    _setup_chinese_font()

    matching = [
        font
        for font in fm.fontManager.ttflist
        if font.name == "Microsoft YaHei" and font.weight in {"bold", 700}
    ]
    assert matching


def test_visualizer_uses_outdoor_reference_density_and_independent_rotor_fields():
    """若可视化入口恢复旧字段或支路密度，本测试必须失败。"""
    runtime = _load_config(
        ROOT / "simulation" / "configs" / "xf_validation_case.yaml"
    )

    W_oa = 5.28 / 1000.0
    p_da = 101325.0 * 0.622 / (0.622 + W_oa)
    rho_da_oa = p_da / (287.05 * (35.0 + 273.15))
    assert runtime["oa"].mdot == pytest.approx(3499.0 * rho_da_oa / 3600.0)
    assert runtime["ra"].mdot == pytest.approx(3333.0 * rho_da_oa / 3600.0)
    assert runtime["regen_oa"].mdot == pytest.approx(
        145.0 * rho_da_oa / 3600.0
    )
    assert runtime["params"].rotor1_cool_blow_mdot == pytest.approx(
        166.0 * rho_da_oa / 3600.0
    )
    assert runtime["params"].rotor2_cool_blow_mdot == pytest.approx(
        800.0 * rho_da_oa / 3600.0
    )
    assert runtime["params"].rotor1_diameter == 1070.0
    assert runtime["params"].rotor2_diameter == 1220.0


def test_visualizer_keeps_json_and_dashboard_per_run(monkeypatch, tmp_path):
    """连续两次绘图不能覆盖旧图，也不能丢掉其源 JSON。"""
    import json
    import sys
    from simulation import visualize
    from simulation.tests.test_cli import _sample_system_result

    results_dir = tmp_path / 'results'
    monkeypatch.setattr(visualize, 'OUTPUT_DIR', results_dir)
    monkeypatch.setattr(visualize, 'build_system', lambda: object())
    monkeypatch.setattr(visualize, 'forward_simulate', lambda *a: _sample_system_result())
    monkeypatch.setattr(sys, 'argv', ['visualize.py', '--config', str(ROOT / 'simulation/configs/default_case.yaml')])
    visualize.main()
    visualize.main()
    runs = list(results_dir.iterdir())
    assert len(runs) == 2
    for run in runs:
        record = json.loads((run / 'result.json').read_text())
        assert record['config']['schema_version'] == 2
        assert (run / 'fig1_dashboard.png').read_bytes().startswith(b'\x89PNG')
