"""CLI 配置到系统运行时参数的映射测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import dual_rotor_sim
import dual_rotor_sim.cli as cli_module
from dual_rotor_sim.cli import _build_from_config, _make_parser, cmd_simulate


def _sample_system_result():
    """构造包含完整节点和两台转轮输出的轻量真实结果。"""
    states = {
        "OA": dual_rotor_sim.AirState(T=35.0, W=18.0, mdot=1.2),
        "AfterFan1": dual_rotor_sim.AirState(T=35.0, W=18.0, mdot=1.2),
        "AfterCC1": dual_rotor_sim.AirState(T=11.0, W=8.0, mdot=1.2),
        "AfterR1_P": dual_rotor_sim.AirState(T=20.0, W=2.0, mdot=1.0),
        "BeforeFan2": dual_rotor_sim.AirState(T=21.0, W=1.8, mdot=2.0),
        "AfterFan2": dual_rotor_sim.AirState(T=21.0, W=1.8, mdot=2.0),
        "AfterCC2": dual_rotor_sim.AirState(T=15.0, W=1.4, mdot=1.8),
        "Supply": dual_rotor_sim.AirState(T=16.0, W=0.2, mdot=1.8),
    }
    rotor1 = dual_rotor_sim.RotorOutputs(
        process_outlet_T=20.0,
        process_outlet_W=2.0,
        process_outlet_Tdp=-8.0,
        regen_outlet_T=60.0,
        regen_outlet_W=22.0,
        cool_blow_outlet_T=40.0,
        cool_blow_outlet_W=9.0,
    )
    rotor2 = dual_rotor_sim.RotorOutputs(
        process_outlet_T=16.0,
        process_outlet_W=0.2,
        process_outlet_Tdp=-39.0,
        regen_outlet_T=70.0,
        regen_outlet_W=5.0,
        cool_blow_outlet_T=50.0,
        cool_blow_outlet_W=1.4,
    )
    return dual_rotor_sim.SystemResult(
        supply_state=states["Supply"],
        cooling_kW=125.0,
        heating_kW=39.0,
        energy_breakdown={
            "CC1_kW": -80.0,
            "CC2_kW": -45.0,
            "Heater_Low_kW": 20.0,
            "Heater_OA_kW": 19.0,
        },
        intermediate_states=states,
        rotor1_outputs=rotor1,
        rotor2_outputs=rotor2,
    )


def _valid_config() -> dict:
    return {
        "schema_version": 2,
        "psychrometrics": {
            "pressure_Pa": 101325.0,
            "dew_point_basis": "water",
        },
        "flow_reference": {"basis": "outdoor_air"},
        "outdoor_air": {
            "T_degC": 35.0,
            "W_g_per_kg": 27.02,
            "flow_m3h": 4200.0,
        },
        "return_air": {
            "T_degC": 20.0,
            "W_g_per_kg": 0.2313,
            "flow_m3h": 8800.0,
        },
        "regen_outdoor_air": {
            "T_degC": 35.0,
            "W_g_per_kg": 27.02,
            "flow_m3h": 333.0,
        },
        "system_params": {
            "fan1_P_kW": 4.0,
            "fan2_P_kW": 18.5,
            "rotor1": {
                "thickness_mm": 200.0,
                "diameter_mm": 1220.0,
                "process_airspeed_m_s": 1.4,
                "cool_blow_airspeed_m_s": 1.2,
                "cool_blow_flow_m3h": 200.0,
                "regen_angle_deg": 90.0,
                "cool_blow_angle_deg": 15.0,
            },
            "rotor2": {
                "thickness_mm": 300.0,
                "diameter_mm": 1220.0,
                "process_airspeed_m_s": 3.5,
                "cool_blow_airspeed_m_s": 3.3,
                "cool_blow_flow_m3h": 800.0,
                "regen_angle_deg": 22.5,
                "cool_blow_angle_deg": 22.5,
            },
        },
        "control_setpoints": {
            "cc1_T_outlet_degC": 11.0,
            "cc2_T_outlet_degC": 15.0,
            "heater_low_T_degC": 130.0,
            "heater_oa_T_degC": 115.0,
            "rotor1_rpm": 8.0,
            "rotor2_rpm": 6.0,
        },
    }


@pytest.fixture
def fake_simulation_api(monkeypatch):
    """替换重量级模型加载，只记录配置映射到系统 API 的真实参数。"""

    captured = {}

    def record(**kwargs):
        return SimpleNamespace(**kwargs)

    system_sentinel = object()
    result_sentinel = object()

    def fake_build_system(**kwargs):
        captured["build_system_kwargs"] = kwargs
        return system_sentinel

    def fake_forward_simulate(system, oa, ra, regen_oa, params, setpoints):
        captured.update(
            system=system,
            oa=oa,
            ra=ra,
            regen_oa=regen_oa,
            params=params,
            setpoints=setpoints,
        )
        return result_sentinel

    monkeypatch.setattr(dual_rotor_sim, "AirState", record)
    monkeypatch.setattr(dual_rotor_sim, "SystemParams", record)
    monkeypatch.setattr(dual_rotor_sim, "ControlSetpoints", record)
    monkeypatch.setattr(dual_rotor_sim, "build_system", fake_build_system)
    monkeypatch.setattr(dual_rotor_sim, "forward_simulate", fake_forward_simulate)
    captured["result_sentinel"] = result_sentinel
    return captured


def test_config_maps_one_outdoor_density_to_all_five_independent_flows(
    fake_simulation_api,
):
    """若回风或再生风又按自身状态换算，本测试必须失败。"""
    cfg = _valid_config()

    result = _build_from_config(cfg)

    W_oa = 27.02 / 1000.0
    p_da = 101325.0 * 0.622 / (0.622 + W_oa)
    rho_da_oa = p_da / (287.05 * (35.0 + 273.15))
    to_mass = lambda flow: flow * rho_da_oa / 3600.0

    assert fake_simulation_api["oa"].mdot == pytest.approx(to_mass(4200.0))
    assert fake_simulation_api["ra"].mdot == pytest.approx(to_mass(8800.0))
    assert fake_simulation_api["regen_oa"].mdot == pytest.approx(to_mass(333.0))
    assert fake_simulation_api["params"].rotor1_cool_blow_mdot == pytest.approx(
        to_mass(200.0)
    )
    assert fake_simulation_api["params"].rotor2_cool_blow_mdot == pytest.approx(
        to_mass(800.0)
    )
    assert result is fake_simulation_api["result_sentinel"]


def test_config_maps_independent_rotor_geometry_and_airspeeds(fake_simulation_api):
    """若两台转轮再次共享风速或几何默认值，本测试必须失败。"""
    _build_from_config(_valid_config())

    params = fake_simulation_api["params"]
    assert (
        params.rotor1_thickness,
        params.rotor1_diameter,
        params.rotor1_process_airspeed,
        params.rotor1_cool_blow_airspeed,
    ) == (200.0, 1220.0, 1.4, 1.2)
    assert (
        params.rotor2_thickness,
        params.rotor2_diameter,
        params.rotor2_process_airspeed,
        params.rotor2_cool_blow_airspeed,
    ) == (300.0, 1220.0, 3.5, 3.3)


def test_config_build_uses_default_model_location(
    fake_simulation_api,
):
    """工况配置只描述系统输入，不携带模型路径。"""
    _build_from_config(_valid_config())

    assert fake_simulation_api["build_system_kwargs"] == {}


def test_simulate_cli_requires_a_config_file():
    """若 CLI 恢复带隐藏默认值的内联工况，本测试必须失败。"""
    parser = _make_parser()
    args = parser.parse_args(["simulate", "--config", "case.yaml"])
    assert args.config == "case.yaml"
    with pytest.raises(SystemExit):
        parser.parse_args(["simulate"])


def test_enterprise_case_reaches_forward_simulation(fake_simulation_api):
    """1220 mm 企业工况不得被模型治理门禁拦截。"""
    result = _build_from_config(_valid_config())

    assert fake_simulation_api["build_system_kwargs"] == {}
    assert result is fake_simulation_api["result_sentinel"]


def test_each_successful_cli_run_writes_its_own_json_and_dashboard(
    monkeypatch,
    tmp_path,
):
    """若后一次运行覆盖前一次，或图和 JSON 分离，本测试必须失败。"""
    config_path = tmp_path / "企业工况.yaml"
    config_path.write_text(
        "schema_version: 2\noutdoor_air:\n  T_degC: 35.0\n",
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    monkeypatch.setattr(cli_module, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(
        cli_module,
        "_build_from_config",
        lambda _cfg: _sample_system_result(),
    )
    args = SimpleNamespace(
        config=str(config_path),
        quiet=True,
        json=True,
    )

    cmd_simulate(args)
    cmd_simulate(args)

    run_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name
    for run_dir in run_dirs:
        assert {path.name for path in run_dir.iterdir()} == {
            "result.json",
            "fig1_dashboard.png",
        }
        assert (run_dir / "fig1_dashboard.png").read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )
        document = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        assert document["config_source"] == str(config_path.resolve())
        assert document["config"]["outdoor_air"]["T_degC"] == 35.0
        assert document["result"]["supply_state"]["W_g_per_kg"] == 0.2
        assert document["result"]["energy_summary_kW"] == {
            "cooling_load": 125.0,
            "heating_load": 39.0,
        }
        serialized = json.dumps(document, ensure_ascii=False)
        assert "electrical_kW" not in serialized
        assert "power_kW" not in serialized
        assert "Fan1_kW" not in serialized
        assert "Rotor1Motor_kW" not in serialized


def test_cli_reports_model_loading_failure_without_traceback(monkeypatch, capsys):
    """模型加载失败应成为简洁中文 CLI 错误，而不是未捕获异常。"""
    def fail_build(_cfg):
        raise RuntimeError("模型文件加载失败")

    monkeypatch.setattr(cli_module, "_build_from_config", fail_build)
    args = SimpleNamespace(
        config=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "xf_validation_case.yaml"
        ),
        quiet=True,
        json=True,
    )

    with pytest.raises(SystemExit) as error:
        cmd_simulate(args)

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "[错误] 模型文件加载失败" in stderr
    assert "Traceback" not in stderr
