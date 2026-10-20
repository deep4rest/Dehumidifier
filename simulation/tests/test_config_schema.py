"""schema v2、流量口径与固定工况契约测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from dual_rotor_sim.config import (
    ConfigError,
    parse_simulation_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(name: str) -> dict:
    with open(ROOT / "simulation" / "configs" / name, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _validation_config() -> dict:
    return {
        "schema_version": 2,
        "psychrometrics": {
            "pressure_Pa": 101325.0,
            "dew_point_basis": "water",
        },
        "flow_reference": {"basis": "outdoor_air"},
        "outdoor_air": {
            "T_degC": 35.0,
            "W_g_per_kg": 5.28,
            "flow_m3h": 3499.0,
        },
        "return_air": {
            "T_degC": 25.0,
            "W_g_per_kg": 0.49,
            "flow_m3h": 3333.0,
        },
        "regen_outdoor_air": {
            "T_degC": 35.0,
            "W_g_per_kg": 5.28,
            "flow_m3h": 145.0,
        },
        "system_params": {
            "fan1_P_kW": 4.0,
            "fan2_P_kW": 18.5,
            "rotor1": {
                "thickness_mm": 300.0,
                "diameter_mm": 1070.0,
                "process_airspeed_m_s": 1.6,
                "cool_blow_airspeed_m_s": 1.3,
                "cool_blow_flow_m3h": 166.0,
                "regen_angle_deg": 90.0,
                "cool_blow_angle_deg": 15.0,
            },
            "rotor2": {
                "thickness_mm": 200.0,
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
            "heater_low_T_degC": 120.0,
            "heater_oa_T_degC": 100.0,
            "rotor1_rpm": 8.0,
            "rotor2_rpm": 6.0,
        },
    }


def test_schema_derives_only_topology_dependent_flows():
    """若派生公式或五个独立流量之一改变，本测试必须失败。"""
    parsed = parse_simulation_config(_validation_config())

    assert parsed.independent_flow_m3h == {
        "outdoor_air": 3499.0,
        "return_air": 3333.0,
        "regen_outdoor_air": 145.0,
        "rotor1_cold_blow": 166.0,
        "rotor2_cold_blow": 800.0,
    }
    assert parsed.derived_flow_m3h == {
        "rotor1_process": 3333.0,
        "rotor2_regen": 800.0,
        "rotor2_process": 5866.0,
        "rotor1_regen": 1111.0,
    }


@pytest.mark.parametrize(
    "mutate, expected_path",
    [
        (lambda cfg: cfg.pop("flow_reference"), "flow_reference"),
        (
            lambda cfg: cfg["system_params"]["rotor1"].pop("cool_blow_flow_m3h"),
            "system_params.rotor1.cool_blow_flow_m3h",
        ),
        (
            lambda cfg: cfg["return_air"].__setitem__("legacy_mdot", 1),
            "return_air.legacy_mdot",
        ),
        (
            lambda cfg: cfg["system_params"].__setitem__("oa_mdot_m3h", 4800),
            "system_params.oa_mdot_m3h",
        ),
        (lambda cfg: cfg.__setitem__("solver", {}), "solver"),
        (lambda cfg: cfg.__setitem__("run_options", {}), "run_options"),
        (lambda cfg: cfg.__setitem__("reporting", {}), "reporting"),
        (lambda cfg: cfg.__setitem__("profile", "legacy"), "profile"),
    ],
)
def test_schema_rejects_missing_unknown_and_bypass_keys(mutate, expected_path):
    """若隐藏默认、重复流量或绕过入口恢复，本测试必须失败。"""
    cfg = deepcopy(_validation_config())
    mutate(cfg)

    with pytest.raises(ConfigError, match=expected_path.replace(".", r"\.")):
        parse_simulation_config(cfg)


def test_schema_rejects_nonpositive_derived_process_flow():
    """R2 冷吹量吞掉全部二级处理风时必须在加载配置阶段失败。"""
    cfg = deepcopy(_validation_config())
    cfg["system_params"]["rotor2"]["cool_blow_flow_m3h"] = 7000.0

    with pytest.raises(ConfigError, match="rotor2_process"):
        parse_simulation_config(cfg)


def test_default_and_validation_configs_have_distinct_roles_and_exact_flows():
    """默认配置必须保持企业工况，回归配置不得覆盖它。"""
    current = parse_simulation_config(_load_yaml("default_case.yaml"))
    validation = parse_simulation_config(_load_yaml("xf_validation_case.yaml"))

    assert current.independent_flow_m3h == {
        "outdoor_air": 4200.0,
        "return_air": 8800.0,
        "regen_outdoor_air": 333.0,
        "rotor1_cold_blow": 200.0,
        "rotor2_cold_blow": 800.0,
    }
    assert current.derived_flow_m3h == {
        "rotor1_process": 4000.0,
        "rotor2_regen": 800.0,
        "rotor2_process": 12000.0,
        "rotor1_regen": 1333.0,
    }
    assert validation.derived_flow_m3h == {
        "rotor1_process": 3333.0,
        "rotor2_regen": 800.0,
        "rotor2_process": 5866.0,
        "rotor1_regen": 1111.0,
    }
    assert current.outdoor_air.W_g_per_kg == 27.02
    assert validation.outdoor_air.W_g_per_kg == 5.28


def test_schema_accepts_enterprise_rotor_geometry_and_derived_flows():
    """企业真实转轮几何和流量必须能进入前向仿真配置。"""
    cfg = _validation_config()
    cfg["outdoor_air"]["flow_m3h"] = 4200.0
    cfg["return_air"]["flow_m3h"] = 8800.0
    cfg["regen_outdoor_air"]["flow_m3h"] = 333.0
    cfg["system_params"]["rotor1"].update(
        diameter_mm=1220.0,
        process_airspeed_m_s=1.4,
        cool_blow_airspeed_m_s=1.2,
        cool_blow_flow_m3h=200.0,
    )

    parsed = parse_simulation_config(cfg)

    assert parsed.rotor1.diameter_mm == 1220.0
    assert parsed.derived_flow_m3h == {
        "rotor1_process": 4000.0,
        "rotor2_regen": 800.0,
        "rotor2_process": 12000.0,
        "rotor1_regen": 1333.0,
    }
