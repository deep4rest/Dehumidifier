"""legacy/main 双模型构造、输入映射与路由测试。"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from dual_rotor_sim.core.system.air_state import AirState
from dual_rotor_sim.core.system.components.rotor import (
    Rotor,
    RotorEnsemble,
    RotorInputs,
    RotorOutputs,
    _load_saved_routing,
    _load_main_projection_columns,
)
from dual_rotor_sim.core.system.system import ControlSetpoints, DualRotorSystem, SystemParams


OUTPUT_COLS = [
    "冷吹出口温度",
    "冷处出口含湿量",
    "处理出口含湿量",
    "处理出口温度",
    "处理出口露点",
    "再生出口含湿量",
    "再生出口温度",
]


class SpyRotor:
    def __init__(self, offset: float = 0.0):
        self.calls = []
        self.offset = offset

    def process(self, inputs: RotorInputs) -> RotorOutputs:
        self.calls.append(inputs)
        return RotorOutputs(
            process_outlet_T=20.0 + self.offset,
            process_outlet_W=5.0,
            process_outlet_Tdp=-5.0,
            regen_outlet_T=70.0,
        regen_outlet_W=10.0,
        cool_blow_outlet_T=30.0,
        cool_blow_outlet_W=7.0,
        )


def _simulate(system: DualRotorSystem):
    return system.simulate(
        AirState(T=35.0, W=18.0, mdot=1.0),
        AirState(T=25.0, W=11.0, mdot=0.5),
        AirState(T=35.0, W=18.0, mdot=0.3),
        SystemParams(
            rotor1_thickness=300.0,
            rotor1_diameter=1070.0,
            rotor1_process_airspeed=1.6,
            rotor1_cool_blow_airspeed=1.3,
            rotor1_cool_blow_mdot=0.3,
            rotor2_thickness=200.0,
            rotor2_diameter=1220.0,
            rotor2_process_airspeed=3.5,
            rotor2_cool_blow_airspeed=3.3,
            rotor2_cool_blow_mdot=0.3,
        ),
        ControlSetpoints(
            cc1_T_outlet=20.0,
            cc2_T_outlet=15.0,
            heater_low_T=140.0,
            heater_oa_T=120.0,
            rotor1_rpm=5.0,
            rotor2_rpm=6.0,
        ),
    )


def test_dual_model_dispatch_and_main_input_mapping():
    rotor1 = SpyRotor()
    rotor2 = SpyRotor(offset=1.0)
    result = _simulate(DualRotorSystem(rotor1=rotor1, rotor2=rotor2))
    assert len(rotor1.calls) == 1
    assert len(rotor2.calls) == 1
    main_inputs = rotor2.calls[0]
    assert main_inputs.rotor_thickness == 200.0
    assert main_inputs.rotor_diameter == 1220.0
    assert main_inputs.airspeed == 3.5
    assert main_inputs.cool_blow_airspeed == 3.3
    assert main_inputs.rpm == 6.0
    assert not hasattr(result.rotor1_outputs, "power_kW")
    assert not hasattr(result.rotor2_outputs, "power_kW")


def test_shared_rotor_constructor_remains_compatible():
    shared = SpyRotor()
    system = DualRotorSystem(rotor=shared)
    _simulate(system)
    assert len(shared.calls) == 2
    assert system.rotor is shared
    assert system.rotor1 is shared
    assert system.rotor2 is shared


def test_dual_constructor_rejects_ambiguous_mix():
    shared = SpyRotor()
    with pytest.raises(ValueError, match="不能"):
        DualRotorSystem(rotor=shared, rotor1=shared, rotor2=shared)


class DummyBundle:
    def __init__(self):
        self.input_cols = [
            "转轮厚度",
            "转轮直径",
            "入口含湿量(g/kg)",
            "转轮转速(rpm)",
            "再生温度(℃)",
            "冷吹入口温度(℃)",
            "处理入口温度(℃)",
            "处理风速(m/s)",
            "冷吹风速(m/s)",
        ]
        self.output_cols = OUTPUT_COLS
        self.input_domain = {
            "转轮转速(rpm)": {"min": 6.0, "max": 6.0},
            "冷吹入口温度(℃)": {"min": 27.0, "max": 27.0},
            "处理入口温度(℃)": {"min": 15.0, "max": 15.0},
        }

    def predict(self, _frame):
        return np.array([[30.0, 7.0, 5.0, 20.0, -5.0, 10.0, 70.0, 2.0]])


def test_main_profile_allows_out_of_domain_forward_prediction_without_warning():
    routing = {col: "v1" for col in OUTPUT_COLS}
    ensemble = RotorEnsemble({"v1": DummyBundle()}, routing, profile="main")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ensemble.predict(
            RotorInputs(
                process_inlet_T=16.0,
                process_inlet_W=0.5,
                regen_T=140.0,
                cool_blow_inlet_T=28.0,
                rpm=5.0,
                airspeed=3.5,
                cool_blow_airspeed=3.3,
            )
        )
    assert not captured


def test_main_profile_projects_nonnegative_outputs():
    bundle = DummyBundle()
    bundle.predict = lambda _frame: np.array(
        [[30.0, 7.0, -0.1, 20.0, -5.0, 10.0, 70.0, -2.0]]
    )
    routing = {col: "v1" for col in OUTPUT_COLS}
    ensemble = RotorEnsemble({"v1": bundle}, routing, profile="main")
    result = ensemble.predict(
        RotorInputs(
            process_inlet_T=15.0,
            process_inlet_W=0.5,
            regen_T=140.0,
            cool_blow_inlet_T=27.0,
            rpm=5.0,
        )
    )
    assert result.process_outlet_W == 0.0
    assert not hasattr(result, "power_kW")


def test_main_routing_reads_best_run_fields(tmp_path: Path):
    reports = tmp_path / "kan_main_ensemble_best" / "reports"
    reports.mkdir(parents=True)
    expected = {col: "v6" for col in OUTPUT_COLS}
    document = {
        "routing": {
            col: {"best_run": alias, "best_r2": 0.5}
            for col, alias in expected.items()
        },
        "physical_projection": {
            "method": "nonnegative_projection",
            "columns": {
                "冷处出口含湿量": {"constraint": "lower_bound", "lower_bound": 0.0},
                "处理出口含湿量": {"constraint": "lower_bound", "lower_bound": 0.0},
                "再生出口含湿量": {"constraint": "lower_bound", "lower_bound": 0.0},
            }
        },
    }
    (reports / "routing.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )
    assert _load_saved_routing(tmp_path, "main") == expected
    assert set(_load_main_projection_columns(tmp_path)) == {
        "冷处出口含湿量",
        "处理出口含湿量",
        "再生出口含湿量",
    }
