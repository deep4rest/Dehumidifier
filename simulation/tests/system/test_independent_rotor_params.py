"""两级转轮独立参数与流量拓扑测试。"""

from __future__ import annotations

import pytest

from dual_rotor_sim.core.system.air_state import AirState
from dual_rotor_sim.core.system.components.rotor import RotorOutputs
from dual_rotor_sim.core.system.system import ControlSetpoints, DualRotorSystem, SystemParams


class RecordingRotor:
    def __init__(self) -> None:
        self.calls = []

    def process(self, inputs):
        self.calls.append(inputs)
        return RotorOutputs(
            process_outlet_T=18.0,
            process_outlet_W=3.0,
            process_outlet_Tdp=-5.0,
            regen_outlet_T=50.0,
            regen_outlet_W=8.0,
            cool_blow_outlet_T=25.0,
            cool_blow_outlet_W=4.0,
        )


def test_system_uses_oa_state_as_only_total_flow_and_maps_rotors_independently():
    """若 OA 流量被第二字段覆盖，或 R1/R2 参数串线，本测试必须失败。"""
    rotor1 = RecordingRotor()
    rotor2 = RecordingRotor()
    system = DualRotorSystem(rotor1=rotor1, rotor2=rotor2)
    params = SystemParams(
        fan1_P_kW=4.0,
        fan2_P_kW=18.5,
        rotor1_thickness=300.0,
        rotor1_diameter=1070.0,
        rotor1_process_airspeed=1.6,
        rotor1_cool_blow_airspeed=1.3,
        rotor1_regen_angle=90.0,
        rotor1_cool_blow_angle=15.0,
        rotor1_cool_blow_mdot=0.2,
        rotor2_thickness=200.0,
        rotor2_diameter=1220.0,
        rotor2_process_airspeed=3.5,
        rotor2_cool_blow_airspeed=3.3,
        rotor2_regen_angle=22.5,
        rotor2_cool_blow_angle=22.5,
        rotor2_cool_blow_mdot=0.3,
    )

    result = system.simulate(
        AirState(T=35.0, W=12.0, mdot=1.4),
        AirState(T=22.0, W=0.49, mdot=0.5),
        AirState(T=35.0, W=12.0, mdot=0.1),
        params,
        ControlSetpoints(
            cc1_T_outlet=11.0,
            cc2_T_outlet=15.0,
            heater_low_T=140.0,
            heater_oa_T=120.0,
            rotor1_rpm=8.0,
            rotor2_rpm=6.0,
        ),
    )

    states = result.intermediate_states
    assert states["OA"].mdot == pytest.approx(1.4)
    assert states["R1_P_in"].mdot == pytest.approx(1.2)
    assert states["R2_CB_in"].mdot == pytest.approx(0.3)
    assert states["Supply"].mdot == pytest.approx(1.4)
    assert states["AfterR2_RG"].mdot == pytest.approx(0.3)
    # 企业风量规则（2026-09-17）：Heater_OA 入口 = R1 处理出口 / 3。
    # 本工况可用量 0.3+0.2=0.5 >= 目标 1.2/3=0.4，走漏风分支。
    assert states["AfterR1_RG"].mdot == pytest.approx(0.4)

    r1_inputs = rotor1.calls[0]
    # R1 KAN 的再生风量输入必须跟随规则执行后的实际入口流量。
    assert r1_inputs.regen_flow_m3h == pytest.approx(r1_inputs.process_flow_m3h / 3.0)
    assert (
        r1_inputs.rotor_thickness,
        r1_inputs.rotor_diameter,
        r1_inputs.airspeed,
        r1_inputs.cool_blow_airspeed,
    ) == (300.0, 1070.0, 1.6, 1.3)
    r2_inputs = rotor2.calls[0]
    assert (
        r2_inputs.rotor_thickness,
        r2_inputs.rotor_diameter,
        r2_inputs.airspeed,
        r2_inputs.cool_blow_airspeed,
    ) == (200.0, 1220.0, 3.5, 3.3)
