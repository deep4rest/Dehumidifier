"""②  加热器单元测试。

验证：
- 出口含湿量 W 不变
- Q_heat = mdot * (h_out - h_in)（焓差法与组件输出一致）
- 当 T_out <= T_in 时，组件直通，Q_kW = 0
"""

from __future__ import annotations

import pytest

from dual_rotor_sim.core.system.air_state import AirState
from dual_rotor_sim.core.system.components.heater import Heater


@pytest.mark.parametrize("T_in, W_in, mdot, T_out_set", [
    (20.0, 10.0, 1.0, 80.0),    # 典型再生加热
    (25.0,  5.0, 0.5, 120.0),   # 高温再生
    (15.0, 14.7, 2.0, 60.0),    # 饱和入口加热
    (-10.0, 1.0, 0.8, 50.0),    # 低温入口
])
def test_heater_W_invariant(T_in, W_in, mdot, T_out_set):
    """加热过程含湿量不变。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    heater = Heater(T_outlet_setpoint=T_out_set)
    outlet, energy = heater.process(inlet)
    assert abs(outlet.W - inlet.W) < 1e-9, (
        f"W changed: inlet.W={inlet.W}, outlet.W={outlet.W}"
    )


@pytest.mark.parametrize("T_in, W_in, mdot, T_out_set", [
    (20.0, 10.0, 1.0, 80.0),
    (25.0,  5.0, 0.5, 120.0),
    (15.0, 14.7, 2.0, 60.0),
])
def test_heater_Q_matches_enthalpy(T_in, W_in, mdot, T_out_set):
    """EnergyTerm.Q_kW 应等于 mdot*(h_out - h_in)（焓差法）。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    heater = Heater(T_outlet_setpoint=T_out_set)
    outlet, energy = heater.process(inlet)
    expected_Q = mdot * (outlet.h - inlet.h)
    assert abs(energy.Q_kW - expected_Q) < 1e-6, (
        f"Q_kW={energy.Q_kW:.6f}, expected={expected_Q:.6f}"
    )


@pytest.mark.parametrize("T_in, W_in, mdot, T_out_set", [
    (20.0, 10.0, 1.0, 80.0),
    (25.0,  5.0, 0.5, 120.0),
])
def test_heater_Q_positive(T_in, W_in, mdot, T_out_set):
    """加热时 Q_kW 应为正值（外部向空气输入热量）。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    heater = Heater(T_outlet_setpoint=T_out_set)
    _, energy = heater.process(inlet)
    assert energy.Q_kW > 0, f"Q_kW={energy.Q_kW} should be > 0"


def test_heater_no_heating_passthrough():
    """当设定出口温度 <= 入口温度时，气流直通，Q_kW = 0。"""
    inlet = AirState(T=50.0, W=10.0, mdot=1.0)
    heater = Heater(T_outlet_setpoint=50.0)   # 等温
    outlet, energy = heater.process(inlet)
    assert outlet is inlet, "直通时应返回原 inlet 对象"
    assert abs(energy.Q_kW) < 1e-9

    heater_cool = Heater(T_outlet_setpoint=40.0)  # 比入口低
    outlet2, energy2 = heater_cool.process(inlet)
    assert outlet2 is inlet
    assert abs(energy2.Q_kW) < 1e-9


def test_heater_outlet_T():
    """出口温度应等于设定值。"""
    inlet = AirState(T=25.0, W=8.0, mdot=1.0)
    T_set = 110.0
    heater = Heater(T_outlet_setpoint=T_set)
    outlet, _ = heater.process(inlet)
    assert abs(outlet.T - T_set) < 1e-9
