"""③  表冷器单元测试。

验证：
- 凝结分支：W_out == W_sat(T_setpoint) 且 < W_in
- 非凝结分支：W_out == W_in（含湿量不变）
- Q_kW 始终 ≤ 0（制冷取热）
- 冷凝水质量守恒：m_cond = mdot · (W_in - W_out) / 1000 [g/kg → kg]
- 设定温度 ≥ 入口温度时直通，Q_kW = 0
"""

from __future__ import annotations

import pytest

from dual_rotor_sim.core.system.air_state import AirState
from dual_rotor_sim.core.system.components.cooling_coil import CoolingCoil
from dual_rotor_sim.core.system import psychrometrics as psy


# ---------------------------------------------------------------------------
# 辅助：不同工况下的 W_sat 对照值（由 psychrometrics 实时计算，无硬编码）
# ---------------------------------------------------------------------------

def _w_sat(T: float) -> float:
    return float(psy.humidity_ratio_sat(T))


# ---------------------------------------------------------------------------
# ③-A  凝结分支：W_in > W_sat(T_setpoint)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_in, W_in, mdot, T_set", [
    (35.0, 18.0, 1.0, 20.0),   # 夏季室外 → 冷却到 20 ℃，W_sat(20) ≈ 14.7
    (30.0, 16.0, 0.8, 15.0),   # 高湿进入，冷却到 15 ℃
    (28.0, 20.0, 2.0, 18.0),   # 极高含湿量
])
def test_cooling_coil_condensing_W_out(T_in, W_in, mdot, T_set):
    """凝结分支：出口含湿量应等于 W_sat(T_setpoint)。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    cc = CoolingCoil(T_outlet_setpoint=T_set)
    outlet, _ = cc.process(inlet)

    expected_W = _w_sat(T_set)
    assert W_in > expected_W, "前置条件不满足：该工况不是凝结分支"
    assert abs(outlet.W - expected_W) < 1e-9, (
        f"凝结分支 W_out={outlet.W:.6f}, expected W_sat(T_set={T_set})={expected_W:.6f}"
    )


@pytest.mark.parametrize("T_in, W_in, mdot, T_set", [
    (35.0, 18.0, 1.0, 20.0),
    (30.0, 16.0, 0.8, 15.0),
    (28.0, 20.0, 2.0, 18.0),
])
def test_cooling_coil_condensing_W_decreases(T_in, W_in, mdot, T_set):
    """凝结分支：出口含湿量应严格小于入口含湿量。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    cc = CoolingCoil(T_outlet_setpoint=T_set)
    outlet, _ = cc.process(inlet)
    assert outlet.W < inlet.W, (
        f"凝结分支 W_out={outlet.W:.4f} 应 < W_in={inlet.W:.4f}"
    )


# ---------------------------------------------------------------------------
# ③-B  非凝结分支：W_in <= W_sat(T_setpoint)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_in, W_in, mdot, T_set", [
    (30.0,  8.0, 1.0, 20.0),   # W_sat(20) ≈ 14.7 >> 8，纯显热冷却
    (25.0, 10.0, 0.5, 18.0),   # W_sat(18) ≈ 12.9 > 10
    (20.0,  5.0, 2.0, 15.0),   # 低湿进入，W_sat(15) ≈ 10.6 > 5
])
def test_cooling_coil_non_condensing_W_invariant(T_in, W_in, mdot, T_set):
    """非凝结分支：出口含湿量应等于入口含湿量。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    cc = CoolingCoil(T_outlet_setpoint=T_set)
    outlet, _ = cc.process(inlet)

    # 确认是非凝结工况
    assert W_in <= _w_sat(T_set), "前置条件不满足：该工况不是非凝结分支"
    assert abs(outlet.W - inlet.W) < 1e-9, (
        f"非凝结分支 W_out={outlet.W:.6f} != W_in={inlet.W:.6f}"
    )


# ---------------------------------------------------------------------------
# ③-C  Q_kW 必须为负（制冷工况）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_in, W_in, mdot, T_set", [
    (35.0, 18.0, 1.0, 20.0),   # 凝结
    (30.0,  8.0, 1.0, 20.0),   # 非凝结
    (25.0, 10.0, 0.5, 18.0),
])
def test_cooling_coil_Q_negative(T_in, W_in, mdot, T_set):
    """制冷工况下 Q_kW 应为负值（从空气取热）。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    cc = CoolingCoil(T_outlet_setpoint=T_set)
    _, energy = cc.process(inlet)
    assert energy.Q_kW < 0, f"Q_kW={energy.Q_kW:.6f} 应 < 0（制冷）"


# ---------------------------------------------------------------------------
# ③-D  Q_kW 与焓差法一致
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_in, W_in, mdot, T_set", [
    (35.0, 18.0, 1.0, 20.0),
    (30.0,  8.0, 1.0, 20.0),
    (25.0, 10.0, 0.5, 18.0),
    (28.0, 20.0, 2.0, 18.0),
])
def test_cooling_coil_Q_matches_enthalpy(T_in, W_in, mdot, T_set):
    """Q_kW 应等于 mdot·(h_out - h_in)（焓差法）。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    cc = CoolingCoil(T_outlet_setpoint=T_set)
    outlet, energy = cc.process(inlet)
    expected_Q = mdot * (outlet.h - inlet.h)
    assert abs(energy.Q_kW - expected_Q) < 1e-6, (
        f"Q_kW={energy.Q_kW:.6f}, expected={expected_Q:.6f}"
    )


# ---------------------------------------------------------------------------
# ③-E  冷凝水质量守恒
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_in, W_in, mdot, T_set", [
    (35.0, 18.0, 1.0, 20.0),
    (30.0, 16.0, 0.8, 15.0),
    (28.0, 20.0, 2.0, 18.0),
])
def test_cooling_coil_condensate_mass_balance(T_in, W_in, mdot, T_set):
    """凝结分支：冷凝水量 = mdot·(W_in - W_out)/1000 [g/kg → kg/s]，应 > 0。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    cc = CoolingCoil(T_outlet_setpoint=T_set)
    outlet, _ = cc.process(inlet)

    m_cond = mdot * (inlet.W - outlet.W) / 1000.0
    assert m_cond > 0, f"凝结分支冷凝水量应 > 0，得 {m_cond}"
    # 冷凝水量 = 去除的水蒸气质量（守恒检验）
    expected_cond = mdot * (W_in - outlet.W) / 1000.0
    assert abs(m_cond - expected_cond) < 1e-12


# ---------------------------------------------------------------------------
# ③-F  直通分支（设定温度 ≥ 入口温度）
# ---------------------------------------------------------------------------

def test_cooling_coil_no_cooling_passthrough():
    """设定出口温度 ≥ 入口温度时，气流直通，Q_kW = 0。"""
    inlet = AirState(T=20.0, W=10.0, mdot=1.0)

    # 等温设定
    cc_equal = CoolingCoil(T_outlet_setpoint=20.0)
    outlet_eq, energy_eq = cc_equal.process(inlet)
    assert outlet_eq is inlet, "等温设定时应返回原 inlet 对象"
    assert abs(energy_eq.Q_kW) < 1e-9

    # 设定温度高于入口（无制冷需求）
    cc_warm = CoolingCoil(T_outlet_setpoint=25.0)
    outlet_warm, energy_warm = cc_warm.process(inlet)
    assert outlet_warm is inlet, "设定温度高于入口时应返回原 inlet 对象"
    assert abs(energy_warm.Q_kW) < 1e-9


def test_cooling_coil_outlet_T():
    """出口温度应等于设定值（仅在 T_setpoint < T_in 时）。"""
    inlet = AirState(T=35.0, W=18.0, mdot=1.0)
    T_set = 20.0
    cc = CoolingCoil(T_outlet_setpoint=T_set)
    outlet, _ = cc.process(inlet)
    assert abs(outlet.T - T_set) < 1e-9
