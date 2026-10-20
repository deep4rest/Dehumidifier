"""⑤  风机单元测试。

验证：
- 风机不改变温度、含湿量和质量流量
- EnergyTerm.Q_kW 直接等于配置的固定 P_kW
- 固定功率不随流量变化
"""

from __future__ import annotations

import pytest

from dual_rotor_sim.core.system.air_state import AirState
from dual_rotor_sim.core.system.components.fan import Fan

# ---------------------------------------------------------------------------
# ⑤-A  空气状态完全直通
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_in, W_in, mdot, P_kW", [
    (20.0, 10.0, 1.0, 1.5),
    (35.0, 18.0, 2.0, 3.0),
    ( 0.0,  2.0, 0.5, 0.75),
])
def test_fan_air_state_passthrough(T_in, W_in, mdot, P_kW):
    """固定功率风机不改变空气状态。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    fan = Fan(P_kW=P_kW)
    outlet, _ = fan.process(inlet)
    assert outlet is inlet
    assert outlet.T == inlet.T
    assert outlet.W == inlet.W
    assert outlet.mdot == inlet.mdot


# ---------------------------------------------------------------------------
# ⑤-B  不计算风机温升
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_in, W_in, mdot, P_kW", [
    (20.0, 10.0, 1.0, 1.5),
    (35.0, 18.0, 2.0, 3.0),
    ( 0.0,  2.0, 0.5, 0.75),
    (25.0,  8.0, 1.5, 2.2),
])
def test_fan_has_no_temperature_rise(T_in, W_in, mdot, P_kW):
    """固定功率只参与电功耗汇总，不换算为空气温升。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    fan = Fan(P_kW=P_kW)
    outlet, _ = fan.process(inlet)
    assert outlet.T == inlet.T


# ---------------------------------------------------------------------------
# ⑤-C  EnergyTerm.Q_kW == 固定 P_kW
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_in, W_in, mdot, P_kW", [
    (20.0, 10.0, 1.0, 1.5),
    (35.0, 18.0, 2.0, 3.0),
    (25.0,  8.0, 1.5, 2.2),
])
def test_fan_Q_equals_P(T_in, W_in, mdot, P_kW):
    """EnergyTerm.Q_kW 直接采用配置的固定风机功率。"""
    inlet = AirState(T=T_in, W=W_in, mdot=mdot)
    fan = Fan(P_kW=P_kW)
    _, energy = fan.process(inlet)
    assert abs(energy.Q_kW - P_kW) < 1e-9, (
        f"Q_kW={energy.Q_kW}, P_kW={P_kW}"
    )
    assert energy.Q_kW > 0, "风机固定功率应为正值"


# ---------------------------------------------------------------------------
# ⑤-D  固定功率不随流量变化
# ---------------------------------------------------------------------------

def test_fan_power_independent_of_flow():
    """不同质量流量下均返回同一配置功率。"""
    fan = Fan(P_kW=2.2)
    powers = [fan.process(AirState(T=25.0, W=8.0, mdot=m))[1].Q_kW for m in (0.2, 1.0, 3.0)]
    assert powers == [2.2, 2.2, 2.2]


# ---------------------------------------------------------------------------
# ⑤-E  流量守恒
# ---------------------------------------------------------------------------

def test_fan_mdot_invariant():
    """风机不改变质量流量。"""
    inlet = AirState(T=25.0, W=10.0, mdot=1.5)
    fan = Fan(P_kW=2.0)
    outlet, _ = fan.process(inlet)
    assert abs(outlet.mdot - inlet.mdot) < 1e-9


# ---------------------------------------------------------------------------
# ⑤-F  负功率应抛异常
# ---------------------------------------------------------------------------

def test_fan_negative_power_raises():
    """负功率风机初始化应抛出 ValueError。"""
    with pytest.raises(ValueError):
        Fan(P_kW=-0.1)


# ---------------------------------------------------------------------------
# ⑤-G  零流量不崩溃，Q_kW 仍记录
# ---------------------------------------------------------------------------

def test_fan_zero_flow():
    """零流量时风机不应崩溃，Q_kW 应等于 P_kW。"""
    inlet = AirState(T=25.0, W=10.0, mdot=0.0)
    fan = Fan(P_kW=1.5)
    outlet, energy = fan.process(inlet)
    assert abs(energy.Q_kW - 1.5) < 1e-9
