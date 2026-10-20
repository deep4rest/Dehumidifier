"""④  混风单元测试。

验证：
- 质量守恒：ṁ₃ = ṁ₁ + ṁ₂
- 含湿量守恒：ṁ₃·W₃ = ṁ₁·W₁ + ṁ₂·W₂
- 焓守恒：ṁ₃·h₃ = ṁ₁·h₁ + ṁ₂·h₂
- 退化情形：两股同状态混合结果为同状态，流量相加
"""

from __future__ import annotations

import pytest

from dual_rotor_sim.core.system.air_state import AirState
from dual_rotor_sim.core.system.components.mixer import Mixer


# ---------------------------------------------------------------------------
# ④-A  质量流量守恒
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T1, W1, m1, T2, W2, m2", [
    (35.0, 18.0, 1.0, 20.0, 10.0, 0.5),
    (15.0,  5.0, 2.0, 30.0, 15.0, 1.0),
    (25.0, 12.0, 0.3, 25.0,  8.0, 0.7),
])
def test_mixer_mass_conservation(T1, W1, m1, T2, W2, m2):
    """混合后流量 = 两路流量之和。"""
    s1 = AirState(T=T1, W=W1, mdot=m1)
    s2 = AirState(T=T2, W=W2, mdot=m2)
    mixer = Mixer()
    out = mixer.mix(s1, s2)
    assert abs(out.mdot - (m1 + m2)) < 1e-9, (
        f"mdot: {out.mdot} != {m1 + m2}"
    )


# ---------------------------------------------------------------------------
# ④-B  含湿量（湿度）守恒：ṁ₃·W₃ = ṁ₁·W₁ + ṁ₂·W₂
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T1, W1, m1, T2, W2, m2", [
    (35.0, 18.0, 1.0, 20.0, 10.0, 0.5),
    (15.0,  5.0, 2.0, 30.0, 15.0, 1.0),
    ( 0.0,  2.0, 0.5, 40.0, 20.0, 1.5),
    (25.0, 12.0, 0.3, 25.0,  8.0, 0.7),
])
def test_mixer_W_conservation(T1, W1, m1, T2, W2, m2):
    """水蒸气质量守恒：ṁ₃·W₃ == ṁ₁·W₁ + ṁ₂·W₂。"""
    s1 = AirState(T=T1, W=W1, mdot=m1)
    s2 = AirState(T=T2, W=W2, mdot=m2)
    mixer = Mixer()
    out = mixer.mix(s1, s2)

    lhs = out.mdot * out.W
    rhs = m1 * W1 + m2 * W2
    assert abs(lhs - rhs) < 1e-9, (
        f"W 守恒: ṁ₃·W₃={lhs:.8f}, ṁ₁·W₁+ṁ₂·W₂={rhs:.8f}"
    )


# ---------------------------------------------------------------------------
# ④-C  焓守恒：ṁ₃·h₃ = ṁ₁·h₁ + ṁ₂·h₂
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T1, W1, m1, T2, W2, m2", [
    (35.0, 18.0, 1.0, 20.0, 10.0, 0.5),
    (15.0,  5.0, 2.0, 30.0, 15.0, 1.0),
    ( 0.0,  2.0, 0.5, 40.0, 20.0, 1.5),
    (25.0, 12.0, 0.3, 25.0,  8.0, 0.7),
])
def test_mixer_enthalpy_conservation(T1, W1, m1, T2, W2, m2):
    """焓守恒：ṁ₃·h₃ == ṁ₁·h₁ + ṁ₂·h₂。"""
    s1 = AirState(T=T1, W=W1, mdot=m1)
    s2 = AirState(T=T2, W=W2, mdot=m2)
    mixer = Mixer()
    out = mixer.mix(s1, s2)

    lhs = out.mdot * out.h
    rhs = m1 * s1.h + m2 * s2.h
    assert abs(lhs - rhs) < 1e-6, (
        f"焓守恒: ṁ₃·h₃={lhs:.8f}, ṁ₁·h₁+ṁ₂·h₂={rhs:.8f}"
    )


# ---------------------------------------------------------------------------
# ④-D  退化情形：两股同状态混合 = 同状态、流量相加
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T, W, m1, m2", [
    (25.0, 10.0, 1.0, 1.0),
    (35.0, 18.0, 0.5, 1.5),
    (15.0,  5.0, 2.0, 0.3),
])
def test_mixer_degenerate_same_state(T, W, m1, m2):
    """两股相同状态混合：出口状态应与入口相同，流量 = m1 + m2。"""
    s1 = AirState(T=T, W=W, mdot=m1)
    s2 = AirState(T=T, W=W, mdot=m2)
    mixer = Mixer()
    out = mixer.mix(s1, s2)

    assert abs(out.T - T) < 1e-6, f"T: {out.T} != {T}"
    assert abs(out.W - W) < 1e-9, f"W: {out.W} != {W}"
    assert abs(out.mdot - (m1 + m2)) < 1e-9, f"mdot: {out.mdot} != {m1 + m2}"


# ---------------------------------------------------------------------------
# ④-E  混合温度物理性（加权平均温度边界检查）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T1, W1, m1, T2, W2, m2", [
    (10.0,  5.0, 1.0, 40.0, 15.0, 1.0),  # 等流量
    (10.0,  5.0, 2.0, 40.0, 15.0, 1.0),  # 冷侧流量大
    (10.0,  5.0, 1.0, 40.0, 15.0, 3.0),  # 热侧流量大
])
def test_mixer_outlet_T_between_inlets(T1, W1, m1, T2, W2, m2):
    """绝热混合出口温度应介于两入口温度之间（焓守恒决定）。"""
    s1 = AirState(T=T1, W=W1, mdot=m1)
    s2 = AirState(T=T2, W=W2, mdot=m2)
    mixer = Mixer()
    out = mixer.mix(s1, s2)

    T_min, T_max = min(T1, T2), max(T1, T2)
    assert T_min <= out.T <= T_max + 1e-6, (
        f"出口温度 {out.T:.4f} 超出输入范围 [{T_min}, {T_max}]"
    )


# ---------------------------------------------------------------------------
# ④-F  零流量输入应抛出异常
# ---------------------------------------------------------------------------

def test_mixer_zero_flow_raises():
    """两股流量之和 ≤ 0 时应抛出 ValueError。"""
    s1 = AirState(T=25.0, W=10.0, mdot=0.0)
    s2 = AirState(T=30.0, W=12.0, mdot=0.0)
    mixer = Mixer()
    with pytest.raises(ValueError):
        mixer.mix(s1, s2)
