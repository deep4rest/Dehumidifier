"""⑦  系统拓扑正确性测试（v2）。

验证：
⑦-A  流量级联约束：m_r2_rg == m_r2_cb；m_r1_rg == m_r2_rg + m_regen_oa + m_r1_cb
⑦-B  质量守恒：处理侧 Mix1 入口 mdot 之和 == BeforeFan2.mdot；
              再生侧三股 mdot 之和 == AfterMix3.mdot
⑦-C  能耗键齐全：energy_breakdown 只公开 4 个热工组件键
⑦-D  金标准回归：固定工况下 cooling_kW / heating_kW / supply_state.Tdp
              与期望值误差 < 1e-3
⑦-E  物理趋势：heater_low_T（主转轮再生温度）升高时 Supply.Tdp 应单调下降

注：⑦-E 测试 heater_low_T（主转轮再生侧加热器），而非 heater_oa_T（新风转轮加热器）。
原因：KAN 模型对 heater_low_T 在 100–150℃ 区间有清晰的单调响应（更高再生温度 →
主转轮处理区出口含湿量更低 → 送风露点更低）；heater_oa_T 对 Rotor1 预处理区的
影响通过两级转轮间接传递，KAN 模型在此段表现出更复杂的非单调行为。
"""

from __future__ import annotations

import pytest

from dual_rotor_sim.core.system import (
    AirState,
    SystemParams,
    ControlSetpoints,
    build_system,
    forward_simulate,
)
from dual_rotor_sim.core.system.system import DualRotorSystem
from dual_rotor_sim.core.system.components.rotor import Rotor


pytestmark = pytest.mark.requires_models


# ---------------------------------------------------------------------------
# 共享 fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def system(rotor_ensemble):
    """复用 conftest 的 rotor_ensemble，构建 DualRotorSystem。"""
    return DualRotorSystem(rotor=Rotor(ensemble=rotor_ensemble))


@pytest.fixture(scope="module")
def base_oa():
    return AirState(T=35.0, W=18.0, mdot=1.0)


@pytest.fixture(scope="module")
def base_ra():
    return AirState(T=25.0, W=11.0, mdot=0.5)


@pytest.fixture(scope="module")
def base_regen_oa():
    return AirState(T=35.0, W=18.0, mdot=0.3)


@pytest.fixture(scope="module")
def base_params():
    return SystemParams(
        fan1_P_kW=1.5,
        fan2_P_kW=1.0,
        rotor1_process_airspeed=2.5,
        rotor1_cool_blow_airspeed=2.5,
        rotor1_regen_angle=60.0,
        rotor1_cool_blow_angle=30.0,
        rotor2_process_airspeed=2.5,
        rotor2_cool_blow_airspeed=2.5,
        rotor2_regen_angle=60.0,
        rotor2_cool_blow_angle=30.0,
        rotor1_cool_blow_mdot=0.3,
        rotor2_cool_blow_mdot=0.3,
    )


@pytest.fixture(scope="module")
def base_setpoints():
    return ControlSetpoints(
        cc1_T_outlet=20.0,
        cc2_T_outlet=15.0,
        heater_low_T=120.0,
        heater_oa_T=120.0,
        rotor1_rpm=5.0,
        rotor2_rpm=5.0,
    )


@pytest.fixture(scope="module")
def base_result(system, base_oa, base_ra, base_regen_oa, base_params, base_setpoints):
    return system.simulate(base_oa, base_ra, base_regen_oa, base_params, base_setpoints)


# ---------------------------------------------------------------------------
# ⑦-A  流量级联约束
# ---------------------------------------------------------------------------

def test_flow_cascade_r2(base_result, base_params, base_ra):
    """主转轮再生区流量应严格等于主转轮冷吹区流量（级联约束）。"""
    states = base_result.intermediate_states

    # m_r2_cb = R2_CB_in.mdot
    m_r2_cb = states["R2_CB_in"].mdot
    # m_r2_rg = AfterR2_RG.mdot（KAN 输出后构造，应等于 m_r2_cb）
    m_r2_rg = states["AfterR2_RG"].mdot

    assert abs(m_r2_rg - m_r2_cb) < 1e-9, (
        f"m_r2_rg={m_r2_rg:.6f} 应 == m_r2_cb={m_r2_cb:.6f}"
    )


def test_flow_cascade_r1(base_result, base_params, base_ra, base_regen_oa):
    """新风转轮再生区流量应等于三股（主转轮再生出 + 再生新风 + 新风转轮冷吹出）之和。"""
    states = base_result.intermediate_states

    m_r2_rg    = states["AfterR2_RG"].mdot
    m_regen_oa = base_regen_oa.mdot
    m_r1_cb    = states["R1_CB_in"].mdot

    expected_r1_rg = m_r2_rg + m_regen_oa + m_r1_cb
    actual_r1_rg   = states["AfterR1_RG"].mdot

    assert abs(actual_r1_rg - expected_r1_rg) < 1e-9, (
        f"AfterR1_RG.mdot={actual_r1_rg:.6f} 应 == "
        f"m_r2_rg+m_regen_oa+m_r1_cb={expected_r1_rg:.6f}"
    )


# ---------------------------------------------------------------------------
# ⑦-B  质量守恒
# ---------------------------------------------------------------------------

def test_mass_conservation_mix1(base_result, base_ra):
    """处理侧 Mix1：新风转轮处理出 + 回风 = BeforeFan2 总流量。"""
    states = base_result.intermediate_states

    m_in = states["AfterR1_P"].mdot + base_ra.mdot
    m_out = states["BeforeFan2"].mdot

    assert abs(m_out - m_in) < 1e-9, (
        f"BeforeFan2.mdot={m_out:.6f} ≠ AfterR1_P+RA={m_in:.6f}"
    )


def test_mass_conservation_mix3(base_result, base_regen_oa):
    """再生侧三股混风：主转轮再生出 + 再生新风 + 新风转轮冷吹出 = AfterMix3 总流量。"""
    states = base_result.intermediate_states

    m_in = (
        states["AfterR2_RG"].mdot
        + base_regen_oa.mdot
        + states["AfterR1_CB"].mdot
    )
    m_out = states["AfterMix3"].mdot

    assert abs(m_out - m_in) < 1e-9, (
        f"AfterMix3.mdot={m_out:.6f} ≠ 三股之和={m_in:.6f}"
    )


# ---------------------------------------------------------------------------
# ⑦-C  能耗键齐全
# ---------------------------------------------------------------------------

_REQUIRED_ENERGY_KEYS = {
    "CC1_kW",
    "CC2_kW",
    "Heater_OA_kW",
    "Heater_Low_kW",
}


def test_energy_breakdown_keys(base_result):
    """energy_breakdown 必须只包含四个公开热工组件键。"""
    missing = _REQUIRED_ENERGY_KEYS - set(base_result.energy_breakdown.keys())
    assert not missing, f"energy_breakdown 缺少键：{missing}"


def test_energy_breakdown_no_extra(base_result):
    """energy_breakdown 不应包含额外的未知键（防止旧字段残留）。"""
    extra = set(base_result.energy_breakdown.keys()) - _REQUIRED_ENERGY_KEYS
    assert not extra, f"energy_breakdown 存在多余键：{extra}"


# ---------------------------------------------------------------------------
# ⑦-D  金标准回归（首次运行后写入 hard-coded 期望值）
# ---------------------------------------------------------------------------
# 期望值由首次通过的仿真结果确定，防止后续拓扑误改。
# 若模型权重更新，应重新记录。
# tolerance: 1e-3
#
# 输出仅保留可由焓差确定的热负荷，不再宣称整机直接电功耗：
#   cooling_kW    = |CC1| + |CC2|                              (kW，冷量，需除以 COP 得电耗)
#   heating_kW    = Heater_OA + Heater_Low                     (kW，热量，需除以加热效率)

# 2026-08-21：湿空气基准由 Magnus 迁移为 Murphy–Koop/ASHRAE water basis；
# 模型权重与工况配置均未改动，独立重复运行后重录系统金标准。容差仍为 1e-3。
_GOLDEN_COOLING_KW:    float | None = 23.9808
_GOLDEN_HEATING_KW:    float | None = 27.0939
_GOLDEN_SUPPLY_TDP:    float | None = -36.5866


def test_golden_regression(base_result):
    """金标准回归：cooling/heating_kW 和 supply_state.Tdp 应与首次期望值一致（容差 1e-3）。

    首次运行时 _GOLDEN_* 为 None，测试自动跳过并打印当前值，供开发者填写。
    """
    goldens = (
        _GOLDEN_COOLING_KW,
        _GOLDEN_HEATING_KW,
        _GOLDEN_SUPPLY_TDP,
    )
    if any(g is None for g in goldens):
        print(
            f"\n[金标准未设定] 当前值：\n"
                f"  cooling_kW    = {base_result.cooling_kW:.6f}\n"
            f"  heating_kW    = {base_result.heating_kW:.6f}\n"
            f"  supply_Tdp    = {base_result.supply_state.Tdp:.6f}\n"
            "  请将上述值填入 test_system_topology.py 的 _GOLDEN_* 常量后重新运行。"
        )
        pytest.skip("金标准期望值尚未填写，请首次运行后记录并填写 _GOLDEN_* 常量。")

    assert abs(base_result.cooling_kW - _GOLDEN_COOLING_KW) < 1e-3, (
        f"cooling_kW={base_result.cooling_kW:.6f} 偏离期望 {_GOLDEN_COOLING_KW:.6f}"
    )
    assert abs(base_result.heating_kW - _GOLDEN_HEATING_KW) < 1e-3, (
        f"heating_kW={base_result.heating_kW:.6f} 偏离期望 {_GOLDEN_HEATING_KW:.6f}"
    )
    assert abs(base_result.supply_state.Tdp - _GOLDEN_SUPPLY_TDP) < 1e-3, (
        f"supply_Tdp={base_result.supply_state.Tdp:.6f} 偏离期望 {_GOLDEN_SUPPLY_TDP:.6f}"
    )


# ---------------------------------------------------------------------------
# ⑦-E  物理趋势：heater_low_T 升高 → Supply.Tdp 单调下降
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_low, t_high", [
    (100.0, 130.0),
    (110.0, 140.0),
    (120.0, 150.0),
])
def test_heater_low_tdp_monotone(system, base_oa, base_ra, base_regen_oa, base_params, t_low, t_high):
    """heater_low_T（主转轮再生温度）升高时，送风露点应单调下降。

    heater_low_T 直接控制主转轮（Rotor2）再生区入口温度：
      主转轮冷吹出 → Heater_Low(heater_low_T) → 主转轮再生区
    温度越高 → 再生越充分 → 主转轮处理区出口含湿量越低 → 送风露点越低。
    KAN 模型在 100–150℃ 区间对该变量有清晰的单调响应。
    """
    sp_low = ControlSetpoints(
        cc1_T_outlet=20.0, cc2_T_outlet=15.0,
        heater_low_T=t_low, heater_oa_T=120.0,
        rotor1_rpm=5.0, rotor2_rpm=5.0,
    )
    sp_high = ControlSetpoints(
        cc1_T_outlet=20.0, cc2_T_outlet=15.0,
        heater_low_T=t_high, heater_oa_T=120.0,
        rotor1_rpm=5.0, rotor2_rpm=5.0,
    )

    res_low  = system.simulate(base_oa, base_ra, base_regen_oa, base_params, sp_low)
    res_high = system.simulate(base_oa, base_ra, base_regen_oa, base_params, sp_high)

    assert res_high.supply_state.Tdp < res_low.supply_state.Tdp, (
        f"heater_low_T={t_high}℃ 时送风露点 {res_high.supply_state.Tdp:.3f}℃ "
        f"应 < heater_low_T={t_low}℃ 时的 {res_low.supply_state.Tdp:.3f}℃"
    )
