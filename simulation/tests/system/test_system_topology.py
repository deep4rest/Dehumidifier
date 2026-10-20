"""⑦  系统拓扑正确性测试（v2）。

验证：
⑦-A  流量级联约束：m_r2_rg == m_r2_cb；Heater_OA 入口流量执行企业风量规则
      （目标 = R1 处理出口 / 3：可用量 >= 目标漏风、否则补风至目标/上限）
⑦-B  质量守恒：处理侧 Mix1 入口 mdot 之和 == BeforeFan2.mdot；
               再生侧 AfterMix3.mdot == 规则执行后的 Heater_OA 入口流量
⑦-C  能耗键齐全：energy_breakdown 只公开 4 个热工组件键
⑦-D  金标准回归：固定工况下 cooling_kW / heating_kW / supply_state.Tdp
               与期望值误差 < 1e-3
⑦-E  物理趋势：heater_low_T（主转轮再生温度）升高时 Supply.Tdp 应单调下降
⑦-F  再生混风规则：漏风/补风/补风上限三分支的风量、成分与缺口记录

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


def test_flow_cascade_r1(base_result, base_params, base_oa):
    """新风转轮再生区流量应执行企业风量规则：Heater_OA 入口 = R1 处理出口 / 3。

    基础工况可用量（R2 再生出 + R1 冷吹出 = 0.6）>= 目标（0.7/3），走漏风
    分支：入口流量 == 目标，超出部分漏掉，不消耗再生补风。
    """
    states = base_result.intermediate_states
    rule = base_result.regen_mix_rule

    m_r1_p = base_oa.mdot - base_params.rotor1_cool_blow_mdot
    target = m_r1_p / 3.0

    assert rule["branch"] == "leak"
    assert rule["threshold_kg_da_s"] == pytest.approx(target, abs=1e-12)
    assert rule["regen_oa_used_kg_da_s"] == 0.0
    assert rule["leaked_kg_da_s"] == pytest.approx(
        rule["available_kg_da_s"] - target, abs=1e-12
    )
    assert rule["shortfall_kg_da_s"] == 0.0
    assert states["AfterR1_RG"].mdot == pytest.approx(target, abs=1e-12)
    assert states["AfterMix3"].mdot == pytest.approx(target, abs=1e-12)


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


def test_mass_conservation_mix3(base_result, base_params, base_oa):
    """再生侧 AfterMix3 流量 == 企业规则执行后的 Heater_OA 入口流量。"""
    states = base_result.intermediate_states
    rule = base_result.regen_mix_rule

    m_out = states["AfterMix3"].mdot
    assert m_out == pytest.approx(rule["heater_oa_inlet_kg_da_s"], abs=1e-12), (
        f"AfterMix3.mdot={m_out:.6f} 应 == 规则入口流量 "
        f"{rule['heater_oa_inlet_kg_da_s']:.6f}"
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
# 2026-09-17：再生混风执行企业风量规则（Heater_OA 入口 = R1 处理出口 / 3，
# 基础工况走漏风分支），湿空气公式迁移为企业参考代码 + DptAH 查表；
# 重录系统金标准。容差仍为 1e-3。
_GOLDEN_COOLING_KW:    float | None = 23.978721
_GOLDEN_HEATING_KW:    float | None = 27.093713
_GOLDEN_SUPPLY_TDP:    float | None = -33.353443


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


# ---------------------------------------------------------------------------
# ⑦-F  再生混风规则分支（2026-09-17 企业风量规则）
# ---------------------------------------------------------------------------

def test_regen_mix_leak_branch_preserves_mixed_composition(base_result):
    """漏风分支：两股先完全混合再漏风，AfterMix3 成分 == 混合成分（仅减流量）。"""
    states = base_result.intermediate_states
    r2rg, r1cb = states["AfterR2_RG"], states["AfterR1_CB"]
    m_sum = r2rg.mdot + r1cb.mdot
    h_weighted = (r2rg.mdot * r2rg.h + r1cb.mdot * r1cb.h) / m_sum
    w_weighted = (r2rg.mdot * r2rg.W + r1cb.mdot * r1cb.W) / m_sum

    assert base_result.regen_mix_rule["branch"] == "leak"
    assert states["AfterMix3"].mdot < m_sum  # 漏掉了多余风量
    assert states["AfterMix3"].h == pytest.approx(h_weighted, rel=1e-9)
    assert states["AfterMix3"].W == pytest.approx(w_weighted, rel=1e-9)


def test_regen_mix_make_up_branch_fills_to_threshold(system, base_ra, base_setpoints):
    """补风分支：可用量 < 目标且补风充足时，补至目标且无缺口。"""
    params = SystemParams(rotor1_cool_blow_mdot=0.10, rotor2_cool_blow_mdot=0.05)
    oa = AirState(T=35.0, W=18.0, mdot=1.0)
    regen_oa = AirState(T=35.0, W=18.0, mdot=0.5)  # 补风上限充足
    result = system.simulate(oa, base_ra, regen_oa, params, base_setpoints)

    rule = result.regen_mix_rule
    target = (1.0 - 0.10) / 3.0
    assert rule["branch"] == "make_up"
    assert rule["available_kg_da_s"] == pytest.approx(0.05 + 0.10, abs=1e-12)
    assert rule["regen_oa_used_kg_da_s"] == pytest.approx(target - 0.15, abs=1e-9)
    assert rule["shortfall_kg_da_s"] == 0.0
    assert result.intermediate_states["AfterMix3"].mdot == pytest.approx(target, abs=1e-9)
    assert result.intermediate_states["AfterR1_RG"].mdot == pytest.approx(target, abs=1e-9)


def test_regen_mix_make_up_capped_by_config(system, base_ra, base_setpoints):
    """补风上限不足：按配置上限补风，记录缺口，Heater_OA 入口低于目标。"""
    params = SystemParams(rotor1_cool_blow_mdot=0.10, rotor2_cool_blow_mdot=0.05)
    oa = AirState(T=35.0, W=18.0, mdot=1.0)
    regen_oa = AirState(T=35.0, W=18.0, mdot=0.05)  # 上限 0.05 < 缺口 0.15
    result = system.simulate(oa, base_ra, regen_oa, params, base_setpoints)

    rule = result.regen_mix_rule
    target = (1.0 - 0.10) / 3.0
    assert rule["branch"] == "make_up"
    assert rule["regen_oa_used_kg_da_s"] == pytest.approx(0.05, abs=1e-12)
    assert rule["shortfall_kg_da_s"] == pytest.approx(target - 0.20, abs=1e-9)
    assert rule["shortfall_kg_da_s"] > 0.0
    assert result.intermediate_states["AfterMix3"].mdot == pytest.approx(0.20, abs=1e-9)
    # 补风空气按上限用量参与混合：成分 == 三股流量加权（两股 + 受限补风）。
    states = result.intermediate_states
    m2rg, m1cb = states["AfterR2_RG"], states["AfterR1_CB"]
    w_three_stream = (
        m2rg.mdot * m2rg.W
        + m1cb.mdot * m1cb.W
        + rule["regen_oa_used_kg_da_s"] * regen_oa.W
    ) / (m2rg.mdot + m1cb.mdot + rule["regen_oa_used_kg_da_s"])
    assert states["AfterMix3"].W == pytest.approx(w_three_stream, rel=1e-9)


def test_regen_mix_zero_makeup_continues_and_records_shortfall(
    system, base_ra, base_setpoints
):
    """补风关闭为零时仍正常计算，使用两股可用风并完整记录缺口。"""
    params = SystemParams(rotor1_cool_blow_mdot=0.10, rotor2_cool_blow_mdot=0.05)
    oa = AirState(T=35.0, W=18.0, mdot=1.0)
    regen_oa = AirState(T=35.0, W=18.0, mdot=0.0)

    result = system.simulate(oa, base_ra, regen_oa, params, base_setpoints)

    rule = result.regen_mix_rule
    assert rule["branch"] == "make_up"
    assert rule["regen_oa_configured_kg_da_s"] == 0.0
    assert rule["regen_oa_used_kg_da_s"] == 0.0
    assert rule["heater_oa_inlet_kg_da_s"] == pytest.approx(0.15, abs=1e-12)
    assert rule["shortfall_kg_da_s"] == pytest.approx(0.15, abs=1e-12)


def test_regen_mix_exact_threshold_does_not_use_makeup(
    system, base_ra, base_setpoints
):
    """可用量恰好等于目标时关闭补风，不产生漏风或缺口。"""
    params = SystemParams(rotor1_cool_blow_mdot=0.10, rotor2_cool_blow_mdot=0.10)
    oa = AirState(T=35.0, W=18.0, mdot=0.70)
    regen_oa = AirState(T=35.0, W=18.0, mdot=0.30)

    result = system.simulate(oa, base_ra, regen_oa, params, base_setpoints)

    rule = result.regen_mix_rule
    assert rule["available_kg_da_s"] == pytest.approx(rule["threshold_kg_da_s"], abs=1e-12)
    assert rule["branch"] == "leak"
    assert rule["regen_oa_used_kg_da_s"] == 0.0
    assert rule["leaked_kg_da_s"] == pytest.approx(0.0, abs=1e-12)
    assert rule["shortfall_kg_da_s"] == 0.0


def test_regen_mix_tiny_shortfall_does_not_warn(
    system, base_ra, base_setpoints
):
    """小于流量容差的浮点缺口归零，联立求解不发补风不足告警。"""
    params = SystemParams(
        rotor1_cool_blow_mdot=0.10,
        rotor2_cool_blow_mdot=0.10 - 5e-10,
    )
    oa = AirState(T=35.0, W=18.0, mdot=0.70)
    regen_oa = AirState(T=35.0, W=18.0, mdot=0.0)

    with pytest.warns(RuntimeWarning) as caught:
        result = system.simulate_coupled_xf(
            oa, base_ra, regen_oa, params, base_setpoints, max_iter=1
        )

    assert result.regen_mix_rule["shortfall_kg_da_s"] == 0.0
    assert all("再生补风不足" not in str(item.message) for item in caught)


def test_regen_mix_zero_makeup_warns_but_returns_result(
    system, base_ra, base_setpoints
):
    """真实补风缺口只告警；即使联立未收敛也返回已计算结果。"""
    params = SystemParams(rotor1_cool_blow_mdot=0.10, rotor2_cool_blow_mdot=0.05)
    oa = AirState(T=35.0, W=18.0, mdot=1.0)
    regen_oa = AirState(T=35.0, W=18.0, mdot=0.0)

    with pytest.warns(RuntimeWarning, match="再生补风不足"):
        result = system.simulate_coupled_xf(
            oa, base_ra, regen_oa, params, base_setpoints, max_iter=1
        )

    assert result.regen_mix_rule["shortfall_kg_da_s"] > 0.0


pytestmark = pytest.mark.requires_models
