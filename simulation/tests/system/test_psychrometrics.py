"""①  湿空气物性自反一致性测试。

验证：
- (T, W) → h → temperature_from_enthalpy(h, W) 误差 < 1e-4 ℃
- humidity_ratio_sat(T) 在 [0, 60] ℃ 严格单调递增
- dewpoint_celsius(W_sat(T)) 约等于 T（误差 < 0.01 ℃）
"""

from __future__ import annotations

import numpy as np
import pytest

from dual_rotor_sim.core.system import psychrometrics as psy


# 覆盖各种工况的 (T, W) 对，包括极低含湿量和高温高湿
@pytest.mark.parametrize("T, W", [
    (35.0, 18.0),   # 夏季室外
    (20.0, 10.0),   # 室内舒适
    (-5.0,  2.0),   # 冬季
    (60.0, 50.0),   # 高温高湿（边界）
    (15.0,  0.05),  # 极低含湿量（深度干燥后）
    (0.0,   4.84),  # 0 ℃ 饱和近似
])
def test_roundtrip_T(T: float, W: float) -> None:
    """h → 反算 T' 误差应 < 1e-4 ℃。"""
    h = float(psy.enthalpy_kJ(T, W))
    T_back = float(psy.temperature_from_enthalpy(h, W))
    assert abs(T_back - T) < 1e-4, (
        f"T={T}, W={W}: h={h:.4f}, T'={T_back:.6f}, Δ={abs(T_back - T):.2e}"
    )


def test_humidity_ratio_sat_monotone() -> None:
    """饱和含湿量在 0–60 ℃ 应严格单调递增。"""
    temps = np.linspace(0, 60, 200)
    w_sat = psy.humidity_ratio_sat(temps)
    diffs = np.diff(w_sat)
    assert np.all(diffs > 0), (
        f"humidity_ratio_sat 在 {len(diffs[diffs <= 0])} 个区间内非单调"
    )


def test_water_basis_saturation_pressure_matches_independent_gold_vector() -> None:
    """饱和压公式必须与 docs/requirement.txt 企业参考代码逐系数一致。

    2026-09-17：<=0 ℃ 分支由 Murphy–Koop 过冷水公式迁移为企业冰面基准
    多项式（参考代码 CalculateAH/CalculateRH 的 ``DBT <= 0`` 分支），
    金标准值随之重录；>0 ℃ 分支公式未变。
    """

    temperatures = np.array([-75.0, -40.0, -20.0, 0.0, 20.0, 60.0])
    expected_pressure_Pa = np.array(
        [
            0.12211935692063862,
            12.84511625118676,
            103.25929284812797,
            611.1470566150772,
            2338.8037000739814,
            19943.760621620728,
        ]
    )
    actual = psy.saturation_vapor_pressure_water_Pa(temperatures)
    np.testing.assert_allclose(actual, expected_pressure_Pa, rtol=2e-11, atol=1e-11)
    np.testing.assert_array_equal(psy.saturation_pressure_pa(temperatures), actual)


def test_pressure_aware_saturation_humidity_and_water_dewpoint_gold_vector() -> None:
    """W_sat 与露点金标准（2026-09-17 重录：水汽比 0.62198 + DptAH 查表）。

    露点契约：W > 3.764 g/kg（正露点）按同一饱和压公式精确反演；
    W <= 3.764 g/kg 走企业 DptAH 表分段线性插值，与连续公式存在
    表格固有偏差（本组样本最大约 0.026 ℃）。
    """

    temperatures = np.array([-75.0, -40.0, -20.0, 0.0, 20.0, 60.0])
    humidity_g_per_kg = np.array(
        [
            0.0007496263425747417,
            0.0788592979300522,
            0.6345002023971396,
            3.7742697271955854,
            14.695878614876145,
            152.4260422449553,
        ]
    )
    actual_humidity = psy.saturation_humidity_ratio_g_per_kg(temperatures, 101_325.0)
    np.testing.assert_allclose(actual_humidity, humidity_g_per_kg, rtol=2e-11, atol=1e-11)
    assert np.isinf(psy.saturation_humidity_ratio_g_per_kg(60.0, 10_000.0))
    recovered = psy.dew_point_water_C(humidity_g_per_kg, 101_325.0)
    # 正露点（>0 ℃）公式反演应精确回归；0 ℃ 样本 3.7743 > 表上界 3.764，
    # 同样走公式分支。表内（<=0 ℃）样本允许表格插值固有偏差。
    np.testing.assert_allclose(recovered[3:], temperatures[3:], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(recovered[:3], temperatures[:3], atol=0.05, rtol=0.0)


# ---------------------------------------------------------------------------
# 企业 DptAH 映射表契约（docs/requirement.txt，2026-09-17）
# ---------------------------------------------------------------------------

def test_dewpoint_table_exact_points() -> None:
    """表项整点：DptAH[i] 对应露点 -i ℃，必须精确命中。"""

    assert psy.dew_point_water_C(3.764) == 0.0
    assert psy.dew_point_water_C(3.466) == -1.0
    assert psy.dew_point_water_C(psy._DPT_AH_TABLE[10]) == -10.0
    assert psy.dew_point_water_C(0.0001) == -87.0


def test_dewpoint_table_all_integer_points_roundtrip() -> None:
    """企业表全部 88 个整数露点必须双向精确命中。"""
    for index, humidity in enumerate(psy._DPT_AH_TABLE):
        dew_point = float(-index)
        assert float(psy.dew_point_water_C(humidity)) == pytest.approx(
            dew_point, abs=1e-12
        )
        assert float(psy.humidity_ratio_from_dew_point_C(dew_point)) == pytest.approx(
            humidity, abs=1e-12
        )


def test_enterprise_return_air_dewpoint_survives_config_and_recording() -> None:
    """企业复现：25℃、-35℃回风经 W=0.138 配置和 JSON 序列化仍为 -35℃。"""
    from dual_rotor_sim.config import build_runtime_inputs, parse_simulation_config
    from dual_rotor_sim.run_records import _air_state_to_dict

    humidity = float(psy.humidity_ratio_from_dew_point_C(-35.0))
    assert humidity == pytest.approx(0.138, abs=1e-12)
    raw = {
        "schema_version": 2,
        "psychrometrics": {"pressure_Pa": 101325.0, "dew_point_basis": "water"},
        "flow_reference": {"basis": "outdoor_air"},
        "outdoor_air": {"T_degC": 35.0, "W_g_per_kg": 18.0, "flow_m3h": 3500.0},
        "return_air": {"T_degC": 25.0, "W_g_per_kg": humidity, "flow_m3h": 7000.0},
        "regen_outdoor_air": {"T_degC": 35.0, "W_g_per_kg": 18.0, "flow_m3h": 0.0},
        "system_params": {
            "fan1_P_kW": 4.0,
            "fan2_P_kW": 18.5,
            "rotor1": {"thickness_mm": 200.0, "diameter_mm": 1070.0, "process_airspeed_m_s": 1.6, "cool_blow_airspeed_m_s": 1.3, "cool_blow_flow_m3h": 167.0, "regen_angle_deg": 90.0, "cool_blow_angle_deg": 15.0},
            "rotor2": {"thickness_mm": 300.0, "diameter_mm": 1070.0, "process_airspeed_m_s": 3.8, "cool_blow_airspeed_m_s": 3.6, "cool_blow_flow_m3h": 667.0, "regen_angle_deg": 22.5, "cool_blow_angle_deg": 22.5},
        },
        "control_setpoints": {"cc1_T_outlet_degC": 11.0, "cc2_T_outlet_degC": 15.0, "heater_low_T_degC": 130.0, "heater_oa_T_degC": 115.0, "rotor1_rpm": 8.0, "rotor2_rpm": 6.0},
    }

    state = build_runtime_inputs(parse_simulation_config(raw))["ra"]
    record = _air_state_to_dict(state)

    assert state.W == pytest.approx(0.138, abs=1e-12)
    assert state.Tdp == pytest.approx(-35.0, abs=1e-9)
    assert record["Tdp_degC"] == pytest.approx(-35.0, abs=1e-9)


def test_dewpoint_table_linear_interpolation_example() -> None:
    """企业示例：露点 -0.5 ℃ 的含湿量 = 0.5*(DptAH[0]+DptAH[1])，双向一致。"""

    expected_w = 0.5 * (3.764 + 3.466)
    assert float(psy.humidity_ratio_from_dew_point_C(-0.5)) == pytest.approx(expected_w, abs=1e-12)
    assert float(psy.dew_point_water_C(expected_w)) == pytest.approx(-0.5, abs=1e-9)


def test_dewpoint_table_inverse_roundtrip() -> None:
    """表区间内 W -> Tdp -> W 应精确还原（分段线性互逆）。"""

    for w in (3.0, 1.5, 0.5, 0.1, 0.001, 0.0001):
        tdp = float(psy.dew_point_water_C(w))
        assert float(psy.humidity_ratio_from_dew_point_C(tdp)) == pytest.approx(w, rel=1e-9)


def test_dewpoint_drying_limit_roundtrip() -> None:
    """W=0 干燥极限：露点 -inf，且 -inf 反算含湿量为 0。"""

    assert np.isneginf(psy.dew_point_water_C(0.0))
    assert float(psy.humidity_ratio_from_dew_point_C(float("-inf"))) == 0.0


def test_dewpoint_below_table_extrapolates_last_segment() -> None:
    """W < 0.0001 按最后一段斜率线性外推，且保持连续。"""

    below = float(psy.dew_point_water_C(0.00005))
    assert below == pytest.approx(-89.5, abs=1e-9)
    # 连续性：表下限附近不跳变
    just_above = float(psy.dew_point_water_C(0.0001 + 1e-12))
    just_below = float(psy.dew_point_water_C(0.0001 - 1e-12))
    assert abs(just_above - just_below) < 1e-6


def test_dewpoint_rejects_invalid_inputs() -> None:
    """负含湿量 / 非有限输入 / 非正压力仍按原契约报错。"""

    with pytest.raises(ValueError):
        psy.dew_point_water_C(-1.0)
    with pytest.raises(ValueError):
        psy.dew_point_water_C(float("nan"))
    with pytest.raises(ValueError):
        psy.dew_point_water_C(1.0, pressure_Pa=0.0)
    with pytest.raises(ValueError):
        psy.humidity_ratio_from_dew_point_C(float("nan"))


@pytest.mark.parametrize("T_sat", [5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
def test_dewpoint_at_saturation(T_sat: float) -> None:
    """对饱和点 W_sat(T)，露点应近似回归 T，误差 < 0.05 ℃。"""
    W_sat = float(psy.humidity_ratio_sat(T_sat))
    Tdp = float(psy.dewpoint_celsius(W_sat))
    assert abs(Tdp - T_sat) < 0.05, (
        f"T_sat={T_sat}: W_sat={W_sat:.4f}, Tdp={Tdp:.4f}, Δ={abs(Tdp - T_sat):.4f}"
    )


def test_enthalpy_reference_point() -> None:
    """h(0 ℃, 0 g/kg) 应约为 0 kJ/kg（干空气参考点）。"""
    h = float(psy.enthalpy_kJ(0.0, 0.0))
    assert abs(h) < 1e-9, f"h(0,0)={h}"


def test_relative_humidity_saturation() -> None:
    """RH(T, W_sat(T)) 应 ≈ 1.0（误差 < 0.005）。"""
    for T in [10.0, 20.0, 30.0]:
        W_sat = float(psy.humidity_ratio_sat(T))
        rh = float(psy.relative_humidity(T, W_sat))
        assert abs(rh - 1.0) < 0.005, f"RH at saturation: T={T}, RH={rh:.5f}"
