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
    """0 ℃ 下不得继续使用 Magnus 裁剪或冰面公式。"""

    temperatures = np.array([-75.0, -40.0, -20.0, 0.0, 20.0, 60.0])
    expected_pressure_Pa = np.array(
        [
            0.229661156469,
            18.912149430064,
            125.504169354940,
            611.212867451189,
            2338.803700073981,
            19943.760621620728,
        ]
    )
    actual = psy.saturation_vapor_pressure_water_Pa(temperatures)
    np.testing.assert_allclose(actual, expected_pressure_Pa, rtol=2e-11, atol=1e-11)
    np.testing.assert_array_equal(psy.saturation_pressure_pa(temperatures), actual)


def test_pressure_aware_saturation_humidity_and_water_dewpoint_gold_vector() -> None:
    """W_sat 与露点共用 pressure-aware 水面基准，不裁剪伪值。"""

    temperatures = np.array([-75.0, -40.0, -20.0, 0.0, 20.0, 60.0])
    humidity_g_per_kg = np.array(
        [
            0.001409690912523,
            0.116106712112218,
            0.771315014652683,
            3.774466214309156,
            14.695051649778357,
            152.417464941057148,
        ]
    )
    assert np.isinf(psy.saturation_humidity_ratio_g_per_kg(60.0, 10_000.0))
    recovered = psy.dew_point_water_C(humidity_g_per_kg, 101_325.0)
    np.testing.assert_allclose(recovered, temperatures, atol=2e-8, rtol=0.0)


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
