"""湿空气热力学物性计算模块。

饱和压和露点统一采用水面基准：0 ℃ 以下使用 Murphy–Koop
过冷液态水公式，0 ℃ 及以上使用 ASHRAE 液态水公式。所有函数
接受标量或 numpy 数组；带压力的函数不得将饱和压裁剪到总压以下。
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

Numeric = Union[float, np.ndarray]
P_ATM = 101325.0
_WATER_RATIO = 0.621945


def _return_scalar_if_scalar(source: object, result: np.ndarray) -> Numeric:
    if np.asarray(source).ndim == 0:
        return float(np.asarray(result))
    return result


def saturation_vapor_pressure_water_Pa(T_celsius: Numeric) -> Numeric:
    """计算水面饱和水汽压 (Pa)。

    - ``T < 0 ℃``：Murphy–Koop 过冷液态水面公式。
    - ``T >= 0 ℃``：ASHRAE 液态水面公式。
    """

    values = np.asarray(T_celsius, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("饱和压计算的温度存在非有限值")
    absolute = values + 273.15
    if np.any(absolute <= 0.0):
        raise ValueError("饱和压计算温度低于绝对零度")

    result = np.empty_like(absolute, dtype=np.float64)
    below_zero = values < 0.0
    below_temperature = absolute[below_zero]
    if below_temperature.size:
        ln_pressure = (
            54.842763
            - 6763.22 / below_temperature
            - 4.210 * np.log(below_temperature)
            + 0.000367 * below_temperature
            + np.tanh(0.0415 * (below_temperature - 218.8))
            * (
                53.878
                - 1331.22 / below_temperature
                - 9.44523 * np.log(below_temperature)
                + 0.014025 * below_temperature
            )
        )
        result[below_zero] = np.exp(ln_pressure)

    above_temperature = absolute[~below_zero]
    if above_temperature.size:
        ln_pressure = (
            -5.8002206e3 / above_temperature
            + 1.3914993
            - 4.8640239e-2 * above_temperature
            + 4.1764768e-5 * above_temperature**2
            - 1.4452093e-8 * above_temperature**3
            + 6.5459673 * np.log(above_temperature)
        )
        result[~below_zero] = np.exp(ln_pressure)
    return _return_scalar_if_scalar(T_celsius, result)


def saturation_pressure_pa(T_celsius: Numeric) -> Numeric:
    """兼容名称；始终委托给规范水面饱和压函数。"""

    return saturation_vapor_pressure_water_Pa(T_celsius)


def saturation_humidity_ratio_g_per_kg(
    T_celsius: Numeric,
    pressure_Pa: float = P_ATM,
) -> Numeric:
    """计算水面饱和含湿量 (g/kg 干空气)。

    ``p_ws >= pressure_Pa`` 时返回 ``+inf``，不通过裁剪伪造有限值。
    """

    if not math.isfinite(pressure_Pa) or pressure_Pa <= 0.0:
        raise ValueError("系统总压必须是正的有限值")
    saturation_pressure = np.asarray(
        saturation_vapor_pressure_water_Pa(T_celsius), dtype=np.float64
    )
    result = np.full_like(saturation_pressure, np.inf, dtype=np.float64)
    finite_mask = saturation_pressure < pressure_Pa
    result[finite_mask] = (
        1000.0
        * _WATER_RATIO
        * saturation_pressure[finite_mask]
        / (pressure_Pa - saturation_pressure[finite_mask])
    )
    return _return_scalar_if_scalar(T_celsius, result)


def humidity_ratio_sat(T_celsius: Numeric) -> Numeric:
    """兼容常压 API；委托给 pressure-aware 规范函数。"""

    return saturation_humidity_ratio_g_per_kg(T_celsius, P_ATM)


def dew_point_water_C(W_g_kg: Numeric, pressure_Pa: float = P_ATM) -> Numeric:
    """由含湿量按同一水面饱和曲线数值反演露点 (℃)。"""

    if not math.isfinite(pressure_Pa) or pressure_Pa <= 0.0:
        raise ValueError("系统总压必须是正的有限值")
    humidity = np.asarray(W_g_kg, dtype=np.float64)
    if not np.isfinite(humidity).all() or np.any(humidity < 0.0):
        raise ValueError("水面露点输入含湿量必须非负且有限")

    flat = humidity.reshape(-1)
    result = np.full_like(flat, -math.inf)
    positive = flat > 0.0
    vapor_pressure = pressure_Pa * flat[positive] / (
        1000.0 * _WATER_RATIO + flat[positive]
    )
    low_pressure = float(saturation_vapor_pressure_water_Pa(-150.0))
    high_pressure = float(saturation_vapor_pressure_water_Pa(100.0))
    outside = (vapor_pressure < low_pressure) | (vapor_pressure > high_pressure)
    if outside.any():
        raise ValueError(
            "水面露点超出数值反演范围: "
            f"vapor_pressure={vapor_pressure[outside][0]} Pa"
        )

    low = np.full_like(vapor_pressure, -150.0)
    high = np.full_like(vapor_pressure, 100.0)
    for _ in range(100):
        midpoint = (low + high) / 2.0
        below_target = np.asarray(
            saturation_vapor_pressure_water_Pa(midpoint), dtype=np.float64
        ) < vapor_pressure
        low = np.where(below_target, midpoint, low)
        high = np.where(below_target, high, midpoint)
    result[positive] = (low + high) / 2.0
    shaped = result.reshape(humidity.shape)
    return _return_scalar_if_scalar(W_g_kg, shaped)


def dewpoint_celsius(W_g_kg: Numeric) -> Numeric:
    """兼容常压 API；委托给规范水面露点反演。"""

    return dew_point_water_C(W_g_kg, P_ATM)


def enthalpy_kJ(T_celsius: Numeric, W_g_kg: Numeric) -> Numeric:
    """湿空气比焓 (kJ/kg 干空气)。"""

    temperature = np.asarray(T_celsius, dtype=float)
    humidity = np.asarray(W_g_kg, dtype=float) / 1000.0
    return 1.006 * temperature + humidity * (2501.0 + 1.86 * temperature)


def temperature_from_enthalpy(h_kJ: Numeric, W_g_kg: Numeric) -> Numeric:
    """由焓和含湿量反算干球温度 (℃)。"""

    enthalpy = np.asarray(h_kJ, dtype=float)
    humidity = np.asarray(W_g_kg, dtype=float) / 1000.0
    return (enthalpy - 2501.0 * humidity) / (1.006 + 1.86 * humidity)


def relative_humidity(T_celsius: Numeric, W_g_kg: Numeric, pressure_Pa: float = P_ATM) -> Numeric:
    """计算相对湿度，并限制为 [0, 1]。"""

    saturation_pressure = np.asarray(
        saturation_vapor_pressure_water_Pa(T_celsius), dtype=np.float64
    )
    humidity = np.maximum(np.asarray(W_g_kg, dtype=np.float64), 0.0)
    if not math.isfinite(pressure_Pa) or pressure_Pa <= 0:
        raise ValueError("系统总压必须为正的有限值")
    vapor_pressure = pressure_Pa * humidity / (1000.0 * _WATER_RATIO + humidity)
    return np.clip(vapor_pressure / saturation_pressure, 0.0, 1.0)
