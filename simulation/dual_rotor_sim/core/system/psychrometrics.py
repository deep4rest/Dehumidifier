"""湿空气热力学物性计算模块。

饱和水汽压公式与 ``docs/requirement.txt`` 中企业参考代码
（CalculateAH/CalculateRH）逐系数一致：

- ``T <= 0 ℃``：企业冰面基准多项式（参考代码 ``DBT <= 0`` 分支）。
- ``T > 0 ℃``：液态水多项式（与参考代码 ``DBT > 0`` 分支相同，
  历史实现即为此公式，未改动）。
- 水汽气体常数比使用参考代码的 0.62198。

露点 ↔ 含湿量换算（企业 DptAH 映射表契约）：

- 露点 <= 0 ℃（W <= 3.764 g/kg）：使用 DptAH 表分段线性插值。表项
  ``DptAH[i]`` 为露点 ``-i ℃`` 对应的含湿量 (g/kg)，88 项覆盖
  0 ~ -87 ℃；区间内线性插值（如露点 -0.5 ℃ 对应
  ``0.5*(DptAH[0]+DptAH[1])``）。该表按 101.325 kPa 标定，当前工况
  契约仅在配置压力 101325 Pa 下物理一致。
- 露点 > 0 ℃（W > 3.764 g/kg）：超出映射表范围，按同一饱和压公式
  在 [0, 100] ℃ 数值反演（混合方案，保证 OA/CC 出口等正露点状态可用）。
- ``0 < W < 0.0001 g/kg``：低于表下限，按最后一段斜率线性外推。
- ``W == 0``：返回 ``-inf`` 干燥极限（合法值，JSON 用 ``null``）。

所有函数接受标量或 numpy 数组；带压力的函数不得将饱和压裁剪到总压以下。
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

Numeric = Union[float, np.ndarray]
P_ATM = 101325.0
_WATER_RATIO = 0.62198

# 企业 DptAH 含湿量-露点映射表：DptAH[i] = 露点 -i ℃ 对应含湿量 (g/kg)。
_DPT_AH_TABLE: tuple[float, ...] = (
    3.764, 3.466, 3.189, 2.933, 2.694, 2.474, 2.271, 2.083, 1.91, 1.749,
    1.601, 1.464, 1.338, 1.223, 1.116, 1.018, 0.928, 0.845, 0.769, 0.741,
    0.636, 0.578, 0.522, 0.475, 0.431, 0.39, 0.353, 0.319, 0.288, 0.26,
    0.233, 0.211, 0.19, 0.171, 0.153, 0.138, 0.123, 0.111, 0.099, 0.089,
    0.079, 0.071, 0.063, 0.056, 0.0499, 0.0444, 0.0394, 0.035, 0.031, 0.0274,
    0.0243, 0.0214, 0.0189, 0.0167, 0.0147, 0.0129, 0.0113, 0.0099, 0.0087, 0.0076,
    0.0067, 0.0058, 0.0051, 0.0044, 0.0038, 0.0033, 0.0029, 0.0025, 0.0022, 0.0019,
    0.0016, 0.0014, 0.0012, 0.001, 0.0009, 0.00075, 0.00064, 0.00055, 0.00047, 0.0004,
    0.00034, 0.00029, 0.00024, 0.0002, 0.00017, 0.00015, 0.00012, 0.0001,
)

# 插值用递增序列（W 从小到大，对应露点 -87 ~ 0 ℃）。
_DPT_AH_REVERSED = np.asarray(_DPT_AH_TABLE[::-1], dtype=np.float64)
_DPT_AH_TEMPERATURES = np.arange(
    -(len(_DPT_AH_TABLE) - 1), 1, dtype=np.float64
)
_DPT_AH_MAX = _DPT_AH_TABLE[0]      # 3.764 g/kg，露点 0 ℃
_DPT_AH_MIN = _DPT_AH_TABLE[-1]     # 0.0001 g/kg，露点 -87 ℃
# 低于表下限时按最后一段（-87 ~ -86 ℃）线性外推的斜率 dTdp/dW。
_DPT_AH_EXTRAPOLATION_SLOPE = (
    (_DPT_AH_TEMPERATURES[1] - _DPT_AH_TEMPERATURES[0])
    / (_DPT_AH_REVERSED[1] - _DPT_AH_REVERSED[0])
)
if np.any(np.diff(_DPT_AH_REVERSED) <= 0.0):  # pragma: no cover - 表数据错误才触发
    raise RuntimeError("DptAH 映射表必须严格单调递增（按含湿量升序）")


def _return_scalar_if_scalar(source: object, result: np.ndarray) -> Numeric:
    if np.asarray(source).ndim == 0:
        return float(np.asarray(result))
    return result


def saturation_vapor_pressure_water_Pa(T_celsius: Numeric) -> Numeric:
    """计算水面饱和水汽压 (Pa)。

    - ``T <= 0 ℃``：企业参考代码冰面基准多项式（``docs/requirement.txt``
      ``DBT <= 0`` 分支，2026-09-17 替换原 Murphy–Koop 过冷水公式）。
    - ``T > 0 ℃``：液态水多项式（与参考代码 ``DBT > 0`` 分支逐系数一致，
      历史实现即为此公式）。
    """

    values = np.asarray(T_celsius, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("饱和压计算的温度存在非有限值")
    absolute = values + 273.15
    if np.any(absolute <= 0.0):
        raise ValueError("饱和压计算温度低于绝对零度")

    result = np.empty_like(absolute, dtype=np.float64)
    at_or_below_zero = values <= 0.0
    low_temperature = absolute[at_or_below_zero]
    if low_temperature.size:
        ln_pressure = (
            -5674.5359 / low_temperature
            + 6.3925247
            - 9.677843e-3 * low_temperature
            + 6.2215701e-7 * low_temperature**2
            + 2.0747825e-9 * low_temperature**3
            - 9.484024e-13 * low_temperature**4
            + 4.1635 * np.log(low_temperature)
        )
        result[at_or_below_zero] = np.exp(ln_pressure)

    above_temperature = absolute[~at_or_below_zero]
    if above_temperature.size:
        ln_pressure = (
            -5.8002206e3 / above_temperature
            + 1.3914993
            - 4.8640239e-2 * above_temperature
            + 4.1764768e-5 * above_temperature**2
            - 1.4452093e-8 * above_temperature**3
            + 6.5459673 * np.log(above_temperature)
        )
        result[~at_or_below_zero] = np.exp(ln_pressure)
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


def humidity_ratio_from_dew_point_C(
    T_celsius: Numeric,
    pressure_Pa: float = P_ATM,
) -> Numeric:
    """由露点温度查表/反演含湿量 (g/kg 干空气)。

    - 露点 <= 0 ℃：DptAH 表分段线性插值；低于 -87 ℃ 按最后一段
      斜率线性外推（外推含湿量 <= 0 时报错）。
    - 露点 > 0 ℃：按饱和含湿量公式计算（映射表不覆盖正露点区间）。
    - ``-inf`` 露点（干燥极限）返回 0，为 ``W == 0 → -inf`` 的反函数。
    """

    if not math.isfinite(pressure_Pa) or pressure_Pa <= 0.0:
        raise ValueError("系统总压必须是正的有限值")
    shaped_input = np.asarray(T_celsius, dtype=np.float64)
    dew_points = shaped_input.reshape(-1)
    result = np.empty_like(dew_points)

    negative_infinite = np.isneginf(dew_points)
    result[negative_infinite] = 0.0

    invalid = np.isnan(dew_points) | np.isposinf(dew_points)
    if invalid.any():
        raise ValueError("露点输入存在非有限值（-inf 干燥极限除外）")

    finite = np.isfinite(dew_points)
    at_or_below_zero = finite & (dew_points <= 0.0)
    if at_or_below_zero.any():
        clipped = np.clip(
            dew_points[at_or_below_zero],
            _DPT_AH_TEMPERATURES[0],
            _DPT_AH_TEMPERATURES[-1],
        )
        interpolated = np.interp(clipped, _DPT_AH_TEMPERATURES, _DPT_AH_REVERSED)
        below_table = dew_points[at_or_below_zero] < _DPT_AH_TEMPERATURES[0]
        if below_table.any():
            extrapolated = _DPT_AH_REVERSED[0] + (
                dew_points[at_or_below_zero][below_table]
                - _DPT_AH_TEMPERATURES[0]
            ) / _DPT_AH_EXTRAPOLATION_SLOPE
            if np.any(extrapolated <= 0.0):
                raise ValueError("露点超出线性外推有效范围，含湿量将为非正值")
            interpolated[below_table] = extrapolated
        result[at_or_below_zero] = interpolated

    above_zero = finite & (dew_points > 0.0)
    if above_zero.any():
        result[above_zero] = np.asarray(
            saturation_humidity_ratio_g_per_kg(
                dew_points[above_zero], pressure_Pa
            ),
            dtype=np.float64,
        )
    return _return_scalar_if_scalar(T_celsius, result.reshape(shaped_input.shape))


def dew_point_water_C(W_g_kg: Numeric, pressure_Pa: float = P_ATM) -> Numeric:
    """由含湿量计算露点 (℃)——企业 DptAH 映射表契约。

    - ``W == 0``：返回 ``-inf``（合法干燥极限）。
    - ``0 < W <= 3.764 g/kg``（露点 <= 0 ℃）：DptAH 表分段线性插值；
      ``W < 0.0001 g/kg`` 低于表下限时按最后一段斜率线性外推。
    - ``W > 3.764 g/kg``（超出映射表）：按同一饱和压公式在 [-150, 100] ℃
      数值反演。区间下界延伸到 0 ℃ 以下，用于覆盖表上界 3.764 与公式
      0 ℃ 饱和含湿量（约 3.774）之间的窄缝，以及低压工况。
    """

    if not math.isfinite(pressure_Pa) or pressure_Pa <= 0.0:
        raise ValueError("系统总压必须是正的有限值")
    humidity = np.asarray(W_g_kg, dtype=np.float64)
    if not np.isfinite(humidity).all() or np.any(humidity < 0.0):
        raise ValueError("水面露点输入含湿量必须非负且有限")

    flat = humidity.reshape(-1)
    result = np.full_like(flat, -math.inf)
    positive = flat > 0.0
    table_mask = positive & (flat <= _DPT_AH_MAX)
    if table_mask.any():
        clipped = np.clip(flat[table_mask], _DPT_AH_REVERSED[0], _DPT_AH_REVERSED[-1])
        interpolated = np.interp(clipped, _DPT_AH_REVERSED, _DPT_AH_TEMPERATURES)
        below_table = flat[table_mask] < _DPT_AH_REVERSED[0]
        if below_table.any():
            interpolated[below_table] = _DPT_AH_TEMPERATURES[0] + (
                flat[table_mask][below_table] - _DPT_AH_REVERSED[0]
            ) * _DPT_AH_EXTRAPOLATION_SLOPE
        result[table_mask] = interpolated

    formula_mask = positive & (flat > _DPT_AH_MAX)
    if formula_mask.any():
        vapor_pressure = pressure_Pa * flat[formula_mask] / (
            1000.0 * _WATER_RATIO + flat[formula_mask]
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
        result[formula_mask] = (low + high) / 2.0
    shaped = result.reshape(humidity.shape)
    return _return_scalar_if_scalar(W_g_kg, shaped)


def dewpoint_celsius(W_g_kg: Numeric) -> Numeric:
    """兼容常压 API；委托给规范水面露点函数。"""

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
    """计算相对湿度，并限制为 [0, 1]。

    物理式与 ``docs/requirement.txt`` 企业参考代码一致
    （``RH = p_v / p_ws(T)``，参考代码以百分数表述，本实现输出 0~1）。
    """

    saturation_pressure = np.asarray(
        saturation_vapor_pressure_water_Pa(T_celsius), dtype=np.float64
    )
    humidity = np.maximum(np.asarray(W_g_kg, dtype=np.float64), 0.0)
    if not math.isfinite(pressure_Pa) or pressure_Pa <= 0:
        raise ValueError("系统总压必须为正的有限值")
    vapor_pressure = pressure_Pa * humidity / (1000.0 * _WATER_RATIO + humidity)
    return np.clip(vapor_pressure / saturation_pressure, 0.0, 1.0)
