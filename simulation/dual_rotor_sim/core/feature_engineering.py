"""派生特征工程：基于 8 个原始输入派生湿空气物理量与工艺/交叉特征。

所有派生特征都注册在 ``DERIVED_FEATURES`` 中，每项形如：
``name -> (依赖的原始列名列表, 计算函数)``。

设计约束：
1. 物理可解释：所有公式均来源于湿空气热力学（Magnus 方程、焓公式）或工艺直觉。
2. 零数据泄漏：派生特征**只**依赖原始输入列（``入口含湿量 / 转轮转速 / 再生温度 /
   冷吹入口温度 / 再生角度 / 风速 / 冷吹角度 / 处理入口温度``），绝不引用任何 ``出口`` / 输出列。
3. 数值稳定：温度计算使用 ``clip`` 约束在物理合理区间，含湿量做下限约束以防 ``log`` 溢出。

使用方式（在 train 入口）：
    from training.src.feature_engineering import apply_derived_features
    df, added = apply_derived_features(df, ["入口饱和水汽压_Ps", ...])
    selection.inputs.extend(added)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---- 原始输入列名（与 configs/kan.yaml 严格一致） ----
COL_D_IN = "入口含湿量(g/kg)"          # 含湿量, g 水 / kg 干空气
COL_RPM = "转轮转速(rpm)"
COL_T_REG = "再生温度(℃)"
COL_T_COOL_IN = "冷吹入口温度(℃)"
COL_ANG_REG = "再生角度(°)"
COL_V = "风速(m/s)"
COL_ANG_COOL = "冷吹角度(°)"
COL_T_PROC_IN = "处理入口温度(℃)"

# 大气压（Pa），实验室假设为标准大气压
P_ATM = 101325.0

# ---- 基础湿空气工具函数 ----
def _saturation_pressure_pa(T_celsius: np.ndarray) -> np.ndarray:
    """Magnus 公式计算饱和水汽压 (Pa)。

    公式：``Ps = 611.2 · exp(17.67 · T / (T + 243.5))``，T 单位为 ℃。
    适用 -45℃ ~ 60℃，工程精度足够。
    """

    T = np.clip(T_celsius.astype(np.float64), -50.0, 120.0)
    return 611.2 * np.exp(17.67 * T / (T + 243.5))


def _vapor_pressure_pa(d_g_per_kg: np.ndarray) -> np.ndarray:
    """由含湿量 d (g/kg) 反算水蒸气分压 (Pa)。

    公式：``Pv = d · P_atm / (0.622 + d)``，d 必须先转为 kg/kg。
    """

    d = np.clip(d_g_per_kg.astype(np.float64), 1e-6, None) / 1000.0  # g/kg -> kg/kg
    return d * P_ATM / (0.622 + d)


def _dewpoint_celsius(Pv_pa: np.ndarray) -> np.ndarray:
    """由水蒸气分压反算露点 (℃)，Magnus 反解。"""

    # 防止 log(0)
    Pv = np.clip(Pv_pa, 1e-3, None)
    ln_ratio = np.log(Pv / 611.2)
    return 243.5 * ln_ratio / (17.67 - ln_ratio)


# ---- 派生特征实现 ----
def _f_ps_in(df: pd.DataFrame) -> pd.Series:
    """处理入口饱和水汽压 (Pa)。"""

    return pd.Series(
        _saturation_pressure_pa(df[COL_T_PROC_IN].to_numpy()),
        index=df.index,
        dtype=np.float64,
    )


def _f_rh_in(df: pd.DataFrame) -> pd.Series:
    """处理入口相对湿度 (0~1)。

    RH = Pv / Ps；按 [0, 1.5] 截断，允许略高于 1 表示数据噪声。
    """

    Ps = _saturation_pressure_pa(df[COL_T_PROC_IN].to_numpy())
    Pv = _vapor_pressure_pa(df[COL_D_IN].to_numpy())
    rh = np.clip(Pv / np.clip(Ps, 1e-3, None), 0.0, 1.5)
    return pd.Series(rh, index=df.index, dtype=np.float64)


def _f_dewpoint_in(df: pd.DataFrame) -> pd.Series:
    """处理入口露点 (℃)。"""

    Pv = _vapor_pressure_pa(df[COL_D_IN].to_numpy())
    return pd.Series(_dewpoint_celsius(Pv), index=df.index, dtype=np.float64)


def _f_enthalpy_in(df: pd.DataFrame) -> pd.Series:
    """处理入口焓 (kJ/kg 干空气)。

    h = 1.006 · T + d · (2501 + 1.86 · T)，d 单位 kg/kg。
    """

    T = df[COL_T_PROC_IN].to_numpy(dtype=np.float64)
    d = df[COL_D_IN].to_numpy(dtype=np.float64) / 1000.0  # g/kg -> kg/kg
    h = 1.006 * T + d * (2501.0 + 1.86 * T)
    return pd.Series(h, index=df.index, dtype=np.float64)


def _f_proc_angle(df: pd.DataFrame) -> pd.Series:
    """处理段角度 (°) = 360 − 再生角度 − 冷吹角度。"""

    proc = 360.0 - df[COL_ANG_REG].to_numpy(dtype=np.float64) - df[COL_ANG_COOL].to_numpy(dtype=np.float64)
    return pd.Series(np.clip(proc, 1.0, 360.0), index=df.index, dtype=np.float64)


def _f_tau_proc(df: pd.DataFrame) -> pd.Series:
    """处理段停留时间常数 (s) ≈ 处理角度占比 / (转速·rpm/60)。

    tau = (proc_angle / 360) / (rpm / 60) = proc_angle / (6 · rpm)。
    """

    proc = 360.0 - df[COL_ANG_REG].to_numpy(dtype=np.float64) - df[COL_ANG_COOL].to_numpy(dtype=np.float64)
    proc = np.clip(proc, 1.0, 360.0)
    rpm = np.clip(df[COL_RPM].to_numpy(dtype=np.float64), 1e-3, None)
    return pd.Series(proc / (6.0 * rpm), index=df.index, dtype=np.float64)


def _f_tau_reg(df: pd.DataFrame) -> pd.Series:
    """再生段停留时间常数 (s) ≈ reg_angle / (6 · rpm)。"""

    ang = np.clip(df[COL_ANG_REG].to_numpy(dtype=np.float64), 1.0, 360.0)
    rpm = np.clip(df[COL_RPM].to_numpy(dtype=np.float64), 1e-3, None)
    return pd.Series(ang / (6.0 * rpm), index=df.index, dtype=np.float64)


def _f_dT_reg(df: pd.DataFrame) -> pd.Series:
    """再生段温度驱动力 (℃) = 再生温度 − 处理入口温度。"""

    return df[COL_T_REG] - df[COL_T_PROC_IN]


def _f_rh_x_tau_proc(df: pd.DataFrame) -> pd.Series:
    """RH × 处理段时间常数：表征吸附潜力 × 接触时间。"""

    return _f_rh_in(df) * _f_tau_proc(df)


def _f_dT_x_tau_reg(df: pd.DataFrame) -> pd.Series:
    """ΔT_reg × 再生段时间常数：表征脱附驱动 × 接触时间。"""

    return _f_dT_reg(df) * _f_tau_reg(df)


# ---- 派生特征注册表 ----
# 每项: name -> (function, 依赖列名列表, 中文说明)
DERIVED_FEATURES: Dict[str, Tuple[Callable[[pd.DataFrame], pd.Series], List[str], str]] = {
    # 组 A：湿空气状态
    "入口饱和水汽压_Ps": (_f_ps_in, [COL_T_PROC_IN], "Magnus 公式饱和水汽压 (Pa)"),
    "入口相对湿度_RH": (_f_rh_in, [COL_T_PROC_IN, COL_D_IN], "相对湿度 (0~1)"),
    "入口露点_Tdp": (_f_dewpoint_in, [COL_D_IN], "由含湿量反算的露点 (℃)"),
    "入口焓_h": (_f_enthalpy_in, [COL_T_PROC_IN, COL_D_IN], "湿空气焓 (kJ/kg)"),
    # 组 B：工艺/几何
    "处理段角度": (_f_proc_angle, [COL_ANG_REG, COL_ANG_COOL], "处理段角度 (°)"),
    "处理段时间常数": (_f_tau_proc, [COL_ANG_REG, COL_ANG_COOL, COL_RPM], "处理段停留时间 (s)"),
    "再生段时间常数": (_f_tau_reg, [COL_ANG_REG, COL_RPM], "再生段停留时间 (s)"),
    "再生驱动力_dT": (_f_dT_reg, [COL_T_REG, COL_T_PROC_IN], "再生温度 − 处理入口温度 (℃)"),
    # 组 C：交叉项
    "RH×处理时间": (_f_rh_x_tau_proc, [COL_T_PROC_IN, COL_D_IN, COL_ANG_REG, COL_ANG_COOL, COL_RPM], "RH × 处理段时间常数"),
    "dT×再生时间": (_f_dT_x_tau_reg, [COL_T_REG, COL_T_PROC_IN, COL_ANG_REG, COL_RPM], "ΔT_reg × 再生段时间常数"),
}


def list_available_features() -> List[str]:
    """返回所有可用的派生特征名称。"""

    return list(DERIVED_FEATURES.keys())


def apply_derived_features(
    df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """在 ``df`` 上添加指定的派生特征列。

    Args:
        df: 已包含原始 8 个输入列的 DataFrame。
        feature_names: 要派生的特征名称列表，必须出现在 ``DERIVED_FEATURES`` 中。

    Returns:
        ``(df_out, added_columns)``：``df_out`` 在原 df 上追加了新列；
        ``added_columns`` 是实际新增的列名（去重、保持顺序）。

    Raises:
        KeyError: 派生特征名未注册或依赖列在 df 中缺失。
        ValueError: 计算结果出现 NaN/Inf。
    """

    if not feature_names:
        return df.copy(), []

    out = df.copy()
    added: List[str] = []
    for name in feature_names:
        if name in out.columns:
            # 已存在则跳过，避免重复计算
            continue
        if name not in DERIVED_FEATURES:
            raise KeyError(
                f"未注册的派生特征 '{name}'，可用列表: {list_available_features()}"
            )
        fn, deps, _desc = DERIVED_FEATURES[name]
        missing = [c for c in deps if c not in out.columns]
        if missing:
            raise KeyError(
                f"派生特征 '{name}' 缺少依赖列: {missing}"
            )
        series = fn(out)
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=out.index)
        # NaN/Inf 自检：派生特征不允许引入新的非数
        arr = series.to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(arr)
        # 允许原始 df 中已有 NaN 行透传 NaN（清洗阶段统一处理），
        # 但若有限值之外出现 Inf 则视为公式问题，直接报错。
        if np.any(np.isinf(arr[finite_mask | ~finite_mask])):
            raise ValueError(f"派生特征 '{name}' 计算出现 Inf，请检查公式或输入范围")
        out[name] = series.astype(np.float64)
        added.append(name)
    return out, added
