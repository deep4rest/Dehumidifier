"""湿空气状态量数据类。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from . import psychrometrics as psy


@dataclass
class AirState:
    """描述一股湿空气的热力学状态。

    Attributes:
        T:    干球温度 (℃)
        W:    含湿量 (g/kg 干空气)
        mdot: 干空气质量流量 (kg/s)，默认 1.0
        pressure_Pa: 系统总压 (Pa)，默认标准大气压

    只读属性（@property）：
        h     比焓 (kJ/kg 干空气)
        Tdp   露点温度 (℃)
        RH    相对湿度 (0 ~ 1)
    """

    T: float     # 干球温度 (℃)
    W: float     # 含湿量 (g/kg)
    mdot: float = 1.0  # 干空气质量流量 (kg/s)

    pressure_Pa: float = psy.P_ATM

    def __post_init__(self) -> None:
        if not math.isfinite(self.pressure_Pa) or self.pressure_Pa <= 0:
            raise ValueError("空气状态压力必须为正的有限值")

    def at_pressure(self, pressure_Pa: float) -> "AirState":
        """采用系统压力上下文，不改变T、W或质量流量。"""
        return replace(self, pressure_Pa=pressure_Pa)

    # ------------------------------------------------------------------ #
    # 只读派生属性                                                         #
    # ------------------------------------------------------------------ #

    @property
    def h(self) -> float:
        """比焓 (kJ/kg 干空气)。"""
        return float(psy.enthalpy_kJ(self.T, self.W))

    @property
    def Tdp(self) -> float:
        """露点温度 (℃)。"""
        return float(psy.dew_point_water_C(self.W, self.pressure_Pa))

    @property
    def RH(self) -> float:
        """相对湿度 (0 ~ 1)。"""
        return float(psy.relative_humidity(self.T, self.W, self.pressure_Pa))

    # ------------------------------------------------------------------ #
    # 辅助方法                                                             #
    # ------------------------------------------------------------------ #

    def at_mdot(self, new_mdot: float) -> "AirState":
        """返回相同热力学状态但不同流量的新 AirState。"""
        return replace(self, mdot=new_mdot)

    def __repr__(self) -> str:
        return (
            f"AirState(T={self.T:.1f}℃, W={self.W:.2f} g/kg, "
            f"mdot={self.mdot:.3f} kg/s | h={self.h:.2f} kJ/kg, "
            f"Tdp={self.Tdp:.1f}℃, RH={self.RH*100:.1f}%)"
        )
