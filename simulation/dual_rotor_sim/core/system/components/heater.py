"""加热器组件。"""

from __future__ import annotations

from typing import Tuple

from ..air_state import AirState
from ..energy_term import EnergyTerm
from .base import Component


class Heater(Component):
    """加热器（电加热或热水盘管）。

    假设：
    - 出口温度 = 设定值 ``T_outlet_setpoint``
    - 含湿量不变（纯显热加热，无加湿/去湿）
    - 若设定出口温度 ≤ 入口温度：视为无加热需求，气流直接通过

    能量计算：
        Q = ṁ · (h_out − h_in)  [kW]，数值为正（向空气输入热量）
    """

    def __init__(self, T_outlet_setpoint: float, name: str = "Heater") -> None:
        """
        Args:
            T_outlet_setpoint: 加热器出口干球温度设定值 (℃)。
            name: 组件名称。
        """
        self.T_out = T_outlet_setpoint
        self.name = name

    def process(self, inlet: AirState) -> Tuple[AirState, EnergyTerm]:
        """处理进入加热器的空气流。

        Args:
            inlet: 进口空气状态。

        Returns:
            (outlet, energy): 出口空气状态及能量项。
        """
        # 若设定温度不高于入口温度，无加热需求，直通
        if self.T_out <= inlet.T:
            return inlet, EnergyTerm(name=self.name, Q_kW=0.0)

        outlet = AirState(T=self.T_out, W=inlet.W, mdot=inlet.mdot, pressure_Pa=inlet.pressure_Pa)
        Q_kW = inlet.mdot * (outlet.h - inlet.h)  # 正值（加热）
        return outlet, EnergyTerm(name=self.name, Q_kW=Q_kW)
