"""风机组件。"""

from __future__ import annotations

from typing import Tuple

from ..air_state import AirState
from ..energy_term import EnergyTerm
from .base import Component


class Fan(Component):
    """风机（固定功率直通模型，固定 50 Hz）。

    假设：
    - 频率固定为 50 Hz，风量由系统质量流量 ``mdot`` 给定（不另建变频模型）
    - 风机功率直接采用配置中的固定值 ``P_kW``，不根据流量、压升或焓差计算
    - 忽略风机对空气的温升，出口温度、含湿量和质量流量均与入口一致

    能量项说明：
        ``EnergyTerm.Q_kW = P_kW``，用于系统固定电功率汇总；它不表示空气吸热量。
    """

    def __init__(self, P_kW: float = 1.5, name: str = "Fan") -> None:
        """
        Args:
            P_kW: 风机电功率 (kW)，铭牌值或实测平均值。
            name: 组件名称。
        """
        if P_kW < 0:
            raise ValueError(f"风机功率必须 ≥ 0，收到 {P_kW}")
        self.P_kW = P_kW
        self.name = name

    def process(self, inlet: AirState) -> Tuple[AirState, EnergyTerm]:
        """返回风机直通后的空气状态和固定功率。

        Args:
            inlet: 风机入口空气状态（温度、含湿量、流量）。

        Returns:
            (outlet, energy): 原入口状态和固定功率项。
        """
        return inlet, EnergyTerm(name=self.name, Q_kW=self.P_kW)
