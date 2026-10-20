"""表冷器组件。"""

from __future__ import annotations

from typing import Tuple

from ..air_state import AirState
from ..energy_term import EnergyTerm
from .. import psychrometrics as psy
from .base import Component


class CoolingCoil(Component):
    """表冷器（冷水盘管）。

    【假设合理性分析】
    需求文档提出"表冷器入口出口空气仅有温度上的变化"的简化假设。
    该假设仅在出口温度高于入口露点时成立（纯显热冷却段），此时含湿量确实不变。
    然而，低露点除湿工况中表冷器出口温度通常低于入口露点，空气中水蒸气达到饱和
    并析出冷凝水，含湿量同时下降（潜热冷却段）。因此原始简化假设不完全成立，
    本实现对两种情形均做了正确处理：

    处理逻辑：
    - T_out < T_dew_in（冷凝段）：W_out = W_sat(T_out)，含湿量下降
    - T_out ≥ T_dew_in（显热段）：W_out = W_in，含湿量不变
    - T_out ≥ T_in（无制冷需求）：气流直通，返回零能量项

    能量计算（两种情形统一）：
        Q = ṁ · (h_out − h_in)  [kW]，数值为负（取走冷量）
    当发生冷凝时，h_out 已反映潜热变化，Q 自动包含潜热分量。
    """

    def __init__(self, T_outlet_setpoint: float, name: str = "CoolingCoil") -> None:
        """
        Args:
            T_outlet_setpoint: 表冷器出口干球温度设定值 (℃)。
            name: 组件名称（用于报告）。
        """
        self.T_out = T_outlet_setpoint
        self.name = name

    def process(self, inlet: AirState) -> Tuple[AirState, EnergyTerm]:
        """处理进入表冷器的空气流。

        Args:
            inlet: 进口空气状态。

        Returns:
            (outlet, energy): 出口空气状态及能量项。
        """
        # 若设定温度不低于入口温度，无制冷需求，直通
        if self.T_out >= inlet.T:
            return inlet, EnergyTerm(name=self.name, Q_kW=0.0)

        W_sat_out = float(psy.saturation_humidity_ratio_g_per_kg(self.T_out, inlet.pressure_Pa))

        # 判断是否发生冷凝
        if W_sat_out < inlet.W:
            # 冷凝除湿：出口含湿量降至饱和值
            W_out = W_sat_out
        else:
            # 纯显热冷却：含湿量不变
            W_out = inlet.W

        outlet = AirState(T=self.T_out, W=W_out, mdot=inlet.mdot, pressure_Pa=inlet.pressure_Pa)
        Q_kW = inlet.mdot * (outlet.h - inlet.h)  # 负值（制冷）
        return outlet, EnergyTerm(name=self.name, Q_kW=Q_kW)
