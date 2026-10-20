"""组件抽象基类。

所有接受单股气流输入、产出单股气流 + 能量项的组件应继承 :class:`Component`。

异构组件（如 :class:`Mixer` 接受两股气流、:class:`Rotor` 接受 :class:`RotorInputs`）
不强制继承，但在实现时应遵循相似的命名惯例。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from ..air_state import AirState
from ..energy_term import EnergyTerm


class Component(ABC):
    """湿空气处理组件抽象基类。

    子类必须实现 :meth:`process`，接收入口 :class:`AirState`，
    返回 ``(outlet: AirState, energy: EnergyTerm)``。
    """

    name: str

    @abstractmethod
    def process(self, inlet: AirState, **kwargs) -> Tuple[AirState, EnergyTerm]:
        """处理进入该组件的空气流。

        Args:
            inlet: 入口空气状态（T, W, mdot）。
            **kwargs: 子类可按需添加额外关键字参数。

        Returns:
            (outlet, energy): 出口空气状态及组件功率/热流项
            （正值 = 固定电功率或输入热量，负值 = 取走冷量）。
        """
        ...
