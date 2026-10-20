"""能量项数据类。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyTerm:
    """单个组件的能量流。

    Attributes:
        name:  组件名称（用于打印和汇总）
        Q_kW:  组件功率或热流 (kW)。
               正值 = 加热器输入热量，或风机的固定电功率；
               负值 = 从空气取走热量/冷量（表冷器）。
    """

    name: str
    Q_kW: float  # kW

    @property
    def type(self) -> str:
        """能量类型：'heating' / 'cooling' / 'neutral'。"""
        if self.Q_kW > 1e-6:
            return "heating"
        elif self.Q_kW < -1e-6:
            return "cooling"
        return "neutral"

    def __repr__(self) -> str:
        direction = "正向功率/热流" if self.Q_kW >= 0 else "←空气(制冷)"
        return f"EnergyTerm({self.name}: {abs(self.Q_kW):.3f} kW {direction})"
