"""混风组件。"""

from __future__ import annotations

from ..air_state import AirState
from .. import psychrometrics as psy


class Mixer:
    """两股湿空气的绝热混合点。

    系统中共有两处混风点：
    - Mix1：新风转轮处理区出口 + 房间回风（RA） → 二级风机入口
    - Mix2：主转轮再生区出口 + 再生新风 + 新风转轮冷吹区出口（三股，通过两次两流混合实现）

    混风公式（质量守恒 + 焓守恒）：
        ṁ_mix = ṁ₁ + ṁ₂
        W_mix = (ṁ₁·W₁ + ṁ₂·W₂) / ṁ_mix        [g/kg 干空气]
        h_mix = (ṁ₁·h₁ + ṁ₂·h₂) / ṁ_mix        [kJ/kg 干空气]
        T_mix = (h_mix − 2501·d) / (1.006 + 1.86·d)  [d = W_mix/1000]

    注意：温度 T_mix **必须**通过焓和含湿量反算，
    不能直接质量加权温度（因为湿空气比热容随含湿量变化，直接加权温度有误差）。
    """

    def __init__(self, name: str = "Mixer") -> None:
        self.name = name

    def mix(self, s1: AirState, s2: AirState) -> AirState:
        """混合两股气流。

        Args:
            s1: 气流 1 的状态（含流量 mdot）。
            s2: 气流 2 的状态（含流量 mdot）。

        Returns:
            混合后的 AirState（mdot = mdot1 + mdot2）。

        Raises:
            ValueError: 若两股气流压力不一致或总流量 ≤ 0。
        """
        if s1.pressure_Pa != s2.pressure_Pa:
            raise ValueError("混风两股气流压力必须一致")
        mdot3 = s1.mdot + s2.mdot
        if mdot3 <= 0:
            raise ValueError(
                f"混风两股流量之和 ≤ 0: ṁ₁={s1.mdot}, ṁ₂={s2.mdot}"
            )

        W3 = (s1.mdot * s1.W + s2.mdot * s2.W) / mdot3
        h3 = (s1.mdot * s1.h + s2.mdot * s2.h) / mdot3
        T3 = float(psy.temperature_from_enthalpy(h3, W3))

        return AirState(T=T3, W=W3, mdot=mdot3, pressure_Pa=s1.pressure_Pa)
