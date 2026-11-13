"""串联末端后表冷与后加热；同一目标决定两者按需启停。"""
import math
from numbers import Real

from .cooling_coil import CoolingCoil
from .heater import Heater


def condition_supply(inlet, target_T):
    if target_T is None:
        return inlet, inlet, {"CC3_kW": 0., "Heater_Post_kW": 0.}
    if isinstance(target_T, bool) or not isinstance(target_T, Real) or not math.isfinite(target_T) or target_T <= -273.15:
        raise ValueError("送风温度目标必须为高于绝对零度的有限温度")
    after_cc3, cooling = CoolingCoil(target_T, name="CC3").process(inlet)
    supply, heating = Heater(target_T, name="Heater_Post").process(after_cc3)
    return after_cc3, supply, {"CC3_kW":cooling.Q_kW, "Heater_Post_kW":heating.Q_kW}
