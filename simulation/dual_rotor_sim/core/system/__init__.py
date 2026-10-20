"""dual_rotor_sim.core.system 包 —— 双转轮低露点除湿机系统级模型（v2 拓扑修正版）。

主要公开接口
----------
* :class:`AirState`            — 湿空气热力状态点
* :class:`EnergyTerm`          — 单组件能量项
* :class:`SystemParams`        — 系统物理参数
* :class:`ControlSetpoints`    — 操作员设定值
* :class:`SystemResult`        — 系统仿真结果
* :class:`DualRotorSystem`     — 双转轮系统模型（前向仿真）
* :func:`build_system`         — 从两个单 KAN artifact 加载模型，构建系统
* :func:`forward_simulate`     — 一次前向仿真的便捷入口

组件子包 :mod:`src.system.components`
--------------------------------------
* :class:`CoolingCoil`   — 表冷器（冷凝去湿 / 显热冷却）
* :class:`Heater`        — 加热器（显热加热）
* :class:`Fan`           — 处理风机（固定 50 Hz）
* :class:`Mixer`         — 混风点（焓-含湿量守恒）
* :class:`Rotor`         — 转轮组件（KAN 集成预测器）
* :class:`SingleRotorModel` — 固定六输出单 KAN 运行时模型
* :class:`RotorInputs`   — 转轮输入数据类
* :class:`RotorOutputs`  — 转轮输出数据类
"""

from .air_state import AirState
from .energy_term import EnergyTerm
from .system import DualRotorSystem, SystemParams, ControlSetpoints, SystemResult
from .solver import build_system, forward_simulate
from .components import (
    CoolingCoil,
    Heater,
    Fan,
    Mixer,
    Rotor,
    RotorEnsemble,
    SingleRotorModel,
    RotorInputs,
    RotorOutputs,
)

__all__ = [
    # 核心数据类
    "AirState",
    "EnergyTerm",
    # 系统级
    "SystemParams",
    "ControlSetpoints",
    "SystemResult",
    "DualRotorSystem",
    # 求解器
    "build_system",
    "forward_simulate",
    # 组件
    "CoolingCoil",
    "Heater",
    "Fan",
    "Mixer",
    "Rotor",
    "RotorEnsemble",
    "SingleRotorModel",
    "RotorInputs",
    "RotorOutputs",
]
