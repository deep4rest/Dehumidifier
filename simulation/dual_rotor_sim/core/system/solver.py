"""双转轮系统求解器（v2 拓扑修正版）。

提供三个主要入口：

* :func:`build_system`      — 从 artifacts 目录构建 ``DualRotorSystem``（加载 KAN 权重）。
* :func:`forward_simulate`  — 薄封装，调用 ``system.simulate``，支持默认参数补全。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .air_state import AirState
from .system import DualRotorSystem, SystemParams, ControlSetpoints, SystemResult
from .components.rotor import Rotor, SingleRotorModel

# 默认 artifacts 目录：<project_root>/artifacts
_DEFAULT_ARTIFACTS = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "artifacts"
)

# 默认房间回风状态（25℃, 11 g/kg, 0.5 kg/s）
_DEFAULT_RA      = AirState(T=25.0, W=11.0, mdot=0.5)
# 默认再生新风（与典型室外状态相同，0.3 kg/s）
_DEFAULT_REGEN_OA = AirState(T=35.0, W=18.0, mdot=0.3)


def build_system(
    artifacts_dir: Optional[Path] = None,
    device: str = "cpu",
) -> DualRotorSystem:
    """加载 XF 新风转轮与 main 主转轮两个集成模型。

    Args:
        artifacts_dir: 包含各 run 子目录的根目录；``None`` 时使用项目内置默认路径
                       ``<project_root>/artifacts``。
        device:        PyTorch 推理设备，默认 ``"cpu"``。

    Returns:
        已加载权重的 ``DualRotorSystem`` 实例，可直接调用 ``simulate``。

    Example::

        from dual_rotor_sim.core.system.solver import build_system, forward_simulate
        from dual_rotor_sim.core.system.air_state import AirState
        from dual_rotor_sim.core.system.system import SystemParams, ControlSetpoints

        system = build_system()
        oa = AirState(T=35.0, W=18.0, mdot=1.0)
        result = forward_simulate(system, oa)
        print(result.summary())
    """
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else _DEFAULT_ARTIFACTS
    # 新契约只有两个固定 artifact。这里不读取 routing.json，也不根据
    # registry/旧模型自动降级；任一 artifact 缺失或列契约不符都直接抛中文错误。
    xf_model = SingleRotorModel.from_artifact(
        artifacts_dir / "kan_xf_single",
        profile="xf",
        device=device,
    )
    main_model = SingleRotorModel.from_artifact(
        artifacts_dir / "kan_main_single",
        profile="main",
        device=device,
    )
    return DualRotorSystem(
        rotor1=Rotor(model=xf_model, name="Rotor1XF"),
        rotor2=Rotor(model=main_model, name="Rotor2Main"),
    )


def forward_simulate(
    system: DualRotorSystem,
    oa_state: AirState,
    ra_state: Optional[AirState] = None,
    regen_oa_state: Optional[AirState] = None,
    params: Optional[SystemParams] = None,
    setpoints: Optional[ControlSetpoints] = None,
) -> SystemResult:
    """执行一次系统前向仿真。

    Args:
        system:        已构建的 ``DualRotorSystem`` 实例（由 :func:`build_system` 返回）。
        oa_state:      室外新风（OA）状态，温度 (℃)、含湿量 (g/kg)、质量流量 (kg/s)。
        ra_state:      房间回风（RA）状态；``None`` 时使用默认值（25℃, 11 g/kg, 0.5 kg/s）。
        regen_oa_state: 再生侧补充新鲜空气状态；``None`` 时使用默认值（35℃, 18 g/kg, 0.3 kg/s）。
        params:        系统物理参数；``None`` 时使用默认值（见 :class:`SystemParams`）。
        setpoints:     控制设定值；``None`` 时使用默认值（见 :class:`ControlSetpoints`）。

    Returns:
        ``SystemResult``，包含送风状态、各组件能耗和关键中间节点空气状态。
    """
    if ra_state is None:
        ra_state = _DEFAULT_RA
    if regen_oa_state is None:
        regen_oa_state = _DEFAULT_REGEN_OA
    if params is None:
        params = SystemParams()
    if setpoints is None:
        setpoints = ControlSetpoints()
    if getattr(getattr(system, "rotor1", None), "model", None) is not None and getattr(
        system.rotor1.model, "profile", None
    ) == "xf":
        return system.simulate_coupled_xf(
            oa_state=oa_state,
            ra_state=ra_state,
            regen_oa_state=regen_oa_state,
            params=params,
            setpoints=setpoints,
        )
    return system.simulate(
        oa_state=oa_state,
        ra_state=ra_state,
        regen_oa_state=regen_oa_state,
        params=params,
        setpoints=setpoints,
    )
