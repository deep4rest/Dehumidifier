"""双转轮低露点除湿机系统级模型（v2 拓扑修正版）。

处理侧拓扑（左→右为气流方向）
==============================

    OA ──→ Fan1 ──→ CC1 ──→ ┬──→ [新风转轮冷吹区]  ─────────────────────────────────┐
                              └──→ [新风转轮处理区]  ──→ mix(RA) ──→ Fan2 ──→ ┬──→ [主转轮冷吹区]         │
                                                                                └──→ CC2 ──→ [主转轮处理区] ──→ Supply

再生侧拓扑（级联换热，右→左走再生路径）
========================================

    [主转轮冷吹区出] ──→ Heater_Low ──→ [主转轮再生区] ──→ mix ──→ Heater_OA ──→ [新风转轮再生区] ──→ 排放
                                                               ↑
                                              再生新风(RegenOA) + 新风转轮冷吹区出

说明
----
* 新风转轮（Rotor1）= 预除湿级；主转轮（Rotor2）= 深度除湿级。
* 主转轮冷吹侧出风流量 ≡ 主转轮再生侧流量（级联约束）。
* 新风转轮再生侧流量 = 主转轮再生出 + 再生新风 + 新风转轮冷吹出（三股汇合）。
* 表冷器（CC1/CC2）保留凝结分支（W_out = min(W_in, W_sat(T_out))）。
  理由：低露点工况处理侧含湿量通常高于设定出口温度对应的饱和含湿量，
  凝结去湿不可忽略，否则下游转轮入口 W 会被严重高估。
  `需求.md` 第 31 行的"仅温度变化"假设在低露点工况下不成立，此处作物理修正。

能耗汇总语义（单 KAN 运行时契约）
-----------------------
energy_breakdown 仅公开 4 项，单位均为 kW（稳态热负荷率，非累计能量）：

    cooling_kW     = |CC1| + |CC2|       # 表冷器焓差冷负荷
    heating_kW     = Heater_OA + Heater_Low  # 加热器焓差热负荷

关键边界与隐含假设：

1. 所有项均为稳态值，调用链中不存在时间乘法（无 dt、kWh）。
2. CC 项构造时为负（"+ = 热进入空气"约定），cooling_kW 已取绝对值。
   设备侧效率折算需另行提供实测设备参数。
3. 风机仍按配置创建并保持空气状态直通，但其铭牌功率不进入公开结果。
4. Heater/CoolingCoil 在设定温度已被自然满足时静默返回 Q_kW=0（被动达标
   无需做功）。再生侧级联换热下 Heater_OA_kW=0 属正常现象；调试时若发现
   某项意外为零，需检查上游温度是否已自然越过设定值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
import math
from typing import ClassVar, Dict

from .air_state import AirState
from .energy_term import EnergyTerm
from .components.cooling_coil import CoolingCoil
from .components.heater import Heater
from .components.fan import Fan
from .components.mixer import Mixer
from .components.rotor import Rotor, RotorInputs, RotorOutputs


# --------------------------------------------------------------------------- #
# 参数与设定值数据类                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class SystemParams:
    """系统固定参数（设备铭牌 / 物理参数，不随工况变化）。

    Attributes:
        fan1_P_kW:              一级风机额定电功率 (kW)（OA 入口侧）。
        fan2_P_kW:              二级风机额定电功率 (kW)（RA 混风后→主转轮前）。
        pressure_Pa:            系统总压 (Pa)，统一用于全部空气状态与元件湿空气计算。
        rotor1_thickness:       新风转轮厚度 (mm)。
        rotor1_diameter:        新风转轮直径 (mm)。
        rotor1_process_airspeed: 新风转轮处理风速 (m/s)。
        rotor1_cool_blow_airspeed: 新风转轮冷吹风速 (m/s)。
        rotor1_regen_angle:     新风转轮再生区扇形角 (°)。
        rotor1_cool_blow_angle: 新风转轮冷吹区扇形角 (°)。
        rotor2_thickness:       主转轮厚度 (mm)。
        rotor2_diameter:        主转轮直径 (mm)。
        rotor2_process_airspeed: 主转轮处理风速 (m/s)。
        rotor2_cool_blow_airspeed: 主转轮冷吹风速 (m/s)。
        rotor2_regen_angle:     主转轮再生区扇形角 (°)。
        rotor2_cool_blow_angle: 主转轮冷吹区扇形角 (°)。
        rotor1_cool_blow_mdot:  新风转轮冷吹区干空气流量 (kg/s)。
                                新风转轮处理区流量 = oa_state.mdot - rotor1_cool_blow_mdot。
        rotor2_cool_blow_mdot:  主转轮冷吹区干空气流量 (kg/s)。
                                同时也是主转轮再生区流量（级联约束）。
                                主转轮处理区（送风）流量 = Fan2出口流量 - rotor2_cool_blow_mdot。
    """
    fan1_P_kW: float = 1.5                  # kW
    fan2_P_kW: float = 1.0                  # kW
    pressure_Pa: float = 101325.0           # Pa
    rotor1_thickness: float = 300.0         # mm
    rotor1_diameter: float = 1070.0          # mm
    rotor1_process_airspeed: float = 1.6     # m/s
    rotor1_cool_blow_airspeed: float = 1.3   # m/s
    rotor1_regen_angle: float = 60.0        # °
    rotor1_cool_blow_angle: float = 30.0    # °
    rotor1_cool_blow_mdot: float = 0.3      # kg/s
    rotor2_thickness: float = 200.0         # mm
    rotor2_diameter: float = 1220.0          # mm
    rotor2_process_airspeed: float = 3.5     # m/s
    rotor2_cool_blow_airspeed: float = 3.3   # m/s
    rotor2_regen_angle: float = 60.0        # °
    rotor2_cool_blow_angle: float = 30.0    # °
    rotor2_cool_blow_mdot: float = 0.3      # kg/s


@dataclass
class ControlSetpoints:
    """操作员可调工况设定值。

    Attributes:
        cc1_T_outlet:   前表冷器（CC1）出口温度设定值 (℃)。
        cc2_T_outlet:   中表冷器（CC2）出口温度设定值 (℃)。
        heater_low_T:   低湿加热器出口温度设定值 (℃)，即主转轮再生侧入口温度。
        heater_oa_T:    新风加热器出口温度设定值 (℃)，即新风转轮再生侧入口温度。
        rotor1_rpm:     新风转轮转速 (rpm)。
        rotor2_rpm:     主转轮转速 (rpm)。
    """
    cc1_T_outlet: float = 20.0    # ℃
    cc2_T_outlet: float = 15.0    # ℃
    heater_low_T: float = 120.0   # ℃
    heater_oa_T: float = 120.0    # ℃
    rotor1_rpm: float = 5.0       # rpm
    rotor2_rpm: float = 6.0       # rpm（主转轮训练数据恒定值）


# --------------------------------------------------------------------------- #
# 仿真结果                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class SystemResult:
    """系统前向仿真结果（稳态）。

    输出只报告可由空气焓差确定的热负荷：

    * ``cooling_kW``     表冷器冷负荷之和 (kW)，正值表示从空气中带走的总热量
                         （显热+潜热）。
    * ``heating_kW``     加热器热负荷之和 (kW)，正值表示加热器供给空气的
                         显热。

    Attributes:
        supply_state:         送风口空气状态（T、W、Tdp、h、RH）。
        cooling_kW:           系统冷负荷 (kW)：|CC1| + |CC2|，正值为带走的总热量。
        heating_kW:           系统热负荷 (kW)：Heater_OA + Heater_Low，正值为输入的显热。
        energy_breakdown:     四个热工组件明细 {CC1/CC2/Heater_Low/Heater_OA: kW}。
        intermediate_states:  关键中间节点的空气状态 {节点名: AirState}。
        rotor1_outputs:       新风转轮 KAN 模型原始预测输出。
        rotor2_outputs:       主转轮 KAN 模型原始预测输出。
    """
    supply_state: AirState
    cooling_kW: float
    heating_kW: float
    energy_breakdown: Dict[str, float]
    intermediate_states: Dict[str, AirState]
    rotor1_outputs: RotorOutputs
    rotor2_outputs: RotorOutputs
    convergence: dict = field(default_factory=lambda: {
        "status": "not_evaluated", "converged": None, "iterations": 0,
        "residuals": {}, "tolerances": {},
    })

    # Public result contract deliberately exposes only thermal components.  A
    # caller may construct a result from an internal/debug mapping that still
    # contains fan or rotor-motor terms; normalize that boundary here so those
    # terms cannot leak into JSON, dashboards, or downstream records.
    _PUBLIC_THERMAL_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"CC1_kW", "CC2_kW", "Heater_Low_kW", "Heater_OA_kW"}
    )

    def __post_init__(self) -> None:
        self.energy_breakdown = {
            str(name): float(value)
            for name, value in self.energy_breakdown.items()
            if str(name) in self._PUBLIC_THERMAL_KEYS
        }

    def summary(self) -> str:
        """返回可打印的结果摘要字符串。"""
        lines = [
            "─" * 60,
            "  双转轮系统仿真结果（v2 拓扑修正版）",
            "─" * 60,
            f"  求解状态  : {convergence_label(self.convergence)}",
            f"  送风状态  : {self.supply_state}",
            f"  送风露点  : {self.supply_state.Tdp:.2f} ℃",
            "",
            "  热工负荷汇总:",
            f"    冷  量   : {self.cooling_kW:>10.3f} kW   (|CC1|+|CC2|)",
            f"    热  量   : {self.heating_kW:>10.3f} kW   (Heater_OA+Heater_Low)",
            "",
            "  热工组件明细 (kW):",
        ]
        for name, val in self.energy_breakdown.items():
            if name.startswith("CC"):
                unit_str = "kW (冷量)"
            elif name.startswith("Heater"):
                unit_str = "kW (热量)"
            else:
                unit_str = "kW"
            sign = "+" if val >= 0 else ""
            lines.append(f"    {name:<26s}: {sign}{val:.3f} {unit_str}")
        lines.append("")
        lines.append("  关键中间状态:")
        for node, state in self.intermediate_states.items():
            lines.append(f"    {node:<18s}: T={state.T:.1f}℃, W={state.W:.2f} g/kg, "
                         f"Tdp={state.Tdp:.1f}℃")
        lines.append("─" * 60)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 系统模型                                                                     #
# --------------------------------------------------------------------------- #

def convergence_label(diagnostics: dict) -> str:
    """实时展示和历史重绘共用求解状态文案。"""
    converged = diagnostics.get("converged")
    if converged is True:
        return "已收敛"
    if converged is False:
        return "未收敛（仅供调试）"
    return "未评估收敛性"


class DualRotorSystem:
    """双转轮低露点除湿机系统模型（v2 稳态前向仿真）。

    拓扑与 `需求.md` 完全对应：
    * 处理侧：OA → Fan1 → CC1 → split(新风转轮冷吹/处理) → 处理侧出+RA混风
              → Fan2 → split(主转轮冷吹/CC2→主转轮处理) → Supply
    * 再生侧：主转轮冷吹出 → Heater_Low → 主转轮再生区
              → (主转轮再生出 + 再生新风 + 新风转轮冷吹出) → Heater_OA → 新风转轮再生区 → 排放

    正式运行由两个 ``Rotor`` 实例分别承载 XF 单 KAN 和 main 单 KAN；历史
    对照代码仍可通过 ``rotor=`` 显式让两级共享同一个模型，但
    ``build_system()`` 不会走该兼容路径。

    Args:
        rotor: 兼容旧调用的共享转轮实例。
        rotor1: 新风转轮实例。
        rotor2: 主转轮实例。
    """

    def __init__(
        self,
        rotor: Rotor | None = None,
        *,
        rotor1: Rotor | None = None,
        rotor2: Rotor | None = None,
    ) -> None:
        if rotor is not None:
            if rotor1 is not None or rotor2 is not None:
                raise ValueError("rotor= 共享方式不能与 rotor1=/rotor2= 同时使用")
            rotor1 = rotor
            rotor2 = rotor
        if rotor1 is None or rotor2 is None:
            raise ValueError("必须提供 rotor=，或同时提供 rotor1= 与 rotor2=")
        self.rotor1 = rotor1
        self.rotor2 = rotor2
        # 仅为旧代码保留；双模型模式下明确为 None，避免误认为仍然共享。
        self.rotor = rotor

    def simulate(
        self,
        oa_state: AirState,
        ra_state: AirState,
        regen_oa_state: AirState,
        params: SystemParams,
        setpoints: ControlSetpoints,
        *,
        r1_regen_T_override: float | None = None,
        r1_regen_W_override: float | None = None,
        r2_regen_T_override: float | None = None,
    ) -> SystemResult:
        """前向仿真（v2 拓扑）。

        按系统拓扑串联各组件，计算送风状态和全局能耗。

        可选 override 用于联立求解 R1/R2 再生入口状态；仅当显式传入时
        覆盖对应设定值/默认值，不改变拓扑结构。

        Args:
            oa_state:       室外新风（OA）状态；mdot 是唯一的新风总流量来源。
            ra_state:       房间回风（RA）状态。mdot 由调用方设定（kg/s）。
            regen_oa_state: 再生侧补充新鲜空气状态。mdot 由调用方设定（kg/s）。
            params:         系统物理参数（流量分配、风速及兼容性的风机铭牌值等）。
            setpoints:      操作员设定值（表冷器出口温度、加热器温度、转速等）。

        Returns:
            SystemResult，包含送风状态、各组件能耗和关键中间状态。

        Raises:
            ValueError: 若流量约束不满足（冷吹流量 ≥ 上游总流量，或任意流量 ≤ 0）。
        """
        oa_state = oa_state.at_pressure(params.pressure_Pa)
        ra_state = ra_state.at_pressure(params.pressure_Pa)
        regen_oa_state = regen_oa_state.at_pressure(params.pressure_Pa)
        energy: Dict[str, float] = {}
        states: Dict[str, AirState] = {}

        # ---------------------------------------------------------------- #
        # 0. 流量级联约束预计算与校验                                        #
        # ---------------------------------------------------------------- #
        m_oa    = oa_state.mdot
        m_r1_cb = params.rotor1_cool_blow_mdot      # 新风转轮冷吹
        m_r1_p  = m_oa - m_r1_cb                    # 新风转轮处理

        m_ra    = ra_state.mdot                      # 房间回风
        m_fan2  = m_r1_p + m_ra                      # 二级风机出口

        m_r2_cb = params.rotor2_cool_blow_mdot       # 主转轮冷吹（= 主转轮再生，级联）
        m_r2_p  = m_fan2 - m_r2_cb                  # 主转轮处理 = 送风流量
        m_r2_rg = m_r2_cb                            # 主转轮再生（级联约束）

        m_regen_oa = regen_oa_state.mdot             # 再生新风
        m_r1_rg = m_r2_rg + m_regen_oa + m_r1_cb    # 新风转轮再生（三股汇合）

        if m_r1_cb >= m_oa:
            raise ValueError(
                f"新风转轮冷吹流量 ({m_r1_cb} kg/s) 必须 < 新风总流量 ({m_oa} kg/s)"
            )
        if m_r2_cb >= m_fan2:
            raise ValueError(
                f"主转轮冷吹流量 ({m_r2_cb} kg/s) 必须 < 二级风机出口流量 ({m_fan2:.4f} kg/s)"
            )
        for name, val in [
            ("m_r1_p", m_r1_p), ("m_r1_cb", m_r1_cb),
            ("m_r2_p", m_r2_p), ("m_r2_cb", m_r2_cb),
            ("m_r2_rg", m_r2_rg), ("m_r1_rg", m_r1_rg),
            ("m_fan2", m_fan2), ("m_regen_oa", m_regen_oa),
        ]:
            if val <= 0:
                raise ValueError(f"流量 {name}={val:.6f} kg/s 必须 > 0")

        # 记录输入状态；OA 自身 mdot 是唯一的新风总流量来源。
        oa = oa_state
        states["OA"]      = oa
        states["RA"]      = ra_state
        states["RegenOA"] = regen_oa_state

        # ================================================================ #
        # 处理侧                                                            #
        # ================================================================ #

        # ---------------------------------------------------------------- #
        # 1. 一级风机（Fan1）                                               #
        # ---------------------------------------------------------------- #
        fan1 = Fan(P_kW=params.fan1_P_kW, name="Fan1")
        # 风机配置仍用于组件兼容性校验，但铭牌功率不进入公开能耗结果。
        after_fan1, _fan1_e = fan1.process(oa)
        states["AfterFan1"] = after_fan1

        # ---------------------------------------------------------------- #
        # 2. 前表冷器（CC1）                                                #
        # ---------------------------------------------------------------- #
        cc1 = CoolingCoil(T_outlet_setpoint=setpoints.cc1_T_outlet, name="CC1")
        after_cc1, cc1_e = cc1.process(after_fan1)
        energy["CC1_kW"] = cc1_e.Q_kW
        states["AfterCC1"] = after_cc1

        # ---------------------------------------------------------------- #
        # 3. 新风转轮前分流                                                  #
        # ---------------------------------------------------------------- #
        s_r1_cb_in = after_cc1.at_mdot(m_r1_cb)   # → 新风转轮冷吹区
        s_r1_p_in  = after_cc1.at_mdot(m_r1_p)    # → 新风转轮处理区
        states["R1_CB_in"] = s_r1_cb_in
        states["R1_P_in"]  = s_r1_p_in

        # ---------------------------------------------------------------- #
        # 4. 新风转轮（Rotor1）                                             #
        # ---------------------------------------------------------------- #
        # XF 新风转轮需要再生入口风量(m³/h)与含湿量(g/kg)。这里用 OA 基准
        # 干空气密度把质量流量换算回体积流量；再生含湿量在单向 v2 拓扑中
        # 先以 RegenOA 含湿量近似，正式稳态求解应替换为混风后的实际值。
        _pressure_pa = params.pressure_Pa
        _humidity_ratio_oa = oa.W / 1000.0
        _dry_air_pressure_pa = _pressure_pa * 0.622 / (0.622 + _humidity_ratio_oa)
        _rho_da_oa = _dry_air_pressure_pa / (287.05 * (oa.T + 273.15))
        r1_regen_flow_m3h = m_r1_rg * 3600.0 / _rho_da_oa
        r1_inputs = RotorInputs(
            process_inlet_T=s_r1_p_in.T,
            process_inlet_W=s_r1_p_in.W,
            regen_T=(
                setpoints.heater_oa_T
                if r1_regen_T_override is None
                else r1_regen_T_override
            ),
            cool_blow_inlet_T=s_r1_cb_in.T,
            regen_angle=params.rotor1_regen_angle,
            cool_blow_angle=params.rotor1_cool_blow_angle,
            rpm=setpoints.rotor1_rpm,
            airspeed=params.rotor1_process_airspeed,
            rotor_thickness=params.rotor1_thickness,
            rotor_diameter=params.rotor1_diameter,
            cool_blow_airspeed=params.rotor1_cool_blow_airspeed,
            regen_flow_m3h=r1_regen_flow_m3h,
            regen_inlet_W=(
                regen_oa_state.W
                if r1_regen_W_override is None
                else r1_regen_W_override
            ),
            pressure_Pa=_pressure_pa,
        )
        r1_out: RotorOutputs = self.rotor1.process(r1_inputs)

        after_r1_p  = AirState(T=r1_out.process_outlet_T,   W=r1_out.process_outlet_W,  mdot=m_r1_p, pressure_Pa=params.pressure_Pa)
        after_r1_cb = AirState(T=r1_out.cool_blow_outlet_T, W=r1_out.cool_blow_outlet_W, mdot=m_r1_cb, pressure_Pa=params.pressure_Pa)
        states["AfterR1_P"]  = after_r1_p
        states["AfterR1_CB"] = after_r1_cb

        # ---------------------------------------------------------------- #
        # 5. 混风1：新风转轮处理出 + 房间回风（RA）→ 二级风机入口             #
        # ---------------------------------------------------------------- #
        mixer1 = Mixer(name="Mix1_RA")
        before_fan2 = mixer1.mix(after_r1_p, ra_state)
        states["BeforeFan2"] = before_fan2

        # ---------------------------------------------------------------- #
        # 6. 二级风机（Fan2）                                               #
        # ---------------------------------------------------------------- #
        fan2 = Fan(P_kW=params.fan2_P_kW, name="Fan2")
        after_fan2, _fan2_e = fan2.process(before_fan2)
        states["AfterFan2"] = after_fan2

        # ---------------------------------------------------------------- #
        # 7. 主转轮前分流                                                    #
        # ---------------------------------------------------------------- #
        s_r2_cb_in = after_fan2.at_mdot(m_r2_cb)   # → 主转轮冷吹区
        s_cc2_in   = after_fan2.at_mdot(m_r2_p)    # → 中表冷器
        states["R2_CB_in"] = s_r2_cb_in

        # ---------------------------------------------------------------- #
        # 8. 中表冷器（CC2）                                                #
        # ---------------------------------------------------------------- #
        cc2 = CoolingCoil(T_outlet_setpoint=setpoints.cc2_T_outlet, name="CC2")
        after_cc2, cc2_e = cc2.process(s_cc2_in)
        energy["CC2_kW"] = cc2_e.Q_kW
        states["AfterCC2"] = after_cc2

        # ---------------------------------------------------------------- #
        # 9. 主转轮（Rotor2）                                               #
        # ---------------------------------------------------------------- #
        r2_inputs = RotorInputs(
            process_inlet_T=after_cc2.T,
            process_inlet_W=after_cc2.W,
            regen_T=(
                setpoints.heater_low_T
                if r2_regen_T_override is None
                else r2_regen_T_override
            ),
            cool_blow_inlet_T=s_r2_cb_in.T,
            regen_angle=params.rotor2_regen_angle,
            cool_blow_angle=params.rotor2_cool_blow_angle,
            rpm=setpoints.rotor2_rpm,
            airspeed=params.rotor2_process_airspeed,
            rotor_thickness=params.rotor2_thickness,
            rotor_diameter=params.rotor2_diameter,
            cool_blow_airspeed=params.rotor2_cool_blow_airspeed,
            pressure_Pa=_pressure_pa,
        )
        r2_out: RotorOutputs = self.rotor2.process(r2_inputs)

        supply = AirState(T=r2_out.process_outlet_T, W=r2_out.process_outlet_W, mdot=m_r2_p, pressure_Pa=params.pressure_Pa)
        after_r2_cb = AirState(T=r2_out.cool_blow_outlet_T, W=r2_out.cool_blow_outlet_W, mdot=m_r2_cb, pressure_Pa=params.pressure_Pa)
        states["Supply"]      = supply
        states["AfterR2_CB"]  = after_r2_cb

        # ================================================================ #
        # 再生侧（级联换热）                                                 #
        # ================================================================ #

        # ---------------------------------------------------------------- #
        # 10. 低湿加热器（Heater_Low）：主转轮冷吹出 → 主转轮再生入           #
        # ---------------------------------------------------------------- #
        heater_low = Heater(T_outlet_setpoint=setpoints.heater_low_T, name="Heater_Low")
        after_heater_low, heater_low_e = heater_low.process(after_r2_cb)
        energy["Heater_Low_kW"] = heater_low_e.Q_kW
        states["AfterHeaterLow"] = after_heater_low

        # 主转轮再生侧出口（由 Rotor2 KAN 预测；regen_T 设定 = heater_low_T，自洽）
        after_r2_rg = AirState(T=r2_out.regen_outlet_T, W=r2_out.regen_outlet_W, mdot=m_r2_rg, pressure_Pa=params.pressure_Pa)
        states["AfterR2_RG"] = after_r2_rg

        # ---------------------------------------------------------------- #
        # 11. 三股混风：主转轮再生出 + 再生新风 + 新风转轮冷吹出              #
        # ---------------------------------------------------------------- #
        mixer2a = Mixer(name="Mix2a_RegenOA")
        mix2a = mixer2a.mix(after_r2_rg, regen_oa_state)   # 主转轮再生出 + 再生新风

        mixer2b = Mixer(name="Mix2b_R1CB")
        s_oa_heater_in = mixer2b.mix(mix2a, after_r1_cb)   # + 新风转轮冷吹出
        states["AfterMix3"] = s_oa_heater_in                # 三股混风后

        # ---------------------------------------------------------------- #
        # 12. 新风加热器（Heater_OA）：混风后空气 → 新风转轮再生入            #
        # ---------------------------------------------------------------- #
        heater_oa = Heater(T_outlet_setpoint=setpoints.heater_oa_T, name="Heater_OA")
        after_heater_oa, heater_oa_e = heater_oa.process(s_oa_heater_in)
        energy["Heater_OA_kW"] = heater_oa_e.Q_kW
        states["AfterHeaterOA"] = after_heater_oa

        # 新风转轮再生侧出口（由 Rotor1 KAN 预测；regen_T 设定 = heater_oa_T，自洽）
        after_r1_rg = AirState(T=r1_out.regen_outlet_T, W=r1_out.regen_outlet_W, mdot=m_r1_rg, pressure_Pa=params.pressure_Pa)
        states["AfterR1_RG"] = after_r1_rg
        # after_r1_rg → 排放（室外），不作进一步计算

        # ================================================================ #
        # 汇总热工负荷                                                       #
        # ================================================================ #
        # 冷量：表冷器构造时为负（散热出系统），取绝对值得到冷负荷量级
        cooling_kW = abs(energy["CC1_kW"]) + abs(energy["CC2_kW"])
        # 热量：加热器构造时为正（输入显热）
        heating_kW = energy["Heater_OA_kW"] + energy["Heater_Low_kW"]

        return SystemResult(
            supply_state=supply,
            cooling_kW=cooling_kW,
            heating_kW=heating_kW,
            energy_breakdown=energy,
            intermediate_states=states,
            rotor1_outputs=r1_out,
            rotor2_outputs=r2_out,
        )

    def simulate_coupled_xf(
        self,
        oa_state: AirState,
        ra_state: AirState,
        regen_oa_state: AirState,
        params: SystemParams,
        setpoints: ControlSetpoints,
        *,
        max_iter: int = 60,
        relax: float = 0.55,
        tol_T_degC: float = 1e-4,
        tol_W_g_per_kg: float = 1e-6,
    ) -> SystemResult:
        """联立求解 R1/R2 再生入口状态的 XF 稳态仿真。

        未知量:
            x0 = R1 再生入口实际温度 T
            x1 = R1 再生入口实际含湿量 W
            x2 = R2 再生入口实际温度 T

        残差:
            r0 = T_R1_regen - max(T_Mix3, heater_oa_setpoint)
            r1 = W_R1_regen - W_Mix3_gas
            r2 = T_R2_regen - max(T_R2_cold_blow_out, heater_low_setpoint)

        采用带松弛的固定点迭代；收敛要求温度误差 <= tol_T_degC、
        含湿量误差 <= tol_W_g_per_kg。
        """
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter 必须是正整数")
        if not math.isfinite(relax) or not 0 < relax <= 1:
            raise ValueError("relax 必须在 (0,1] 范围")
        if any(not math.isfinite(v) or v <= 0 for v in (tol_T_degC, tol_W_g_per_kg)):
            raise ValueError("收敛容差必须为正的有限值")
        r1_T = float(setpoints.heater_oa_T)
        r1_W = float(regen_oa_state.W)
        r2_T = float(setpoints.heater_low_T)
        result: SystemResult | None = None

        for iteration in range(1, max_iter + 1):
            result = self.simulate(
                oa_state,
                ra_state,
                regen_oa_state,
                params,
                setpoints,
                r1_regen_T_override=r1_T,
                r1_regen_W_override=r1_W,
                r2_regen_T_override=r2_T,
            )
            mix3 = result.intermediate_states["AfterMix3"]
            r2_cb = result.intermediate_states["AfterR2_CB"]
            new_r1_T = max(mix3.T, setpoints.heater_oa_T)
            new_r1_W = mix3.W
            new_r2_T = max(r2_cb.T, setpoints.heater_low_T)
            err_T1 = abs(new_r1_T - r1_T)
            err_W1 = abs(new_r1_W - r1_W)
            err_T2 = abs(new_r2_T - r2_T)
            if not all(math.isfinite(v) for v in (err_T1, err_W1, err_T2)):
                raise ValueError("联立求解残差包含非有限值")
            converged = err_T1 <= tol_T_degC and err_W1 <= tol_W_g_per_kg and err_T2 <= tol_T_degC
            result.convergence = {
                "status": "converged" if converged else "not_converged",
                "converged": converged, "iterations": iteration,
                "residuals": {"r1_T_degC": err_T1, "r1_W_g_per_kg": err_W1, "r2_T_degC": err_T2},
                "tolerances": {"T_degC": tol_T_degC, "W_g_per_kg": tol_W_g_per_kg},
                "last_evaluated_inputs": {"r1_T_degC": r1_T, "r1_W_g_per_kg": r1_W, "r2_T_degC": r2_T},
            }
            if converged:
                return result
            r1_T = r1_T + relax * (new_r1_T - r1_T)
            r1_W = r1_W + relax * (new_r1_W - r1_W)
            r2_T = r2_T + relax * (new_r2_T - r2_T)

        warnings.warn(
            "XF 联立稳态求解未收敛，返回最后迭代结果："
            f"residuals={result.convergence['residuals']}",
            RuntimeWarning,
            stacklevel=2,
        )
        return result
