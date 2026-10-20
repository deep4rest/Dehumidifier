"""双转轮稳态前向仿真的 schema v2 配置解析。

所有体积流量都按同一份配置的室外新风状态和系统压力标定。配置只保存
五个独立流量，其余支路流量由系统拓扑派生。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """配置不符合 schema v2 时抛出的中文错误。"""


@dataclass(frozen=True)
class AirStreamConfig:
    T_degC: float
    W_g_per_kg: float
    flow_m3h: float


@dataclass(frozen=True)
class RotorConfig:
    thickness_mm: float
    diameter_mm: float
    process_airspeed_m_s: float
    cool_blow_airspeed_m_s: float
    cool_blow_flow_m3h: float
    regen_angle_deg: float
    cool_blow_angle_deg: float


@dataclass(frozen=True)
class ControlConfig:
    cc1_T_outlet_degC: float
    cc2_T_outlet_degC: float
    heater_low_T_degC: float
    heater_oa_T_degC: float
    rotor1_rpm: float
    rotor2_rpm: float


@dataclass(frozen=True)
class SimulationConfig:
    schema_version: int
    pressure_Pa: float
    dew_point_basis: str
    flow_reference_basis: str
    outdoor_air: AirStreamConfig
    return_air: AirStreamConfig
    regen_outdoor_air: AirStreamConfig
    fan1_P_kW: float
    fan2_P_kW: float
    rotor1: RotorConfig
    rotor2: RotorConfig
    control_setpoints: ControlConfig

    @property
    def reference_dry_air_density_kg_m3(self) -> float:
        """该配置室外新风标定状态下的干空气密度。"""
        humidity_ratio = self.outdoor_air.W_g_per_kg / 1000.0
        dry_air_pressure = (
            self.pressure_Pa * 0.622 / (0.622 + humidity_ratio)
        )
        return dry_air_pressure / (
            287.05 * (self.outdoor_air.T_degC + 273.15)
        )

    def m3h_to_kg_da_s(self, flow_m3h: float) -> float:
        """使用唯一室外新风基准密度换算干空气质量流量。"""
        return float(flow_m3h) * self.reference_dry_air_density_kg_m3 / 3600.0

    @property
    def independent_flow_m3h(self) -> dict[str, float]:
        return {
            "outdoor_air": self.outdoor_air.flow_m3h,
            "return_air": self.return_air.flow_m3h,
            "regen_outdoor_air": self.regen_outdoor_air.flow_m3h,
            "rotor1_cold_blow": self.rotor1.cool_blow_flow_m3h,
            "rotor2_cold_blow": self.rotor2.cool_blow_flow_m3h,
        }

    @property
    def derived_flow_m3h(self) -> dict[str, float]:
        independent = self.independent_flow_m3h
        rotor1_process = (
            independent["outdoor_air"] - independent["rotor1_cold_blow"]
        )
        rotor2_regen = independent["rotor2_cold_blow"]
        rotor2_process = (
            rotor1_process
            + independent["return_air"]
            - independent["rotor2_cold_blow"]
        )
        # 企业风量规则（2026-09-17）：Heater_OA 入口目标 = R1 处理出口 / 3。
        # 可用量（R2 冷吹出 + R1 冷吹出）>= 目标时漏风至目标；否则按
        # regen_outdoor_air 补风（以配置流量为上限）。实际入口取
        # min(目标, 可用量 + 补风配置)。
        rotor1_regen = min(
            rotor1_process / 3.0,
            independent["rotor2_cold_blow"]
            + independent["rotor1_cold_blow"]
            + independent["regen_outdoor_air"],
        )
        return {
            "rotor1_process": rotor1_process,
            "rotor2_regen": rotor2_regen,
            "rotor2_process": rotor2_process,
            "rotor1_regen": rotor1_regen,
        }


_ROOT_KEYS = {
    "schema_version",
    "psychrometrics",
    "flow_reference",
    "outdoor_air",
    "return_air",
    "regen_outdoor_air",
    "system_params",
    "control_setpoints",
}
_AIR_KEYS = {"T_degC", "W_g_per_kg", "flow_m3h"}
_ROTOR_KEYS = {
    "thickness_mm",
    "diameter_mm",
    "process_airspeed_m_s",
    "cool_blow_airspeed_m_s",
    "cool_blow_flow_m3h",
    "regen_angle_deg",
    "cool_blow_angle_deg",
}

def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"配置字段 {path} 必须是映射")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], path: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        field = f"{path}.{missing[0]}" if path else missing[0]
        raise ConfigError(f"缺少配置字段：{field}")
    unknown = sorted(set(value) - required)
    if unknown:
        field = f"{path}.{unknown[0]}" if path else unknown[0]
        raise ConfigError(f"未知配置字段：{field}")


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"配置字段 {path} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"配置字段 {path} 必须是有限数值")
    if positive and result <= 0.0:
        raise ConfigError(f"配置字段 {path} 必须 > 0")
    return result


def _air_stream(
    root: Mapping[str, Any], key: str, *, allow_zero_flow: bool = False
) -> AirStreamConfig:
    section = _mapping(root[key], key)
    _exact_keys(section, _AIR_KEYS, key)
    temperature = _number(section["T_degC"], f"{key}.T_degC")
    if temperature <= -273.15:
        raise ConfigError(f"配置字段 {key}.T_degC 必须高于绝对零度")
    humidity = _number(section["W_g_per_kg"], f"{key}.W_g_per_kg")
    if humidity < 0.0:
        raise ConfigError(f"配置字段 {key}.W_g_per_kg 必须 >= 0")
    flow = _number(section["flow_m3h"], f"{key}.flow_m3h")
    invalid_flow = flow < 0.0 if allow_zero_flow else flow <= 0.0
    if invalid_flow:
        comparator = ">= 0" if allow_zero_flow else "> 0"
        raise ConfigError(f"配置字段 {key}.flow_m3h 必须 {comparator}")
    return AirStreamConfig(
        T_degC=temperature,
        W_g_per_kg=humidity,
        flow_m3h=flow,
    )


def _rotor(section: Mapping[str, Any], path: str) -> RotorConfig:
    _exact_keys(section, _ROTOR_KEYS, path)
    return RotorConfig(
        thickness_mm=_number(section["thickness_mm"], f"{path}.thickness_mm", positive=True),
        diameter_mm=_number(section["diameter_mm"], f"{path}.diameter_mm", positive=True),
        process_airspeed_m_s=_number(
            section["process_airspeed_m_s"],
            f"{path}.process_airspeed_m_s",
            positive=True,
        ),
        cool_blow_airspeed_m_s=_number(
            section["cool_blow_airspeed_m_s"],
            f"{path}.cool_blow_airspeed_m_s",
            positive=True,
        ),
        cool_blow_flow_m3h=_number(
            section["cool_blow_flow_m3h"],
            f"{path}.cool_blow_flow_m3h",
            positive=True,
        ),
        regen_angle_deg=_number(
            section["regen_angle_deg"], f"{path}.regen_angle_deg", positive=True
        ),
        cool_blow_angle_deg=_number(
            section["cool_blow_angle_deg"],
            f"{path}.cool_blow_angle_deg",
            positive=True,
        ),
    )


def parse_simulation_config(raw: Mapping[str, Any]) -> SimulationConfig:
    """精确解析一份 schema v2 前向仿真配置，不补隐藏工况默认值。"""
    root = _mapping(raw, "<root>")
    _exact_keys(root, _ROOT_KEYS, "")

    if root["schema_version"] != 2:
        raise ConfigError("配置字段 schema_version 必须等于 2")

    psychrometrics = _mapping(root["psychrometrics"], "psychrometrics")
    _exact_keys(
        psychrometrics, {"pressure_Pa", "dew_point_basis"}, "psychrometrics"
    )
    pressure = _number(
        psychrometrics["pressure_Pa"], "psychrometrics.pressure_Pa", positive=True
    )
    if psychrometrics["dew_point_basis"] != "water":
        raise ConfigError("配置字段 psychrometrics.dew_point_basis 必须为 water")

    flow_reference = _mapping(root["flow_reference"], "flow_reference")
    _exact_keys(flow_reference, {"basis"}, "flow_reference")
    if flow_reference["basis"] != "outdoor_air":
        raise ConfigError("配置字段 flow_reference.basis 必须为 outdoor_air")

    outdoor_air = _air_stream(root, "outdoor_air")
    return_air = _air_stream(root, "return_air")
    regen_outdoor_air = _air_stream(
        root, "regen_outdoor_air", allow_zero_flow=True
    )

    system = _mapping(root["system_params"], "system_params")
    _exact_keys(system, {"fan1_P_kW", "fan2_P_kW", "rotor1", "rotor2"}, "system_params")
    rotor1 = _rotor(_mapping(system["rotor1"], "system_params.rotor1"), "system_params.rotor1")
    rotor2 = _rotor(_mapping(system["rotor2"], "system_params.rotor2"), "system_params.rotor2")

    control = _mapping(root["control_setpoints"], "control_setpoints")
    control_keys = {
        "cc1_T_outlet_degC",
        "cc2_T_outlet_degC",
        "heater_low_T_degC",
        "heater_oa_T_degC",
        "rotor1_rpm",
        "rotor2_rpm",
    }
    _exact_keys(control, control_keys, "control_setpoints")
    control_setpoints = ControlConfig(
        **{
            key: _number(control[key], f"control_setpoints.{key}")
            for key in control_keys
        }
    )
    if control_setpoints.rotor1_rpm <= 0 or control_setpoints.rotor2_rpm <= 0:
        raise ConfigError("配置字段 control_setpoints.rotor1_rpm/rotor2_rpm 必须 > 0")

    parsed = SimulationConfig(
        schema_version=2,
        pressure_Pa=pressure,
        dew_point_basis="water",
        flow_reference_basis="outdoor_air",
        outdoor_air=outdoor_air,
        return_air=return_air,
        regen_outdoor_air=regen_outdoor_air,
        fan1_P_kW=_number(system["fan1_P_kW"], "system_params.fan1_P_kW", positive=True),
        fan2_P_kW=_number(system["fan2_P_kW"], "system_params.fan2_P_kW", positive=True),
        rotor1=rotor1,
        rotor2=rotor2,
        control_setpoints=control_setpoints,
    )

    if not math.isfinite(parsed.reference_dry_air_density_kg_m3) or parsed.reference_dry_air_density_kg_m3 <= 0:
        raise ConfigError("室外新风状态无法得到正的有限干空气密度")
    for name, flow in parsed.derived_flow_m3h.items():
        if flow <= 0.0:
            raise ConfigError(f"拓扑派生流量 {name}={flow:g} m³/h 必须 > 0")
    return parsed


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """从 YAML 文件加载并解析 schema v2 前向仿真配置。"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 安装依赖时才触发
        raise ConfigError("缺少 PyYAML，无法读取仿真配置") from exc

    config_path = Path(path)
    with open(config_path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return parse_simulation_config(raw)


def build_runtime_inputs(parsed: SimulationConfig) -> dict[str, Any]:
    """把已校验配置映射为系统内部对象；五股流量共用一个基准密度。"""
    from dual_rotor_sim import AirState, ControlSetpoints, SystemParams

    to_mass = parsed.m3h_to_kg_da_s
    r1 = parsed.rotor1
    r2 = parsed.rotor2
    control = parsed.control_setpoints
    return {
        "oa": AirState(
            T=parsed.outdoor_air.T_degC,
            W=parsed.outdoor_air.W_g_per_kg,
            mdot=to_mass(parsed.outdoor_air.flow_m3h),
            pressure_Pa=parsed.pressure_Pa,
        ),
        "ra": AirState(
            T=parsed.return_air.T_degC,
            W=parsed.return_air.W_g_per_kg,
            mdot=to_mass(parsed.return_air.flow_m3h),
            pressure_Pa=parsed.pressure_Pa,
        ),
        "regen_oa": AirState(
            T=parsed.regen_outdoor_air.T_degC,
            W=parsed.regen_outdoor_air.W_g_per_kg,
            mdot=to_mass(parsed.regen_outdoor_air.flow_m3h),
            pressure_Pa=parsed.pressure_Pa,
        ),
        "params": SystemParams(
            fan1_P_kW=parsed.fan1_P_kW,
            fan2_P_kW=parsed.fan2_P_kW,
            pressure_Pa=parsed.pressure_Pa,
            rotor1_thickness=r1.thickness_mm,
            rotor1_diameter=r1.diameter_mm,
            rotor1_process_airspeed=r1.process_airspeed_m_s,
            rotor1_cool_blow_airspeed=r1.cool_blow_airspeed_m_s,
            rotor1_regen_angle=r1.regen_angle_deg,
            rotor1_cool_blow_angle=r1.cool_blow_angle_deg,
            rotor1_cool_blow_mdot=to_mass(r1.cool_blow_flow_m3h),
            rotor2_thickness=r2.thickness_mm,
            rotor2_diameter=r2.diameter_mm,
            rotor2_process_airspeed=r2.process_airspeed_m_s,
            rotor2_cool_blow_airspeed=r2.cool_blow_airspeed_m_s,
            rotor2_regen_angle=r2.regen_angle_deg,
            rotor2_cool_blow_angle=r2.cool_blow_angle_deg,
            rotor2_cool_blow_mdot=to_mass(r2.cool_blow_flow_m3h),
        ),
        "setpoints": ControlSetpoints(
            cc1_T_outlet=control.cc1_T_outlet_degC,
            cc2_T_outlet=control.cc2_T_outlet_degC,
            heater_low_T=control.heater_low_T_degC,
            heater_oa_T=control.heater_oa_T_degC,
            rotor1_rpm=control.rotor1_rpm,
            rotor2_rpm=control.rotor2_rpm,
        ),
    }
