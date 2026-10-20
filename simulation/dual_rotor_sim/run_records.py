"""单工况仿真结果的独立留档。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .core.system.air_state import AirState
from .core.system.system import SystemResult

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "simulation"


_PUBLIC_ENERGY_KEYS = {
    "CC1_kW",
    "CC2_kW",
    "Heater_Low_kW",
    "Heater_OA_kW",
}


def _json_number(value: float) -> float | None:
    """把干燥极限露点等 ±inf 转成 JSON ``null``，不输出非标准常量。"""

    numeric = float(value)
    return numeric if numeric == numeric and abs(numeric) != float("inf") else None


def _air_state_to_dict(state: AirState) -> dict[str, float | None]:
    return {
        "pressure_Pa": state.pressure_Pa,
        "T_degC": _json_number(state.T),
        "W_g_per_kg": _json_number(state.W),
        "mdot_kg_da_s": _json_number(state.mdot),
        "Tdp_degC": _json_number(state.Tdp),
        "RH_fraction": _json_number(state.RH),
        "h_kJ_per_kg_da": _json_number(state.h),
    }


def result_to_dict(result: SystemResult) -> dict[str, Any]:
    """将完整系统结果转换为不损失展示精度的 JSON 数据。"""
    return {
        "convergence": result.convergence,
        "supply_state": _air_state_to_dict(result.supply_state),
        "energy_summary_kW": {
            "cooling_load": float(result.cooling_kW),
            "heating_load": float(result.heating_kW),
        },
        "energy_breakdown_kW": {
            str(name): float(value)
            for name, value in result.energy_breakdown.items()
            if name in _PUBLIC_ENERGY_KEYS
        },
        "intermediate_states": {
            str(name): _air_state_to_dict(state)
            for name, state in result.intermediate_states.items()
        },
        "rotor1_outputs": {
            key: _json_number(value)
            for key, value in asdict(result.rotor1_outputs).items()
        },
        "rotor2_outputs": {
            key: _json_number(value)
            for key, value in asdict(result.rotor2_outputs).items()
        },
        "regen_mix_rule": {
            str(key): (_json_number(value) if isinstance(value, (int, float)) else value)
            for key, value in result.regen_mix_rule.items()
        },
    }


def _allocate_run_dir(
    results_dir: Path,
    config_path: Path,
    simulated_at: datetime,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"{simulated_at:%Y%m%d_%H%M%S_%f}_{config_path.stem}"
    )
    candidate = results_dir / base_name
    sequence = 2
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = results_dir / f"{base_name}_{sequence:02d}"
            sequence += 1


def save_simulation_run(
    result: SystemResult,
    config: dict[str, Any],
    config_path: Path,
    results_dir: Path,
    *,
    simulated_at: datetime | None = None,
) -> Path:
    """把单次仿真的 JSON 与 dashboard 写入同一个新目录。"""
    from .dashboard import save_dashboard

    timestamp = simulated_at or datetime.now().astimezone()
    source = config_path.resolve()
    run_dir = _allocate_run_dir(results_dir, source, timestamp)
    try:
        document = {
            "run_id": run_dir.name,
            "simulated_at": timestamp.isoformat(),
            "config_source": str(source),
            "config": config,
            "result": result_to_dict(result),
        }
        json_path = run_dir / "result.json"
        json_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        save_dashboard(
            result=result,
            config=config,
            target=run_dir / "fig1_dashboard.png",
        )
    except Exception:
        # 仅清理由本次调用新建的目录，防止留下缺图或缺 JSON 的伪记录。
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir
