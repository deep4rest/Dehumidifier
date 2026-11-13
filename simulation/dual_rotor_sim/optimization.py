"""Bounded control search with automatic terminal temperature conditioning.

The physical energy term is the sum of six thermal loads, not electricity.
The optimization loss adds the user's piecewise absolute dew-point distance.
Training-domain warnings remain diagnostic; convergence and output-quality
constraints decide whether a trial can be selected.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import product
import copy
import math
from typing import Callable
import warnings

import numpy as np

from .config import build_runtime_inputs, parse_simulation_config
from .core.system import psychrometrics as psy
from .core.system.solver import build_system, forward_simulate

CONTROL_NAMES = (
    "cc1_T_outlet_degC", "cc2_T_outlet_degC",
    "heater_oa_T_degC", "heater_low_T_degC",
)
DEFAULT_BOUNDS = ((9., 13.), (13., 17.), (50., 140.), (50., 140.))
COOLING_LOAD_KEYS = ("CC1_kW", "CC2_kW", "CC3_kW")
HEATING_LOAD_KEYS = ("Heater_Low_kW", "Heater_OA_kW", "Heater_Post_kW")
SUPPLY_T_TARGET_TOLERANCE_DEGC = 1e-4
DEW_POINT_DISTANCE_REFERENCE_DEGC = 1.0


def dew_point_region_weight(error_degC: float) -> float | None:
    """User zones; the shared -.5 and +.3 boundaries belong to the middle."""
    if not math.isfinite(error_degC) or not -1. <= error_degC <= 1.:
        return None
    if error_degC < -.5:
        return .35
    if error_degC <= .3:
        return .1
    return .5


def loss_contract(energy_reference_kW: float, coefficient: float) -> dict:
    return {
        "formula":"J = Q_sum / Q_ref + coefficient * region_weight(error) * abs(error) / (1 degC)",
        "error_definition":"supply_dew_point_degC - target_dew_point_degC",
        "distance":"absolute",
        "energy_reference_kW":energy_reference_kW,
        "dew_point_distance_reference_degC":DEW_POINT_DISTANCE_REFERENCE_DEGC,
        "dew_point_penalty_coefficient":coefficient,
        "normalization":"fixed energy reference, shared by baseline and every candidate",
        "physical_thermal_load_units":"kW",
        "optimization_loss_units":"dimensionless; not a physical thermal load or electricity",
        "zones":[
            {"lower_degC":-1.,"upper_degC":-.5,"lower_inclusive":True,"upper_inclusive":False,"weight":.35},
            {"lower_degC":-.5,"upper_degC":.3,"lower_inclusive":True,"upper_inclusive":True,"weight":.1},
            {"lower_degC":.3,"upper_degC":1.,"lower_inclusive":False,"upper_inclusive":True,"weight":.5},
        ],
        "cooling_load_keys":list(COOLING_LOAD_KEYS),
        "heating_load_keys":list(HEATING_LOAD_KEYS),
    }


@dataclass(frozen=True)
class SearchSettings:
    dew_point_target_degC: float
    supply_T_target_degC: float
    supply_T_min_degC: float | None = None
    supply_T_max_degC: float | None = None
    dew_point_tolerance_degC: float = 1.
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS
    step_degC: float = .1
    cooling_weight: float = 1.
    heating_weight: float = 1.
    seeds: tuple[int, ...] = (20261110, 20261111, 20261112)
    maxiter: int = 40
    popsize: int = 8
    coarse_grid: bool = True
    initial_controls: tuple[float, ...] | None = None
    energy_reference_kW: float | None = None
    dew_point_penalty_coefficient: float = .1

    def __post_init__(self):
        numeric = [self.dew_point_target_degC, self.supply_T_target_degC, self.dew_point_tolerance_degC, self.supply_T_min_degC,
                   self.supply_T_max_degC, self.step_degC,
                   self.cooling_weight, self.heating_weight, self.energy_reference_kW,
                   self.dew_point_penalty_coefficient]
        if not all(math.isfinite(v) for v in numeric if v is not None):
            raise ValueError("优化目标、温度及权重必须为有限值")
        if self.supply_T_min_degC is not None and self.supply_T_max_degC is not None and self.supply_T_min_degC > self.supply_T_max_degC:
            raise ValueError("送风温度下限不能超过上限")
        if self.supply_T_min_degC is not None and self.supply_T_target_degC < self.supply_T_min_degC:
            raise ValueError("送风温度目标不能低于附加下限")
        if self.supply_T_max_degC is not None and self.supply_T_target_degC > self.supply_T_max_degC:
            raise ValueError("送风温度目标不能高于附加上限")
        if not 0 < self.dew_point_tolerance_degC <= 1.:
            raise ValueError("露点误差容限必须大于0且不超过1℃")
        if self.dew_point_target_degC-self.dew_point_tolerance_degC <= -273.15 or self.supply_T_target_degC <= -273.15 or (self.supply_T_min_degC is not None and self.supply_T_min_degC <= -273.15) or (self.supply_T_max_degC is not None and self.supply_T_max_degC <= -273.15):
            raise ValueError("目标温度必须高于绝对零度")
        if self.step_degC <= 0 or min(self.cooling_weight, self.heating_weight) <= 0:
            raise ValueError("搜索步长和冷热负荷权重必须为正数")
        if self.cooling_weight != 1. or self.heating_weight != 1.:
            raise ValueError("当前优化能耗定义为六项冷热负荷之和，冷热负荷权重必须均为1")
        if self.energy_reference_kW is not None and self.energy_reference_kW <= 0.:
            raise ValueError("能耗归一化基准必须是正的有限kW数值")
        if self.dew_point_penalty_coefficient <= 0.:
            raise ValueError("露点距离惩罚系数必须为正的有限数值")
        if len(self.bounds) != len(CONTROL_NAMES):
            raise ValueError("仅支持CC1、CC2、R1再生、R2再生四个温度变量")
        for lo, hi in self.bounds:
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi or lo <= -273.15:
                raise ValueError("每个控制边界必须有限且下限小于上限")
            count = (hi-lo)/self.step_degC
            if abs(count-round(count)) > 1e-7:
                raise ValueError("控制区间须为搜索步长的整数倍")
        if not self.seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("至少提供一个非负整数随机种子")
        if self.maxiter < 0 or self.popsize < 2:
            raise ValueError("maxiter不能为负，popsize至少为2")
        if self.initial_controls is not None:
            if len(self.initial_controls) != len(CONTROL_NAMES) or any(not math.isfinite(v) or not lo <= v <= hi for v,(lo,hi) in zip(self.initial_controls,self.bounds)):
                raise ValueError("初始控制参数须为边界内的四个有限温度")


def assess_result(result, settings: SearchSettings, *, energy_reference_kW: float | None = None) -> dict:
    """Evaluate actual saved states, using unclipped RH and original convergence."""
    reference = energy_reference_kW if energy_reference_kW is not None else settings.energy_reference_kW
    if reference is not None and (not math.isfinite(reference) or reference <= 0.):
        raise ValueError("能耗归一化基准必须是正的有限kW数值")
    reasons = []; violation = 0.
    if result.convergence.get("converged") is not True:
        reasons.append("unconverged"); violation += 1.
    states = result.intermediate_states
    max_rh = 0.; bad_states = []; supersaturated = []
    for name, state in states.items():
        values = (state.T, state.W, state.mdot, state.pressure_Pa)
        if not all(math.isfinite(v) for v in values) or state.T <= -273.15 or state.W < 0 or state.mdot < 0 or state.pressure_Pa <= 0:
            bad_states.append(name); continue
        rh = state.pressure_Pa * state.W / (621.98 + state.W) / float(psy.saturation_pressure_pa(state.T))
        if not math.isfinite(rh):
            bad_states.append(name); continue
        max_rh = max(max_rh, rh)
        if rh > 1. + 1e-6:
            supersaturated.append(name)
    if bad_states:
        reasons.append("invalid_state"); violation += len(bad_states)
    if supersaturated:
        reasons.append("supersaturated"); violation += min(max_rh-1., 1e6)
    supply = result.supply_state
    T, W, dp = supply.T, supply.W, supply.Tdp
    valid_dp = math.isfinite(dp) or (dp == -math.inf and W == 0.)
    if not math.isfinite(T) or not math.isfinite(W) or not valid_dp:
        reasons.append("invalid_supply"); violation += 1.
    else:
        lower = settings.dew_point_target_degC-settings.dew_point_tolerance_degC
        upper = settings.dew_point_target_degC+settings.dew_point_tolerance_degC
        if dp < lower:
            reasons.append("dew_point_below_band")
            # W=0 and dp=-inf are legal forward states but cannot satisfy a
            # finite target band. Keep their penalty finite and JSON-safe.
            violation += (lower-dp)/10. if math.isfinite(dp) else 1e6
        elif dp > upper:
            reasons.append("dew_point_above_band"); violation += (dp-upper)/10.
        temperature_error = T-settings.supply_T_target_degC
        if abs(temperature_error) > SUPPLY_T_TARGET_TOLERANCE_DEGC:
            reasons.append("supply_temperature_target")
            violation += abs(temperature_error)/10.
        below = settings.supply_T_min_degC is not None and T < settings.supply_T_min_degC
        above = settings.supply_T_max_degC is not None and T > settings.supply_T_max_degC
        if below or above:
            reasons.append("supply_temperature")
            violation += ((settings.supply_T_min_degC-T) if below else (T-settings.supply_T_max_degC))/10.
    cooling, heating = result.cooling_kW, result.heating_kW
    loads_valid = all(math.isfinite(v) and v >= 0 for v in (cooling, heating))
    energy = getattr(result,"energy_breakdown",{})
    keys = set(COOLING_LOAD_KEYS+HEATING_LOAD_KEYS)
    load_contract_valid = set(energy)==keys and all(math.isfinite(energy[k]) for k in keys)
    if load_contract_valid:
        load_contract_valid = all(energy[k]<=0 for k in COOLING_LOAD_KEYS) and all(energy[k]>=0 for k in HEATING_LOAD_KEYS)
        expected_cooling = sum(-energy[k] for k in COOLING_LOAD_KEYS)
        expected_heating = sum(energy[k] for k in HEATING_LOAD_KEYS)
        load_contract_valid = load_contract_valid and math.isclose(cooling,expected_cooling,rel_tol=0.,abs_tol=1e-9) and math.isclose(heating,expected_heating,rel_tol=0.,abs_tol=1e-9)
    if not load_contract_valid:
        reasons.append("invalid_load_contract"); violation += 1.
        loads_valid = False
    if not loads_valid:
        reasons.append("invalid_load"); violation += 1.
    objective = cooling+heating if loads_valid else None
    if objective is not None and not math.isfinite(objective):
        reasons.append("invalid_load"); violation += 1.; objective = None
    dew_point_error = dp-settings.dew_point_target_degC if math.isfinite(dp) else None
    region_weight = dew_point_region_weight(dew_point_error) if dew_point_error is not None else None
    normalized_energy = objective/reference if objective is not None and reference is not None else None
    normalized_distance = abs(dew_point_error)/DEW_POINT_DISTANCE_REFERENCE_DEGC if dew_point_error is not None else None
    weighted_distance = region_weight*normalized_distance if region_weight is not None else None
    penalty = settings.dew_point_penalty_coefficient*weighted_distance if weighted_distance is not None else None
    optimization_loss = normalized_energy+penalty if normalized_energy is not None and penalty is not None else None
    if normalized_energy is not None and not math.isfinite(normalized_energy):
        reasons.append("invalid_objective"); violation += 1.
        normalized_energy = None; optimization_loss = None
    elif optimization_loss is not None and not math.isfinite(optimization_loss):
        reasons.append("invalid_objective"); violation += 1.; optimization_loss = None
    feasible = not reasons
    # All feasible scores are < 1; all infeasible scores are >= 1. This avoids
    # selecting a failed or wet trial merely because its load is very small.
    score = optimization_loss/(1.+optimization_loss) if feasible and optimization_loss is not None else (None if feasible else 1.+min(violation,1e6))
    return {"feasible":feasible, "reasons":reasons, "score":score,
            "objective_weighted_thermal_kW":objective,
            "thermal_load_sum_kW":objective,
            "energy_breakdown_kW":dict(energy) if load_contract_valid else None,
            "dew_point_region_weight":region_weight,
            "dew_point_distance_penalty":penalty,
            "energy_reference_kW":reference,
            "normalized_energy":normalized_energy,
            "normalized_absolute_dewpoint_error":normalized_distance,
            "weighted_normalized_dewpoint_distance":weighted_distance,
            "dew_point_penalty_coefficient":settings.dew_point_penalty_coefficient,
            "objective_calibrated":reference is not None,
            "optimization_loss":optimization_loss,
            "cooling_kW":cooling if math.isfinite(cooling) else None,
            "heating_kW":heating if math.isfinite(heating) else None,
            "supply_T_degC":T if math.isfinite(T) else None,
            "supply_T_target_degC":settings.supply_T_target_degC,
            "supply_temperature_error_degC":T-settings.supply_T_target_degC if math.isfinite(T) else None,
            "supply_temperature_tolerance_degC":SUPPLY_T_TARGET_TOLERANCE_DEGC,
            "supply_W_g_per_kg":W if math.isfinite(W) else None,
            "supply_Tdp_degC":dp if math.isfinite(dp) else None,
            "dew_point_error_degC":dew_point_error,
            "dew_point_band_degC":[settings.dew_point_target_degC-settings.dew_point_tolerance_degC,
                                    settings.dew_point_target_degC+settings.dew_point_tolerance_degC],
            "dry_limit_dew_point":dp == -math.inf and W == 0.,
            "max_unclipped_RH":max_rh, "supersaturated_nodes":supersaturated,
            "invalid_state_nodes":bad_states,
            "converged":result.convergence.get("converged"),
            "iterations":result.convergence.get("iterations"),
            "residuals":result.convergence.get("residuals")}


def optimize_controls(config: dict, settings: SearchSettings, *, system=None,
                      evaluation_callback: Callable[[dict], None] | None = None) -> dict:
    """Return the lowest feasible candidate found; never claims a global proof.

    All flows, inlet states, geometry and both rotation fields stay fixed.
    A missing model or incompatible package fails before search begins.
    """
    from scipy.optimize import differential_evolution

    original = copy.deepcopy(config)
    runtime = build_runtime_inputs(parse_simulation_config(original))
    runtime["setpoints"] = replace(runtime["setpoints"],supply_T_target=settings.supply_T_target_degC)
    if system is None:
        system = build_system()
    cache = {}; records = []; best = None; phase = "baseline"
    fixed_energy_reference = settings.energy_reference_kW

    def canonical(values):
        result = []
        for value, (lo, hi) in zip(values, settings.bounds):
            index = round((float(value)-lo)/settings.step_degC)
            result.append(round(min(hi, max(lo, lo+index*settings.step_degC)), 10))
        return tuple(result)

    def evaluate(values, *, quantize=True):
        nonlocal best, fixed_energy_reference
        vector = canonical(values) if quantize else tuple(float(v) for v in values)
        if vector in cache:
            return cache[vector]
        control = dict(zip(CONTROL_NAMES, vector))
        # YAML names carry unit suffixes; runtime dataclass names do not.
        setpoints = replace(runtime["setpoints"], **{k.removesuffix("_degC"):v for k,v in control.items()})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = forward_simulate(system, runtime["oa"], runtime["ra"],
                                          runtime["regen_oa"], runtime["params"], setpoints)
                if fixed_energy_reference is None:
                    if phase != "baseline":
                        raise RuntimeError("自动能耗归一化基准只能来自首次baseline计算")
                    physical = assess_result(result,settings)
                    value = physical["thermal_load_sum_kW"]
                    if value is None or not math.isfinite(value) or value <= 0.:
                        raise RuntimeError("原配置在送风温度目标下的六项总负荷不能作为正归一化基准；请显式提供energy_reference_kW")
                    fixed_energy_reference = value
                assessment = assess_result(result, settings,energy_reference_kW=fixed_energy_reference)
            except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError) as exc:
                if fixed_energy_reference is None:
                    raise RuntimeError("首次baseline计算发生数值异常，不能自动确定能耗基准；请显式提供energy_reference_kW") from exc
                assessment = {"feasible":False, "reasons":["numerical_error"], "score":1e6,
                              "objective_weighted_thermal_kW":None,"thermal_load_sum_kW":None,"optimization_loss":None,
                              "energy_reference_kW":fixed_energy_reference,"normalized_energy":None,
                              "dew_point_distance_penalty":None,"dew_point_region_weight":None,
                              "error":f"{type(exc).__name__}: {exc}"}
        record = {"evaluation":len(records)+1, "phase":phase, "controls":control,
                  **assessment, "warnings":list(dict.fromkeys(str(w.message) for w in caught))}
        cache[vector] = record; records.append(record)
        on_grid = all(abs(a-b)<1e-9 for a,b in zip(canonical(vector), vector))
        in_bounds = all(lo <= v <= hi for v,(lo,hi) in zip(vector,settings.bounds))
        record["eligible_for_selection"] = on_grid and in_bounds
        if record["feasible"] and on_grid and in_bounds:
            if best is None or record["optimization_loss"] < best["optimization_loss"]:
                best = record
        if evaluation_callback:
            evaluation_callback(record)
        return record

    base_vector = tuple(getattr(runtime["setpoints"], k.removesuffix("_degC")) for k in CONTROL_NAMES)
    baseline = evaluate(base_vector, quantize=False)
    if settings.initial_controls is not None:
        phase = "initial_controls"
        evaluate(settings.initial_controls)
    if settings.coarse_grid:
        phase = "coarse_grid"
        for point in product(*[(lo, (lo+hi)/2, hi) for lo,hi in settings.bounds]):
            evaluate(point)
    seed_runs = []
    for seed in settings.seeds:
        phase = f"differential_evolution_seed_{seed}"
        x0 = list(best["controls"].values()) if best else canonical(base_vector)
        solved = differential_evolution(lambda x:evaluate(x)["score"], settings.bounds,
                    maxiter=settings.maxiter, popsize=settings.popsize, seed=seed,
                    x0=x0, polish=False, workers=1, updating="immediate", tol=0., atol=0.)
        seed_runs.append({"seed":seed, "optimizer_success":bool(solved.success),
                          "message":str(solved.message), "generations":int(solved.nit),
                          "objective_calls_including_cached":int(solved.nfev),
                          "best_feasible_thermal_kW_after_seed":None if best is None else best["thermal_load_sum_kW"],
                          "best_feasible_optimization_loss_after_seed":None if best is None else best["optimization_loss"]})
    # Discrete coordinate refinement checks actual realizable settings after DE.
    # Accepted points are re-evaluated on the 0.1 C (configurable) search lattice.
    if best is not None:
        phase = "coordinate_refinement"
        for step in (1., .5, settings.step_degC):
            for _ in range(30):
                previous = best["optimization_loss"]
                center = tuple(best["controls"].values())
                for i in range(len(CONTROL_NAMES)):
                    for direction in (-1., 1.):
                        candidate = list(center); candidate[i] += direction*step
                        evaluate(candidate)
                if best["optimization_loss"] >= previous:
                    break
    best_config = None
    if best:
        best_config = copy.deepcopy(original)
        best_config["control_setpoints"].update(best["controls"])
        best_config["control_setpoints"]["supply_T_target_degC"] = settings.supply_T_target_degC
    assert config == original, "优化不得修改调用方原工况"
    minimum_thermal = min((r for r in records if r["feasible"] and r["eligible_for_selection"]),
                          key=lambda r:r["thermal_load_sum_kW"],default=None)
    return {"schema_version":3, "dew_point_constraint":"absolute_error_to_target",
            "supply_temperature_constraint":"absolute_error_to_target_1e-4_degC",
            "status":"feasible_candidate_found" if best else "no_feasible_candidate_found",
            "objective":"normalized_six_thermal_load_sum_plus_piecewise_dewpoint_absolute_distance",
            "energy_reference_kW":fixed_energy_reference,
            "energy_reference_source":"explicit" if settings.energy_reference_kW is not None else "original_configuration_at_supply_temperature_target",
            "loss_contract":loss_contract(fixed_energy_reference,settings.dew_point_penalty_coefficient),
            "actual_electricity_optimized":False, "global_minimum_proven":False,
            "settings":asdict(settings), "control_names":list(CONTROL_NAMES),
            "baseline":baseline, "best":best, "best_config":best_config,
            "best_selection_criterion":"minimum normalized composite optimization_loss among eligible feasible candidates",
            "minimum_thermal_load_candidate":minimum_thermal,
            "minimum_thermal_load_candidate_scope":"eligible feasible candidates within the configured control bounds and lattice",
            "unique_forward_evaluations":len(records),
            "feasible_evaluations":sum(r["feasible"] for r in records),
            "rejections_by_reason":dict(Counter(reason for r in records for reason in r["reasons"])),
            "seed_runs":seed_runs, "evaluations":records,
            "model_coverage_policy":"diagnostic only; no main model retraining or domain restriction"}
