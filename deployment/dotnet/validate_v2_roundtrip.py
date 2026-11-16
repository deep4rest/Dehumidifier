"""Run the delivered search DLL, then replay its full BestConfig in a new forward process.

The default fixture is QY1070. Target error and replay error are reported separately.
Optional contrast runs show the effects of changing the case or bypassing the terminal.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CONTROL_FIELDS = {
    "Cc1OutletC": "cc1_T_outlet_degC", "Cc2OutletC": "cc2_T_outlet_degC",
    "HeaterOaC": "heater_oa_T_degC", "HeaterLowC": "heater_low_T_degC",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(verifier: Path, models: Path, config: Path, output: Path, *,
             dew_point: float, supply_temperature: float, maxiter: int = 40,
             contrast_config: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)

    def invoke(name: str, mode: str, case: Path, extra=()) -> dict:
        summary = output / f"{name}_summary.json"
        command = [str(verifier), "--mode", mode, "--models", str(models),
                   "--config", str(case), "--results-dir", str(output / name),
                   "--summary-output", str(summary), *extra]
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=1800)
        (output / f"{name}_stdout.log").write_text(process.stdout, encoding="utf-8")
        (output / f"{name}_stderr.log").write_text(process.stderr, encoding="utf-8")
        write(output / f"{name}_command.json", {"command": command, "exit_code": process.returncode})
        if process.returncode != 0:
            raise RuntimeError(f"{name} exited {process.returncode}: {process.stderr}")
        return read(summary)

    search = invoke("search", "search", config, ["--dew-point-target", str(dew_point),
                    "--supply-temp-target", str(supply_temperature), "--maxiter", str(maxiter)])
    report = read(Path(search["result_json"]))
    best_path = Path(search["result_json"]).parent / "best_config.json"
    best = read(best_path)
    expected = read(config)
    expected["control_setpoints"].update({field: search["best_controls"][name] for name, field in CONTROL_FIELDS.items()})
    expected["control_setpoints"]["supply_T_target_degC"] = supply_temperature
    if best != expected:
        raise AssertionError("BestConfig changed fields other than the four controls and terminal target")
    forward = invoke("replay", "forward", best_path)["prediction"]
    assessment = search["best_assessment"]
    differences = {
        "temperature_degC": abs(forward["supply_T_degC"] - assessment["SupplyTemperatureC"]),
        "dew_point_degC": abs(forward["supply_Tdp_degC"] - assessment["SupplyDewPointC"]),
        "humidity_g_per_kg": abs(forward["supply_W_g_per_kg"] - assessment["SupplyHumidityGPerKg"]),
        "thermal_load_kW": abs(forward["thermal_load_sum_kW"] - assessment["ThermalLoadSumKW"]),
        **{name: abs(value - assessment["EnergyBreakdownKW"][name])
           for name, value in forward["energy_breakdown_kW"].items()},
    }
    search_record = read(Path(report["BestForwardJsonPath"]))["result"]
    forward_record = read(Path(forward["result_json"]))["result"]
    # Same DLL/core/models/config/solver: every recorded physical field and
    # convergence history must agree, not just the total objective.
    exact_record_match = search_record == forward_record
    passed = (search["best_independently_verified"] and forward["converged"]
              and exact_record_match and all(value <= 1e-9 for value in differences.values())
              and abs(forward["supply_Tdp_degC"] - dew_point) <= report["Settings"]["DewPointToleranceC"])
    contrasts = {}
    if contrast_config is not None:
        altered = read(contrast_config)
        altered["control_setpoints"].update(best["control_setpoints"])
        case = output / "different_case_same_controls.json"
        write(case, altered)
        contrasts["different_case_same_controls"] = invoke("different_case", "forward", case)["prediction"]
        for name, target in (("terminal_bypassed", None), ("terminal_condensing", -55.0)):
            altered = copy.deepcopy(best)
            altered["control_setpoints"]["supply_T_target_degC"] = target
            case = output / f"{name}.json"
            write(case, altered)
            prediction = invoke(name, "forward", case)["prediction"]
            record = read(Path(prediction["result_json"]))["result"]
            upstream = {key: value for key, value in record["intermediate_states"].items()
                        if key not in ("AfterCC3", "Supply")}
            original_upstream = {key: value for key, value in search_record["intermediate_states"].items()
                                 if key not in ("AfterCC3", "Supply")}
            same_upstream = (upstream == original_upstream
                             and record["rotor1_outputs"] == search_record["rotor1_outputs"]
                             and record["rotor2_outputs"] == search_record["rotor2_outputs"]
                             and record["convergence"] == search_record["convergence"]
                             and record["regen_mix_rule"] == search_record["regen_mix_rule"]
                             and all(record["energy_breakdown_kW"][key] == value
                                     for key, value in search_record["energy_breakdown_kW"].items()
                                     if key not in ("CC3_kW", "Heater_Post_kW")))
            contrasts[name] = {"prediction": prediction, "upstream_exact_match": same_upstream}
            passed = passed and same_upstream
    result = {"passed": passed, "config": str(config), "config_sha256": sha(config),
              "model_sha256": {p.name: sha(p) for p in sorted(models.iterdir()) if p.is_file()},
              "assembly_sha256": {name: sha(verifier.parent / name) for name in (
                  "露点与能耗预测模型v2.dll", "最优能耗工况搜索模型v2.dll", "Dehumidifier.Simulation.dll")},
              "target_dew_point_degC": dew_point, "target_temperature_degC": supply_temperature,
              "best_config": str(best_path), "best_controls": search["best_controls"],
              "search": assessment, "forward": forward,
              "target_error_degC": forward["supply_Tdp_degC"] - dew_point,
              "search_to_forward_absolute_errors": differences,
              "full_physical_record_exact_match": exact_record_match, "contrasts": contrasts}
    write(output / "roundtrip_validation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "deployment/dotnet/examples/qy1070_case.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dew-point-target", type=float, required=True)
    parser.add_argument("--supply-temp-target", type=float, required=True)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--contrast-config", type=Path)
    args = parser.parse_args()
    result = validate(args.verifier.resolve(), args.models.resolve(), args.config.resolve(), args.output.resolve(),
                      dew_point=args.dew_point_target, supply_temperature=args.supply_temp_target,
                      maxiter=args.maxiter, contrast_config=args.contrast_config.resolve() if args.contrast_config else None)
    print(json.dumps({key: result[key] for key in ("passed", "best_controls", "target_error_degC",
          "search_to_forward_absolute_errors", "full_physical_record_exact_match")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
