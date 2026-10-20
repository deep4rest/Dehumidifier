#!/usr/bin/env python3
"""Compare C# result records with the Python reference records.

This checker deliberately compares every saved node, both rotor output records,
the four thermal terms, totals, and convergence histories.  It does not average
away a bad node and it never changes a tolerance to make a run pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "iteration_count": 0,
    "temperature": 1e-3,
    "humidity": 1e-5,
    "dew_point": 1e-2,
    "flow_pressure": 1e-10,
    "enthalpy": 1e-2,
    "relative_humidity": 1e-4,
    "load": 1e-2,
    "convergence_temperature": 1e-3,
    "convergence_humidity": 1e-5,
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _category(path: str) -> str:
    if path == "convergence.iterations" or path.endswith(".iteration"):
        return "iteration_count"
    if path.endswith("Tdp_degC") or path.endswith("process_outlet_Tdp"):
        return "dew_point"
    if path.endswith("mdot_kg_da_s") or path.endswith("pressure_Pa"):
        return "flow_pressure"
    if path.startswith("regen_mix_rule.") and (path.endswith("_kg_da_s") or path.endswith("_m3h")):
        return "flow_pressure"
    if path.endswith("RH_fraction"):
        return "relative_humidity"
    if path.endswith("h_kJ_per_kg_da"):
        return "enthalpy"
    if "energy_" in path or path.endswith("_kW") or "loads." in path:
        return "load"
    if path.endswith("W_g_per_kg") or path.endswith("_W"):
        return "humidity"
    if path.endswith("T_degC") or path.endswith("_T"):
        return "temperature"
    if ".inputs." in path or ".residuals." in path:
        return "convergence_humidity" if "W" in path else "convergence_temperature"
    return "temperature"


def _record_diff(
    py_value: Any,
    cs_value: Any,
    path: str,
    diffs: list[dict[str, Any]],
    missing: list[str],
) -> None:
    if py_value is None or cs_value is None:
        if py_value is not None or cs_value is not None:
            diffs.append({"path": path, "python": py_value, "csharp": cs_value, "abs_diff": None, "category": "classification"})
        return
    if _num(py_value) and _num(cs_value):
        py_float = float(py_value)
        cs_float = float(cs_value)
        diffs.append({"path": path, "python": py_float, "csharp": cs_float, "abs_diff": abs(py_float - cs_float), "category": _category(path)})
        return
    if isinstance(py_value, dict) and isinstance(cs_value, dict):
        for key in py_value:
            child = f"{path}.{key}" if path else key
            if key not in cs_value:
                missing.append(child)
            else:
                _record_diff(py_value[key], cs_value[key], child, diffs, missing)
        for key in cs_value:
            if key not in py_value:
                missing.append(f"{path}.{key}" if path else key)
        return
    if isinstance(py_value, list) and isinstance(cs_value, list):
        if len(py_value) != len(cs_value):
            missing.append(f"{path} (list length {len(py_value)} vs {len(cs_value)})")
        for index, value in enumerate(py_value[: min(len(py_value), len(cs_value))]):
            _record_diff(value, cs_value[index], f"{path}[{index}]", diffs, missing)
        return
    if py_value != cs_value:
        diffs.append({"path": path, "python": py_value, "csharp": cs_value, "abs_diff": None, "category": "classification"})


def _compare_case(py_path: Path, cs_path: Path, name: str) -> dict[str, Any]:
    py = _read(py_path)["result"]
    cs = _read(cs_path)["result"]
    diffs: list[dict[str, Any]] = []
    missing: list[str] = []

    # The C# record contains an extra warning list; compare all numerical and
    # status content explicitly and leave deployment warnings out of the diff.
    fields = [
        "supply_state",
        "energy_summary_kW",
        "energy_breakdown_kW",
        "intermediate_states",
        "rotor1_outputs",
        "rotor2_outputs",
        "convergence",
    ]
    if "regen_mix_rule" in py or "regen_mix_rule" in cs:
        fields.append("regen_mix_rule")
    for field in fields:
        if field not in py or field not in cs:
            missing.append(field)
        else:
            _record_diff(py[field], cs[field], field, diffs, missing)

    checks: list[dict[str, Any]] = []
    for category, tolerance in THRESHOLDS.items():
        entries = [item for item in diffs if item.get("category") == category and item.get("abs_diff") is not None]
        worst = max(entries, key=lambda item: item["abs_diff"], default=None)
        max_abs = float(worst["abs_diff"]) if worst else 0.0
        checks.append({
            "category": category,
            "max_abs": max_abs,
            "tolerance": tolerance,
            "pass": max_abs <= tolerance,
            "worst": worst,
        })
    status_py = py.get("convergence", {}).get("converged")
    status_cs = cs.get("convergence", {}).get("converged")
    iterations_py = py.get("convergence", {}).get("iterations")
    iterations_cs = cs.get("convergence", {}).get("iterations")
    checks.append({"category": "convergence_status", "python": status_py, "csharp": status_cs, "pass": status_py == status_cs})
    checks.append({"category": "convergence_iterations", "python": iterations_py, "csharp": iterations_cs, "pass": iterations_py == iterations_cs})

    failed_paths = [item for item in diffs if item.get("category") == "classification" or (item.get("abs_diff") is not None and item["abs_diff"] > THRESHOLDS.get(item["category"], 0.0))]
    passed = not missing and not failed_paths and all(bool(item["pass"]) for item in checks)
    return {
        "name": name,
        "python_reference": str(py_path),
        "csharp_result": str(cs_path),
        "passed": passed,
        "missing_or_extra": missing,
        "checks": checks,
        "worst_differences": sorted(diffs, key=lambda item: item.get("abs_diff") or float("inf"), reverse=True)[:20],
        "difference_count": len(diffs),
        "failed_difference_count": len(failed_paths),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# C# + ONNX 数值迁移验证报告",
        "",
        "本报告逐节点比较当前 Python 基准与 Windows x64 C# + ONNX 结果。阈值来自实施计划，未为通过验证而放宽。该结果属于部署/数值迁移验证，不是工程物理验收，也不修复训练域外输入或水分收支偏差。",
        "",
        f"总体：**{'通过' if report['passed'] else '未通过'}**；工况数：{len(report['cases'])}。",
        "",
    ]
    for case in report["cases"]:
        lines.append(f"## {case['name']}：{'通过' if case['passed'] else '未通过'}")
        lines.append("")
        lines.append("| 类别 | 最大绝对差 | 阈值 | 结果 | 最差路径 |")
        lines.append("|---|---:|---:|---|---|")
        for check in case["checks"]:
            worst = check.get("worst") or {}
            max_abs = check.get("max_abs")
            threshold = check.get("tolerance", "—")
            value = f"{max_abs:.12g}" if isinstance(max_abs, (int, float)) else "—"
            limit = f"{threshold:.12g}" if isinstance(threshold, (int, float)) else str(threshold)
            lines.append(f"| {check['category']} | {value} | {limit} | {'通过' if check['pass'] else '失败'} | {worst.get('path', '—')} |")
        if case["missing_or_extra"]:
            lines.append("")
            lines.append("缺失或多余字段：" + ", ".join(case["missing_or_extra"]))
        lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "两台 KAN 仍是既有 artifacts 导出的固定模型；此验证只证明导出、预处理/逆变换、C# 物性/元件/拓扑/迭代及留档行为与 Python 基准在给定样本上的数值一致。它不代表模型在训练域外的预测精度、设备性能或水分守恒已经通过工程验收。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-root", type=Path, required=True, help="contains <case>/result.json")
    parser.add_argument("--csharp-root", type=Path, required=True, help="contains <case>/result.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for name in ("qy1070_case", "default_case"):
        py_path = args.python_root / name / "result.json"
        cs_path = args.csharp_root / name / "result.json"
        if not py_path.exists() or not cs_path.exists():
            raise SystemExit(f"缺少对照记录: {py_path} / {cs_path}")
        cases.append(_compare_case(py_path, cs_path, name))
    report = {
        "schema_version": 1,
        "passed": all(case["passed"] for case in cases),
        "strict_migration_passed": all(case["passed"] for case in cases),
        "thresholds": THRESHOLDS,
        "cases": cases,
        "scope": "deployment_and_numerical_migration_only",
        "engineering_acceptance": "not_evaluated",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
