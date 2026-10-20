#!/usr/bin/env python3
"""Create immutable Python reference records used by the C# migration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "simulation") not in sys.path:
    sys.path.insert(0, str(ROOT / "simulation"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from dual_rotor_sim.config import build_runtime_inputs, load_simulation_config
from dual_rotor_sim.core.system.solver import build_system
from dual_rotor_sim.run_records import result_to_dict
from dual_rotor_sim.dashboard import save_dashboard


def create_one(config_path: Path, output_dir: Path) -> Path:
    parsed = load_simulation_config(config_path)
    runtime = build_runtime_inputs(parsed)
    system = build_system(ROOT / "artifacts", device="cpu")
    result = system.simulate_coupled_xf(
        oa_state=runtime["oa"],
        ra_state=runtime["ra"],
        regen_oa_state=runtime["regen_oa"],
        params=runtime["params"],
        setpoints=runtime["setpoints"],
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document = {
        "run_id": output_dir.name,
        "simulated_at": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path.resolve()),
        "config": config,
        "result": result_to_dict(result),
    }
    (output_dir / "result.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_dashboard(result=result, config=config, target=output_dir / "fig1_dashboard.png")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"拒绝覆盖非空参考目录: {root}")
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        create_one(ROOT / "simulation/configs/qy1070_case.yaml", root / "qy1070_case"),
        create_one(ROOT / "simulation/configs/default_case.yaml", root / "default_case"),
    ]
    all_converged = True
    for path in paths:
        document = json.loads((path / "result.json").read_text(encoding="utf-8"))
        converged = document["result"].get("convergence", {}).get("converged")
        label = "已收敛" if converged is True else ("未收敛" if converged is False else "未评估收敛")
        print(f"Python 参考工况 {path.name}：{label}；JSON/PNG 已保存到 {path}")
        all_converged = all_converged and converged is True
    if not all_converged:
        print("参考工况包含未收敛或未评估状态；保留全部 JSON/PNG，退出码为 4。", file=sys.stderr)
    return 0 if all_converged else 4


if __name__ == "__main__":
    raise SystemExit(main())
