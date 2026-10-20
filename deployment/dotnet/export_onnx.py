#!/usr/bin/env python3
"""Export the two current single-KAN artifacts to ONNX and verify numerics.

The exported graph contains the trained KAN only.  The C# runtime owns the
documented MinMax/log1p preprocessing contract, which is recorded beside the
graph in ``preprocessing.json``.  No model is retrained or selected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "simulation") not in sys.path:
    sys.path.insert(0, str(ROOT / "simulation"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.single_kan.model import Preprocessor  # noqa: E402
from dual_rotor_sim.core.system.components.rotor import SingleRotorModel  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_float_list(values: Any) -> list[float]:
    return [float(x) for x in np.asarray(values, dtype=np.float64).reshape(-1)]


def _load(profile: str) -> tuple[torch.nn.Module, Any, dict[str, Any], Path]:
    artifact = ROOT / "artifacts" / f"kan_{profile}_single"
    # 正式仿真加载器验证最终风量输入、六输出、架构及 scaler；旧 XLS
    # load_artifact 仍有历史风速契约，不能作为最终风量包的加载入口。
    runtime = SingleRotorModel.from_artifact(artifact, profile, device="cpu")
    import pickle
    with (artifact / "model/scalers.pkl").open("rb") as stream:
        preprocessor = Preprocessor.from_state_dict(pickle.load(stream))
    return runtime.model, preprocessor, runtime.metadata, artifact


def _make_samples(profile: str, preprocessor: Any, count: int = 257) -> np.ndarray:
    """Deterministic in-domain, boundary, and mild out-of-domain rows."""
    rng = np.random.default_rng(20260910 + (0 if profile == "xf" else 1))
    scaler = preprocessor.input_scaler
    lo = np.asarray(scaler.data_min_, dtype=np.float64)
    hi = np.asarray(scaler.data_max_, dtype=np.float64)
    rows = [lo, hi, (lo + hi) / 2.0]
    # A few rows just beyond the training range exercise the no-clip contract.
    rows.extend([lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo)])
    rows.extend(rng.uniform(lo, hi, size=(count - len(rows), lo.size)))
    return np.asarray(rows, dtype=np.float64)


def _export_one(profile: str, target: Path) -> dict[str, Any]:
    model, preprocessor, metadata, artifact = _load(profile)
    target.mkdir(parents=True, exist_ok=True)
    onnx_path = target / ("xf.onnx" if profile == "xf" else "main.onnx")

    dummy = torch.zeros((1, len(metadata["input_cols"])), dtype=torch.float32)
    # The model's public forward accepts scaled float32 input and returns scaled
    # float32 output.  Keeping this boundary explicit prevents accidental
    # duplicate scaling in a host application.
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    checked = onnx.load(str(onnx_path))
    onnx.checker.check_model(checked)

    samples = _make_samples(profile, preprocessor)
    scaled = preprocessor.transform_features(samples).astype(np.float32)
    with torch.no_grad():
        torch_scaled = model(torch.from_numpy(scaled)).cpu().numpy().astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_scaled = session.run(["output"], {"input": scaled})[0].astype(np.float32)
    scaled_error = np.abs(torch_scaled.astype(np.float64) - ort_scaled.astype(np.float64))
    physical_torch = preprocessor.inverse_targets(torch_scaled)
    physical_ort = preprocessor.inverse_targets(ort_scaled)
    physical_error = np.abs(physical_torch - physical_ort)
    if not np.isfinite(physical_ort).all():
        raise RuntimeError(f"{profile} ONNX 输出包含非有限值")
    if float(physical_error.max()) >= 1e-3:
        raise RuntimeError(f"{profile} ONNX 导出物理量最大绝对误差超出 1e-3：{physical_error.max()}")

    xs = preprocessor.input_scaler
    ys = preprocessor.target_scaler
    preprocessing = {
        "feature_range": [-1.0, 1.0],
        "input_columns": list(metadata["input_cols"]),
        "output_columns": list(metadata["direct_outputs"]),
        "input_scale": _json_float_list(xs.scale_),
        "input_min": _json_float_list(xs.min_),
        "output_scale": _json_float_list(ys.scale_),
        "output_min": _json_float_list(ys.min_),
        "log1p_output_columns": list(preprocessor.target_transform.log1p_targets),
        "clip": False,
        "numeric_boundary": "physical double -> input float32 -> ONNX float32 -> physical double",
    }
    (target / f"preprocessing_{profile}.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "profile": profile,
        "model_type": "KAN",
        "onnx_file": onnx_path.name,
        "opset": 17,
        "input_name": "input",
        "output_name": "output",
        "input_shape": ["batch", len(metadata["input_cols"])],
        "output_shape": ["batch", 6],
        "input_cols": list(metadata["input_cols"]),
        "direct_outputs": list(metadata["direct_outputs"]),
        "derived_outputs": ["处理出口露点"],
        "architecture": metadata["architecture"],
        "source_artifact": str(artifact.relative_to(ROOT)),
        "source_hashes": {
            "model_metadata.json": sha256(artifact / "model_metadata.json"),
            "model/best_model.pt": sha256(artifact / "model" / "best_model.pt"),
            "model/scalers.pkl": sha256(artifact / "model" / "scalers.pkl"),
        },
        "verification": {
            "samples": int(len(samples)),
            "max_abs_scaled": float(scaled_error.max()),
            "max_abs_physical": float(physical_error.max()),
            "per_output_max_abs_physical": [float(x) for x in physical_error.max(axis=0)],
            "nonfinite_count": int((~np.isfinite(physical_ort)).sum()),
        },
    }
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description="按当前训练契约导出两台 KAN；导出通过不代表历史 C# DLL 已完成适配。")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"拒绝覆盖非空导出目录: {target}")
    target.mkdir(parents=True, exist_ok=True)
    contracts = [_export_one(profile, target) for profile in ("xf", "main")]
    preprocessing = {
        "schema_version": 1,
        "profiles": {
            profile: json.loads((target / f"preprocessing_{profile}.json").read_text(encoding="utf-8"))
            for profile in ("xf", "main")
        },
    }
    (target / "preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    merged = {
        "schema_version": 1,
        "runtime": "Microsoft.ML.OnnxRuntime CPU",
        "preprocessing_file": "preprocessing.json",
        "models": contracts,
    }
    (target / "model_contract.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(merged, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
