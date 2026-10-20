from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.requires_models


@pytest.fixture(autouse=True)
def require_external_export_dependencies():
    missing = [str(ROOT / "artifacts" / f"kan_{profile}_single" / name)
               for profile in ("xf", "main")
               for name in ("model_metadata.json", "model/best_model.pt", "model/scalers.pkl")
               if not (ROOT / "artifacts" / f"kan_{profile}_single" / name).is_file()]
    missing.extend(f"Python 模块 {name}" for name in ("onnx", "onnxruntime") if importlib.util.find_spec(name) is None)
    if missing:
        reason = "缺少导出依赖或外部模型：" + ", ".join(missing)
        if os.environ.get("REQUIRE_EXTERNAL_FIXTURES") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


def test_exported_onnx_contract_matches_current_artifacts(tmp_path: Path):
    output = tmp_path / "models"
    subprocess.run(
        [sys.executable, str(ROOT / "deployment/dotnet/export_onnx.py"), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )
    contract = json.loads((output / "model_contract.json").read_text(encoding="utf-8"))
    assert {item["profile"] for item in contract["models"]} == {"xf", "main"}
    assert (output / "xf.onnx").is_file()
    assert (output / "main.onnx").is_file()
    assert (output / "preprocessing.json").is_file()
    for item in contract["models"]:
        assert item["verification"]["nonfinite_count"] == 0
        assert item["verification"]["max_abs_physical"] < 1e-3


def test_preprocessing_contract_has_no_clip_and_log1p_only_on_process_humidity(tmp_path: Path):
    output = tmp_path / "models"
    subprocess.run([sys.executable, str(ROOT / "deployment/dotnet/export_onnx.py"), "--output", str(output)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    payload = json.loads((output / "preprocessing.json").read_text(encoding="utf-8"))
    for profile in ("xf", "main"):
        item = payload["profiles"][profile]
        assert item["clip"] is False
        assert item["log1p_output_columns"] == ["处理出口含湿量"]
        assert len(item["input_scale"]) == (10 if profile == "xf" else 9)
        assert len(item["output_scale"]) == 6
