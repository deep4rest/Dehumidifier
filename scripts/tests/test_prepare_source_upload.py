"""验证上传快照不会携带本机训练/部署资料。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "prepare_source_upload", Path(__file__).resolve().parents[1] / "prepare_source_upload.py"
)
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


def _write(root, name, content="源码示例"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_snapshot_excludes_history_models_data_and_build_outputs(tmp_path):
    source, target = tmp_path / "repo", tmp_path / "snapshot"
    keep = {
        "README.md", "data/README.md", "artifacts/README.md", "results/README.md",
        "training/single_kan/train.py", "simulation/configs/case.yaml",
        "deployment/dotnet/src/Simulation/build/Simulation.targets",
        "deployment/dotnet/examples/case.json",
    }
    reject = {
        ".git/config", ".claude/settings.json", ".agents/skills/local.md",
        "data/dataset.xls", "artifacts/model/metadata.json", "archive/artifacts/model/config.yaml",
        "results/simulation/run/result.json", "training/model.pt", "training/scalers.pkl",
        "deployment/dotnet/src/Simulation/models/model_contract.json",
        "deployment/dotnet/src/Simulation/bin/Release/generated.cs",
        "deployment/dotnet/src/Simulation/obj/generated.cs", "deployment/dotnet/result.zip",
    }
    for name in keep | reject:
        _write(source, name)
    report = upload.prepare(source, target)
    assert {item["path"] for item in report["files"]} == keep
    assert {str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()} == keep


def test_snapshot_refuses_nonempty_output(tmp_path):
    _write(tmp_path / "repo", "README.md")
    _write(tmp_path / "snapshot", "existing.md")
    with pytest.raises(ValueError, match="非空"):
        upload.prepare(tmp_path / "repo", tmp_path / "snapshot")


def test_snapshot_rejects_symlink_and_binary_in_text_files(tmp_path):
    source = tmp_path / "repo"
    _write(source, "README.md")
    (source / "training").mkdir()
    (source / "training/linked.py").symlink_to(source / "README.md")
    with pytest.raises(ValueError, match="符号链接"):
        upload.source_files(source)
    (source / "training/linked.py").unlink()
    (source / "training/fake.py").write_bytes(b"not\0source")
    with pytest.raises(ValueError, match="二进制"):
        upload.source_files(source)
