"""无模型、无 .NET 环境下验证源码封装边界与严格迁移判定。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


DEPLOYMENT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DEPLOYMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_copy_preserves_msbuild_targets_and_excludes_generated_assets(tmp_path):
    package = _load("package_delivery")
    source = tmp_path / "source"
    paths = [
        "src/Test/build/Test.targets", "src/Test/Test.cs", "examples/case.json",
        "src/Test/bin/Release/generated.cs", "src/Test/obj/generated.cs",
        "src/Test/models/model_contract.json", "src/Test/models/model.onnx",
        "build/previous/old.ps1", "src/Test/weights.pt",
    ]
    for name in paths:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    target = tmp_path / "target"
    package.copy_source(source, target)
    copied = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    assert copied == {"src/Test/build/Test.targets", "src/Test/Test.cs", "examples/case.json"}


def test_strict_migration_rejects_iteration_difference(tmp_path):
    migration = _load("validate_migration")
    result = {
        "supply_state": {}, "energy_summary_kW": {}, "energy_breakdown_kW": {},
        "intermediate_states": {}, "rotor1_outputs": {}, "rotor2_outputs": {},
        "convergence": {"converged": True, "iterations": 12},
    }
    py = tmp_path / "python.json"
    cs = tmp_path / "csharp.json"
    py.write_text(json.dumps({"result": result}), encoding="utf-8")
    cs.write_text(json.dumps({"result": {**result, "convergence": {"converged": True, "iterations": 13}}}), encoding="utf-8")
    report = migration._compare_case(py, cs, "fixture")
    assert report["passed"] is False
    assert report["missing_or_extra"] == []
    assert next(check for check in report["checks"] if check["category"] == "convergence_iterations")["pass"] is False


def test_delivery_hashes_fixture_and_never_claims_unperformed_validation(tmp_path, monkeypatch):
    package = _load("package_delivery")
    repository = tmp_path / "repository"
    for name in ("README.md", "AGENTS.md", "requirements.txt", "deployment/dotnet/README.md"):
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    (repository / "deployment/dotnet/examples").mkdir()
    monkeypatch.setattr(package, "ROOT", repository)
    models = tmp_path / "models"
    models.mkdir()
    for name in package.MODEL_FILES:
        (models / name).write_bytes(b"fixture-model")
    build = tmp_path / "build"
    for subdir, executable in (("verify", "Dehumidifier.Verify.exe"), ("winforms", "WinFormsDemo.exe")):
        published = build / subdir
        (published / "models").mkdir(parents=True)
        (published / executable).write_bytes(b"fixture-executable")
        (published / "Dehumidifier.Simulation.dll").write_bytes(b"fixture-dll")
        for name in package.MODEL_FILES:
            (published / "models" / name).write_bytes((models / name).read_bytes())
    (build / "nuget").mkdir()
    (build / "nuget/Dehumidifier.Simulation.1.0.1.nupkg").write_bytes(b"fixture-package")
    output = tmp_path / "delivery"
    summary = package.assemble(build, models, output)
    metadata = json.loads((output / "package_metadata.json").read_text())
    assert metadata["strict_migration_evaluated"] is False
    assert metadata["strict_migration_passed"] is False
    assert metadata["source_contract_aligned"] is True
    assert metadata["current_python_compatible"] == "requires_current_cross_language_validation"
    assert metadata["engineering_acceptance"] == "not_evaluated"
    assert summary["sha256"] == package.sha256(output.with_suffix(".zip"))
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert all(package.sha256(output / name) == digest for name, digest in manifest.items())
    with pytest.raises(FileExistsError):
        package.assemble(build, models, output)
    (build / "winforms/models/xf.onnx").write_bytes(b"different-model")
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="发布模型与本次导出不一致"):
        package.assemble(build, models, rejected)
    assert not rejected.exists()


@pytest.mark.parametrize("first_converged, expected_exit", [(True, 0), (False, 4), (None, 4)])
def test_reference_saves_all_cases_and_returns_convergence_exit_code(tmp_path, monkeypatch, capsys, first_converged, expected_exit):
    reference = _load("create_reference")
    created = []

    def fake_create(config_path, output_dir):
        output_dir.mkdir(parents=True)
        converged = first_converged if not created else True
        (output_dir / "result.json").write_text(json.dumps({"result": {"convergence": {"converged": converged}}}), encoding="utf-8")
        (output_dir / "fig1_dashboard.png").write_bytes(b"fixture-png")
        created.append(output_dir)
        return output_dir

    monkeypatch.setattr(reference, "create_one", fake_create)
    monkeypatch.setattr(sys, "argv", ["create_reference.py", "--output", str(tmp_path / "reference")])
    assert reference.main() == expected_exit
    assert len(created) == 2
    assert all((directory / "result.json").is_file() and (directory / "fig1_dashboard.png").is_file() for directory in created)
    captured = capsys.readouterr()
    assert "Python 参考工况" in captured.out
    if expected_exit == 4:
        assert "退出码为 4" in captured.err
