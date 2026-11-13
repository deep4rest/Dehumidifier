"""Exercise package identity and evidence gates using a tiny synthetic build."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

DEPLOYMENT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(DEPLOYMENT))
    spec = importlib.util.spec_from_file_location("package_v2_delivery", DEPLOYMENT / "package_v2_delivery.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    repository = tmp_path / "repo"
    for name in ("README.md", "AGENTS.md", "requirements.txt", "deployment/dotnet/docs/v2模型DLL调用说明.md", "deployment/dotnet/src/Core/build/Core.targets"):
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source-fixture", encoding="utf-8")
    monkeypatch.setattr(package, "ROOT", repository)
    models = tmp_path / "models"
    models.mkdir()
    for name in package.MODEL_FILES:
        (models / name).write_text("current-model:" + name, encoding="utf-8")
    build = tmp_path / "build"
    for folder, executable in (("v2verify", "Dehumidifier.V2Verify.exe"), ("verify", "Dehumidifier.Verify.exe"), ("winforms", "WinFormsDemo.exe")):
        p = build / folder
        (p / "models").mkdir(parents=True)
        (p / executable).write_bytes(b"fixture-exe")
        (p / "Dehumidifier.Simulation.dll").write_bytes(b"current-core")
        for name in package.MODEL_FILES:
            (p / "models" / name).write_bytes((models / name).read_bytes())
    for name in package.COMMON_DLLS:
        if name != "Dehumidifier.Simulation.dll":
            (build / "v2verify" / name).write_bytes(b"runtime-fixture")
    for _, title, _ in package.PRODUCTS:
        (build / "v2verify" / (title + ".dll")).write_text(title, encoding="utf-8")
    # Framework files stay with the self-contained verifier, not the DLL lib.
    (build / "v2verify/System.Private.CoreLib.dll").write_bytes(b"framework-fixture")
    (build / "nuget").mkdir()
    for name in ("Dehumidifier.Simulation", "Dehumidifier.Prediction", "Dehumidifier.Optimization"):
        (build / "nuget" / (name + ".2.0.0.nupkg")).write_bytes(b"current-package")
    return package, build, models, tmp_path


def test_two_names_hashes_minimal_lib_and_no_unperformed_claims(fixture):
    package, build, models, tmp = fixture
    output = tmp / "delivery"
    result = package.assemble(build, models, output)
    assert [p["package_name"] for p in result["products"]] == ["露点与能耗预测模型v2", "最优能耗工况搜索模型v2"]
    for entry in result["products"]:
        product = Path(entry["directory"])
        metadata = json.loads((product / "package_metadata.json").read_text(encoding="utf-8"))
        assert metadata["strict_migration_evaluated"] is False
        assert metadata["strict_migration_passed"] is False
        assert metadata["windows_runtime_validation_evaluated"] is False
        assert metadata["windows_runtime_validation_passed"] is False
        assert metadata["engineering_acceptance"] == "not_evaluated"
        assert (product / metadata["main_assembly"]).is_file()
        assert not (product / "dll/System.Private.CoreLib.dll").exists()
        assert (product / "verify/System.Private.CoreLib.dll").exists()
        manifest = json.loads(product.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert all(package.sha256(product / name) == sha for name, sha in manifest.items())
        with ZipFile(entry["archive"]) as zip:
            assert zip.testzip() is None
            assert set(zip.namelist()) == {product.name + "/" + name for name in manifest}
        assert entry["sha256"] == package.sha256(Path(entry["archive"]))
    assert (output / "露点与能耗预测模型v2/examples/WinFormsDemo/publish/WinFormsDemo.exe").is_file()
    assert not (output / "最优能耗工况搜索模型v2/examples/WinFormsDemo").exists()
    with pytest.raises(FileExistsError):
        package.assemble(build, models, output)


def test_strict_failure_and_runtime_success_are_kept_separate(fixture):
    package, build, models, tmp = fixture
    strict = tmp / "validation.json"
    strict.write_text(json.dumps({"passed": False, "strict_migration_passed": False, "cases": []}), encoding="utf-8")
    runtime = tmp / "runtime.json"
    runtime.write_text(json.dumps({"windows_runtime_validation_passed": True, "checks": ["actual-calls"]}), encoding="utf-8")
    out = tmp / "reviewed_delivery"
    result = package.assemble(build, models, out, strict, runtime)
    assert result["strict_migration_passed"] is False
    assert result["windows_runtime_validation_passed"] is True
    for entry in result["products"]:
        p = Path(entry["directory"])
        m = json.loads((p / "package_metadata.json").read_text(encoding="utf-8"))
        assert m["windows_runtime_validation_evaluated"] is True
        assert m["windows_runtime_validation_passed"] is True
        assert m["strict_migration_evaluated"] is True
        assert m["strict_migration_passed"] is False
        assert (p / "reports/validation.json").read_bytes() == strict.read_bytes()
        assert (p / "reports/runtime_validation.json").read_bytes() == runtime.read_bytes()


@pytest.mark.parametrize("corruption", ["model", "core", "missing-entry"])
def test_rejects_mixed_build_before_creating_any_delivery(fixture, corruption):
    package, build, models, tmp = fixture
    if corruption == "model":
        (build / "v2verify/models/xf.onnx").write_bytes(b"stale-model")
    elif corruption == "core":
        (build / "winforms/Dehumidifier.Simulation.dll").write_bytes(b"old-core")
    else:
        (build / "v2verify/最优能耗工况搜索模型v2.dll").unlink()
    out = tmp / "must-not-exist"
    with pytest.raises((ValueError, FileNotFoundError)):
        package.assemble(build, models, out)
    assert not out.exists()
