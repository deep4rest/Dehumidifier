#!/usr/bin/env python3
"""从同一次Windows构建组装两个中文命名v2 DLL成品，保留真实验证边界。"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from package_delivery import MODEL_FILES, copy_source, sha256
from zipfile import ZIP_DEFLATED, ZipFile
import hashlib

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = (
    ("prediction", "露点与能耗预测模型v2", "Dehumidifier.Prediction"),
    ("optimization", "最优能耗工况搜索模型v2", "Dehumidifier.Optimization"),
)
COMMON_DLLS = (
    "Dehumidifier.Simulation.dll", "Microsoft.ML.OnnxRuntime.dll",
    "System.Drawing.Common.dll", "Microsoft.Win32.SystemEvents.dll",
    "onnxruntime.dll", "onnxruntime_providers_shared.dll",
)
PACKAGE_VERSION = "2.0.2"


def _validate_packages(nuget: Path, verifier: Path, models: Path) -> None:
    """NuGet existence alone cannot prove it contains this build/model set."""
    assemblies = {"Dehumidifier.Simulation": "Dehumidifier.Simulation.dll",
                  **{package: title + ".dll" for _, title, package in PRODUCTS}}
    for package, assembly in assemblies.items():
        archive = _require(nuget / f"{package}.{PACKAGE_VERSION}.nupkg")
        with ZipFile(archive) as packed:
            if packed.read(f"lib/net8.0/{assembly}") != (verifier / assembly).read_bytes():
                raise ValueError(f"NuGet内程序集与本次发布不一致：{archive.name}")
            if package == "Dehumidifier.Simulation":
                for name in MODEL_FILES:
                    path = f"contentFiles/any/any/Dehumidifier/models/{name}"
                    if packed.read(path) != (models / name).read_bytes():
                        raise ValueError(f"NuGet内模型与本次导出不一致：{name}")


def _validate_runtime_identity(report: dict | None, verifier: Path, models: Path) -> None:
    if report is None or report.get("windows_runtime_validation_passed") is not True:
        return
    expected_models = {name: sha256(models / name) for name in MODEL_FILES}
    expected_dlls = {name: sha256(verifier / name) for name in (
        "Dehumidifier.Simulation.dll", *(p[1] + ".dll" for p in PRODUCTS))}
    if report.get("model_sha256") != expected_models or report.get("assembly_sha256") != expected_dlls:
        raise ValueError("运行报告缺少或不匹配本次DLL/模型SHA256，拒绝借用其他构建的通过结论")


def _published_models(root: Path, models: Path) -> None:
    embedded = next((p for p in (root / "models", root / "Dehumidifier/models")
                     if (p / "model_contract.json").is_file()), None)
    if embedded is None:
        raise FileNotFoundError(f"发布目录缺少模型：{root}")
    for name in MODEL_FILES:
        path = embedded / name
        if not path.is_file() or sha256(path) != sha256(models / name):
            raise ValueError(f"发布模型与本次导出不一致：{path}")


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _json_report(path: Path | None, strict: bool = False) -> dict | None:
    if path is None:
        return None
    report = json.loads(_require(path).read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError(f"验证报告必须为JSON对象：{path}")
    if strict and (not isinstance(report.get("passed"), bool) or not isinstance(report.get("cases"), list)):
        raise ValueError("严格迁移报告须来自validate_migration.py，含passed和cases")
    if "windows_runtime_validation_passed" in report and not isinstance(report["windows_runtime_validation_passed"], bool):
        raise ValueError("windows_runtime_validation_passed必须是显式布尔值")
    return report


def _archive(directory: Path) -> dict:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    manifest = {p.relative_to(directory).as_posix(): sha256(p) for p in files}
    manifest_path = directory.with_suffix(".manifest.json")
    archive = directory.with_suffix(".zip")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with ZipFile(archive, "x", ZIP_DEFLATED, compresslevel=6) as packed:
        for path in files:
            packed.write(path, directory.name + "/" + path.relative_to(directory).as_posix())
    with ZipFile(archive) as packed:
        if packed.testzip() is not None:
            raise ValueError(f"ZIP CRC校验失败：{archive}")
        expected_names = {directory.name + "/" + name for name in manifest}
        if set(packed.namelist()) != expected_names:
            raise ValueError("ZIP内文件清单与manifest不一致")
        for name, expected in manifest.items():
            actual = hashlib.sha256(packed.read(directory.name + "/" + name)).hexdigest()
            if actual != expected:
                raise ValueError(f"ZIP内文件哈希不一致：{name}")
    checksum = sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    return {"directory": str(directory.resolve()), "archive": str(archive.resolve()),
            "sha256": checksum, "manifest": str(manifest_path.resolve()), "files": len(files),
            "zip_crc_and_file_hashes_verified": True}


def assemble(build: Path, models: Path, output: Path, validation: Path | None = None,
             runtime: Path | None = None, examples: Path | None = None,
             include_winforms: bool = True) -> dict:
    """先校验全部输入，再创建尚不存在的父目录和两个命名成品。"""
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有交付：{output}")
    if not build.is_dir() or not models.is_dir():
        raise FileNotFoundError("须指定本次构建和导出模型目录")
    for name in MODEL_FILES:
        _require(models / name)
    verifier = build / "v2verify"
    _require(verifier / "Dehumidifier.V2Verify.exe")
    for name in COMMON_DLLS:
        _require(verifier / name)
    for _, title, _ in PRODUCTS:
        _require(verifier / (title + ".dll"))
    publication_roots = [verifier, build / "verify"]
    _require(build / "verify/Dehumidifier.Verify.exe")
    if include_winforms:
        _require(build / "winforms/WinFormsDemo.exe")
        publication_roots.append(build / "winforms")
    core_hash = sha256(verifier / "Dehumidifier.Simulation.dll")
    for published in publication_roots:
        _published_models(published, models)
        if sha256(_require(published / "Dehumidifier.Simulation.dll")) != core_hash:
            raise ValueError(f"发布目录携带的整机业务DLL不一致：{published}")
    nuget = build / "nuget"
    _require(nuget / f"Dehumidifier.Simulation.{PACKAGE_VERSION}.nupkg")
    for _, _, package_id in PRODUCTS:
        _require(nuget / f"{package_id}.{PACKAGE_VERSION}.nupkg")
    _validate_packages(nuget, verifier, models)
    strict_report = _json_report(validation, strict=True)
    runtime_report = _json_report(runtime)
    _validate_runtime_identity(runtime_report, verifier, models)
    strict_passed = strict_report is not None and strict_report["passed"] is True and strict_report.get("strict_migration_passed", True) is True
    runtime_evaluated = runtime_report is not None and isinstance(runtime_report.get("windows_runtime_validation_passed"), bool)
    runtime_passed = runtime_evaluated and runtime_report["windows_runtime_validation_passed"] is True
    if examples is not None and not examples.is_dir():
        raise FileNotFoundError(examples)
    quickstart = _require(ROOT / "deployment/dotnet/docs/快速使用.md")
    run_example = _require(ROOT / "deployment/dotnet/samples/run_example.ps1")
    sample = ROOT / "deployment/dotnet/samples/V2DllConsumer"
    for name in ("Program.cs", "V2DllConsumer.csproj"):
        _require(sample / name)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for kind, title, package_id in PRODUCTS:
        product = output / title
        dll = product / "dll"
        dll.mkdir(parents=True)
        # Both entries are required by the documented search->prediction sample.
        # Ship a matched pair in each lib directory so hosts never borrow an
        # unrelated entry assembly from another delivery/version.
        for name in (*(p[1] + ".dll" for p in PRODUCTS), *COMMON_DLLS):
            shutil.copy2(verifier / name, dll / name)
        for name in (*(p[1] + ".xml" for p in PRODUCTS), "Dehumidifier.Simulation.xml"):
            if (verifier / name).is_file():
                shutil.copy2(verifier / name, dll / name)
        # The universal verifier uses both public entry assemblies; retain its
        # complete self-contained output even when the lib entry is one product.
        shutil.copytree(verifier, product / "verify")
        (product / "models").mkdir()
        for name in MODEL_FILES:
            shutil.copy2(models / name, product / "models" / name)
        (product / "examples").mkdir()
        default_examples = ROOT / "deployment/dotnet/examples"
        for source in (default_examples, examples):
            if source is not None and source.is_dir():
                for path in source.glob("*.json"):
                    shutil.copy2(path, product / "examples" / path.name)
        if kind == "prediction" and include_winforms:
            shutil.copytree(build / "winforms", product / "examples/WinFormsDemo/publish")
        (product / "nuget").mkdir()
        for name in (f"{package}.{PACKAGE_VERSION}.nupkg" for package in ("Dehumidifier.Simulation", "Dehumidifier.Prediction", "Dehumidifier.Optimization")):
            shutil.copy2(nuget / name, product / "nuget" / name)
        copy_source(ROOT / "deployment/dotnet", product / "source/deployment/dotnet")
        for name in ("README.md", "AGENTS.md", "requirements.txt"):
            shutil.copy2(ROOT / name, product / "source" / name)
        handoff = ROOT / "docs/handoff"
        if handoff.is_dir():
            shutil.copytree(handoff, product / "source/docs/handoff")
        instructions = ROOT / "deployment/dotnet/docs/v2模型DLL调用说明.md"
        if instructions.is_file():
            shutil.copy2(instructions, product / "v2模型DLL调用说明.md")
        reports = product / "reports"
        reports.mkdir()
        for report_path, filename in ((validation, "validation.json"), (runtime, "runtime_validation.json")):
            if report_path is not None:
                shutil.copy2(report_path, reports / filename)
                if report_path.with_suffix(".md").is_file():
                    shutil.copy2(report_path.with_suffix(".md"), reports / Path(filename).with_suffix(".md"))
        shutil.copy2(quickstart, product / "快速使用.md")
        copy_source(sample, product / "接入示例")
        for filename, run_mode in (("01_前向预测.cmd", "forward"), ("02_搜索并前向复算.cmd", "search")):
            # ASCII content works with the default Windows cmd code page; the
            # PowerShell file itself is UTF-8 BOM for Windows PowerShell 5.1.
            script = ("@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "
                      '"%~dp0run_example.ps1" -Mode ' + run_mode + "\r\nset runExit=%ERRORLEVEL%\r\npause\r\nexit /b %runExit%\r\n")
            (product / filename).write_bytes(script.encode("ascii"))
        # Use an ASCII path from cmd, preserve Chinese content for PS5.1.
        (product / "run_example.ps1").write_text(run_example.read_text(encoding="utf-8"), encoding="utf-8-sig")
        metadata = {
            "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
            "package_name": title, "package_kind": kind, "package_version": PACKAGE_VERSION,
            "main_assembly": f"dll/{title}.dll", "managed_framework": "net8.0", "platform": "win-x64",
            "runtime_requires_python": False,
            "recommended_configuration": "examples/qy1070_case.json",
            "companion_assemblies_included": [p[1] + ".dll" for p in PRODUCTS],
            "dll_host_requires_dotnet8": True, "verification_executable_self_contained": True,
            "requires_visual_cpp_v14_x64_runtime": True,
            "visual_cpp_runtime_installer_included": False,
            "source_contract_aligned": True,
            "topology": "R2 process outlet -> CC3 -> Heater_Post -> Supply; six thermal loads",
            "current_python_compatible": "strict_passed" if strict_passed else "strict_migration_not_passed_or_not_evaluated",
            "windows_runtime_validation_evaluated": runtime_evaluated,
            "windows_runtime_validation_passed": runtime_passed,
            "strict_migration_evaluated": strict_report is not None,
            "strict_migration_passed": strict_passed,
            "engineering_acceptance": "not_evaluated",
            "offline_nuget_dependencies_included": False,
            "global_minimum_proven": False,
            "optimization_algorithm_note": "C# DE uses System.Random; same objective/hard constraints, not an identical SciPy/NumPy trajectory",
            "physical_limitations": "Existing model coverage and rotor water/enthalpy imbalances retained; ideal terminal components lack capacity/frost/COP constraints",
            "build_input": str(build.resolve()), "core_assembly_sha256": core_hash,
            "main_assembly_sha256": sha256(dll / (title + ".dll")),
            "model_sha256": {name: sha256(models / name) for name in MODEL_FILES},
            "strict_report_sha256": sha256(validation) if validation is not None else None,
            "runtime_report_sha256": sha256(runtime) if runtime is not None else None,
            "source_sha256": {p.relative_to(product / "source").as_posix(): sha256(p) for p in sorted((product / "source").rglob("*")) if p.is_file()},
        }
        (product / "package_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        runtime_label = ('通过' if runtime_passed else '未通过') if runtime_evaluated else '未评估'
        strict_label = ('通过' if strict_passed else '未通过') if strict_report is not None else '未评估'
        readme = f"# {title}（{PACKAGE_VERSION}）\n\n先读[快速使用.md](快速使用.md)。双击 `01_前向预测.cmd` 或 `02_搜索并前向复算.cmd` 即可运行，无需Python或预装.NET。开发接入示例位于 `接入示例`，只引用本包DLL。两个入口和共享核心已配套放入 `dll`，请整套使用。\n\n本次Windows运行验证：{runtime_label}；严格Python迁移：{strict_label}。详细证据见 `reports`；工程实测未评估。\n"
        (product / "README.md").write_text(readme, encoding="utf-8")
        summaries.append({"package_name": title, "kind": kind, **_archive(product)})
    result = {"output": str(output.resolve()), "products": summaries,
              "strict_migration_passed": strict_passed, "windows_runtime_validation_passed": runtime_passed}
    (output / "交付索引.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="尚不存在的父目录；内含两个中文命名成品")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    parser.add_argument("--examples", type=Path, help="可选附加工况JSON目录")
    parser.add_argument("--without-winforms", action="store_true")
    args = parser.parse_args()
    print(json.dumps(assemble(args.build.resolve(), args.models.resolve(), args.output.resolve(),
                              args.validation_report, args.runtime_report, args.examples,
                              include_winforms=not args.without_winforms), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
