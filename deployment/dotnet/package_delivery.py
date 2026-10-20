#!/usr/bin/env python3
"""从显式构建目录组装新交付；不依赖旧模板，不推断 Windows 或工程验收通过。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = (
    "xf.onnx", "main.onnx", "model_contract.json", "preprocessing.json",
    "preprocessing_xf.json", "preprocessing_main.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_source(source: Path, target: Path) -> None:
    """源码 allowlist 保留 MSBuild targets，同时排除生成目录及模型。"""
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in {"bin", "obj", "models", "__pycache__", "nuget-cache", "results"} for part in relative.parts):
            continue
        if relative.parts[0] == "build":
            continue
        allowed = path.suffix in {".py", ".cs", ".csproj", ".targets", ".ps1", ".md", ".txt"}
        allowed |= relative.parts[0] == "examples" and path.suffix == ".json"
        if allowed:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def assemble(build: Path, models: Path, output: Path, validation: Path | None = None) -> dict:
    archive = output.with_suffix(".zip")
    manifest_path = output.with_suffix(".manifest.json")
    checksum = archive.with_suffix(".zip.sha256")
    for existing in (output, archive, manifest_path, checksum):
        if existing.exists():
            raise FileExistsError(f"拒绝覆盖已有交付：{existing}")
    for file in MODEL_FILES:
        if not (models / file).is_file():
            raise FileNotFoundError(models / file)
    for path in (build / "verify/Dehumidifier.Verify.exe", build / "verify/Dehumidifier.Simulation.dll", build / "winforms/WinFormsDemo.exe"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not list((build / "nuget").glob("Dehumidifier.Simulation.*.nupkg")):
        raise FileNotFoundError("构建目录缺少 Dehumidifier.Simulation NuGet 主包")
    # 构建输出中的模型必须与本次指定导出完全一致；不将旧 DLL 搭配新模型。
    for root in (build / "verify", build / "winforms"):
        candidates = (root / "models", root / "Dehumidifier/models")
        embedded = next((candidate for candidate in candidates if (candidate / "model_contract.json").is_file()), None)
        if embedded is None:
            raise FileNotFoundError(f"发布目录没有完整模型：{root}")
        for file in MODEL_FILES:
            if not (embedded / file).is_file() or sha256(embedded / file) != sha256(models / file):
                raise ValueError(f"发布模型与本次导出不一致：{embedded / file}")
    if sha256(build / "verify/Dehumidifier.Simulation.dll") != sha256(build / "winforms/Dehumidifier.Simulation.dll"):
        raise ValueError("验证程序与 WinForms 携带的业务 DLL 不一致")

    report = None
    if validation is not None:
        report = json.loads(validation.read_text(encoding="utf-8"))
        if not isinstance(report.get("passed"), bool) or not isinstance(report.get("cases"), list):
            raise ValueError("数值迁移报告必须来自 validate_migration.py")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(build / "verify", output / "verify")
    shutil.copytree(build / "winforms", output / "examples/WinFormsDemo/publish")
    shutil.copytree(build / "nuget", output / "nuget")
    (output / "models").mkdir()
    for file in MODEL_FILES:
        shutil.copy2(models / file, output / "models" / file)
    for example in (ROOT / "deployment/dotnet/examples").glob("*.json"):
        shutil.copy2(example, output / "examples" / example.name)
    copy_source(ROOT / "deployment/dotnet", output / "source/deployment/dotnet")
    for file in ("README.md", "AGENTS.md", "requirements.txt"):
        shutil.copy2(ROOT / file, output / "source" / file)
    handoff = ROOT / "docs/handoff"
    if handoff.is_dir():
        shutil.copytree(handoff, output / "source/docs/handoff")
    (output / "reports").mkdir()
    if validation is not None:
        shutil.copy2(validation, output / "reports/validation.json")
        if validation.with_suffix(".md").is_file():
            shutil.copy2(validation.with_suffix(".md"), output / "reports/validation.md")
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_name": output.name,
        "source_baseline": "final_volume_inputs_enterprise_regen_mix_dptah_20260919",
        "source_contract_aligned": True,
        "current_python_compatible": "requires_current_cross_language_validation",
        "windows_runtime_validation": "not_evaluated_by_this_packager",
        "strict_migration_evaluated": report is not None,
        "strict_migration_passed": report is not None and report["passed"] is True,
        "engineering_acceptance": "not_evaluated",
        "offline_nuget_dependencies_included": False,
        "source_hashes": {str(p.relative_to(output / "source")): sha256(p) for p in sorted((output / "source").rglob("*")) if p.is_file()},
        "build_input": str(build.resolve()),
        "model_contract_sha256": sha256(models / "model_contract.json"),
    }
    (output / "package_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {p.relative_to(output).as_posix(): sha256(p) for p in files}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with ZipFile(archive, "x", ZIP_DEFLATED, compresslevel=6) as packed:
        for path in files:
            packed.write(path, output.name + "/" + path.relative_to(output).as_posix())
    with ZipFile(archive) as packed:
        if packed.testzip() is not None:
            raise ValueError("ZIP CRC 校验失败")
        for name, expected in manifest.items():
            if hashlib.sha256(packed.read(output.name + "/" + name)).hexdigest() != expected:
                raise ValueError(f"ZIP 内文件哈希不一致：{name}")
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return {"directory": str(output), "archive": str(archive), "sha256": sha256(archive), "files": len(files)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="本次构建的 verify、winforms、nuget 父目录")
    parser.add_argument("--models", type=Path, required=True, help="本次导出的 ONNX 与契约目录")
    parser.add_argument("--output", type=Path, required=True, help="尚不存在的交付目录，建议位于 results/deployment/")
    parser.add_argument("--validation-report", type=Path, help="可选的本次严格数值迁移 validation.json；失败报告也如实保留")
    args = parser.parse_args()
    result = assemble(args.build.resolve(), args.models.resolve(), args.output.resolve(), args.validation_report)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
