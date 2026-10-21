#!/usr/bin/env python3
"""创建可审计的纯源码目录；不复制 Git 历史、外部数据或运行产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT_FILES = {
    ".gitignore", "AGENTS.md", "CLAUDE.md", "README.md", "需求.md",
    "requirements.txt", "pytest.ini", "environment.yml",
}
SOURCE_ROOTS = (
    "training", "simulation", "deployment", "scripts", "docs",
    "archive/training", "archive/xf_rotor1_retrain",
)
EXTRA_FILES = (
    "archive/__init__.py", "archive/README.md", "artifacts/README.md",
    "results/README.md", "data/README.md",
)
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".ini", ".yaml", ".yml", ".json",
    ".cs", ".csproj", ".sln", ".props", ".targets", ".config", ".ps1",
    ".sh", ".cmd", ".bat",
}
EXCLUDED_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", "bin", "obj", "dist",
    "artifacts", "results", "reports", "checkpoints", "models", "publish",
    "packages", ".nuget", "node_modules",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
)


def source_files(root: Path) -> list[Path]:
    """只接受明确列出的源码区域及 UTF-8 文本扩展名。"""
    candidates = {root / name for name in ROOT_FILES | set(EXTRA_FILES)}
    for directory in SOURCE_ROOTS:
        base = root / directory
        if base.exists():
            candidates.update(base.rglob("*"))
    selected = []
    for path in sorted(candidates):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        explicit = rel.as_posix() in ROOT_FILES | set(EXTRA_FILES)
        if not explicit and any(part in EXCLUDED_DIRS or part.endswith('.egg-info') for part in rel.parts[:-1]):
            continue
        if not explicit and path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError(f"拒绝符号链接：{rel}")
        content = path.read_text(encoding="utf-8")
        if "\0" in content:
            raise ValueError(f"拒绝二进制内容：{rel}")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise ValueError(f"发现疑似凭据，停止上传：{rel}")
        selected.append(path)
    return selected


def prepare(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("输出目录不能是仓库本身或其父目录")
    for directory in SOURCE_ROOTS:
        if output.is_relative_to(root / directory):
            raise ValueError("输出目录不能位于源码目录内；建议 results/source_upload/<运行编号>/")
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，拒绝覆盖")
    files = source_files(root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in files:
        rel = path.relative_to(root)
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append({"path": rel.as_posix(), "bytes": target.stat().st_size,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    return {"schema_version": 1, "file_count": len(manifest),
            "total_bytes": sum(item["bytes"] for item in manifest), "files": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="新的纯源码目录")
    parser.add_argument("--manifest", type=Path, help="可选清单路径，放在输出目录之外")
    args = parser.parse_args()
    if args.manifest and args.manifest.resolve().is_relative_to(args.output.resolve()):
        parser.error("清单必须位于纯源码输出目录之外")
    try:
        report = prepare(args.root, args.output)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, f"源码快照失败：{exc}\n")
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"纯源码快照完成：{report['file_count']} 个文件，{report['total_bytes']} 字节；{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
