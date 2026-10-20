#!/usr/bin/env python3
"""Convert the two frozen schema-v2 YAML cases to exact JSON objects."""
from __future__ import annotations

import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "deployment" / "dotnet" / "examples"
TARGET.mkdir(parents=True, exist_ok=True)
for name in ("qy1070_case", "default_case"):
    source = ROOT / "simulation" / "configs" / f"{name}.yaml"
    target = TARGET / f"{name}.json"
    if target.exists():
        raise SystemExit(f"拒绝覆盖已有示例: {target}")
    target.write_text(json.dumps(yaml.safe_load(source.read_text(encoding="utf-8")), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
