"""训练数据来源与字段范围清单。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_domain(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            continue
        result[col] = {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "n_unique": int(finite.nunique()),
            "n_missing": int(values.isna().sum()),
        }
    return result


def build_dataset_manifest(
    source_path: Path,
    *,
    sheet_name: str,
    header_row: int,
    column_aliases: Dict[str, str],
    df: pd.DataFrame,
    input_columns: List[str],
    output_columns: List[str],
    group_columns: List[str],
) -> Dict:
    """生成不修改原始工作簿的数据来源与质量清单。"""

    source_path = Path(source_path)
    group_count = 0
    if group_columns and all(c in df.columns for c in group_columns):
        group_count = int(df[group_columns].drop_duplicates().shape[0])
    return {
        "source_path": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "source_size_bytes": int(source_path.stat().st_size),
        "sheet_name": sheet_name,
        "header_row_zero_based": int(header_row),
        "column_aliases": dict(column_aliases),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "input_columns": list(input_columns),
        "output_columns": list(output_columns),
        "group_columns": list(group_columns),
        "group_count": group_count,
        "input_domain": _numeric_domain(df, input_columns),
        "output_domain": _numeric_domain(df, output_columns),
    }


def save_dataset_manifest(manifest: Dict, target_dir: Path) -> Path:
    target = Path(target_dir) / "dataset_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return target
