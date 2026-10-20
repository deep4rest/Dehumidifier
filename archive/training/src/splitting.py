"""数据划分与划分报告。

支持三种划分方式：
- time   : 按时间顺序按比例切分；
- batch  : 按指定批次留出测试集，其余按时间比例划分训练/验证；
- rolling: 滚动验证（产生多组训练/验证窗口，目前作为索引列表返回）。

划分操作在“行级 DataFrame”上进行，避免训练集滑窗跨越测试集边界。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    """单次训练/验证/测试划分结果。"""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    method: str
    notes: List[str] = field(default_factory=list)
    rolling_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    group_columns: List[str] = field(default_factory=list)
    group_assignments: Dict[str, List[str]] = field(default_factory=dict)


def _group_key_frame(df: pd.DataFrame, group_columns: List[str]) -> pd.Series:
    """将多个几何字段编码为稳定、可审计的分组键。"""

    missing = [c for c in group_columns if c not in df.columns]
    if missing:
        raise ValueError(f"group 划分字段不存在: {missing}")

    def _key(row: pd.Series) -> str:
        values = [_to_jsonable(v) for v in row.tolist()]
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    return df[group_columns].apply(_key, axis=1)


def _slice_by_ratio(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("划分比例之和必须为正")
    tr = train_ratio / total
    vr = val_ratio / total
    n = len(df)
    n_train = int(n * tr)
    n_val = int(n * vr)
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train, val, test


def split_dataframe(
    df: pd.DataFrame,
    *,
    method: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_column: Optional[str],
    group_columns: Optional[List[str]] = None,
    test_groups: Optional[List[Any]] = None,
    rolling_folds: int = 5,
    seed: int = 42,
) -> SplitResult:
    """对已排序的 DataFrame 进行划分。"""

    if method == "time":
        train, val, test = _slice_by_ratio(df, train_ratio, val_ratio, test_ratio)
        return SplitResult(train, val, test, method="time")

    if method == "batch":
        if not group_column or group_column not in df.columns:
            raise ValueError("batch 划分需要提供有效的 group_column")
        test_groups = list(test_groups or [])
        if not test_groups:
            # 默认取最后若干个分组作为测试集
            unique_groups = list(pd.unique(df[group_column]))
            n_test = max(1, int(len(unique_groups) * test_ratio))
            test_groups = unique_groups[-n_test:]

        test_mask = df[group_column].isin(test_groups)
        rest = df.loc[~test_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)
        # 在剩余部分按时间比例切分训练/验证
        denom = train_ratio + val_ratio
        if denom <= 0:
            raise ValueError("train_ratio + val_ratio 必须为正")
        n_rest = len(rest)
        n_train = int(n_rest * (train_ratio / denom))
        train = rest.iloc[:n_train].reset_index(drop=True)
        val = rest.iloc[n_train:].reset_index(drop=True)
        notes = [f"按批次留出测试集，测试批次数量: {len(test_groups)}"]
        return SplitResult(train, val, test, method="batch", notes=notes)

    if method == "group":
        cols = list(group_columns or [])
        if not cols:
            if group_column:
                cols = [group_column]
            else:
                raise ValueError("group 划分需要提供 group_columns")
        keys = _group_key_frame(df, cols)
        unique_groups = sorted(str(v) for v in pd.unique(keys))
        if len(unique_groups) < 3:
            raise ValueError(
                f"group 划分至少需要 3 个设备组，当前仅 {len(unique_groups)} 个"
            )

        total = train_ratio + val_ratio + test_ratio
        if total <= 0:
            raise ValueError("划分比例之和必须为正")
        ratios = np.asarray(
            [train_ratio, val_ratio, test_ratio], dtype=np.float64
        )
        if np.any(ratios <= 0):
            raise ValueError("group 划分的 train/val/test 比例都必须为正")
        rng = np.random.default_rng(seed)
        shuffled = list(np.asarray(unique_groups, dtype=object)[rng.permutation(len(unique_groups))])
        raw_counts = len(shuffled) * ratios / ratios.sum()
        counts = np.floor(raw_counts).astype(int)
        counts = np.maximum(counts, 1)
        while counts.sum() < len(shuffled):
            candidates = raw_counts - counts
            counts[int(np.argmax(candidates))] += 1
        while counts.sum() > len(shuffled):
            candidates = counts - raw_counts
            candidates[counts <= 1] = -np.inf
            counts[int(np.argmax(candidates))] -= 1
        n_train, n_val, _n_test = (int(v) for v in counts)
        train_groups = set(shuffled[:n_train])
        val_groups = set(shuffled[n_train:n_train + n_val])
        test_group_set = set(shuffled[n_train + n_val:])

        key_str = keys.astype(str)
        train = df.loc[key_str.isin(train_groups)].reset_index(drop=True)
        val = df.loc[key_str.isin(val_groups)].reset_index(drop=True)
        test = df.loc[key_str.isin(test_group_set)].reset_index(drop=True)
        assignments = {
            "train": sorted(train_groups),
            "val": sorted(val_groups),
            "test": sorted(test_group_set),
        }
        notes = [
            f"按设备几何组合划分，seed={seed}，"
            f"train/val/test 组数={len(train_groups)}/{len(val_groups)}/{len(test_group_set)}"
        ]
        return SplitResult(
            train,
            val,
            test,
            method="group",
            notes=notes,
            group_columns=cols,
            group_assignments=assignments,
        )

    if method == "rolling":
        # 简化实现：先按 time 划分得到 train/val/test，再在 train 上生成 rolling fold 索引
        train, val, test = _slice_by_ratio(df, train_ratio, val_ratio, test_ratio)
        n = len(train)
        if rolling_folds < 2:
            rolling_folds = 2
        fold_size = n // (rolling_folds + 1)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(rolling_folds):
            tr_end = fold_size * (i + 1)
            vl_end = fold_size * (i + 2)
            folds.append(
                (np.arange(0, tr_end), np.arange(tr_end, vl_end))
            )
        notes = [f"启用滚动验证，折数: {rolling_folds}"]
        return SplitResult(
            train, val, test, method="rolling", notes=notes, rolling_folds=folds
        )

    raise ValueError(f"未知划分方式: {method}")


def _summarize_part(
    df: pd.DataFrame,
    name: str,
    output_cols: List[str],
    time_column: Optional[str],
    group_column: Optional[str],
    group_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"name": name, "n_rows": int(len(df))}
    if time_column and time_column in df.columns and len(df) > 0:
        summary["time_min"] = _to_jsonable(df[time_column].min())
        summary["time_max"] = _to_jsonable(df[time_column].max())
    if group_column and group_column in df.columns:
        summary["n_groups"] = int(df[group_column].nunique())
    elif group_columns:
        present = [c for c in group_columns if c in df.columns]
        if len(present) == len(group_columns):
            summary["n_groups"] = int(df[present].drop_duplicates().shape[0])
    if output_cols and len(df) > 0:
        stats = {}
        for col in output_cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                stats[col] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()) if len(s) > 1 else 0.0,
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
        summary["output_stats"] = stats
    return summary


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_split_report(
    split: SplitResult,
    *,
    output_cols: List[str],
    time_column: Optional[str],
    group_column: Optional[str],
    group_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """生成数据集划分报告，含外推风险提示。"""

    parts = [
        _summarize_part(
            split.train_df, "train", output_cols, time_column, group_column, group_columns
        ),
        _summarize_part(
            split.val_df, "val", output_cols, time_column, group_column, group_columns
        ),
        _summarize_part(
            split.test_df, "test", output_cols, time_column, group_column, group_columns
        ),
    ]
    warnings: List[str] = list(split.notes)

    # 外推风险检查：测试集输出范围超出训练集 5%
    train_stats = parts[0].get("output_stats", {})
    test_stats = parts[2].get("output_stats", {})
    for col, ts in test_stats.items():
        rs = train_stats.get(col)
        if not rs:
            continue
        span = max(rs["max"] - rs["min"], 1e-9)
        low_diff = (rs["min"] - ts["min"]) / span
        high_diff = (ts["max"] - rs["max"]) / span
        if low_diff > 0.05 or high_diff > 0.05:
            warnings.append(
                f"输出 '{col}' 测试集范围超出训练集（下界差 {low_diff:.2%}，上界差 {high_diff:.2%}），存在外推风险"
            )

    assignments = split.group_assignments
    train_groups = set(assignments.get("train", []))
    val_groups = set(assignments.get("val", []))
    test_groups = set(assignments.get("test", []))
    overlaps = {
        "train_val": sorted(train_groups & val_groups),
        "train_test": sorted(train_groups & test_groups),
        "val_test": sorted(val_groups & test_groups),
    }
    group_leakage = any(overlaps.values())
    if group_leakage:
        warnings.append(f"检测到设备组跨集合泄漏: {overlaps}")

    return {
        "method": split.method,
        "parts": parts,
        "warnings": warnings,
        "group_columns": list(split.group_columns),
        "group_assignments": assignments,
        "group_overlaps": overlaps,
        "group_leakage": group_leakage,
        "rolling_folds_count": (
            len(split.rolling_folds) if split.rolling_folds else 0
        ),
    }


def save_split_report(report: Dict[str, Any], target_dir: Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "split_report.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return target
