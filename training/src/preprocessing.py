"""数据清洗、异常处理与缩放器封装。

提供：
- 非数值清洗与缺失值处理；
- IQR / Z-Score 异常值标记；
- 仅使用训练集拟合的标准化/归一化器；
- 缩放器与预处理配置的保存与加载。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class OutputTransformer:
    """输出处理管道：可选 log1p 变换 + sklearn 缩放器。

    处理顺序：
    - ``transform(arr)``: 先对 ``log_columns`` 指定的列做 ``log1p``，再调用内部 sklearn 缩放器。
    - ``inverse_transform(arr)``: 先调用 sklearn ``inverse_transform``，再对 log 列做 ``expm1``。

    外部接口保持与 sklearn scaler 一致（``fit`` / ``transform`` / ``inverse_transform``），
    以避免侵入评估与训练代码。``log_columns`` 接受列名列表，在 ``fit`` 时转为索引。
    """

    def __init__(
        self,
        scaler: Any,
        output_cols: List[str],
        log_columns: Optional[List[str]] = None,
    ) -> None:
        self.scaler = scaler
        self.output_cols = list(output_cols)
        self.log_columns = [c for c in (log_columns or []) if c in self.output_cols]
        self._log_indices = [self.output_cols.index(c) for c in self.log_columns]

    # ---- log1p 变换辅助 -----------------------------------------
    def _apply_log(self, arr: np.ndarray) -> np.ndarray:
        if not self._log_indices:
            return arr
        out = arr.astype(np.float64, copy=True)
        for idx in self._log_indices:
            col = out[..., idx]
            if np.any(col < 0):
                raise ValueError(
                    f"输出列 '{self.output_cols[idx]}' 含负值，无法进行 log1p 变换"
                )
            out[..., idx] = np.log1p(col)
        return out.astype(np.float32)

    def _apply_expm1(self, arr: np.ndarray) -> np.ndarray:
        if not self._log_indices:
            return arr
        out = arr.astype(np.float64, copy=True)
        for idx in self._log_indices:
            out[..., idx] = np.expm1(out[..., idx])
        return out.astype(np.float32)

    # ---- sklearn 兼容接口 -----------------------------------------
    def fit(self, arr: np.ndarray) -> "OutputTransformer":
        self.scaler.fit(self._apply_log(arr))
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self._apply_log(arr))

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        unscaled = self.scaler.inverse_transform(arr)
        return self._apply_expm1(unscaled)


@dataclass
class CleaningReport:
    """记录清洗过程产生的统计信息。"""

    rows_before: int = 0
    rows_after: int = 0
    missing_filled: Dict[str, int] = field(default_factory=dict)
    outliers_flagged: Dict[str, int] = field(default_factory=dict)
    dropped_rows: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "missing_filled": self.missing_filled,
            "outliers_flagged": self.outliers_flagged,
            "dropped_rows": self.dropped_rows,
            "notes": self.notes,
        }


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """将指定列强制转为数值类型，无法转换的位置变为 NaN。"""

    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fill_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str,
    report: CleaningReport,
) -> pd.DataFrame:
    """按指定策略填充或删除缺失值。"""

    out = df.copy()
    if strategy == "drop":
        before = len(out)
        out = out.dropna(subset=columns).reset_index(drop=True)
        report.dropped_rows += before - len(out)
        report.notes.append(f"按策略 drop 删除含缺失行 {before - len(out)} 行")
        return out

    for col in columns:
        if col not in out.columns:
            continue
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            continue
        if strategy == "ffill":
            out[col] = out[col].ffill().bfill()
        elif strategy == "linear":
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
            # 兜底：仍存在的空值用列均值填充
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].mean())
        elif strategy == "mean":
            out[col] = out[col].fillna(out[col].mean())
        elif strategy == "median":
            out[col] = out[col].fillna(out[col].median())
        else:
            raise ValueError(f"未知缺失值填充策略: {strategy}")
        report.missing_filled[col] = n_missing
    return out


def detect_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str,
    factor: float,
    report: CleaningReport,
) -> pd.DataFrame:
    """异常值标记（不删除原始数据，只在 report 中记录）。"""

    if method == "none":
        return df
    out = df.copy()
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        series = out[col]
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            low = q1 - factor * iqr
            high = q3 + factor * iqr
            mask = (series < low) | (series > high)
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0 or pd.isna(std):
                continue
            mask = ((series - mean).abs() / std) > factor
        else:
            raise ValueError(f"未知异常值检测方法: {method}")
        n_flag = int(mask.sum())
        if n_flag > 0:
            report.outliers_flagged[col] = n_flag
    return out


def clean_dataframe(
    df: pd.DataFrame,
    columns: List[str],
    *,
    missing_strategy: str = "linear",
    outlier_method: str = "iqr",
    outlier_factor: float = 3.0,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """整合清洗流程：数值化 -> 缺失填充 -> 异常标记。"""

    report = CleaningReport(rows_before=len(df))
    cleaned = coerce_numeric(df, columns)
    cleaned = fill_missing(cleaned, columns, missing_strategy, report)
    cleaned = detect_outliers(
        cleaned, columns, outlier_method, outlier_factor, report
    )
    report.rows_after = len(cleaned)
    return cleaned, report


def build_scaler(kind: str, feature_range: Optional[Tuple[float, float]] = None):
    """根据配置创建 sklearn 缩放器。

    Args:
        kind: ``standard`` 或 ``minmax``。
        feature_range: 仅在 ``minmax`` 时生效的目标区间，默认 ``(0, 1)``。
    """

    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        if feature_range is None:
            return MinMaxScaler()
        lo, hi = float(feature_range[0]), float(feature_range[1])
        return MinMaxScaler(feature_range=(lo, hi))
    raise ValueError(f"未知缩放器类型: {kind}")


def fit_scalers(
    train_df: pd.DataFrame,
    input_cols: List[str],
    output_cols: List[str],
    kind: str,
    feature_range: Optional[Tuple[float, float]] = None,
    log_transform_outputs: Optional[List[str]] = None,
) -> Tuple[Any, Any]:
    """仅使用训练集拟合输入与输出缩放器。

    如果提供 ``log_transform_outputs``，输出缩放器会被包装为 ``OutputTransformer``，
    在拟合 / transform 时对指定列依次做 log1p 变换，在 inverse_transform 时 expm1 还原。
    """

    x_scaler = build_scaler(kind, feature_range=feature_range)
    base_y_scaler = build_scaler(kind, feature_range=feature_range)
    x_scaler.fit(train_df[input_cols].to_numpy(dtype=np.float32))

    log_cols = [c for c in (log_transform_outputs or []) if c in output_cols]
    y_scaler: Any = OutputTransformer(
        base_y_scaler, output_cols=output_cols, log_columns=log_cols,
    )
    y_scaler.fit(train_df[output_cols].to_numpy(dtype=np.float32))
    return x_scaler, y_scaler


def transform_split(
    df: pd.DataFrame,
    input_cols: List[str],
    output_cols: List[str],
    x_scaler: Any,
    y_scaler: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """将一个数据划分应用缩放器。"""

    x = x_scaler.transform(df[input_cols].to_numpy(dtype=np.float32))
    y = y_scaler.transform(df[output_cols].to_numpy(dtype=np.float32))
    return x.astype(np.float32), y.astype(np.float32)


def save_scalers(
    x_scaler: Any,
    y_scaler: Any,
    input_cols: List[str],
    output_cols: List[str],
    target_dir: Path,
    input_domain: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Path]:
    """保存缩放器与对应字段顺序。"""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = target_dir / "scalers.pkl"
    meta_path = target_dir / "scaler_meta.json"
    with open(pkl_path, "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)
    log_cols: List[str] = []
    if isinstance(y_scaler, OutputTransformer):
        log_cols = list(y_scaler.log_columns)

    def _describe_scaler(scaler: Any) -> Dict[str, Any]:
        inner = scaler.scaler if isinstance(scaler, OutputTransformer) else scaler
        kind = type(inner).__name__
        info: Dict[str, Any] = {"kind": kind}
        fr = getattr(inner, "feature_range", None)
        if fr is not None:
            info["feature_range"] = [float(fr[0]), float(fr[1])]
        return info

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_cols": input_cols,
                "output_cols": output_cols,
                "log_transform_outputs": log_cols,
                "x_scaler": _describe_scaler(x_scaler),
                "y_scaler": _describe_scaler(y_scaler),
                "input_domain": input_domain or {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {"pickle": pkl_path, "meta": meta_path}


def save_cleaning_report(report: CleaningReport, target_dir: Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "cleaning_report.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    return target
