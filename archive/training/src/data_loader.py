"""Excel 数据读取与工作表扫描。

此模块负责：
- 列出 Excel 文件中的所有工作表；
- 为每个工作表收集字段名、字段顺序、行列规模、空值比例、推断的时间字段；
- 选择目标工作表并读取为 DataFrame，供后续字段画像与建模使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


# 用于推断时序排序字段的关键词
_TIME_KEYWORDS = (
    "时间", "时刻", "日期", "采样", "序号", "编号",
    "time", "timestamp", "date", "index", "step", "sample",
)


@dataclass
class SheetSummary:
    """单个工作表的扫描结果。"""

    sheet_name: str
    n_rows: int
    n_cols: int
    columns: List[str] = field(default_factory=list)
    null_ratio: Dict[str, float] = field(default_factory=dict)
    time_candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "sheet_name": self.sheet_name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": self.columns,
            "null_ratio": self.null_ratio,
            "time_candidates": self.time_candidates,
        }


def _detect_time_candidates(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    for col in df.columns:
        col_str = str(col).lower()
        if any(k.lower() in col_str for k in _TIME_KEYWORDS):
            candidates.append(str(col))
            continue
        # datetime 类型直接视为时间候选
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(str(col))
    return candidates


def list_sheets(excel_path: Path) -> List[str]:
    """列出 Excel 中的所有工作表名称。"""

    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")
    xls = pd.ExcelFile(excel_path)
    return list(xls.sheet_names)


def _read_sheet(
    excel_path: Path,
    sheet_name: str,
    *,
    header_row: int = 0,
    column_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """按统一的数据契约读取工作表并规范化列名。"""

    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    if column_aliases:
        df = df.rename(columns=dict(column_aliases))
    return df


def coerce_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """将声明为模型字段的文本数字转为数值，无法转换的内容记为 NaN。"""

    out = df.copy()
    for col in dict.fromkeys(str(c) for c in columns):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def scan_sheet(
    excel_path: Path,
    sheet_name: str,
    *,
    header_row: int = 0,
    column_aliases: Optional[Dict[str, str]] = None,
) -> SheetSummary:
    """扫描单个工作表，返回基础统计信息。"""

    df = _read_sheet(
        excel_path,
        sheet_name,
        header_row=header_row,
        column_aliases=column_aliases,
    )
    n_rows, n_cols = df.shape
    columns = [str(c) for c in df.columns]
    null_ratio = {
        str(col): float(df[col].isna().mean()) if n_rows > 0 else 0.0
        for col in df.columns
    }
    time_candidates = _detect_time_candidates(df)
    return SheetSummary(
        sheet_name=sheet_name,
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        columns=columns,
        null_ratio=null_ratio,
        time_candidates=time_candidates,
    )


def scan_workbook(
    excel_path: Path,
    *,
    header_row: int = 0,
    column_aliases: Optional[Dict[str, str]] = None,
) -> List[SheetSummary]:
    """扫描 Excel 中的所有工作表。"""

    summaries: List[SheetSummary] = []
    for sheet in list_sheets(excel_path):
        summaries.append(
            scan_sheet(
                excel_path,
                sheet,
                header_row=header_row,
                column_aliases=column_aliases,
            )
        )
    return summaries


def select_sheet(
    summaries: List[SheetSummary], requested: Optional[str]
) -> SheetSummary:
    """根据用户配置选择目标工作表。

    若指定了 requested，则按名称匹配；否则选择行数最多的工作表。
    """

    if not summaries:
        raise ValueError("Excel 文件中不存在任何工作表")

    if requested:
        for s in summaries:
            if s.sheet_name == requested:
                return s
        raise ValueError(
            f"指定的工作表 '{requested}' 不存在，可选: {[s.sheet_name for s in summaries]}"
        )

    return max(summaries, key=lambda s: s.n_rows)


def load_sheet(
    excel_path: Path,
    sheet_name: str,
    *,
    header_row: int = 0,
    column_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """读取目标工作表为 DataFrame。"""

    return _read_sheet(
        excel_path,
        sheet_name,
        header_row=header_row,
        column_aliases=column_aliases,
    )
