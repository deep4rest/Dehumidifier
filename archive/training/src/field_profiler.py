"""字段画像与字段字典生成。

为每个字段统计：
- 推断的数据类型；
- 缺失比例；
- 数值范围（min/max/mean/std）与唯一值数量；
- 是否可作为时序排序字段、分组字段；
- 是否为输入候选、输出候选、待确认字段。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# 用于识别 ID/分组字段的关键词
_ID_KEYWORDS = ("id", "实验id", "批次", "batch", "trial", "case")
_TIME_KEYWORDS = (
    "时间", "时刻", "日期", "采样", "序号", "编号",
    "time", "timestamp", "date", "step", "sample",
)


@dataclass
class FieldProfile:
    """单个字段的画像信息。"""

    name: str
    dtype: str
    is_numeric: bool
    n_total: int
    n_missing: int
    missing_ratio: float
    n_unique: int
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    role_hint: str = "unknown"  # input / output / id / time / group / unknown
    candidate_input: bool = False
    candidate_output: bool = False
    needs_confirm: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _matches(name: str, keywords) -> bool:
    lower = name.lower()
    return any(k.lower() in lower for k in keywords)


def profile_field(
    series: pd.Series,
    name: str,
    input_keywords: List[str],
    output_keywords: List[str],
    missing_threshold: float = 0.3,
) -> FieldProfile:
    """对单个字段生成画像。"""

    n_total = int(len(series))
    n_missing = int(series.isna().sum())
    missing_ratio = float(n_missing / n_total) if n_total > 0 else 0.0
    n_unique = int(series.nunique(dropna=True))
    is_numeric = pd.api.types.is_numeric_dtype(series)

    profile = FieldProfile(
        name=name,
        dtype=str(series.dtype),
        is_numeric=bool(is_numeric),
        n_total=n_total,
        n_missing=n_missing,
        missing_ratio=missing_ratio,
        n_unique=n_unique,
    )

    if is_numeric and n_total > 0:
        non_null = series.dropna()
        if len(non_null) > 0:
            profile.min = float(non_null.min())
            profile.max = float(non_null.max())
            profile.mean = float(non_null.mean())
            profile.std = float(non_null.std()) if len(non_null) > 1 else 0.0

    # 角色推断：先识别 id/time，再识别输入/输出
    # 优先级说明：当字段同时匹配输入与输出关键词时（例如 "入口含湿量"），
    # 视为输入字段，因为入口/进口侧本质属于工况输入。
    if _matches(name, _ID_KEYWORDS):
        profile.role_hint = "id"
        profile.notes.append("名称含 ID/批次关键词，建议作为分组字段")
    elif _matches(name, _TIME_KEYWORDS) or pd.api.types.is_datetime64_any_dtype(series):
        profile.role_hint = "time"
        profile.notes.append("识别为时序排序候选字段")
    elif _matches(name, input_keywords):
        profile.role_hint = "input"
        profile.candidate_input = True
        if _matches(name, output_keywords):
            profile.notes.append("名称同时含输入与输出关键词，按输入侧优先采纳")
    elif _matches(name, output_keywords):
        profile.role_hint = "output"
        profile.candidate_output = True
    else:
        if is_numeric:
            profile.needs_confirm = True
            profile.notes.append("名称未匹配输入/输出关键词，需要人工确认角色")

    if not is_numeric and profile.role_hint not in ("id", "time"):
        profile.notes.append("非数值类型，默认从训练特征中排除")
        profile.candidate_input = False
        profile.candidate_output = False
        profile.needs_confirm = True

    if missing_ratio > missing_threshold:
        profile.notes.append(
            f"缺失比例 {missing_ratio:.2%} 超过阈值 {missing_threshold:.0%}，标记为低可信"
        )

    if is_numeric and n_unique == 1:
        profile.notes.append("唯一值数量为 1，作为常量字段不参与训练")
        profile.candidate_input = False
        profile.candidate_output = False

    return profile


def build_field_dictionary(
    df: pd.DataFrame,
    input_keywords: List[str],
    output_keywords: List[str],
    missing_threshold: float = 0.3,
) -> List[FieldProfile]:
    """对 DataFrame 中的全部字段生成字段字典。"""

    profiles: List[FieldProfile] = []
    for col in df.columns:
        profiles.append(
            profile_field(
                df[col],
                str(col),
                input_keywords,
                output_keywords,
                missing_threshold=missing_threshold,
            )
        )
    return profiles


def save_field_dictionary(
    profiles: List[FieldProfile], target_dir: Path
) -> Dict[str, Path]:
    """将字段字典保存为 CSV 与 JSON 两种格式，方便后续浏览与代码引用。"""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    records = [p.to_dict() for p in profiles]
    csv_path = target_dir / "field_dictionary.csv"
    json_path = target_dir / "field_dictionary.json"

    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return {"csv": csv_path, "json": json_path}
