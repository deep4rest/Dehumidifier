"""KAN 训练前字段与数据质量评估模块。

本模块只做数据读取与分析，不修改任何原始数据或训练配置。

主要能力：
- 字段存在性校验，并对缺失字段给出相似候选（去除空格/单位/括号差异等）；
- 遗漏候选字段与角色冲突分析；
- 字段数值统计（缺失、非数值、min/max/mean/median/std/分位数）；
- 基于 IQR 与 Z-score 的异常值检测与高风险标记；
- 缺失/异常的预处理建议；
- 归一化策略建议（针对 KAN）；
- 结构化报告生成（Markdown + JSON）。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class FieldStat:
    """单个字段的数值统计结果。"""

    name: str
    role: str  # input / output
    actual_name: Optional[str]  # 数据中实际匹配到的字段名
    matched: bool
    n_total: int = 0
    n_missing: int = 0
    missing_ratio: float = 0.0
    n_non_numeric: int = 0
    non_numeric_ratio: float = 0.0
    non_numeric_examples: List[str] = field(default_factory=list)
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    iqr_outlier_count: int = 0
    iqr_outlier_ratio: float = 0.0
    iqr_lower: Optional[float] = None
    iqr_upper: Optional[float] = None
    iqr_outlier_examples: List[float] = field(default_factory=list)
    zscore_outlier_count: int = 0
    zscore_outlier_ratio: float = 0.0
    high_risk: bool = False
    risk_reasons: List[str] = field(default_factory=list)
    preprocess_suggestion: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExistenceCheck:
    """字段存在性检查结果。"""

    requested: str
    role: str  # input / output
    matched: bool
    actual_name: Optional[str]
    similar_candidates: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MissingCandidate:
    """未被选择但可能与建模相关的字段。"""

    column: str
    reason: str
    role_conflict: bool = False
    suggested_role: str = "unknown"  # input / output / unknown

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 字段名归一化与相似度
# ---------------------------------------------------------------------------


def _strip_unit_and_spaces(name: str) -> str:
    """去除字段名中的单位、空格、换行、括号差异，得到归一化的简化名。"""

    if name is None:
        return ""
    s = str(name)
    # 去除换行符
    s = s.replace("\r", "").replace("\n", "")
    # 中文括号 -> 英文括号
    s = s.replace("（", "(").replace("）", ")")
    # 去除括号及其内容（通常是单位）
    s = re.sub(r"\([^)]*\)", "", s)
    # 去除已知的常见单位后缀（防止有些字段没有用括号包裹单位）
    s = re.sub(r"(g/kg|kg/kg|m/s|rpm|℃|°c|°|%|kw|w|pa|kpa)", "", s, flags=re.IGNORECASE)
    # 去除空白字符
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _build_actual_index(actual_columns: List[str]) -> Dict[str, str]:
    """构建归一化键 -> 原字段名 的索引。"""

    index: Dict[str, str] = {}
    for col in actual_columns:
        key = _strip_unit_and_spaces(col)
        # 仅在键非空且尚未被记录时写入，避免重复列名导致冲突
        if key and key not in index:
            index[key] = col
    return index


def _find_similar_candidates(
    requested: str, actual_columns: List[str], top_n: int = 3
) -> List[str]:
    """为未匹配字段挑选相似候选，依据归一化后字符串的 difflib 相似度。"""

    norm_requested = _strip_unit_and_spaces(requested)
    norm_to_col: Dict[str, str] = {}
    for col in actual_columns:
        norm_to_col[_strip_unit_and_spaces(col)] = col

    candidates = list(norm_to_col.keys())
    matches = get_close_matches(norm_requested, candidates, n=top_n, cutoff=0.5)
    return [norm_to_col[m] for m in matches if m in norm_to_col]


# ---------------------------------------------------------------------------
# 字段存在性校验
# ---------------------------------------------------------------------------


def check_existence(
    requested_inputs: List[str],
    requested_outputs: List[str],
    actual_columns: List[str],
) -> List[ExistenceCheck]:
    """检查请求的输入输出字段是否存在于实际数据列中。

    匹配优先级：完全匹配 > 归一化匹配（去除单位/空格/换行/括号差异）。
    若都不匹配则给出相似字段候选，但不静默替换。
    """

    actual_set = set(actual_columns)
    actual_index = _build_actual_index(actual_columns)
    results: List[ExistenceCheck] = []

    def _check(req: str, role: str) -> ExistenceCheck:
        if req in actual_set:
            return ExistenceCheck(
                requested=req, role=role, matched=True, actual_name=req,
                note="精确匹配",
            )
        norm = _strip_unit_and_spaces(req)
        if norm in actual_index:
            actual = actual_index[norm]
            return ExistenceCheck(
                requested=req, role=role, matched=True, actual_name=actual,
                note=f"归一化匹配（去除单位/空格/换行差异），实际字段名='{actual}'",
            )
        # 未匹配，提供相似候选
        similar = _find_similar_candidates(req, actual_columns)
        return ExistenceCheck(
            requested=req, role=role, matched=False, actual_name=None,
            similar_candidates=similar,
            note="未在数据中匹配到字段，请确认字段名是否正确",
        )

    for r in requested_inputs:
        results.append(_check(r, "input"))
    for r in requested_outputs:
        results.append(_check(r, "output"))
    return results


# ---------------------------------------------------------------------------
# 遗漏候选字段与角色冲突
# ---------------------------------------------------------------------------


_INPUT_HINTS = (
    "入口", "进口", "再生角度", "冷吹角度", "再生温度", "转速", "风速",
    "环境", "设定", "inlet", "regen", "speed",
)
_OUTPUT_HINTS = (
    "出口", "处理后", "露点", "含湿量", "焓", "效率", "power", "功率", "outlet",
)


def detect_missing_candidates(
    actual_columns: List[str],
    matched_inputs: List[str],
    matched_outputs: List[str],
    df: pd.DataFrame,
) -> List[MissingCandidate]:
    """识别未被选择但可能与建模相关的字段。

    规则：
    - 已匹配的字段不再视为遗漏；
    - 字段名包含输入/输出语义关键词且为数值列，则提示为候选；
    - 若同时命中输入与输出语义，标记角色冲突。
    """

    selected = set(matched_inputs) | set(matched_outputs)
    candidates: List[MissingCandidate] = []
    for col in actual_columns:
        if col in selected:
            continue
        # ID 字段直接跳过
        lower = str(col).lower()
        if "id" in lower or "实验" in str(col) or "批次" in str(col):
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) if col in df.columns else False
        hits_input = any(h in str(col).lower() for h in [k.lower() for k in _INPUT_HINTS])
        hits_output = any(h in str(col).lower() for h in [k.lower() for k in _OUTPUT_HINTS])

        if not (hits_input or hits_output):
            continue

        suggested = "unknown"
        if hits_input and not hits_output:
            suggested = "input"
        elif hits_output and not hits_input:
            suggested = "output"

        reason_parts = []
        if hits_input:
            reason_parts.append("含输入语义关键词")
        if hits_output:
            reason_parts.append("含输出语义关键词")
        if is_numeric:
            reason_parts.append("为数值列")
        else:
            reason_parts.append("非数值列，需要确认")

        candidates.append(MissingCandidate(
            column=col,
            reason="；".join(reason_parts),
            role_conflict=hits_input and hits_output,
            suggested_role=suggested,
        ))
    return candidates


# ---------------------------------------------------------------------------
# 字段统计与异常检测
# ---------------------------------------------------------------------------


def _coerce_numeric_with_examples(
    series: pd.Series, max_examples: int = 5
) -> Tuple[pd.Series, int, List[str]]:
    """尝试将一列转换为数值，返回数值序列、非数值条目数、示例值。

    若原列已是数值则不再尝试转换。
    """

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), 0, []

    coerced = pd.to_numeric(series, errors="coerce")
    # 非数值: 原始非空但转换后为 NaN
    raw_not_null = series.notna()
    coerced_is_nan = coerced.isna()
    bad_mask = raw_not_null & coerced_is_nan
    examples = (
        series[bad_mask].astype(str).head(max_examples).tolist()
        if bad_mask.any() else []
    )
    return coerced, int(bad_mask.sum()), examples


def compute_field_stat(
    series: pd.Series,
    requested_name: str,
    role: str,
    actual_name: str,
    missing_threshold: float = 0.3,
    outlier_ratio_threshold: float = 0.1,
    iqr_factor: float = 1.5,
    zscore_threshold: float = 3.0,
) -> FieldStat:
    """对单个已匹配字段计算统计、异常与风险标记。"""

    n_total = int(len(series))
    coerced, n_non_numeric, non_numeric_examples = _coerce_numeric_with_examples(series)
    n_missing = int(coerced.isna().sum())
    missing_ratio = float(n_missing / n_total) if n_total > 0 else 0.0

    stat = FieldStat(
        name=requested_name,
        role=role,
        actual_name=actual_name,
        matched=True,
        n_total=n_total,
        n_missing=n_missing,
        missing_ratio=missing_ratio,
        n_non_numeric=n_non_numeric,
        non_numeric_ratio=float(n_non_numeric / n_total) if n_total > 0 else 0.0,
        non_numeric_examples=non_numeric_examples,
    )

    valid = coerced.dropna()
    if len(valid) == 0:
        stat.high_risk = True
        stat.risk_reasons.append("无有效数值样本")
        return stat

    stat.min = float(valid.min())
    stat.max = float(valid.max())
    stat.mean = float(valid.mean())
    stat.median = float(valid.median())
    stat.std = float(valid.std()) if len(valid) > 1 else 0.0
    stat.q25 = float(valid.quantile(0.25))
    stat.q75 = float(valid.quantile(0.75))

    # IQR 异常
    iqr = stat.q75 - stat.q25
    lower = stat.q25 - iqr_factor * iqr
    upper = stat.q75 + iqr_factor * iqr
    iqr_mask = (valid < lower) | (valid > upper)
    stat.iqr_lower = float(lower)
    stat.iqr_upper = float(upper)
    stat.iqr_outlier_count = int(iqr_mask.sum())
    stat.iqr_outlier_ratio = float(stat.iqr_outlier_count / len(valid))
    if stat.iqr_outlier_count > 0:
        stat.iqr_outlier_examples = [
            float(v) for v in valid[iqr_mask].head(5).tolist()
        ]

    # Z-score 异常
    if stat.std and stat.std > 0:
        z = (valid - stat.mean) / stat.std
        z_mask = z.abs() > zscore_threshold
        stat.zscore_outlier_count = int(z_mask.sum())
        stat.zscore_outlier_ratio = float(stat.zscore_outlier_count / len(valid))

    # 风险标记
    if missing_ratio > missing_threshold:
        stat.high_risk = True
        stat.risk_reasons.append(
            f"缺失率 {missing_ratio:.2%} 超过阈值 {missing_threshold:.0%}"
        )
    if stat.iqr_outlier_ratio > outlier_ratio_threshold:
        stat.high_risk = True
        stat.risk_reasons.append(
            f"IQR 异常率 {stat.iqr_outlier_ratio:.2%} 超过阈值 {outlier_ratio_threshold:.0%}"
        )
    if n_non_numeric > 0:
        stat.high_risk = True
        stat.risk_reasons.append(
            f"存在 {n_non_numeric} 个非数值样本，需要人工确认或预处理"
        )

    return stat


# ---------------------------------------------------------------------------
# 预处理建议
# ---------------------------------------------------------------------------


def build_preprocess_suggestions(
    stat: FieldStat,
    missing_threshold: float = 0.3,
    outlier_ratio_threshold: float = 0.1,
) -> List[str]:
    """基于统计结果生成预处理建议（文本列表）。"""

    suggestions: List[str] = []

    # 缺失值
    if stat.missing_ratio == 0.0 and stat.n_non_numeric == 0:
        suggestions.append("无缺失，无需填充")
    elif stat.missing_ratio > missing_threshold:
        if stat.role == "output":
            suggestions.append(
                f"输出字段缺失率过高（{stat.missing_ratio:.2%}），建议人工确认是否可用，"
                f"或考虑剔除该输出"
            )
        else:
            suggestions.append(
                f"输入字段缺失率过高（{stat.missing_ratio:.2%}），建议人工确认或从训练特征中剔除"
            )
    elif stat.missing_ratio > 0:
        if stat.role == "output":
            suggestions.append(
                f"输出字段存在少量缺失（{stat.missing_ratio:.2%}），建议直接剔除缺失样本，"
                f"避免对目标变量进行插值"
            )
        else:
            suggestions.append(
                f"输入字段存在少量缺失（{stat.missing_ratio:.2%}），可使用线性插值、"
                f"前后向填充或中位数填充"
            )

    # 非数值
    if stat.n_non_numeric > 0:
        suggestions.append(
            f"存在 {stat.n_non_numeric} 个无法转换为数值的样本（示例: {stat.non_numeric_examples}），"
            f"建议人工核对原始数据后清洗"
        )

    # 异常值
    if stat.iqr_outlier_count > 0:
        ratio = stat.iqr_outlier_ratio
        if ratio < 0.01:
            action = "可考虑保留或剔除少量样本"
        elif ratio < outlier_ratio_threshold:
            action = "建议使用缩尾（winsorize）或截尾处理"
        else:
            action = "异常比例较高，建议人工核对是否为真实工况，再决定保留或剔除"
        suggestions.append(
            f"IQR 异常 {stat.iqr_outlier_count} 条（{ratio:.2%}），范围 "
            f"[{stat.iqr_lower:.4g}, {stat.iqr_upper:.4g}]：{action}"
        )

    # 物理意义提示
    if stat.role == "output" and (stat.iqr_outlier_count > 0 or stat.n_non_numeric > 0):
        suggestions.append(
            "目标变量的清洗策略可能影响物理意义，必须由用户确认后再执行"
        )

    return suggestions


# ---------------------------------------------------------------------------
# 归一化建议
# ---------------------------------------------------------------------------


def build_normalization_suggestion(
    input_stats: List[FieldStat],
    output_stats: List[FieldStat],
    grid_range: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, Any]:
    """对比尺度并给出归一化建议。"""

    def _scales(stats: List[FieldStat]) -> List[Tuple[str, float, float]]:
        rows = []
        for s in stats:
            if s.min is None or s.max is None:
                continue
            rows.append((s.actual_name or s.name, s.min, s.max))
        return rows

    def _scale_diff(rows: List[Tuple[str, float, float]]) -> float:
        ranges = [abs(mx - mn) for _, mn, mx in rows if mx is not None and mn is not None]
        ranges = [r for r in ranges if r > 0]
        if not ranges:
            return 0.0
        return max(ranges) / max(min(ranges), 1e-12)

    in_rows = _scales(input_stats)
    out_rows = _scales(output_stats)
    in_diff = _scale_diff(in_rows)
    out_diff = _scale_diff(out_rows)

    suggestion: Dict[str, Any] = {
        "input_scale_diff_ratio": float(in_diff),
        "output_scale_diff_ratio": float(out_diff),
        "input_recommendation": "",
        "output_recommendation": "",
        "fit_policy": "缩放器仅基于训练集拟合，再应用到验证/测试集，避免数据泄漏",
        "grid_alignment": (
            f"KAN 默认 grid_range={list(grid_range)}，建议输入归一化到该区间以保证 B-spline 覆盖"
        ),
    }

    # 输入：尺度差异显著就建议 minmax 到 grid_range
    if in_diff >= 10.0:
        suggestion["input_recommendation"] = (
            f"输入字段尺度差异显著（最大/最小区间比 ≈ {in_diff:.1f}），"
            f"强烈建议使用 MinMaxScaler 归一化到 {list(grid_range)}"
        )
    elif in_diff >= 3.0:
        suggestion["input_recommendation"] = (
            f"输入字段尺度差异中等（≈ {in_diff:.1f}），建议使用 MinMaxScaler 归一化到 "
            f"{list(grid_range)} 或 StandardScaler 标准化"
        )
    else:
        suggestion["input_recommendation"] = (
            f"输入字段尺度差异较小（≈ {in_diff:.1f}），可不归一化；"
            f"但 KAN 训练仍推荐归一化到 {list(grid_range)} 以与 grid_range 对齐"
        )

    # 输出：建议 MinMax 到 [0,1] 或 Standard，并强调要保存缩放器以便反归一化
    if out_diff >= 3.0:
        suggestion["output_recommendation"] = (
            f"输出字段尺度差异 ≈ {out_diff:.1f}，建议对每个输出独立做缩放（MinMax 或 Standard），"
            f"必须保存缩放器并在预测后反归一化以回到原始物理尺度"
        )
    else:
        suggestion["output_recommendation"] = (
            f"输出字段尺度差异 ≈ {out_diff:.1f}，可选择是否缩放；若缩放则必须保存缩放器并支持反归一化"
        )

    suggestion["input_fields"] = [
        {"name": n, "min": mn, "max": mx} for n, mn, mx in in_rows
    ]
    suggestion["output_fields"] = [
        {"name": n, "min": mn, "max": mx} for n, mn, mx in out_rows
    ]
    return suggestion


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


def _existence_section_md(checks: List[ExistenceCheck]) -> str:
    lines = ["## 1. 字段存在性检查", ""]
    inputs = [c for c in checks if c.role == "input"]
    outputs = [c for c in checks if c.role == "output"]

    def _block(title: str, items: List[ExistenceCheck]) -> List[str]:
        rows = [f"### {title}", "", "| 请求字段 | 是否匹配 | 实际字段 | 说明 | 相似候选 |",
                "|---|---|---|---|---|"]
        for c in items:
            actual = c.actual_name or "-"
            similar = ", ".join(c.similar_candidates) if c.similar_candidates else "-"
            ok = "✅" if c.matched else "❌"
            rows.append(
                f"| {c.requested} | {ok} | {actual} | {c.note} | {similar} |"
            )
        rows.append("")
        return rows

    lines += _block("输入字段", inputs)
    lines += _block("输出字段", outputs)

    n_unmatched = sum(1 for c in checks if not c.matched)
    if n_unmatched == 0:
        lines.append("**结论：** 所有请求字段均匹配成功（含归一化匹配）。")
    else:
        lines.append(
            f"**结论：** 存在 {n_unmatched} 个未匹配字段，属于阻止训练的高风险问题，"
            f"必须在训练前确认字段名。"
        )
    lines.append("")
    return "\n".join(lines)


def _missing_candidates_section_md(
    cands: List[MissingCandidate], all_columns: List[str],
    matched_inputs: List[str], matched_outputs: List[str],
) -> str:
    selected = set(matched_inputs) | set(matched_outputs)
    lines = ["## 2. 遗漏候选字段与全字段一览", ""]
    lines.append("### 2.1 全部字段角色映射")
    lines.append("")
    lines.append("| 字段 | 当前角色 |")
    lines.append("|---|---|")
    for col in all_columns:
        if col in matched_inputs:
            role = "输入"
        elif col in matched_outputs:
            role = "输出"
        else:
            role = "未选择"
        lines.append(f"| {col} | {role} |")
    lines.append("")

    lines.append("### 2.2 可能遗漏的候选字段")
    lines.append("")
    if not cands:
        lines.append("无显著的遗漏候选字段。")
    else:
        lines.append("| 字段 | 建议角色 | 角色冲突 | 说明 |")
        lines.append("|---|---|---|---|")
        for c in cands:
            conflict = "是" if c.role_conflict else "否"
            lines.append(f"| {c.column} | {c.suggested_role} | {conflict} | {c.reason} |")
        lines.append("")
        lines.append("**说明：** 系统仅生成建议，不会自动修改训练字段配置，请由用户最终确认。")
    lines.append("")
    return "\n".join(lines)


def _stats_section_md(stats: List[FieldStat]) -> str:
    lines = ["## 3. 字段数值统计与异常检测", ""]
    lines.append("| 字段 | 角色 | 样本数 | 缺失数 | 缺失率 | 非数值 | 最小 | 最大 | 均值 | 中位数 | 标准差 | IQR异常 | IQR异常率 | Z>3 异常 | 高风险 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in stats:
        lines.append(
            f"| {s.actual_name or s.name} | {s.role} | {s.n_total} | {s.n_missing} | "
            f"{s.missing_ratio:.2%} | {s.n_non_numeric} | "
            f"{_fmt(s.min)} | {_fmt(s.max)} | {_fmt(s.mean)} | {_fmt(s.median)} | "
            f"{_fmt(s.std)} | {s.iqr_outlier_count} | {s.iqr_outlier_ratio:.2%} | "
            f"{s.zscore_outlier_count} | {'⚠️ 是' if s.high_risk else '否'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if abs(x) >= 1000 or abs(x) < 0.001:
        return f"{x:.4g}"
    return f"{x:.4f}"


def _preprocess_section_md(stats: List[FieldStat]) -> str:
    lines = ["## 4. 预处理建议", ""]
    for s in stats:
        title = f"### {s.actual_name or s.name}（{s.role}）"
        lines.append(title)
        lines.append("")
        if not s.preprocess_suggestion:
            lines.append("- 无特殊处理建议")
        else:
            for sg in s.preprocess_suggestion:
                lines.append(f"- {sg}")
        if s.risk_reasons:
            lines.append(f"- ⚠️ 风险提示：{'；'.join(s.risk_reasons)}")
        lines.append("")
    return "\n".join(lines)


def _normalization_section_md(norm: Dict[str, Any]) -> str:
    lines = ["## 5. 归一化与标准化建议", ""]
    lines.append(f"- 输入字段尺度差异比：{norm['input_scale_diff_ratio']:.2f}")
    lines.append(f"- 输出字段尺度差异比：{norm['output_scale_diff_ratio']:.2f}")
    lines.append(f"- 输入建议：{norm['input_recommendation']}")
    lines.append(f"- 输出建议：{norm['output_recommendation']}")
    lines.append(f"- 拟合策略：{norm['fit_policy']}")
    lines.append(f"- 与 KAN 网格对齐：{norm['grid_alignment']}")
    lines.append("")
    lines.append("### 5.1 输入字段量级")
    lines.append("")
    lines.append("| 字段 | 最小值 | 最大值 |")
    lines.append("|---|---|---|")
    for f_ in norm["input_fields"]:
        lines.append(f"| {f_['name']} | {_fmt(f_['min'])} | {_fmt(f_['max'])} |")
    lines.append("")
    lines.append("### 5.2 输出字段量级")
    lines.append("")
    lines.append("| 字段 | 最小值 | 最大值 |")
    lines.append("|---|---|---|")
    for f_ in norm["output_fields"]:
        lines.append(f"| {f_['name']} | {_fmt(f_['min'])} | {_fmt(f_['max'])} |")
    lines.append("")
    return "\n".join(lines)


def _conclusion_section_md(
    checks: List[ExistenceCheck], stats: List[FieldStat]
) -> str:
    lines = ["## 6. 结论", ""]
    blockers: List[str] = []
    n_unmatched = sum(1 for c in checks if not c.matched)
    if n_unmatched > 0:
        blockers.append(f"{n_unmatched} 个字段未匹配，必须先修正字段名")
    high_risk = [s for s in stats if s.high_risk]
    if high_risk:
        names = ", ".join(s.actual_name or s.name for s in high_risk)
        blockers.append(f"{len(high_risk)} 个字段被标记为高风险：{names}")

    if not blockers:
        lines.append("✅ **可以进入 KAN 训练阶段。** 所有字段匹配成功且无高风险问题。")
    else:
        lines.append("❌ **暂不建议直接进入训练阶段。** 需要先处理以下问题：")
        for b in blockers:
            lines.append(f"- {b}")
    lines.append("")
    lines.append("> 本报告仅作为建议，不会自动修改训练配置或原始数据。")
    return "\n".join(lines)


def render_markdown_report(
    excel_path: Path,
    sheet_name: str,
    requested_inputs: List[str],
    requested_outputs: List[str],
    actual_columns: List[str],
    checks: List[ExistenceCheck],
    candidates: List[MissingCandidate],
    stats: List[FieldStat],
    normalization: Dict[str, Any],
) -> str:
    """生成 Markdown 报告字符串。"""

    matched_inputs = [c.actual_name for c in checks if c.role == "input" and c.matched and c.actual_name]
    matched_outputs = [c.actual_name for c in checks if c.role == "output" and c.matched and c.actual_name]
    header = [
        "# KAN 训练前字段与数据质量评估报告",
        "",
        f"- 数据文件：`{excel_path}`",
        f"- 工作表：`{sheet_name}`",
        f"- 请求输入字段（{len(requested_inputs)}）：{requested_inputs}",
        f"- 请求输出字段（{len(requested_outputs)}）：{requested_outputs}",
        f"- 数据列总数：{len(actual_columns)}",
        "",
    ]
    parts = [
        "\n".join(header),
        _existence_section_md(checks),
        _missing_candidates_section_md(candidates, actual_columns, matched_inputs, matched_outputs),
        _stats_section_md(stats),
        _preprocess_section_md(stats),
        _normalization_section_md(normalization),
        _conclusion_section_md(checks, stats),
    ]
    return "\n".join(parts)


def build_machine_readable_report(
    excel_path: Path,
    sheet_name: str,
    requested_inputs: List[str],
    requested_outputs: List[str],
    actual_columns: List[str],
    checks: List[ExistenceCheck],
    candidates: List[MissingCandidate],
    stats: List[FieldStat],
    normalization: Dict[str, Any],
) -> Dict[str, Any]:
    """生成机器可读的 JSON 数据结构。"""

    n_unmatched = sum(1 for c in checks if not c.matched)
    high_risk = [s for s in stats if s.high_risk]
    can_train = (n_unmatched == 0) and (len(high_risk) == 0)

    return {
        "excel_path": str(excel_path),
        "sheet_name": sheet_name,
        "actual_columns": actual_columns,
        "requested_inputs": requested_inputs,
        "requested_outputs": requested_outputs,
        "existence_checks": [c.to_dict() for c in checks],
        "missing_candidates": [c.to_dict() for c in candidates],
        "field_stats": [s.to_dict() for s in stats],
        "normalization": normalization,
        "summary": {
            "n_unmatched_fields": n_unmatched,
            "n_high_risk_fields": len(high_risk),
            "high_risk_fields": [s.actual_name or s.name for s in high_risk],
            "can_proceed_to_training": can_train,
        },
    }


# ---------------------------------------------------------------------------
# 一键运行入口
# ---------------------------------------------------------------------------


@dataclass
class AssessmentArtifacts:
    """评估流程产物路径集合。"""

    markdown_path: Path
    json_path: Path
    csv_path: Path


def run_assessment(
    df: pd.DataFrame,
    excel_path: Path,
    sheet_name: str,
    requested_inputs: List[str],
    requested_outputs: List[str],
    output_dir: Path,
    missing_threshold: float = 0.3,
    outlier_ratio_threshold: float = 0.1,
    iqr_factor: float = 1.5,
    zscore_threshold: float = 3.0,
    grid_range: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[Dict[str, Any], AssessmentArtifacts]:
    """执行完整的字段与数据质量评估并落盘报告。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_columns = [str(c) for c in df.columns]
    checks = check_existence(requested_inputs, requested_outputs, actual_columns)

    # 已匹配的字段 -> 计算统计与建议
    stats: List[FieldStat] = []
    matched_inputs: List[str] = []
    matched_outputs: List[str] = []
    for c in checks:
        if not c.matched or not c.actual_name:
            stats.append(FieldStat(
                name=c.requested, role=c.role, actual_name=None, matched=False,
                risk_reasons=["字段未匹配"], high_risk=True,
            ))
            continue
        series = df[c.actual_name]
        s = compute_field_stat(
            series, c.requested, c.role, c.actual_name,
            missing_threshold=missing_threshold,
            outlier_ratio_threshold=outlier_ratio_threshold,
            iqr_factor=iqr_factor, zscore_threshold=zscore_threshold,
        )
        s.preprocess_suggestion = build_preprocess_suggestions(
            s, missing_threshold=missing_threshold,
            outlier_ratio_threshold=outlier_ratio_threshold,
        )
        stats.append(s)
        if c.role == "input":
            matched_inputs.append(c.actual_name)
        else:
            matched_outputs.append(c.actual_name)

    candidates = detect_missing_candidates(
        actual_columns, matched_inputs, matched_outputs, df
    )

    input_stats = [s for s in stats if s.role == "input" and s.matched]
    output_stats = [s for s in stats if s.role == "output" and s.matched]
    normalization = build_normalization_suggestion(input_stats, output_stats, grid_range=grid_range)

    md = render_markdown_report(
        excel_path, sheet_name, requested_inputs, requested_outputs,
        actual_columns, checks, candidates, stats, normalization,
    )
    machine = build_machine_readable_report(
        excel_path, sheet_name, requested_inputs, requested_outputs,
        actual_columns, checks, candidates, stats, normalization,
    )

    md_path = output_dir / "field_quality_report.md"
    json_path = output_dir / "field_quality_report.json"
    csv_path = output_dir / "field_quality_stats.csv"

    md_path.write_text(md, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2, default=_json_default)
    pd.DataFrame([s.to_dict() for s in stats]).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    return machine, AssessmentArtifacts(
        markdown_path=md_path, json_path=json_path, csv_path=csv_path,
    )


def _json_default(o: Any) -> Any:
    """numpy 数据类型 -> 原生 Python 类型，避免 JSON 序列化失败。"""

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"不可序列化的对象类型：{type(o)}")
