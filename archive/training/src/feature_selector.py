"""输入输出字段推荐与泄漏检查。

基于字段画像与配置中的关键词列表，推荐 LSTM 输入与输出字段。
同时检查潜在的泄漏风险，例如：
- 多个候选输入字段间的强相关性；
- 输出派生字段被同时纳入输入；
- 输入字段中存在与输出名称高度相似的字段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from archive.training.src.field_profiler import FieldProfile


# 输出派生字段关键词，用于交叉检查
_DERIVED_OUTPUT_TOKENS = (
    "出口", "处理后", "outlet", "露点", "效率", "性能",
)

# 高相关性阈值，用于提示泄漏风险
_HIGH_CORR_THRESHOLD = 0.95


@dataclass
class FeatureSelection:
    """特征选择结果。"""

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)  # 字段 -> 排除原因
    warnings: List[str] = field(default_factory=list)
    time_column: Optional[str] = None
    group_column: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "excluded": self.excluded,
            "warnings": self.warnings,
            "time_column": self.time_column,
            "group_column": self.group_column,
        }


def _looks_like_output(name: str) -> bool:
    lower = name.lower()
    return any(t.lower() in lower for t in _DERIVED_OUTPUT_TOKENS)


def select_features(
    profiles: List[FieldProfile],
    df: pd.DataFrame,
    *,
    explicit_inputs: Optional[List[str]] = None,
    explicit_outputs: Optional[List[str]] = None,
    explicit_time: Optional[str] = None,
    explicit_group: Optional[str] = None,
    missing_threshold: float = 0.3,
) -> FeatureSelection:
    """根据字段画像与配置选择输入输出字段。"""

    sel = FeatureSelection()
    profile_map = {p.name: p for p in profiles}

    # ---- 时间/分组字段 ----
    if explicit_time and explicit_time in profile_map:
        sel.time_column = explicit_time
    else:
        time_candidates = [p.name for p in profiles if p.role_hint == "time"]
        # 若没有明显的时间字段，使用 id 字段作为时序排序代理
        if not time_candidates:
            time_candidates = [p.name for p in profiles if p.role_hint == "id"]
        sel.time_column = time_candidates[0] if time_candidates else None

    if explicit_group and explicit_group in profile_map:
        sel.group_column = explicit_group
    else:
        group_candidates = [p.name for p in profiles if p.role_hint == "id"]
        sel.group_column = None
        for cand in group_candidates:
            # 若每个分组的行数过少（极端情况下每组仅 1 行），分组无意义，跳过
            try:
                avg_rows = len(df) / max(df[cand].nunique(), 1)
            except Exception:
                avg_rows = 0
            if avg_rows >= 2:
                sel.group_column = cand
                break
        if not sel.group_column and group_candidates:
            sel.warnings.append(
                f"分组候选字段 '{group_candidates[0]}' 平均每组样本不足 2，已忽略分组"
            )

    # ---- 输出字段 ----
    if explicit_outputs:
        for name in explicit_outputs:
            if name in profile_map:
                sel.outputs.append(name)
            else:
                sel.warnings.append(f"显式输出字段 '{name}' 不在数据中，已忽略")
    else:
        sel.outputs = [p.name for p in profiles if p.candidate_output and p.is_numeric]

    if not sel.outputs:
        sel.warnings.append(
            "未识别到任何输出字段，请在配置中显式指定 output_columns"
        )

    # ---- 输入字段 ----
    if explicit_inputs:
        candidate_inputs = []
        for name in explicit_inputs:
            if name in profile_map:
                candidate_inputs.append(name)
            else:
                sel.warnings.append(f"显式输入字段 '{name}' 不在数据中，已忽略")
    else:
        candidate_inputs = [
            p.name for p in profiles if p.candidate_input and p.is_numeric
        ]

    output_set = set(sel.outputs)
    reserved = {sel.time_column, sel.group_column}
    explicit_set = set(explicit_inputs or [])

    final_inputs: List[str] = []
    for name in candidate_inputs:
        if name in output_set:
            sel.excluded[name] = "字段同时被识别为输出，已从输入中排除"
            continue
        if name in reserved:
            sel.excluded[name] = "字段被用作时间或分组列，不参与训练特征"
            continue
        prof = profile_map[name]
        if not prof.is_numeric:
            sel.excluded[name] = "非数值字段不能直接作为 LSTM 输入"
            continue
        if prof.missing_ratio > missing_threshold:
            sel.excluded[name] = (
                f"缺失比例 {prof.missing_ratio:.2%} 超过阈值 {missing_threshold:.0%}"
            )
            continue
        if prof.n_unique <= 1:
            if name in explicit_set:
                sel.warnings.append(
                    f"输入字段 '{name}' 唯一值仅 {prof.n_unique} 个（常量），"
                    "已按显式配置保留，但不会贡献信息"
                )
            else:
                sel.excluded[name] = "唯一值数量不足，作为常量字段排除"
                continue
        if _looks_like_output(name):
            sel.warnings.append(
                f"输入字段 '{name}' 名称含输出/派生关键词，存在泄漏风险，请确认"
            )
        final_inputs.append(name)

    sel.inputs = final_inputs

    if not sel.inputs:
        sel.warnings.append(
            "未识别到任何输入字段，请在配置中显式指定 input_columns"
        )

    # ---- 泄漏检查：输入与输出之间的强相关性 ----
    if sel.inputs and sel.outputs:
        try:
            sub = df[sel.inputs + sel.outputs].apply(pd.to_numeric, errors="coerce")
            corr = sub.corr().abs()
            for inp in sel.inputs:
                for out in sel.outputs:
                    if inp in corr.index and out in corr.columns:
                        c = corr.loc[inp, out]
                        if pd.notna(c) and c >= _HIGH_CORR_THRESHOLD:
                            sel.warnings.append(
                                f"输入 '{inp}' 与输出 '{out}' 相关系数 {c:.3f} 过高，"
                                "可能存在泄漏，请人工确认"
                            )
        except Exception as exc:  # pragma: no cover - 安全兜底
            sel.warnings.append(f"相关性检查失败: {exc}")

    return sel


def save_feature_selection(sel: FeatureSelection, target_dir: Path) -> Path:
    """保存特征选择结果为 JSON。"""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "feature_selection.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(sel.to_dict(), f, ensure_ascii=False, indent=2)
    return target
