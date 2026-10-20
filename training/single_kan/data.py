"""双转轮单 KAN 的唯一数据入口。

数据入口只读权威 XLS 的原始 bytes，先校验 SHA-256，再解析技术表头和中文
说明行。所有训练字段在这里归一为中文 canonical 名称；源技术键和说明行
映射通过 ``DataFrame.attrs`` 保留下来用于审计。原始 workbook 永远不在本
模块中写回，也不执行插值或通用 IQR 删除。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    AUDIT_COLUMNS,
    ALLOWED_GEOMETRY_GROUPS,
    CANONICAL_LABELS,
    DIRECT_OUTPUT_COLUMNS,
    EXPECTED_DATA_ROWS,
    EXPECTED_SOURCE_SHA256 as _CONFIG_SOURCE_HASHES,
    FIXED_XF_FOLD_1_GROUPS,
    FIXED_XF_OUTER_GROUP,
    ID_COLUMN,
    INPUT_COLUMNS_BY_PROFILE,
    MAIN_INPUT_COLUMNS,
    PROFILE_CONFIGS,
    SOURCE_AUDIT_COLUMNS,
    SOURCE_PATHS,
    SOURCE_TO_CANONICAL_BY_PROFILE,
    XF_INPUT_COLUMNS,
    get_profile,
)

__all__ = [
    "DataContractError",
    "PreparedDataset",
    "EXPECTED_SOURCE_SHA256",
    "MAIN_MODEL_INPUT_COLUMNS",
    "MODEL_INPUT_COLUMNS_BY_PROFILE",
    "MODEL_OUTPUT_COLUMNS_BY_PROFILE",
    "XF_MODEL_INPUT_COLUMNS",
    "clean_dataset",
    "make_model_arrays",
    "normalize_frame",
    "prepare_profile_dataset",
    "prepare_sample",
    "read_authoritative_dataset",
    "split_dataset",
    "validate_dataset",
]


# Exposed as a module value intentionally: audit/tests can monkeypatch this map
# to prove that a source-hash mismatch is a hard failure.
EXPECTED_SOURCE_SHA256 = dict(_CONFIG_SOURCE_HASHES)
MODEL_INPUT_COLUMNS_BY_PROFILE = INPUT_COLUMNS_BY_PROFILE
MODEL_OUTPUT_COLUMNS_BY_PROFILE = {
    "xf": DIRECT_OUTPUT_COLUMNS,
    "main": DIRECT_OUTPUT_COLUMNS,
}
XF_MODEL_INPUT_COLUMNS = XF_INPUT_COLUMNS
MAIN_MODEL_INPUT_COLUMNS = MAIN_INPUT_COLUMNS


class DataContractError(ValueError):
    """数据源、字段或几何划分违反锁定契约。"""


@dataclass(frozen=True)
class PreparedDataset:
    """同一 source snapshot 经验证、清洗和分组后的离线快照。"""

    profile: str
    raw: pd.DataFrame
    clean: pd.DataFrame
    assignments: pd.DataFrame
    source_sha256: str | None
    header_mapping: Mapping[str, str]
    cleaning_report: Mapping[str, Any]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _profile_name(profile: str) -> str:
    return get_profile(profile).name


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def _description_mapping(frame: pd.DataFrame) -> dict[str, str]:
    """从技术表头与第一条说明记录建立稳定的 source->中文映射。"""

    if frame.empty:
        return {}
    first = frame.iloc[0]
    mapping: dict[str, str] = {}
    for column in frame.columns:
        value = first.get(column)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        mapping[str(column).strip()] = str(value).strip()
    return mapping


def _read_excel_snapshot(content: bytes, *, sheet_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        technical = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=0)
    except Exception as exc:  # pragma: no cover - pandas/xlrd boundary
        raise DataContractError(f"无法解析工作表 {sheet_name}: {exc}") from exc
    technical.columns = [str(column).strip() for column in technical.columns]
    if technical.empty:
        raise DataContractError(f"工作表 {sheet_name} 为空")
    mapping = _description_mapping(technical)
    if "ID" not in technical.columns:
        raise DataContractError("技术表头缺少 ID 字段")
    # 第二行是中文说明；若 ID 能转成数字，通常意味着表头行错位。
    first_id = pd.to_numeric(pd.Series([technical.iloc[0]["ID"]]), errors="coerce")
    if first_id.notna().any():
        raise DataContractError("预期第二行为中文说明，但 ID 可被解析为数值")
    data = technical.iloc[1:].reset_index(drop=True)
    return data, mapping


def read_authoritative_dataset(
    profile: str,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """读取并归一一个权威 XLS。

    ``path`` 仅用于显式传入同一权威路径；任何临时复制或替代文件均拒绝，
    从而避免“看起来可复现、实际已经换源”的训练。
    """

    cfg = get_profile(profile)
    expected_path = SOURCE_PATHS[cfg.name].resolve()
    selected = expected_path if path is None else Path(path).resolve()
    if selected != expected_path:
        raise DataContractError(
            f"{cfg.name} 数据入口固定为权威文件 {expected_path}，拒绝替代文件: {selected}"
        )
    if not selected.is_file():
        raise DataContractError(f"权威数据文件不存在: {selected}")
    try:
        content = selected.read_bytes()
    except OSError as exc:
        raise DataContractError(f"权威数据文件读取失败: {selected}: {exc}") from exc
    actual = _sha256(content)
    expected = EXPECTED_SOURCE_SHA256[cfg.name]
    if actual != expected:
        raise DataContractError(
            f"{cfg.name} 数据源 SHA-256 与锁定契约不一致: expected={expected}, actual={actual}"
        )
    data, description = _read_excel_snapshot(content, sheet_name=cfg.sheet_name)
    normalized = normalize_frame(cfg.name, data, header_mapping=description)
    normalized.attrs.update(
        {
            "profile": cfg.name,
            "source_path": str(selected),
            "source_sha256": actual,
            "source_size_bytes": len(content),
            "header_mapping": description,
            "expected_row_count": EXPECTED_DATA_ROWS[cfg.name],
        }
    )
    if len(normalized) != EXPECTED_DATA_ROWS[cfg.name]:
        raise DataContractError(
            f"{cfg.name} 有效行数与锁定契约不一致: expected={EXPECTED_DATA_ROWS[cfg.name]}, actual={len(normalized)}"
        )
    return normalized


def normalize_frame(
    profile: str,
    frame: pd.DataFrame,
    *,
    header_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """把技术字段或 canonical 字段归一到统一中文字段。

    不删除未知列，便于审计；模型数组只会从 profile 的 canonical 输入和六
    个直接输出中选列。main 数据中技术字段 ``Label_React_Inlet_AbsHum`` 的
    中文说明是“冷处出口含湿量”，因此被映射为 ``冷吹出口含湿量``。
    """

    cfg = get_profile(profile)
    if not isinstance(frame, pd.DataFrame):
        raise DataContractError("数据必须是 pandas.DataFrame")
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    mapping = SOURCE_TO_CANONICAL_BY_PROFILE[cfg.name]
    for source, canonical in mapping.items():
        if source in result.columns and canonical in result.columns and source != canonical:
            source_values = pd.to_numeric(result[source], errors="coerce").to_numpy()
            canonical_values = pd.to_numeric(result[canonical], errors="coerce").to_numpy()
            if source_values.shape != canonical_values.shape or not np.allclose(
                source_values, canonical_values, equal_nan=True
            ):
                raise DataContractError(
                    f"技术字段 {source} 与 canonical 字段 {canonical} 数值不一致"
                )
        elif canonical not in result.columns and source in result.columns:
            result[canonical] = result[source]

    # 常见中文别名只在 canonical 缺失时使用，不覆盖稳定键已有值。
    aliases = {
        "实验ID": "ID",
        "转轮厚度(mm)": "转轮厚度",
        "转轮直径(mm)": "转轮直径",
        "处理入口温度": "处理入口温度(℃)",
        "再生入口风量(m³/h)": "再生风量",
        "再生风量(m³/h)": "再生风量",
        "再生入口含湿量(g/kg)": "再生入口含湿量",
        "再生出口含湿量(g/kg)": "再生出口含湿量",
        "冷处出口含湿量": "冷吹出口含湿量",
        "处理出口露点(℃)": "处理出口露点",
    }
    for alias, canonical in aliases.items():
        if canonical not in result.columns and alias in result.columns:
            result[canonical] = result[alias]

    # Preserve source mapping for report writers. attrs are not used for model
    # values and survive ordinary DataFrame copies.
    result.attrs["header_mapping"] = dict(header_mapping or result.attrs.get("header_mapping", {}))
    result.attrs["profile"] = cfg.name
    result.attrs["source_to_canonical"] = dict(mapping)
    return result


def _required_columns(profile: str) -> tuple[str, ...]:
    cfg = get_profile(profile)
    return tuple(dict.fromkeys((ID_COLUMN, *cfg.input_columns, *cfg.direct_outputs, *cfg.audit_columns)))


def validate_dataset(profile: str, frame: pd.DataFrame) -> pd.DataFrame:
    """验证字段、类型、缺失/非有限值和重复 ID，并返回 canonical 副本。"""

    cfg = get_profile(profile)
    normalized = normalize_frame(cfg.name, frame)
    required = _required_columns(cfg.name)
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise DataContractError(f"{cfg.name} 缺少必需字段: {missing}")

    converted = normalized.copy()
    numeric_columns = list(required)
    for column in numeric_columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    values = converted[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad_columns = [
            column
            for index, column in enumerate(numeric_columns)
            if not np.isfinite(values[:, index]).all()
        ]
        raise DataContractError(f"模型或审计字段存在缺失/非数字/非有限值: {bad_columns}")

    ids = converted[ID_COLUMN].to_numpy(dtype=np.float64)
    if not np.equal(ids, np.floor(ids)).all():
        raise DataContractError("ID 必须为有限整数")
    converted[ID_COLUMN] = ids.astype(np.int64)
    duplicated = converted.loc[converted[ID_COLUMN].duplicated(False), ID_COLUMN].unique()
    if len(duplicated):
        shown = [int(value) for value in sorted(duplicated.tolist())[:20]]
        raise DataContractError(f"ID 重复: {shown}")

    for column in ("转轮厚度", "转轮直径"):
        geometry = converted[column].to_numpy(dtype=np.float64)
        if not np.equal(geometry, np.floor(geometry)).all():
            raise DataContractError(f"几何字段 {column} 必须为整数")
        converted[column] = geometry.astype(np.int64)
    geometry_groups = set(
        zip(converted["转轮厚度"].astype(int), converted["转轮直径"].astype(int))
    )
    illegal_groups = sorted(geometry_groups - set(ALLOWED_GEOMETRY_GROUPS[cfg.name]))
    if illegal_groups:
        raise DataContractError(f"{cfg.name} 存在非法厚度×直径几何组: {illegal_groups[:20]}")
    converted.attrs.update(normalized.attrs)
    converted.attrs["validated_profile"] = cfg.name
    converted.attrs["input_columns"] = list(cfg.input_columns)
    converted.attrs["direct_outputs"] = list(cfg.direct_outputs)
    converted.attrs["audit_columns"] = list(cfg.audit_columns)
    converted.attrs["expected_row_count"] = EXPECTED_DATA_ROWS[cfg.name]
    return converted


def clean_dataset(profile: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只执行契约验证和 XF 已知异常 ID 1407 的审计排除。"""

    cfg = get_profile(profile)
    converted = validate_dataset(cfg.name, frame)
    excluded_ids: list[int] = []
    clean = converted
    if cfg.name == "xf":
        mask = converted[ID_COLUMN].eq(1407)
        if bool(mask.any()):
            excluded_ids = [1407]
            clean = converted.loc[~mask].copy().reset_index(drop=True)
    clean.attrs.update(converted.attrs)
    report: dict[str, Any] = {
        "profile": cfg.name,
        "source_sha256": converted.attrs.get("source_sha256"),
        "raw_row_count": int(len(converted)),
        "clean_row_count": int(len(clean)),
        "excluded_ids": excluded_ids,
        "interpolation": False,
        "iqr_filtering": False,
        "source_workbook_modified": False,
        "audit_columns": list(cfg.audit_columns),
    }
    return clean, report


def _assignment_rows(
    frame: pd.DataFrame,
    *,
    group_to_partition: Mapping[tuple[int, int], str],
    validation_groups: Sequence[tuple[int, int]] = (),
) -> pd.DataFrame:
    rows = frame[[ID_COLUMN, "转轮厚度", "转轮直径"]].copy()
    rows["rotor_thickness_mm"] = pd.to_numeric(rows["转轮厚度"], errors="raise").astype(np.int64)
    rows["rotor_diameter_mm"] = pd.to_numeric(rows["转轮直径"], errors="raise").astype(np.int64)
    rows["geometry_group"] = list(zip(rows["rotor_thickness_mm"], rows["rotor_diameter_mm"]))
    rows["partition"] = [
        group_to_partition[(int(depth), int(diameter))]
        for depth, diameter in zip(rows["rotor_thickness_mm"], rows["rotor_diameter_mm"])
    ]
    rows["split"] = rows["partition"]
    validation_set = set(validation_groups)
    rows["validation_fold"] = [
        "fold_1" if (int(depth), int(diameter)) in validation_set else None
        for depth, diameter in zip(rows["rotor_thickness_mm"], rows["rotor_diameter_mm"])
    ]
    return rows[[
        ID_COLUMN,
        "rotor_thickness_mm",
        "rotor_diameter_mm",
        "geometry_group",
        "partition",
        "split",
        "validation_fold",
    ]]


def _main_geometry_partitions(frame: pd.DataFrame) -> dict[tuple[int, int], str]:
    groups = sorted(
        {
            (int(depth), int(diameter))
            for depth, diameter in zip(frame["转轮厚度"], frame["转轮直径"])
        }
    )
    if len(groups) < 3:
        raise DataContractError("主转轮几何组少于 3 组，无法执行 70/15/15 隔离划分")
    n_groups = len(groups)
    n_train = max(1, int(round(n_groups * 0.70)))
    n_val = max(1, int(round(n_groups * 0.15)))
    n_test = n_groups - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1
    if n_train + n_val + n_test != n_groups or min(n_train, n_val, n_test) < 1:
        raise DataContractError("主转轮几何组无法形成非空 70/15/15 划分")
    return {
        group: (
            "train" if index < n_train else
            "validation" if index < n_train + n_val else "test"
        )
        for index, group in enumerate(groups)
    }


def split_dataset(profile: str, frame: pd.DataFrame) -> pd.DataFrame:
    """构造不跨几何组的固定划分。"""

    cfg = get_profile(profile)
    normalized = validate_dataset(cfg.name, frame)
    groups = {
        (int(depth), int(diameter))
        for depth, diameter in zip(normalized["转轮厚度"], normalized["转轮直径"])
    }
    if cfg.name == "xf":
        if FIXED_XF_OUTER_GROUP not in groups:
            raise DataContractError(f"XF 数据缺少固定测试几何组 {FIXED_XF_OUTER_GROUP}")
        group_to_partition = {
            group: (
                "test" if group == FIXED_XF_OUTER_GROUP else
                "validation" if group in set(FIXED_XF_FOLD_1_GROUPS) else "train"
            )
            for group in groups
        }
        assignments = _assignment_rows(
            normalized,
            group_to_partition=group_to_partition,
            validation_groups=FIXED_XF_FOLD_1_GROUPS,
        )
    else:
        assignments = _assignment_rows(
            normalized,
            group_to_partition=_main_geometry_partitions(normalized),
        )
    if assignments[ID_COLUMN].duplicated().any():
        raise DataContractError("划分结果 ID 重复")
    assignments.attrs.update(
        {
            "profile": cfg.name,
            "split_policy": cfg.split_policy,
            "source_sha256": normalized.attrs.get("source_sha256"),
        }
    )
    return assignments


def make_model_arrays(
    profile: str,
    frame: pd.DataFrame,
    assignments: pd.DataFrame | None = None,
    split: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """按 canonical 契约提取训练矩阵与仅审计矩阵。

    返回的第三项只包含 ``power`` 和 Excel 露点，明确不会被送入模型、
    scaler、loss 或 checkpoint。
    """

    cfg = get_profile(profile)
    normalized = validate_dataset(cfg.name, frame)
    selected = normalized
    if assignments is not None:
        assignment = assignments.copy()
        if ID_COLUMN not in assignment.columns or "partition" not in assignment.columns:
            raise DataContractError("划分表必须包含 ID 和 partition")
        if assignment[ID_COLUMN].duplicated().any():
            raise DataContractError("划分表 ID 重复")
        if split is not None:
            key = {"val": "validation", "valid": "validation", "outer_test": "test"}.get(
                str(split), str(split)
            )
            assignment = assignment.loc[assignment["partition"].eq(key)]
        selected = normalized.merge(
            assignment[[ID_COLUMN]], on=ID_COLUMN, how="inner", validate="one_to_one"
        )
    if selected.empty:
        raise DataContractError(f"{cfg.name} split={split!r} 为空")
    x = selected[list(cfg.input_columns)].to_numpy(dtype=np.float32)
    y = selected[list(cfg.direct_outputs)].to_numpy(dtype=np.float32)
    audit = selected[list(cfg.audit_columns)].copy()
    if x.ndim != 2 or x.shape[1] != cfg.in_features:
        raise DataContractError(f"{cfg.name} 输入张量必须为 [N,{cfg.in_features}]")
    if y.ndim != 2 or y.shape[1] != cfg.out_features:
        raise DataContractError(f"{cfg.name} 目标张量必须为 [N,{cfg.out_features}]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise DataContractError("模型张量存在非有限值")
    if set(audit.columns) != set(AUDIT_COLUMNS):
        raise DataContractError("审计字段必须且只能包含 power 与处理出口露点")
    return x, y, audit


def prepare_profile_dataset(profile: str) -> PreparedDataset:
    """读取、校验、清洗和划分一个 profile（不训练）。"""

    cfg = get_profile(profile)
    raw = read_authoritative_dataset(cfg.name)
    clean, report = clean_dataset(cfg.name, raw)
    assignments = split_dataset(cfg.name, clean)
    return PreparedDataset(
        profile=cfg.name,
        raw=raw,
        clean=clean,
        assignments=assignments,
        source_sha256=str(raw.attrs.get("source_sha256")),
        header_mapping=dict(raw.attrs.get("header_mapping", {})),
        cleaning_report=report,
    )


def prepare_sample(profile: str, n: int = 500) -> pd.DataFrame:
    """从权威源构造只读的小样本副本，覆盖几何组和每组边界值。"""

    if int(n) <= 0:
        raise DataContractError("小样本行数必须为正")
    cfg = get_profile(profile)
    raw = read_authoritative_dataset(cfg.name)
    clean, report = clean_dataset(cfg.name, raw)
    group_columns = ["转轮厚度", "转轮直径"]
    pieces: list[pd.DataFrame] = []
    for _, group in clean.groupby(group_columns, sort=True, dropna=False):
        pieces.append(group.head(1))
        if len(group) > 1:
            pieces.append(group.tail(1))
    boundary = pd.concat(pieces, ignore_index=False).drop_duplicates(subset=[ID_COLUMN])
    remaining = clean.loc[~clean[ID_COLUMN].isin(boundary[ID_COLUMN])]
    need = max(0, int(n) - len(boundary))
    if need:
        sampled = remaining.sample(n=min(need, len(remaining)), random_state=42)
        sample = pd.concat([boundary, sampled], ignore_index=True)
    else:
        sample = boundary.head(int(n)).copy()
    sample = sample.head(int(n)).reset_index(drop=True)
    sample.attrs.update(clean.attrs)
    sample.attrs["sample_size"] = len(sample)
    sample.attrs["cleaning_report"] = report
    sample.attrs["is_copy"] = True
    return sample
