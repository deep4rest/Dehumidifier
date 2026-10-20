"""XF.xls 的唯一、严格且可审计的数据入口。

本模块不插值、不执行 IQR 删除、不改写原始 XLS。任何契约偏离均立即失败。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from archive.training.src.data_loader import coerce_numeric_columns

from .config import (
    AUDIT_NUMERIC_COLUMNS,
    EXCLUDED_ID,
    EXPECTED_CLEAN_ROWS,
    EXPECTED_RAW_ROWS,
    EXPECTED_SOURCE_SHA256,
    FIXED_FOLD_GROUPS,
    MODEL_INPUT_COLUMNS,
    MODEL_INPUT_LABELS,
    MODEL_INPUT_RUNTIME_FIELDS,
    MODEL_OUTPUT_COLUMNS,
    MODEL_OUTPUT_LABELS,
    MODEL_OUTPUT_RUNTIME_FIELDS,
    OUTER_GROUP,
    REPORTS_DIR,
    REPOSITORY_ROOT,
    SHEET_NAME,
    SOURCE_PATH,
)

__all__ = [
    "DataContractError",
    "XFDataset",
    "EXPECTED_CLEAN_ROWS",
    "EXPECTED_RAW_ROWS",
    "EXPECTED_SOURCE_SHA256",
    "FIXED_FOLD_GROUPS",
    "MODEL_INPUT_COLUMNS",
    "MODEL_INPUT_LABELS",
    "MODEL_OUTPUT_COLUMNS",
    "MODEL_OUTPUT_LABELS",
    "prepare_xf_dataset",
    "read_xf_source",
    "validate_and_clean_xf",
    "validate_report_snapshot",
]


class DataContractError(ValueError):
    """XF 数据与锁定契约不一致。"""


@dataclass(frozen=True)
class XFDataset:
    """数据、固定划分和已回写报告的同一快照。"""

    raw: pd.DataFrame
    clean: pd.DataFrame
    split_assignments: pd.DataFrame
    source_sha256: str
    dataset_contract: Mapping[str, Any]
    cleaning_report: Mapping[str, Any]
    input_domain: Mapping[str, Any]
    report_dir: Path


def _sha256_bytes(content: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(content)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _replace_file(source: Path, destination: Path) -> None:
    """单文件原子替换边界，便于发布故障注入测试。"""

    os.replace(source, destination)


def read_xf_source(source_path: Path = SOURCE_PATH) -> pd.DataFrame:
    """对同一 bytes 快照校验哈希并解析技术表头。"""

    path = Path(source_path)
    if path.resolve() != SOURCE_PATH.resolve():
        raise DataContractError(
            f"XF 数据入口固定为 {SOURCE_PATH}，拒绝替代文件: {path}"
        )
    if not path.is_file():
        raise DataContractError(f"XF 数据文件不存在: {path}")

    try:
        with path.open("rb") as stream:
            snapshot = stream.read()
    except OSError as exc:
        raise DataContractError(f"XF 数据文件读取失败: {path}: {exc}") from exc

    actual_hash = _sha256_bytes(snapshot)
    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise DataContractError(
            "XF.xls SHA-256 与锁定契约不一致: "
            f"expected={EXPECTED_SOURCE_SHA256}, actual={actual_hash}"
        )

    try:
        with_description = pd.read_excel(
            io.BytesIO(snapshot), sheet_name=SHEET_NAME, header=0
        )
    except Exception as exc:
        raise DataContractError(
            f"无法从已校验 bytes 快照解析工作表 {SHEET_NAME}: {exc}"
        ) from exc
    with_description.columns = [str(column).strip() for column in with_description.columns]
    if with_description.empty:
        raise DataContractError(f"工作表 {SHEET_NAME} 为空")
    if pd.to_numeric(
        pd.Series([with_description.iloc[0].get("ID")]), errors="coerce"
    ).notna().any():
        raise DataContractError("预期第二行为说明文字，但 ID 可被解析为数值")
    result = with_description.iloc[1:].reset_index(drop=True)
    result.attrs["source_sha256"] = actual_hash
    result.attrs["source_size_bytes"] = len(snapshot)
    return result


def _require_columns(raw: pd.DataFrame) -> tuple[str, ...]:
    required = tuple(
        dict.fromkeys(
            ("ID",)
            + MODEL_INPUT_COLUMNS
            + MODEL_OUTPUT_COLUMNS
            + AUDIT_NUMERIC_COLUMNS
        )
    )
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise DataContractError(f"缺少必需字段: {missing}")
    return required


def _coerce_and_validate_numeric(
    raw: pd.DataFrame, required: tuple[str, ...]
) -> pd.DataFrame:
    converted = coerce_numeric_columns(raw, required)
    model_columns = list(MODEL_INPUT_COLUMNS + MODEL_OUTPUT_COLUMNS)
    model_values = converted[model_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(model_values).all():
        bad_mask = ~np.isfinite(model_values)
        bad_columns = sorted(
            {model_columns[index] for index in np.where(bad_mask)[1].tolist()}
        )
        raise DataContractError(
            f"17 个模型字段存在非数字、NaN 或 Inf: {bad_columns}"
        )

    audit_values = converted[list(AUDIT_NUMERIC_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(audit_values).all():
        raise DataContractError("审计字段存在非数字、NaN 或 Inf")
    return converted


def _assert_constant(series: pd.Series, expected: float, label: str) -> None:
    values = set(float(value) for value in pd.unique(series))
    if values != {float(expected)}:
        raise DataContractError(f"{label}必须固定为 {expected}，实际为 {sorted(values)}")


def validate_and_clean_xf(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """验证全量契约，且唯一删除 ID 1407。"""

    required = _require_columns(raw)
    if len(raw) != EXPECTED_RAW_ROWS:
        raise DataContractError(
            f"XF 原始有效样本应为 {EXPECTED_RAW_ROWS}，实际为 {len(raw)}"
        )
    converted = _coerce_and_validate_numeric(raw, required)

    ids = converted["ID"].to_numpy(dtype=np.float64)
    if not np.equal(ids, np.floor(ids)).all():
        raise DataContractError("ID 必须是整数")
    converted["ID"] = converted["ID"].astype("int64")
    duplicate_ids = sorted(
        int(value) for value in converted.loc[converted["ID"].duplicated(False), "ID"].unique()
    )
    if duplicate_ids:
        raise DataContractError(f"ID 重复: {duplicate_ids[:20]}")

    excluded = converted.loc[converted["ID"].eq(EXCLUDED_ID)]
    if len(excluded) != 1:
        raise DataContractError(
            f"必须恰好存在一条待删除 ID {EXCLUDED_ID}，实际 {len(excluded)} 条"
        )

    process_temperature = converted["TextBox_Process_inlet_Temp"].to_numpy()
    purge_temperature = converted["TextBox_Purge_Inlet_Temp"].to_numpy()
    mismatch = ~np.equal(process_temperature, purge_temperature)
    if mismatch.any():
        mismatch_ids = converted.loc[mismatch, "ID"].head(20).tolist()
        raise DataContractError(
            f"处理入口温度与冷吹入口温度不一致，ID: {mismatch_ids}"
        )

    _assert_constant(converted["TextBox_Rotation_Speed"], 8, "转速(rpm)")
    _assert_constant(converted["Label_Purge_Zone_Angle"], 15, "冷吹角(°)")
    _assert_constant(converted["Label_React_Zone_Angle"], 90, "再生角(°)")

    excluded_row = excluded.iloc[0]
    excluded_flow_residual = float(
        excluded_row["Label_React_CMH"]
        - excluded_row["TextBox_OA_CMH"]
        - excluded_row["TextBox_Purge_CMH"]
    )
    removed_record = {
        "ID": int(excluded_row["ID"]),
        "TextBox_Purge_CMH": _json_value(excluded_row["TextBox_Purge_CMH"]),
        "TextBox_Process_CMH": _json_value(excluded_row["TextBox_Process_CMH"]),
        "TextBox_OA_CMH": _json_value(excluded_row["TextBox_OA_CMH"]),
        "Label_React_CMH": _json_value(excluded_row["Label_React_CMH"]),
        "TextBox_OA_AbsHum": _json_value(excluded_row["TextBox_OA_AbsHum"]),
        "TextBox_Purge_Inlet_AbsHum": _json_value(
            excluded_row["TextBox_Purge_Inlet_AbsHum"]
        ),
        "Label_React_Inlet_AbsHum": _json_value(
            excluded_row["Label_React_Inlet_AbsHum"]
        ),
        "flow_balance_residual_m3h": _json_value(excluded_flow_residual),
    }

    clean = converted.loc[~converted["ID"].eq(EXCLUDED_ID)].copy().reset_index(drop=True)
    if len(clean) != EXPECTED_CLEAN_ROWS:
        raise DataContractError(
            f"删除 ID {EXCLUDED_ID} 后应为 {EXPECTED_CLEAN_ROWS} 条，实际 {len(clean)}"
        )

    flow_residual = (
        clean["Label_React_CMH"]
        - clean["TextBox_OA_CMH"]
        - clean["TextBox_Purge_CMH"]
    ).to_numpy(dtype=np.float64)
    if not np.equal(flow_residual, 0.0).all():
        bad_ids = clean.loc[~np.equal(flow_residual, 0.0), "ID"].head(20).tolist()
        raise DataContractError(
            "清洗后 Label_React_CMH != TextBox_OA_CMH + TextBox_Purge_CMH，"
            f"ID: {bad_ids}"
        )

    calculated_mixed_humidity = (
        clean["TextBox_OA_CMH"] * clean["TextBox_OA_AbsHum"]
        + clean["TextBox_Purge_CMH"] * clean["TextBox_Purge_Inlet_AbsHum"]
    ) / clean["Label_React_CMH"]
    mixing_deviation = (
        clean["Label_React_Inlet_AbsHum"] - calculated_mixed_humidity
    ).to_numpy(dtype=np.float64)

    cleaning_report: dict[str, Any] = {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "raw_row_count": int(len(converted)),
        "clean_row_count": int(len(clean)),
        "cleaning_policy": {
            "removed_ids": [EXCLUDED_ID],
            "interpolation": False,
            "iqr_filtering": False,
            "source_workbook_modified": False,
        },
        "removed_records": [removed_record],
        "post_clean_regen_flow_balance": {
            "formula": "Label_React_CMH = TextBox_OA_CMH + TextBox_Purge_CMH",
            "nonzero_residual_count": int(np.count_nonzero(flow_residual)),
            "max_abs_residual_m3h": float(np.max(np.abs(flow_residual))),
        },
        "substream_mixing_humidity_deviation": {
            "formula": (
                "Label_React_Inlet_AbsHum - "
                "(TextBox_OA_CMH*TextBox_OA_AbsHum + "
                "TextBox_Purge_CMH*TextBox_Purge_Inlet_AbsHum) / Label_React_CMH"
            ),
            "unit": "g/kg",
            "minimum": float(np.min(mixing_deviation)),
            "maximum": float(np.max(mixing_deviation)),
            "mean": float(np.mean(mixing_deviation)),
            "mean_absolute": float(np.mean(np.abs(mixing_deviation))),
            "max_absolute": float(np.max(np.abs(mixing_deviation))),
            "nonzero_count": int(np.count_nonzero(mixing_deviation)),
            "authoritative_field": "Label_React_Inlet_AbsHum",
            "authoritative_field_modified": False,
        },
    }
    return clean, cleaning_report


def _build_split_assignments(clean: pd.DataFrame) -> pd.DataFrame:
    group_to_fold: dict[tuple[int, int], str] = {}
    for fold_name, groups in FIXED_FOLD_GROUPS.items():
        for group in groups:
            if group in group_to_fold:
                raise DataContractError(f"内层几何组重复分配: {group}")
            group_to_fold[group] = fold_name

    rows = clean[
        ["ID", "TextBox_Rotor_depth", "TextBox_Rotor_diameter"]
    ].copy()
    rows = rows.rename(
        columns={
            "TextBox_Rotor_depth": "rotor_thickness_mm",
            "TextBox_Rotor_diameter": "rotor_diameter_mm",
        }
    )
    for column in ("rotor_thickness_mm", "rotor_diameter_mm"):
        values = rows[column].to_numpy(dtype=np.float64)
        if not np.equal(values, np.floor(values)).all():
            raise DataContractError(f"几何分组字段必须为整数: {column}")
        rows[column] = rows[column].astype("int64")

    groups = list(zip(rows["rotor_thickness_mm"], rows["rotor_diameter_mm"]))
    outer_mask = np.asarray([group == OUTER_GROUP for group in groups], dtype=bool)
    rows["partition"] = np.where(outer_mask, "outer_test", "development")
    rows["validation_fold"] = [
        None if is_outer else group_to_fold.get(group)
        for group, is_outer in zip(groups, outer_mask)
    ]

    unmapped = rows.loc[
        rows["partition"].eq("development") & rows["validation_fold"].isna()
    ]
    if not unmapped.empty:
        bad_groups = sorted(
            set(zip(unmapped["rotor_thickness_mm"], unmapped["rotor_diameter_mm"]))
        )
        raise DataContractError(f"开发集几何组未分配内层 fold: {bad_groups}")
    if int(outer_mask.sum()) != 4_750 or int((~outer_mask).sum()) != 52_435:
        raise DataContractError(
            "固定划分行数不符: "
            f"outer={int(outer_mask.sum())}, development={int((~outer_mask).sum())}"
        )
    if rows["ID"].duplicated().any():
        raise DataContractError("划分表中 ID 重复")
    return rows.sort_values("ID", kind="stable").reset_index(drop=True)


def _domain_payload(clean: pd.DataFrame) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for column in MODEL_INPUT_COLUMNS:
        values = clean[column]
        unique = sorted(_json_value(value) for value in pd.unique(values))
        entry: dict[str, Any] = {
            "label": MODEL_INPUT_LABELS[column],
            "runtime_field": MODEL_INPUT_RUNTIME_FIELDS[column],
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "n_unique": int(values.nunique()),
        }
        if len(unique) <= 30:
            entry["supported_values"] = unique
        domains[column] = entry

    group_counts = (
        clean.groupby(
            ["TextBox_Rotor_depth", "TextBox_Rotor_diameter"], sort=True
        )
        .size()
        .reset_index(name="row_count")
    )
    return {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "clean_row_count": int(len(clean)),
        "inputs": domains,
        "geometry_groups": [
            {
                "rotor_thickness_mm": int(row.TextBox_Rotor_depth),
                "rotor_diameter_mm": int(row.TextBox_Rotor_diameter),
                "row_count": int(row.row_count),
            }
            for row in group_counts.itertuples(index=False)
        ],
    }


def _snapshot_id(members: Mapping[str, Any]) -> str:
    identity = {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "members": members,
    }
    canonical = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _contract_payload(
    report_members: Mapping[str, Any], *, source_size_bytes: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_path": "data/XF.xls",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_size_bytes": int(source_size_bytes),
        "sheet_name": SHEET_NAME,
        "technical_header_row_one_based": 1,
        "skipped_description_row_one_based": 2,
        "raw_row_count": EXPECTED_RAW_ROWS,
        "clean_row_count": EXPECTED_CLEAN_ROWS,
        "id_column": "ID",
        "excluded_ids": [EXCLUDED_ID],
        "derived_model_inputs": [],
        "inputs": [
            {
                "source_column": column,
                "label": MODEL_INPUT_LABELS[column],
                "runtime_field": MODEL_INPUT_RUNTIME_FIELDS[column],
            }
            for column in MODEL_INPUT_COLUMNS
        ],
        "outputs": [
            {
                "source_column": column,
                "label": MODEL_OUTPUT_LABELS[column],
                "runtime_field": MODEL_OUTPUT_RUNTIME_FIELDS[column],
            }
            for column in MODEL_OUTPUT_COLUMNS
        ],
        "outer_holdout": {
            "rotor_thickness_mm": OUTER_GROUP[0],
            "rotor_diameter_mm": OUTER_GROUP[1],
            "row_count": 4_750,
        },
        "development_row_count": 52_435,
        "fixed_envelope": {
            "rotation_speed_rpm": 8,
            "regen_angle_deg": 90,
            "cool_blow_angle_deg": 15,
        },
        "report_snapshot": {
            "schema_version": 1,
            "committed": True,
            "commit_marker_file": "dataset_contract.json",
            "snapshot_id": _snapshot_id(report_members),
            "members": dict(report_members),
        },
    }


def _report_member_metadata(
    payloads: Mapping[str, bytes], *, clean_row_count: int
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": _sha256_bytes(content),
            "size_bytes": len(content),
            "row_count": int(clean_row_count),
        }
        for name, content in payloads.items()
    }


def _logical_report_row_count(name: str, content: bytes) -> int:
    try:
        if name == "split_assignments.csv":
            frame = pd.read_csv(io.BytesIO(content))
            expected_columns = [
                "ID",
                "rotor_thickness_mm",
                "rotor_diameter_mm",
                "partition",
                "validation_fold",
            ]
            if list(frame.columns) != expected_columns:
                raise DataContractError(
                    f"报告快照 CSV 字段不符: {list(frame.columns)}"
                )
            return int(len(frame))
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise DataContractError(f"报告快照无法解析: {name}: {exc}") from exc

    if not isinstance(payload, dict):
        raise DataContractError(f"报告快照 JSON 根必须为对象: {name}")
    if payload.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise DataContractError(f"报告快照源哈希不符: {name}")
    try:
        return int(payload["clean_row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(f"报告快照缺少有效 clean_row_count: {name}") from exc


def validate_report_snapshot(report_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    """以 dataset_contract.json 为最终提交标记验证整组报告。"""

    target = Path(report_dir)
    marker_path = target / "dataset_contract.json"
    if not marker_path.is_file():
        raise DataContractError(
            f"报告快照 commit marker 缺失: {marker_path}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"报告快照 commit marker 无法解析: {exc}") from exc
    if not isinstance(marker, dict):
        raise DataContractError("报告快照 commit marker 根必须为对象")
    if marker.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise DataContractError("报告快照 commit marker 源哈希不符")

    snapshot = marker.get("report_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("committed") is not True:
        raise DataContractError("报告快照 commit marker 未提交")
    members = snapshot.get("members")
    expected_names = {
        "cleaning_report.json",
        "input_domain.json",
        "split_assignments.csv",
    }
    if not isinstance(members, dict) or set(members) != expected_names:
        raise DataContractError(
            f"报告快照成员不符: expected={sorted(expected_names)}, "
            f"actual={sorted(members) if isinstance(members, dict) else type(members).__name__}"
        )

    for name in sorted(expected_names):
        metadata = members[name]
        if not isinstance(metadata, dict):
            raise DataContractError(f"报告快照成员元数据无效: {name}")
        member_path = target / name
        if not member_path.is_file():
            raise DataContractError(f"报告快照成员缺失: {member_path}")
        try:
            content = member_path.read_bytes()
        except OSError as exc:
            raise DataContractError(f"报告快照成员无法读取: {member_path}: {exc}") from exc
        actual_hash = _sha256_bytes(content)
        if actual_hash != metadata.get("sha256"):
            raise DataContractError(
                f"报告快照成员哈希不符: {name}, "
                f"expected={metadata.get('sha256')}, actual={actual_hash}"
            )
        if len(content) != metadata.get("size_bytes"):
            raise DataContractError(f"报告快照成员字节数不符: {name}")
        actual_rows = _logical_report_row_count(name, content)
        if actual_rows != metadata.get("row_count"):
            raise DataContractError(
                f"报告快照成员行数不符: {name}, "
                f"expected={metadata.get('row_count')}, actual={actual_rows}"
            )

    actual_snapshot_id = _snapshot_id(members)
    if snapshot.get("snapshot_id") != actual_snapshot_id:
        raise DataContractError(
            "报告快照 ID 不符: "
            f"expected={snapshot.get('snapshot_id')}, actual={actual_snapshot_id}"
        )
    return marker


def _publish_report_snapshot(
    target: Path,
    *,
    member_payloads: Mapping[str, bytes],
    contract_payload: bytes,
) -> None:
    """先在 staging 完整写入/回读，再把 commit marker 最后发布。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        for name, content in member_payloads.items():
            (staging / name).write_bytes(content)
        (staging / "dataset_contract.json").write_bytes(contract_payload)
        validate_report_snapshot(staging)

        target.mkdir(parents=True, exist_ok=True)
        for name in member_payloads:
            _replace_file(staging / name, target / name)
        _replace_file(
            staging / "dataset_contract.json", target / "dataset_contract.json"
        )
        validate_report_snapshot(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_xf_dataset(
    *, reports_dir: Path = REPORTS_DIR, source_path: Path = SOURCE_PATH
) -> XFDataset:
    """执行完整读取/验证/划分，并以 UTF-8 写出四份固定报告。"""

    raw = read_xf_source(source_path)
    clean, cleaning_report = validate_and_clean_xf(raw)
    assignments = _build_split_assignments(clean)
    input_domain = _domain_payload(clean)

    target = Path(reports_dir)
    csv_buffer = io.StringIO(newline="")
    assignments.to_csv(
        csv_buffer,
        index=False,
        lineterminator="\n",
    )
    member_payloads = {
        "cleaning_report.json": _json_bytes(cleaning_report),
        "input_domain.json": _json_bytes(input_domain),
        "split_assignments.csv": csv_buffer.getvalue().encode("utf-8"),
    }
    report_members = _report_member_metadata(
        member_payloads, clean_row_count=len(clean)
    )
    contract = _contract_payload(
        report_members, source_size_bytes=int(raw.attrs["source_size_bytes"])
    )
    _publish_report_snapshot(
        target,
        member_payloads=member_payloads,
        contract_payload=_json_bytes(contract),
    )

    return XFDataset(
        raw=raw,
        clean=clean,
        split_assignments=assignments,
        source_sha256=EXPECTED_SOURCE_SHA256,
        dataset_contract=contract,
        cleaning_report=cleaning_report,
        input_domain=input_domain,
        report_dir=target,
    )


if __name__ == "__main__":
    prepared = prepare_xf_dataset()
    print(
        f"XF 数据契约验证通过: raw={len(prepared.raw)}, "
        f"clean={len(prepared.clean)}, reports={prepared.report_dir}"
    )
