"""XF 数据契约的可执行规格。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archive.xf_rotor1_retrain import data as data_module
from archive.xf_rotor1_retrain.data import (
    EXPECTED_CLEAN_ROWS,
    EXPECTED_RAW_ROWS,
    EXPECTED_SOURCE_SHA256,
    FIXED_FOLD_GROUPS,
    MODEL_INPUT_COLUMNS,
    MODEL_INPUT_LABELS,
    MODEL_OUTPUT_COLUMNS,
    MODEL_OUTPUT_LABELS,
    DataContractError,
    prepare_xf_dataset,
    read_xf_source,
    validate_and_clean_xf,
    validate_report_snapshot,
)


@pytest.fixture(scope="session")
def raw_xf() -> pd.DataFrame:
    """全会话只读一次 XLS，避免单元测试重复消耗 I/O。"""

    return read_xf_source()


@pytest.fixture(scope="session")
def prepared_xf(tmp_path_factory: pytest.TempPathFactory):
    report_dir = tmp_path_factory.mktemp("xf_reports")
    return prepare_xf_dataset(reports_dir=report_dir)


def test_reads_technical_header_and_skips_description_row(raw_xf: pd.DataFrame) -> None:
    """若表头行或说明行读取错位，首条 ID 和样本数会错。"""

    assert len(raw_xf) == EXPECTED_RAW_ROWS == 57_186
    assert raw_xf.iloc[0]["ID"] == 1
    assert list(raw_xf.columns[:4]) == [
        "ID",
        "TextBox_Rotor_depth",
        "TextBox_Rotor_diameter",
        "TextBox_Purge_CMH",
    ]
    assert "实验ID" not in raw_xf["ID"].astype(str).tolist()


def test_contract_has_exactly_ten_raw_inputs_and_seven_direct_outputs() -> None:
    """捕获输入派生特征或旧冷吹湿量标签混入新产物。"""

    assert len(MODEL_INPUT_COLUMNS) == 10
    assert len(MODEL_OUTPUT_COLUMNS) == 7
    assert MODEL_OUTPUT_COLUMNS[3] == "Label_Purge_Outlet_AbsHum"
    assert MODEL_OUTPUT_LABELS["Label_Purge_Outlet_AbsHum"] == "冷吹出口含湿量"
    assert "冷处出口含湿量" not in MODEL_OUTPUT_LABELS.values()
    assert set(MODEL_INPUT_LABELS) == set(MODEL_INPUT_COLUMNS)
    assert set(MODEL_OUTPUT_LABELS) == set(MODEL_OUTPUT_COLUMNS)


def test_cleaning_removes_only_id_1407_and_preserves_audit_evidence(prepared_xf) -> None:
    """捕获多删样本、修改异常行或遗失原始证据。"""

    result = prepared_xf
    assert len(result.raw) == EXPECTED_RAW_ROWS == 57_186
    assert len(result.clean) == EXPECTED_CLEAN_ROWS == 57_185
    assert 1407 in set(result.raw["ID"])
    assert 1407 not in set(result.clean["ID"])
    assert set(result.raw["ID"]) - set(result.clean["ID"]) == {1407}

    removed = result.cleaning_report["removed_records"][0]
    assert removed["ID"] == 1407
    assert removed["TextBox_Purge_CMH"] == 3395
    assert removed["TextBox_OA_CMH"] == 189
    assert removed["Label_React_CMH"] == 222
    assert removed["flow_balance_residual_m3h"] == -3362


def test_clean_data_enforces_shared_inlet_temperature_and_fixed_envelope(prepared_xf) -> None:
    """捕获处理/冷吹入口错位或固定转速与角度被放宽。"""

    clean = prepared_xf.clean
    np.testing.assert_array_equal(
        clean["TextBox_Process_inlet_Temp"].to_numpy(),
        clean["TextBox_Purge_Inlet_Temp"].to_numpy(),
    )
    assert set(clean["TextBox_Rotation_Speed"]) == {8}
    assert set(clean["Label_Purge_Zone_Angle"]) == {15}
    assert set(clean["Label_React_Zone_Angle"]) == {90}


def test_clean_data_has_finite_numeric_model_fields_and_balanced_regen_flow(prepared_xf) -> None:
    """捕获数字转换漏洞、非有限值或权威总再生流量不守恒。"""

    clean = prepared_xf.clean
    model = clean[list(MODEL_INPUT_COLUMNS) + list(MODEL_OUTPUT_COLUMNS)].to_numpy()
    assert np.issubdtype(model.dtype, np.number)
    assert np.isfinite(model).all()
    np.testing.assert_array_equal(
        clean["Label_React_CMH"].to_numpy(),
        (clean["TextBox_OA_CMH"] + clean["TextBox_Purge_CMH"]).to_numpy(),
    )
    mixing_audit = prepared_xf.cleaning_report[
        "substream_mixing_humidity_deviation"
    ]
    assert mixing_audit["nonzero_count"] == 57_182
    assert mixing_audit["authoritative_field_modified"] is False


def test_rejects_missing_model_column(raw_xf: pd.DataFrame) -> None:
    """删除任一建模字段必须硬失败，不允许默认填充。"""

    with pytest.raises(DataContractError, match="缺少必需字段"):
        validate_and_clean_xf(raw_xf.drop(columns=[MODEL_INPUT_COLUMNS[0]]))


def test_rejects_alternative_source_path(tmp_path: Path) -> None:
    """捕获以临时复制、123.xls 或其他路径替换唯一权威入口。"""

    substitute = tmp_path / "XF.xls"
    substitute.write_bytes(b"not-the-authoritative-workbook")
    with pytest.raises(DataContractError, match="数据入口固定"):
        read_xf_source(substitute)


def test_rejects_source_hash_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """捕获权威路径上内容变化却被自动接受的回归。"""

    monkeypatch.setattr(data_module, "EXPECTED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(DataContractError, match="SHA-256"):
        read_xf_source()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, "not-a-number"])
def test_rejects_non_numeric_or_nonfinite_model_values(
    raw_xf: pd.DataFrame, bad_value: object
) -> None:
    """任一建模字段的 NaN/Inf/文本均必须硬失败。"""

    bad = raw_xf.copy()
    bad[MODEL_OUTPUT_COLUMNS[0]] = bad[MODEL_OUTPUT_COLUMNS[0]].astype(object)
    bad.loc[0, MODEL_OUTPUT_COLUMNS[0]] = bad_value
    with pytest.raises(DataContractError, match="非数字、NaN 或 Inf"):
        validate_and_clean_xf(bad)


def test_rejects_duplicate_ids(raw_xf: pd.DataFrame) -> None:
    """捕获重复 ID 导致 OOF 连接重复或漏行。"""

    bad = raw_xf.copy()
    bad.loc[bad.index[1], "ID"] = bad.loc[bad.index[0], "ID"]
    with pytest.raises(DataContractError, match="ID 重复"):
        validate_and_clean_xf(bad)


def test_rejects_process_and_purge_inlet_temperature_mismatch(raw_xf: pd.DataFrame) -> None:
    """捕获把处理入口温度错当为处理/冷吹共享温度。"""

    bad = raw_xf.copy()
    bad["TextBox_Purge_Inlet_Temp"] = bad["TextBox_Purge_Inlet_Temp"].astype(object)
    first = bad.index[0]
    bad.loc[first, "TextBox_Purge_Inlet_Temp"] = (
        float(bad.loc[first, "TextBox_Purge_Inlet_Temp"]) + 0.1
    )
    with pytest.raises(DataContractError, match="处理入口温度与冷吹入口温度不一致"):
        validate_and_clean_xf(bad)


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("TextBox_Rotation_Speed", 7, "转速"),
        ("Label_Purge_Zone_Angle", 14, "冷吹角"),
        ("Label_React_Zone_Angle", 89, "再生角"),
    ],
)
def test_rejects_each_fixed_envelope_violation(
    raw_xf: pd.DataFrame, column: str, bad_value: float, message: str
) -> None:
    """固定转速、冷吹角、再生角任一项被篡改都必须独立硬失败。"""

    bad = raw_xf.copy()
    bad[column] = bad[column].astype(object)
    bad.loc[bad.index[0], column] = bad_value
    with pytest.raises(DataContractError, match=message):
        validate_and_clean_xf(bad)


def test_rejects_post_clean_regen_flow_imbalance(raw_xf: pd.DataFrame) -> None:
    """捕获删除 1407 后任一样本的权威总再生流量不守恒。"""

    bad = raw_xf.copy()
    bad["Label_React_CMH"] = bad["Label_React_CMH"].astype(object)
    first = bad.index[0]
    bad.loc[first, "Label_React_CMH"] = float(bad.loc[first, "Label_React_CMH"]) + 1
    with pytest.raises(DataContractError, match="Label_React_CMH !="):
        validate_and_clean_xf(bad)


def test_outer_holdout_and_fixed_folds_are_group_and_id_disjoint(prepared_xf) -> None:
    """捕获 300×1070 泄漏到开发集，或开发几何组未被恰好一折覆盖。"""

    assignments = prepared_xf.split_assignments
    outer = assignments.loc[assignments["partition"] == "outer_test"]
    development = assignments.loc[assignments["partition"] == "development"]

    assert len(outer) == 4_750
    assert len(development) == 52_435
    assert set(outer["ID"]).isdisjoint(development["ID"])
    assert set(zip(outer["rotor_thickness_mm"], outer["rotor_diameter_mm"])) == {
        (300, 1070)
    }
    assert outer["validation_fold"].isna().all()
    assert development["validation_fold"].notna().all()
    assert set(development["validation_fold"]) == set(FIXED_FOLD_GROUPS)

    seen_groups: set[tuple[int, int]] = set()
    for fold_name, expected_groups in FIXED_FOLD_GROUPS.items():
        fold = development.loc[development["validation_fold"] == fold_name]
        actual_groups = set(zip(fold["rotor_thickness_mm"], fold["rotor_diameter_mm"]))
        assert actual_groups == set(expected_groups)
        assert seen_groups.isdisjoint(actual_groups)
        seen_groups.update(actual_groups)
    assert len(development["ID"].unique()) == len(development)


def test_reports_round_trip_with_locked_hash_and_utf8_assignments(prepared_xf) -> None:
    """捕获报告路径/编码/哈希/行数漂移，确保后续任务可回读。"""

    report_dir = Path(prepared_xf.report_dir)
    expected_names = {
        "dataset_contract.json",
        "cleaning_report.json",
        "input_domain.json",
        "split_assignments.csv",
    }
    assert {path.name for path in report_dir.iterdir()} == expected_names

    contract = json.loads((report_dir / "dataset_contract.json").read_text(encoding="utf-8"))
    cleaning = json.loads((report_dir / "cleaning_report.json").read_text(encoding="utf-8"))
    domain = json.loads((report_dir / "input_domain.json").read_text(encoding="utf-8"))
    csv_text = (report_dir / "split_assignments.csv").read_text(encoding="utf-8")
    round_trip = pd.read_csv(report_dir / "split_assignments.csv")

    assert contract["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert cleaning["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert domain["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert contract["raw_row_count"] == 57_186
    assert contract["clean_row_count"] == 57_185
    assert len(domain["inputs"]) == 10
    assert len(round_trip) == 57_185
    assert csv_text.startswith("ID,rotor_thickness_mm,rotor_diameter_mm,partition,validation_fold")
    assert validate_report_snapshot(report_dir)["report_snapshot"]["committed"] is True


def test_failed_report_publish_is_not_accepted_as_a_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """捕获发布中断后将部分新报告当成完整快照的回归。"""

    report_dir = tmp_path / "reports"
    real_replace = data_module._replace_file
    replace_calls = 0

    def fail_during_publish(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(data_module, "_replace_file", fail_during_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        prepare_xf_dataset(reports_dir=report_dir)
    with pytest.raises(DataContractError, match="commit marker|快照"):
        validate_report_snapshot(report_dir)
