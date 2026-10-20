"""双转轮单 KAN 数据契约的 TDD 规格。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.single_kan.config import (
    AUDIT_COLUMNS,
    DIRECT_OUTPUT_COLUMNS,
    EXPECTED_SOURCE_SHA256,
    FIXED_XF_FOLD_1_GROUPS,
    MAIN_INPUT_COLUMNS,
    PROFILE_CONFIGS,
    XF_INPUT_COLUMNS,
)
from training.single_kan.data import (
    DataContractError,
    clean_dataset,
    make_model_arrays,
    read_authoritative_dataset,
    split_dataset,
    validate_dataset,
)


def _frame(profile: str, *, n_per_group: int = 4) -> pd.DataFrame:
    """生成不依赖 XLS 的小型 canonical frame。"""

    cfg = PROFILE_CONFIGS[profile]
    groups = [(200, 550), (300, 1070), (400, 965), (200, 1070)]
    rows: list[dict[str, float | int]] = []
    index = 1
    for depth, diameter in groups:
        for _ in range(n_per_group):
            row: dict[str, float | int] = {
                "ID": index,
                "转轮厚度": depth,
                "转轮直径": diameter,
                "入口含湿量(g/kg)": 1.0 + index / 100,
                "转轮转速(rpm)": 8.0 if profile == "xf" else 6.0,
                "再生温度(℃)": 50.0 + index,
                "冷吹入口温度(℃)": 20.0,
                "处理入口温度(℃)": 20.0,
                "处理风速(m/s)": 1.0,
                "冷吹风速(m/s)": 1.0,
                "再生风量": 200.0,
                "再生入口含湿量": 0.5,
                "power": 2.0,
                "处理出口露点": -20.0,
                "处理出口温度": 18.0,
                "处理出口含湿量": 0.5,
                "冷吹出口温度": 22.0,
                "冷吹出口含湿量": 0.7,
                "再生出口温度": 30.0,
                "再生出口含湿量": 5.0,
            }
            # Main's canonical input contract has exactly nine fields; the
            # extra XF-only values remain harmless audit/source columns.
            rows.append({k: v for k, v in row.items() if k in {
                "ID", "转轮厚度", "转轮直径", "入口含湿量(g/kg)",
                "转轮转速(rpm)", "再生温度(℃)", "冷吹入口温度(℃)",
                "处理入口温度(℃)", "处理风速(m/s)", "冷吹风速(m/s)",
                "再生风量", "再生入口含湿量", "power", "处理出口露点",
                "处理出口温度", "处理出口含湿量", "冷吹出口温度",
                "冷吹出口含湿量", "再生出口温度", "再生出口含湿量",
            }})
            index += 1
    return pd.DataFrame(rows)


def test_contract_is_two_fixed_raw_to_six_direct_outputs() -> None:
    assert len(XF_INPUT_COLUMNS) == 10
    assert len(MAIN_INPUT_COLUMNS) == 9
    assert len(DIRECT_OUTPUT_COLUMNS) == 6
    assert "power" not in XF_INPUT_COLUMNS
    assert "power" not in MAIN_INPUT_COLUMNS
    assert "处理出口露点" not in DIRECT_OUTPUT_COLUMNS
    assert set(AUDIT_COLUMNS) == {"power", "处理出口露点"}
    assert PROFILE_CONFIGS["main"].derived_features == ()


def test_main_input_names_are_the_nine_plan_raw_chinese_fields() -> None:
    assert MAIN_INPUT_COLUMNS == (
        "转轮厚度",
        "转轮直径",
        "入口含湿量(g/kg)",
        "转轮转速(rpm)",
        "再生温度(℃)",
        "冷吹入口温度(℃)",
        "处理入口温度(℃)",
        "处理风速(m/s)",
        "冷吹风速(m/s)",
    )


def test_model_arrays_exclude_power_and_dew_point() -> None:
    frame = _frame("main")
    assignments = split_dataset("main", frame)
    x, y, audit = make_model_arrays("main", frame, assignments, "train")
    assert x.shape[1] == 9
    assert y.shape[1] == 6
    assert audit.shape[1] == 2
    assert set(audit.columns) == {"power", "处理出口露点"}
    assert "power" not in MAIN_INPUT_COLUMNS
    assert "处理出口露点" not in DIRECT_OUTPUT_COLUMNS
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_clean_dataset_only_excludes_xf_id_1407() -> None:
    frame = _frame("xf")
    frame.loc[0, "ID"] = 1407
    clean, report = clean_dataset("xf", frame)
    assert 1407 not in set(clean["ID"])
    assert report["excluded_ids"] == [1407]
    assert report["interpolation"] is False
    assert report["iqr_filtering"] is False


def test_main_geometry_groups_never_cross_partitions() -> None:
    frame = _frame("main", n_per_group=12)
    assignments = split_dataset("main", frame)
    seen: dict[tuple[int, int], str] = {}
    for row in assignments.itertuples(index=False):
        group = (int(row.rotor_thickness_mm), int(row.rotor_diameter_mm))
        old = seen.setdefault(group, str(row.partition))
        assert old == str(row.partition)
    assert set(assignments["partition"]) == {"train", "validation", "test"}


def test_xf_fixed_test_and_fold_one_validation_groups() -> None:
    frame = _frame("xf")
    # Include all relevant groups once; group assignment is the invariant.
    assignments = split_dataset("xf", frame)
    by_group = assignments.groupby(["rotor_thickness_mm", "rotor_diameter_mm"])[
        "partition"
    ].unique()
    assert set(by_group.loc[(300, 1070)]) == {"test"}
    for group in FIXED_XF_FOLD_1_GROUPS:
        if group in by_group.index:
            assert set(by_group.loc[group]) == {"validation"}


def test_validate_rejects_duplicate_ids_and_nonfinite_model_value() -> None:
    frame = _frame("main")
    frame.loc[1, "ID"] = frame.loc[0, "ID"]
    with pytest.raises(DataContractError, match="ID.*重复"):
        validate_dataset("main", frame)
    bad = _frame("main")
    bad.loc[0, "处理出口含湿量"] = np.nan
    with pytest.raises(DataContractError, match="非有限|缺失"):
        validate_dataset("main", bad)


def test_source_hash_constants_are_locked_and_authoritative_path_is_fixed() -> None:
    assert EXPECTED_SOURCE_SHA256["xf"] == (
        "116ea18458bf831225765472440a21da8045cb8e9bfadd898f677b004ce026cb"
    )
    assert EXPECTED_SOURCE_SHA256["main"] == (
        "a1173dee9789832e54087bd2a0f7d8865b8371fa7789d4a191449a45731f704f"
    )
    with pytest.raises(DataContractError, match="固定|权威"):
        read_authoritative_dataset("xf", Path("/tmp/other.xls"))
