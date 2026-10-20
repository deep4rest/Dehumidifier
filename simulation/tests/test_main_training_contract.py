"""主转轮新数据集、配置与几何分组契约测试。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from archive.training.src.config import load_config
from archive.training.src.data_loader import coerce_numeric_columns, load_sheet
from archive.training.src.splitting import build_split_report, split_dataframe
from archive.training.ensemble_predict import (
    apply_physical_projection,
    assemble_ensemble,
    build_routing,
    validate_main_ensemble,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "主转轮数据集.xls"
CONFIGS = [
    ROOT / "archive" / "artifacts" / "kan_main_v1" / "config_snapshot.yaml",
    ROOT / "archive" / "artifacts" / "kan_main_v3_balanced" / "config_snapshot.yaml",
    ROOT / "archive" / "artifacts" / "kan_main_v4_weighted" / "config_snapshot.yaml",
    ROOT / "archive" / "artifacts" / "kan_main_v5_physics" / "config_snapshot.yaml",
    ROOT / "archive" / "training" / "configs" / "kan_main_v6_full.yaml",
]


def _requires_external_files(*paths: Path):
    missing = [str(path) for path in paths if not path.is_file()]
    return pytest.mark.skipif(
        bool(missing) and os.environ.get("REQUIRE_EXTERNAL_FIXTURES") != "1",
        reason="缺少单独提供的历史模型配置快照：" + "、".join(missing),
    )

EXPECTED_INPUTS = [
    "转轮厚度",
    "转轮直径",
    "入口含湿量(g/kg)",
    "转轮转速(rpm)",
    "再生温度(℃)",
    "冷吹入口温度(℃)",
    "处理入口温度(℃)",
    "处理风速(m/s)",
    "冷吹风速(m/s)",
]
EXPECTED_OUTPUTS = [
    "冷吹出口温度",
    "冷处出口含湿量",
    "处理出口含湿量",
    "处理出口温度",
    "处理出口露点",
    "再生出口含湿量",
    "再生出口温度",
    "power",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def main_df() -> pd.DataFrame:
    if not SOURCE.is_file() and os.environ.get("REQUIRE_EXTERNAL_FIXTURES") != "1":
        pytest.skip(f"缺少单独提供的权威主转轮 XLS：{SOURCE}")
    before_hash = _sha256(SOURCE)
    before_mtime = SOURCE.stat().st_mtime_ns
    df = load_sheet(SOURCE, "CollectRecord", header_row=1)
    after_hash = _sha256(SOURCE)
    assert after_hash == before_hash
    assert SOURCE.stat().st_mtime_ns == before_mtime
    return df


@pytest.mark.requires_data
def test_main_xls_shape_and_numeric_contract(main_df):
    assert main_df.shape == (60610, 22)
    converted = coerce_numeric_columns(
        main_df, EXPECTED_INPUTS + EXPECTED_OUTPUTS
    )
    assert list(converted[EXPECTED_INPUTS].columns) == EXPECTED_INPUTS
    assert list(converted[EXPECTED_OUTPUTS].columns) == EXPECTED_OUTPUTS
    assert all(pd.api.types.is_numeric_dtype(converted[c]) for c in EXPECTED_INPUTS)
    assert all(pd.api.types.is_numeric_dtype(converted[c]) for c in EXPECTED_OUTPUTS)
    assert not converted[EXPECTED_INPUTS + EXPECTED_OUTPUTS].isna().any().any()


@pytest.mark.parametrize("config_path", [
    pytest.param(path, marks=[pytest.mark.requires_models, _requires_external_files(path)])
    if "artifacts" in path.relative_to(ROOT).parts else path
    for path in CONFIGS
])
def test_main_configs_share_raw_contract_and_split(config_path):
    cfg = load_config(config_path)
    assert cfg.header_row == 1
    assert cfg.input_columns == EXPECTED_INPUTS
    assert cfg.output_columns == EXPECTED_OUTPUTS
    assert cfg.group_columns == ["转轮厚度", "转轮直径"]
    assert cfg.split_method == "group"
    assert (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio) == (0.70, 0.15, 0.15)
    assert cfg.seed == 42
    excluded = {"处理风量", "冷吹风量", "再生角度(°)", "冷吹角度(°)", "实验ID"}
    assert excluded.isdisjoint(cfg.input_columns)


@pytest.mark.requires_models
@_requires_external_files(CONFIGS[3])
def test_v5_v6_derived_features_exclude_angles_and_time_constants():
    v5 = load_config(CONFIGS[3])
    v6 = load_config(CONFIGS[4])
    wet_air = {
        "入口饱和水汽压_Ps",
        "入口相对湿度_RH",
        "入口露点_Tdp",
        "入口焓_h",
    }
    assert set(v5.derived_features) == wet_air
    assert set(v6.derived_features) == wet_air | {"再生驱动力_dT"}
    forbidden = {
        "处理段角度",
        "处理段时间常数",
        "再生段时间常数",
        "RH×处理时间",
        "dT×再生时间",
    }
    assert forbidden.isdisjoint(v5.derived_features)
    assert forbidden.isdisjoint(v6.derived_features)


@pytest.mark.requires_data
def test_geometry_group_split_is_reproducible_and_leak_free(main_df):
    small = main_df[["转轮厚度", "转轮直径", "实验ID"]].copy()
    kwargs = dict(
        method="group",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        group_column=None,
        group_columns=["转轮厚度", "转轮直径"],
        seed=42,
    )
    split1 = split_dataframe(small, **kwargs)
    split2 = split_dataframe(small, **kwargs)
    assert split1.group_assignments == split2.group_assignments
    assert sum(map(len, [split1.train_df, split1.val_df, split1.test_df])) == 60610

    groups = {
        name: set(map(tuple, part[["转轮厚度", "转轮直径"]].drop_duplicates().to_numpy()))
        for name, part in (
            ("train", split1.train_df),
            ("val", split1.val_df),
            ("test", split1.test_df),
        )
    }
    assert len(groups["train"]) == 40
    assert len(groups["val"]) == 9
    assert len(groups["test"]) == 8
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])

    report = build_split_report(
        split1,
        output_cols=[],
        time_column=None,
        group_column=None,
        group_columns=["转轮厚度", "转轮直径"],
    )
    assert report["group_leakage"] is False
    assert all(not values for values in report["group_overlaps"].values())


def test_main_routing_selects_highest_r2_per_output():
    import numpy as np

    y_true = np.column_stack(
        [np.linspace(1.0 + j, 10.0 + j, 20) for j in range(8)]
    )
    pred_v1 = y_true.copy()
    pred_v6 = y_true.copy()
    pred_v1[:, 4] += np.linspace(-4.0, 4.0, 20)
    pred_v6[:, 0] += np.linspace(-4.0, 4.0, 20)
    bundles = {
        "v1": {
            "y_true": y_true,
            "y_pred": pred_v1,
            "output_cols": EXPECTED_OUTPUTS,
        },
        "v6": {
            "y_true": y_true,
            "y_pred": pred_v6,
            "output_cols": EXPECTED_OUTPUTS,
        },
    }
    routing = build_routing(bundles)
    assert routing[EXPECTED_OUTPUTS[0]]["best_run"] == "v1"
    assert routing[EXPECTED_OUTPUTS[4]]["best_run"] == "v6"
    for col, info in routing.items():
        assert info["best_r2"] == max(info["r2_per_run"].values()), col

    _, prediction, cols = assemble_ensemble(bundles, routing)
    validation = validate_main_ensemble(prediction, cols, routing)
    assert validation["passed"] is True


def test_main_physical_projection_is_nonnegative_and_audited():
    import numpy as np

    prediction = np.ones((3, len(EXPECTED_OUTPUTS)), dtype=float)
    prediction[0, 2] = -0.002
    prediction[1, 7] = -5.0
    projected, report = apply_physical_projection(
        prediction, EXPECTED_OUTPUTS
    )
    assert projected[0, 2] == 0.0
    assert projected[1, 7] == 0.0
    assert report["total_adjusted_values"] == 2
    assert report["columns"]["处理出口含湿量"]["adjusted_count"] == 1
    assert report["columns"]["power"]["adjusted_count"] == 1
