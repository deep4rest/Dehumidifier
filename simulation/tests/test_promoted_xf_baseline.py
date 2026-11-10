"""Installed XF9 baseline contract, runtime reconstruction and package isolation."""
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dual_rotor_sim.core.system.components.rotor import SingleRotorModel
from training.single_kan.config import VOLUME_XF_INPUT_COLUMNS

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/"artifacts/kan_xf_single"
pytestmark=pytest.mark.skipif(not (PACKAGE/"model_metadata.json").is_file() and os.environ.get("REQUIRE_EXTERNAL_FIXTURES")!="1",reason="需要带外恢复新风基线包")


def test_promoted_runtime_is_xf9_and_rpm_has_no_prediction_effect():
    runtime=SingleRotorModel.from_artifact(PACKAGE,"xf")
    assert tuple(runtime.input_cols)==VOLUME_XF_INPUT_COLUMNS
    assert runtime.metadata["schema_version"]==4
    assert runtime.model.humidity_head.metadata()==runtime.metadata["output_constraint"]
    row={"转轮厚度":400,"转轮直径":1850,"处理入口温度(℃)":9,"入口含湿量(g/kg)":3.9,
         "处理风量":11667,"冷吹风量":583,"再生温度(℃)":140,"再生风量":3889,"再生入口含湿量":.31}
    a=runtime.predict_frame(pd.DataFrame([{**row,"转轮转速(rpm)":8}]))
    b=runtime.predict_frame(pd.DataFrame([{**row,"转轮转速(rpm)":20}]))
    assert a==b
    assert all(np.isfinite(list(a.values())))
    assert all(value>=0 for name,value in a.items() if "含湿量" in name)


def test_runtime_refuses_wrong_softplus_configuration(tmp_path):
    target=tmp_path/"wrong_head"
    shutil.copytree(PACKAGE,target)
    metadata_path=target/"model_metadata.json"
    meta=json.loads(metadata_path.read_text(encoding="utf-8"))
    meta["output_constraint"]["beta"]=1.
    metadata_path.write_text(json.dumps(meta,ensure_ascii=False),encoding="utf-8")
    with pytest.raises(RuntimeError,match="非负头元数据"):
        SingleRotorModel.from_artifact(target,"xf")
