"""The installed main9 model carries its humidity head through raw inference."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dual_rotor_sim.core.system.components.rotor import SingleRotorModel
from training.single_kan.config import VOLUME_MAIN_INPUT_COLUMNS

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'artifacts/kan_main_single'
DATA=ROOT/'data/转轮数据集_尺寸隔离_8比1比1/主转轮_测试集.csv'
pytestmark=pytest.mark.skipif(
    (not (PACKAGE/'model_metadata.json').is_file() or not DATA.is_file()) and os.environ.get('REQUIRE_EXTERNAL_FIXTURES')!='1',
    reason='需要带外恢复主转轮Softplus包和测试CSV',
)


@pytest.mark.parametrize('cold_inlet_T',[23.6,27.,30.])
def test_main_raw_humidity_is_nonnegative_and_temperature_bypasses_head(cold_inlet_T):
    runtime=SingleRotorModel.from_artifact(PACKAGE,'main')
    assert tuple(runtime.input_cols)==VOLUME_MAIN_INPUT_COLUMNS
    assert runtime.metadata['schema_version']==4
    assert runtime.metadata['output_constraint']==runtime.model.humidity_head.metadata()
    assert sum(p.numel() for p in runtime.model.parameters())==2160
    frame=pd.read_csv(DATA).nsmallest(1,'处理出口含湿量').copy()
    frame['冷吹入口温度(℃)']=cold_inlet_T
    output=runtime.predict_frame(frame)
    assert all(np.isfinite(list(output.values())))
    assert all(value>=0 for col,value in output.items() if '含湿量' in col)
    x=runtime.x_scaler.transform(frame[runtime.input_cols].to_numpy(np.float32))
    with torch.no_grad():
        features=torch.as_tensor(x,dtype=torch.float32)
        raw=runtime.model.network(features)
        constrained=runtime.model(features)
    torch.testing.assert_close(constrained[:,[0,2,4]],raw[:,[0,2,4]],atol=0,rtol=0)
