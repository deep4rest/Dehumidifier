"""单 KAN 运行时契约与公开结果语义的红绿测试。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from dual_rotor_sim.core.kan_model import build_kan
from dual_rotor_sim.core.system.air_state import AirState
from dual_rotor_sim.core.system.components.rotor import (
    Rotor,
    RotorInputs,
    RotorOutputs,
    SingleRotorModel,
)
from dual_rotor_sim.core.system.psychrometrics import dew_point_water_C
from dual_rotor_sim.core.system.solver import build_system
from dual_rotor_sim.core.system.system import SystemResult
from dual_rotor_sim.run_records import result_to_dict


XF_INPUTS = [
    "转轮厚度",
    "转轮直径",
    "处理入口温度(℃)",
    "入口含湿量(g/kg)",
    "处理风量",
    "冷吹风量",
    "转轮转速(rpm)",
    "再生温度(℃)",
    "再生风量",
    "再生入口含湿量",
]
XF_OUTPUTS = [
    "处理出口温度",
    "处理出口含湿量",
    "冷吹出口温度",
    "冷吹出口含湿量",
    "再生出口温度",
    "再生出口含湿量",
]
MAIN_INPUTS = [
    "转轮厚度",
    "转轮直径",
    "入口含湿量(g/kg)",
    "转轮转速(rpm)",
    "再生温度(℃)",
    "冷吹入口温度(℃)",
    "处理入口温度(℃)",
    "处理风量",
    "冷吹风量",
]
MAIN_OUTPUTS = [
    "处理出口温度",
    "处理出口含湿量",
    "冷吹出口温度",
    "冷吹出口含湿量",
    "再生出口温度",
    "再生出口含湿量",
]


class IdentityScaler:
    """测试用可逆缩放器，避免把运行时测试绑定到训练实现。"""

    def __init__(self, columns, role):
        setattr(self, "input_cols" if role == "input" else "output_cols", list(columns))
        self.n_features_in_ = len(columns)

    def transform(self, values):
        return np.asarray(values, dtype=np.float32)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32)


def _write_artifact(
    root: Path,
    *,
    profile: str,
    metadata_update: dict | None = None,
) -> Path:
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    input_cols = XF_INPUTS if profile == "xf" else MAIN_INPUTS
    direct_outputs = XF_OUTPUTS if profile == "xf" else MAIN_OUTPUTS
    widths = [24] if profile == "xf" else [16]
    grid_size = 6 if profile == "xf" else 5
    model = build_kan(
        len(input_cols),
        len(direct_outputs),
        hidden_widths=widths,
        grid_size=grid_size,
        spline_order=3,
    )
    torch.save(model.state_dict(), model_dir / "best_model.pt")
    with (model_dir / "scalers.pkl").open("wb") as stream:
        pickle.dump(
            {"x_scaler": IdentityScaler(input_cols, "input"), "y_scaler": IdentityScaler(direct_outputs, "output")},
            stream,
        )
    metadata = {
        "profile": profile,
        "input_cols": input_cols,
        "output_cols": direct_outputs,
        "direct_outputs": direct_outputs,
        "derived_outputs": ["处理出口露点"],
        "audit_columns": ["power", "处理出口露点"],
        "architecture": {
            "hidden_widths": widths,
            "grid_size": grid_size,
            "spline_order": 3,
        },
    }
    if metadata_update:
        metadata.update(metadata_update)
    (root / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    return root


def _sample_inputs() -> RotorInputs:
    return RotorInputs(
        process_inlet_T=11.0,
        process_inlet_W=6.0,
        regen_T=120.0,
        cool_blow_inlet_T=11.0,
        rpm=6.0,
        airspeed=3.5,
        rotor_thickness=200.0,
        rotor_diameter=1220.0,
        cool_blow_airspeed=3.3,
        regen_flow_m3h=600.0,
        regen_inlet_W=1.0,
        pressure_Pa=90000.0,
    )


def test_single_artifact_requires_direct_output_contract(tmp_path: Path):
    """旧 output_cols/路由契约不能冒充新的六输出单模型。"""
    artifact = _write_artifact(tmp_path / "main", profile="main")
    metadata_path = artifact / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("direct_outputs")
    metadata["output_cols"] = MAIN_OUTPUTS
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    with pytest.raises((ValueError, KeyError, RuntimeError), match="直接|direct_outputs"):
        SingleRotorModel.from_artifact(artifact, profile="main")


def test_single_artifact_rejects_output_cols_disagreeing_with_direct_outputs(
    tmp_path: Path,
):
    artifact = _write_artifact(tmp_path / "main", profile="main")
    metadata_path = artifact / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_cols"] = ["power"]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="output_cols|direct_outputs"):
        SingleRotorModel.from_artifact(artifact, profile="main")


def test_single_model_predicts_six_direct_outputs_and_derives_dew_point(
    tmp_path: Path,
):
    artifact = _write_artifact(tmp_path / "main", profile="main")
    model = SingleRotorModel.from_artifact(artifact, profile="main")
    model.predict_frame = lambda _frame: {
        "处理出口温度": 16.0,
        "处理出口含湿量": 2.0,
        "冷吹出口温度": 27.0,
        "冷吹出口含湿量": 3.0,
        "再生出口温度": 75.0,
        "再生出口含湿量": 8.0,
    }

    output = Rotor(model=model).process(_sample_inputs())

    assert not hasattr(output, "power_kW")
    assert output.process_outlet_Tdp == pytest.approx(
        dew_point_water_C(2.0, 90000.0)
    )
    assert output.process_outlet_W == 2.0
    assert output.regen_outlet_W == 8.0


def test_build_system_loads_only_two_fixed_single_artifacts(monkeypatch, tmp_path: Path):
    calls: list[tuple[Path, str]] = []

    class FakeSingle:
        def __init__(self, profile):
            self.profile = profile

    def fake_from_artifact(path, *, profile, device="cpu"):
        calls.append((Path(path), profile))
        return FakeSingle(profile)

    monkeypatch.setattr(SingleRotorModel, "from_artifact", fake_from_artifact)
    system = build_system(artifacts_dir=tmp_path)

    assert calls == [
        (tmp_path / "kan_xf_single", "xf"),
        (tmp_path / "kan_main_single", "main"),
    ]
    assert system.rotor1.model.profile == "xf"
    assert system.rotor2.model.profile == "main"


def test_public_result_and_json_have_only_thermal_load_fields():
    rotor = RotorOutputs(
        process_outlet_T=16.0,
        process_outlet_W=2.0,
        process_outlet_Tdp=-5.0,
        regen_outlet_T=75.0,
        regen_outlet_W=8.0,
        cool_blow_outlet_T=27.0,
        cool_blow_outlet_W=3.0,
    )
    result = SystemResult(
        supply_state=AirState(T=16.0, W=2.0),
        cooling_kW=10.0,
        heating_kW=20.0,
        energy_breakdown={
            "CC1_kW": -4.0,
            "CC2_kW": -6.0,
            "Heater_Low_kW": 8.0,
            "Heater_OA_kW": 12.0,
            "Fan1_kW": 99.0,
            "Rotor1Motor_kW": 99.0,
        },
        intermediate_states={},
        rotor1_outputs=rotor,
        rotor2_outputs=rotor,
    )

    assert not hasattr(result, "electrical_kW")
    assert set(result.energy_breakdown) == {
        "CC1_kW",
        "CC2_kW",
        "Heater_Low_kW",
        "Heater_OA_kW",
    }
    document = result_to_dict(result)
    assert "electrical_kW" not in json.dumps(document, ensure_ascii=False)
    assert "Fan1_kW" not in document["energy_breakdown_kW"]
    assert "Rotor1Motor_kW" not in document["energy_breakdown_kW"]
    assert "power_kW" not in document["rotor1_outputs"]
