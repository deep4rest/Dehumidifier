"""转轮 KAN 组件。

正式前向使用两个固定六输出 ``SingleRotorModel``；历史
``RotorEnsemble`` 仅保留作离线对照。
"""

from __future__ import annotations

import json
import math
import pickle
from ....scalers import TargetScalerAdapter as _ArtifactTargetScaler, validate_scaler_columns
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from ...feature_engineering import DERIVED_FEATURES, apply_derived_features
from ...kan_model import build_kan
from ..psychrometrics import dew_point_water_C, saturation_humidity_ratio_g_per_kg


# 新实验的运行时列契约。旧 ``RotorEnsemble`` 下面仍保留历史常量和实现，
# 但正式前向只允许 ``SingleRotorModel`` 使用这两组固定列。
_XF_SINGLE_INPUT_COLS = (
    "转轮厚度",
    "转轮直径",
    "处理入口温度(℃)",
    "入口含湿量(g/kg)",
    "处理风速(m/s)",
    "冷吹风速(m/s)",
    "转轮转速(rpm)",
    "再生温度(℃)",
    "再生风量",
    "再生入口含湿量",
)
_XF_SINGLE_OUTPUT_COLS = (
    "处理出口温度",
    "处理出口含湿量",
    "冷吹出口温度",
    "冷吹出口含湿量",
    "再生出口温度",
    "再生出口含湿量",
)
_MAIN_SINGLE_INPUT_COLS = (
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
_MAIN_SINGLE_OUTPUT_COLS = (
    "处理出口温度",
    "处理出口含湿量",
    "冷吹出口温度",
    "冷吹出口含湿量",
    "再生出口温度",
    "再生出口含湿量",
)
_SINGLE_ARCHITECTURES = {
    "xf": {"hidden_widths": [24], "grid_size": 6, "spline_order": 3},
    "main": {"hidden_widths": [16], "grid_size": 5, "spline_order": 3},
}

def _ensure_scaler_pickle_compat() -> None:
    """历史 scalers.pkl 以旧顶层包名 ``src.preprocessing`` 序列化。

    包重命名后仿真侧不再提供顶层 ``src``；若其不可导入，则建立指向
    ``dual_rotor_sim.core.preprocessing`` 的模块别名，保证产物可反序列化。
    """
    if "src.preprocessing" in sys.modules:
        return
    try:
        import src.preprocessing  # noqa: F401
        return
    except ImportError:
        pass
    import dual_rotor_sim.core.preprocessing as _prep

    src_pkg = sys.modules.setdefault("src", types.ModuleType("src"))
    src_pkg.preprocessing = _prep
    sys.modules["src.preprocessing"] = _prep


# legacy 表保持原训练产物契约不变。
_RUN_REGISTRY: Dict[str, Dict] = {
    "v1": {
        "dir": "kan_full_v1",
        "hidden_widths": [16],
        "grid_size": 5,
        "spline_order": 3,
    },
    "v3": {
        "dir": "kan_v3_balanced",
        "hidden_widths": [24],
        "grid_size": 6,
        "spline_order": 3,
    },
    "v4": {
        "dir": "kan_v4_weighted",
        "hidden_widths": [24],
        "grid_size": 6,
        "spline_order": 3,
    },
    "v5": {
        "dir": "kan_v5_physics",
        "hidden_widths": [24],
        "grid_size": 6,
        "spline_order": 3,
    },
    "v6": {
        "dir": "kan_v6_full",
        "hidden_widths": [32],
        "grid_size": 6,
        "spline_order": 3,
    },
}

_MAIN_RUN_REGISTRY: Dict[str, Dict] = {
    "v1": {
        "dir": "kan_main_v1",
        "hidden_widths": [16],
        "grid_size": 5,
        "spline_order": 3,
    },
    "v3": {
        "dir": "kan_main_v3_balanced",
        "hidden_widths": [24],
        "grid_size": 6,
        "spline_order": 3,
    },
    "v4": {
        "dir": "kan_main_v4_weighted",
        "hidden_widths": [24],
        "grid_size": 6,
        "spline_order": 3,
    },
    "v5": {
        "dir": "kan_main_v5_physics",
        "hidden_widths": [24],
        "grid_size": 6,
        "spline_order": 3,
    },
    "v6": {
        "dir": "kan_main_v6_full",
        "hidden_widths": [32],
        "grid_size": 6,
        "spline_order": 3,
    },
}

_XF_RUN_REGISTRY: Dict[str, Dict] = {
    "xf_v1_baseline": {
        "dir": "kan_xf_ensemble_best/xf_v1_baseline",
        "hidden_widths": [16],
        "grid_size": 5,
        "spline_order": 3,
    },
    "xf_v2_balanced": {
        "dir": "kan_xf_ensemble_best/xf_v2_balanced",
        "hidden_widths": [24],
        "grid_size": 6,
        "spline_order": 3,
    },
    "xf_v5_robust": {
        "dir": "kan_xf_ensemble_best/xf_v5_robust",
        "hidden_widths": [32, 16],
        "grid_size": 6,
        "spline_order": 3,
    },
}

_PROFILE_REGISTRIES = {
    "legacy": _RUN_REGISTRY,
    "main": _MAIN_RUN_REGISTRY,
    "xf": _XF_RUN_REGISTRY,
}

_DEFAULT_ROUTING: Dict[str, str] = {
    "冷吹出口温度": "v3",
    "冷处出口含湿量": "v1",
    "处理出口含湿量": "v1",
    "处理出口温度": "v4",
    "处理出口露点": "v6",
    "再生出口含湿量": "v1",
    "再生出口温度": "v4",
}

_ENSEMBLE_DIRS = {
    "legacy": "kan_ensemble_best",
    "main": "kan_main_ensemble_best",
    "xf": "kan_xf_ensemble_best",
}

_MAIN_NONNEGATIVE_OUTPUTS = {
    "冷处出口含湿量",
    "处理出口含湿量",
    "再生出口含湿量",
}

_XF_NONNEGATIVE_OUTPUTS = {
    "Label_Process_Outlet_AbsHum",
    "Label_Purge_Outlet_AbsHum",
    "Label_React_Outlet_AbsHum",
}

@dataclass
class RotorInputs:
    """转轮稳态输入。

    ``airspeed`` 是处理风速；``cool_blow_airspeed`` 是冷吹风速。厚度和直径
    仅被 main 模型使用，legacy 模型会忽略它们。
    """

    process_inlet_T: float
    process_inlet_W: float
    regen_T: float
    cool_blow_inlet_T: float
    regen_angle: float = 60.0
    cool_blow_angle: float = 30.0
    rpm: float = 5.0
    airspeed: float = 2.5
    rotor_thickness: float = 200.0
    rotor_diameter: float = 1220.0
    cool_blow_airspeed: float = 3.3
    # XF 新风转轮 10 输入所需的再生侧状态
    regen_flow_m3h: float = 0.0
    regen_inlet_W: float = 0.0
    # 系统总压，用于由处理出口含湿量统一派生水面露点。
    pressure_Pa: float = 101325.0


@dataclass
class RotorOutputs:
    process_outlet_T: float
    process_outlet_W: float
    process_outlet_Tdp: float
    regen_outlet_T: float
    regen_outlet_W: float
    cool_blow_outlet_T: float
    cool_blow_outlet_W: float


class _RunBundle:
    """单个 KAN run 的模型、缩放器和列契约。"""

    def __init__(
        self,
        model: torch.nn.Module,
        x_scaler,
        y_scaler,
        input_cols: List[str],
        output_cols: List[str],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.device = device

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.input_cols if c not in df.columns]
        if missing:
            raise KeyError(f"在线推理缺少模型输入列: {missing}")
        x = df[self.input_cols].to_numpy(dtype=np.float32)
        x_scaled = self.x_scaler.transform(x).astype(np.float32)
        with torch.no_grad():
            tensor = torch.tensor(
                x_scaled, dtype=torch.float32, device=self.device
            )
            y_scaled = self.model(tensor).detach().cpu().numpy()
        return self.y_scaler.inverse_transform(y_scaled.astype(np.float32))


def _raw_row(inputs: RotorInputs) -> Dict[str, float]:
    """把系统转轮输入映射为两份单 KAN 的原始物理列。

    这里刻意只提供原始量。特别是没有调用 ``feature_engineering``，也不
    根据露点、焓或相对湿度补列；主转轮的 9 列输入契约由 artifact 严格
    校验，任何派生列都应在加载阶段被拒绝。
    """

    return {
        # 主转轮中文技术表头
        "转轮厚度": inputs.rotor_thickness,
        "转轮厚度(mm)": inputs.rotor_thickness,
        "转轮直径": inputs.rotor_diameter,
        "转轮直径(mm)": inputs.rotor_diameter,
        "入口含湿量(g/kg)": inputs.process_inlet_W,
        "转轮转速(rpm)": inputs.rpm,
        "再生温度(℃)": inputs.regen_T,
        "冷吹入口温度(℃)": inputs.cool_blow_inlet_T,
        "再生角度(°)": inputs.regen_angle,
        "风速(m/s)": inputs.airspeed,
        "处理风速(m/s)": inputs.airspeed,
        "冷吹风速(m/s)": inputs.cool_blow_airspeed,
        "冷吹角度(°)": inputs.cool_blow_angle,
        "处理入口温度(℃)": inputs.process_inlet_T,
        "再生风量": inputs.regen_flow_m3h,
        "再生入口含湿量": inputs.regen_inlet_W,
        # XF 数据集原始技术表头
        "TextBox_Rotor_depth": inputs.rotor_thickness,
        "TextBox_Rotor_diameter": inputs.rotor_diameter,
        "TextBox_Process_inlet_Temp": inputs.process_inlet_T,
        "TextBox_Purge_Inlet_AbsHum": inputs.process_inlet_W,
        "Label_Process_FV": inputs.airspeed,
        "Label_Purge_FV": inputs.cool_blow_airspeed,
        "TextBox_Rotation_Speed": inputs.rpm,
        "TextBox_React_Heater_Temp": inputs.regen_T,
        "Label_React_CMH": inputs.regen_flow_m3h,
        "Label_React_Inlet_AbsHum": inputs.regen_inlet_W,
        # 双表头中的中文说明列只用于兼容旧产物，不是派生量。
        "处理/冷吹共享入口温度(℃)": inputs.process_inlet_T,
        "处理/冷吹共享入口含湿量(g/kg)": inputs.process_inlet_W,
        "再生入口温度(℃)": inputs.regen_T,
        "再生入口风量(m³/h)": inputs.regen_flow_m3h,
        "再生入口含湿量(g/kg)": inputs.regen_inlet_W,
    }


def _single_metadata_path(artifact_dir: Path) -> Path:
    """定位新单模型元数据文件。

    训练侧目前把模型元数据放在 artifact 根目录；同时接受 ``model`` 子目录
    的同名文件，方便在不改变运行时契约的情况下重打包产物。历史
    ``scaler_meta.json`` 只作为候选位置，因其没有 ``direct_outputs`` 时会
    在后续严格契约检查中失败，不能被当作旧模型自动降级加载。
    """

    candidates = (
        artifact_dir / "model_metadata.json",
        artifact_dir / "metadata.json",
        artifact_dir / "model" / "model_metadata.json",
        artifact_dir / "model" / "metadata.json",
        artifact_dir / "model" / "scaler_meta.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    names = ", ".join(str(path) for path in candidates[:4])
    raise FileNotFoundError(f"单 KAN artifact 缺少模型元数据文件（应为 {names}）")


def _metadata_architecture(metadata: Mapping[str, Any], profile: str) -> Dict[str, Any]:
    """从元数据中读取并校验单 KAN 架构参数。"""

    expected = _SINGLE_ARCHITECTURES[profile]
    raw = metadata.get("architecture")
    if raw is None:
        raw = metadata.get("model")
    if not isinstance(raw, Mapping):
        raise RuntimeError("单 KAN artifact 元数据缺少 architecture/model 架构字段")

    missing = [key for key in expected if key not in raw]
    if missing:
        raise RuntimeError(f"单 KAN artifact 架构字段缺失: {missing}")
    actual = {
        "hidden_widths": [int(x) for x in raw["hidden_widths"]],
        "grid_size": int(raw["grid_size"]),
        "spline_order": int(raw["spline_order"]),
    }
    if "in_features" in raw and int(raw["in_features"]) != len(
        _XF_SINGLE_INPUT_COLS if profile == "xf" else _MAIN_SINGLE_INPUT_COLS
    ):
        raise RuntimeError("单 KAN architecture.in_features 与输入列数不一致")
    if "out_features" in raw and int(raw["out_features"]) != 6:
        raise RuntimeError("单 KAN architecture.out_features 必须为 6")
    if "grid_range" in raw:
        grid_range = [float(value) for value in raw["grid_range"]]
        if grid_range != [-1.0, 1.0]:
            raise RuntimeError("单 KAN architecture.grid_range 必须为 [-1, 1]")
    if actual != expected:
        raise RuntimeError(
            f"{profile} 单 KAN 架构契约不匹配: artifact={actual}, expected={expected}"
        )
    return actual


class SingleRotorModel:
    """一个固定六输出 KAN 及其可逆缩放器。

    ``from_artifact`` 是正式运行时唯一的模型加载入口。它不读取 registry、
    routing.json，也不会按输出列选择候选模型；输入和输出顺序全部由新 artifact
    元数据声明并与 profile 的固定契约逐项比较。
    """

    def __init__(
        self,
        *,
        profile: str,
        model: torch.nn.Module,
        x_scaler: Any,
        y_scaler: Any,
        input_cols: List[str],
        direct_outputs: List[str],
        metadata: Mapping[str, Any],
        device: str = "cpu",
    ) -> None:
        if profile not in _SINGLE_ARCHITECTURES:
            raise ValueError(f"未知单 KAN profile: {profile}")
        self.profile = profile
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.input_cols = list(input_cols)
        self.direct_outputs = list(direct_outputs)
        self.output_cols = list(direct_outputs)
        self.derived_outputs = list(
            metadata.get("derived_outputs", ["处理出口露点"])
        )
        self.audit_columns = list(metadata.get("audit_columns", []))
        self.metadata = dict(metadata)
        self.device = device

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: Path,
        profile: str,
        *,
        device: str = "cpu",
    ) -> "SingleRotorModel":
        if profile not in _SINGLE_ARCHITECTURES:
            raise ValueError(f"未知单 KAN profile: {profile}，可选 xf/main")
        artifact_dir = Path(artifact_dir)
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"单 KAN artifact 目录不存在: {artifact_dir}")

        metadata_path = _single_metadata_path(artifact_dir)
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, Mapping):
            raise RuntimeError("单 KAN artifact 元数据必须是 JSON 对象")

        expected_inputs = (
            _XF_SINGLE_INPUT_COLS if profile == "xf" else _MAIN_SINGLE_INPUT_COLS
        )
        expected_outputs = (
            _XF_SINGLE_OUTPUT_COLS if profile == "xf" else _MAIN_SINGLE_OUTPUT_COLS
        )
        if metadata.get("profile") not in (None, profile):
            raise RuntimeError(
                f"单 KAN artifact profile 不匹配: artifact={metadata.get('profile')!r}, "
                f"runtime={profile!r}"
            )
        if "input_cols" not in metadata:
            raise RuntimeError("单 KAN artifact 元数据缺少 input_cols 字段")
        if "output_cols" not in metadata:
            raise RuntimeError("单 KAN artifact 元数据缺少 output_cols 字段")
        if "direct_outputs" not in metadata:
            raise RuntimeError("单 KAN artifact 元数据缺少 direct_outputs 字段")
        input_cols = [str(value) for value in metadata["input_cols"]]
        output_cols = [str(value) for value in metadata["output_cols"]]
        direct_outputs = [str(value) for value in metadata["direct_outputs"]]
        if tuple(input_cols) != expected_inputs:
            raise RuntimeError(
                f"{profile} 单 KAN 输入列契约不匹配: artifact={input_cols}, "
                f"expected={list(expected_inputs)}"
            )
        if tuple(direct_outputs) != expected_outputs:
            raise RuntimeError(
                f"{profile} 单 KAN direct_outputs 契约不匹配: "
                f"artifact={direct_outputs}, expected={list(expected_outputs)}"
            )
        if tuple(output_cols) != tuple(direct_outputs):
            raise RuntimeError(
                f"{profile} 单 KAN output_cols 与 direct_outputs 列契约不一致: "
                f"output_cols={output_cols}, direct_outputs={direct_outputs}"
            )
        if len(set(input_cols)) != len(input_cols) or len(set(direct_outputs)) != len(direct_outputs):
            raise RuntimeError("单 KAN artifact 列契约包含重复字段")
        forbidden = set(input_cols) & (
            {"power", "Label_PowerConsumption", "处理出口露点", "Label_Process_Outlet_DP"}
        )
        if forbidden:
            raise RuntimeError(f"单 KAN 输入列包含禁用审计字段: {sorted(forbidden)}")
        forbidden = set(direct_outputs) & (
            {"power", "Label_PowerConsumption", "处理出口露点", "Label_Process_Outlet_DP"}
        )
        if forbidden:
            raise RuntimeError(f"单 KAN direct_outputs 包含禁用审计字段: {sorted(forbidden)}")
        if "derived_features" in metadata and metadata["derived_features"]:
            raise RuntimeError("单 KAN artifact 不允许声明 derived_features")
        derived_outputs = metadata.get("derived_outputs")
        if derived_outputs is None or list(derived_outputs) != ["处理出口露点"]:
            raise RuntimeError(
                "单 KAN artifact derived_outputs 必须严格为 ['处理出口露点']"
            )
        audit_columns = metadata.get("audit_columns")
        if audit_columns is None or set(audit_columns) != {"power", "处理出口露点"}:
            raise RuntimeError(
                "单 KAN artifact audit_columns 必须包含 power 和处理出口露点"
            )
        _metadata_architecture(metadata, profile)

        model_dir = artifact_dir / "model"
        model_path = model_dir / "best_model.pt"
        scalers_path = model_dir / "scalers.pkl"
        missing = [str(path) for path in (model_path, scalers_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"单 KAN artifact 缺少模型文件: {missing}")
        for scaler_meta_path in (
            model_dir / "scaler_meta.json",
            artifact_dir / "scaler_meta.json",
        ):
            if not scaler_meta_path.is_file():
                continue
            with scaler_meta_path.open(encoding="utf-8") as stream:
                scaler_meta = json.load(stream)
            if "input_cols" in scaler_meta and list(scaler_meta["input_cols"]) != input_cols:
                raise RuntimeError(
                    "单 KAN scaler_meta 输入列契约与 artifact 不一致"
                )
            if "output_cols" in scaler_meta and list(scaler_meta["output_cols"]) != direct_outputs:
                raise RuntimeError(
                    "单 KAN scaler_meta 输出列契约与 direct_outputs 不一致"
                )
        with scalers_path.open("rb") as stream:
            _ensure_scaler_pickle_compat()
            scalers = pickle.load(stream)
        if not isinstance(scalers, Mapping):
            raise RuntimeError("单 KAN scalers.pkl 必须是对象")
        if {"x_scaler", "y_scaler"} <= set(scalers):
            x_scaler = scalers["x_scaler"]
            y_scaler = scalers["y_scaler"]
        elif {"input_scaler", "target_scaler", "target_transform"} <= set(scalers):
            # training/single_kan/model.py 的正式保存格式。
            transform_state = scalers["target_transform"]
            if not isinstance(transform_state, Mapping):
                raise RuntimeError("单 KAN target_transform 状态无效")
            output_columns = [str(value) for value in scalers.get("target_columns", ())]
            if output_columns != direct_outputs:
                raise RuntimeError(
                    "单 KAN target scaler 列契约与 direct_outputs 不一致: "
                    f"scaler={output_columns}, artifact={direct_outputs}"
                )
            transform_columns = [
                str(value) for value in transform_state.get("output_columns", ())
            ]
            if transform_columns != direct_outputs:
                raise RuntimeError(
                    "单 KAN target_transform 列契约与 direct_outputs 不一致: "
                    f"transform={transform_columns}, artifact={direct_outputs}"
                )
            log_columns = [
                str(value) for value in transform_state.get("log1p_targets", ())
            ]
            x_scaler = scalers["input_scaler"]
            y_scaler = _ArtifactTargetScaler(
                scalers["target_scaler"], output_columns, log_columns
            )
        else:
            raise RuntimeError(
                "单 KAN scalers.pkl 必须包含 x_scaler/y_scaler，或训练侧的 "
                "input_scaler/target_scaler/target_transform"
            )
        try:
            validate_scaler_columns(x_scaler, input_cols, role="输入")
            validate_scaler_columns(y_scaler, direct_outputs, role="输出")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        scaler_output_cols = getattr(y_scaler, "output_cols", None)
        if scaler_output_cols is not None and list(scaler_output_cols) != direct_outputs:
            raise RuntimeError(
                "单 KAN 目标 scaler 列契约与 direct_outputs 不一致: "
                f"scaler={list(scaler_output_cols)}, artifact={direct_outputs}"
            )
        log_columns = set(getattr(y_scaler, "log_columns", []))
        if log_columns & {"power", "Label_PowerConsumption", "处理出口露点"}:
            raise RuntimeError("单 KAN 目标 scaler 不得包含 power 或处理出口露点")

        architecture = _metadata_architecture(metadata, profile)
        model = build_kan(
            in_features=len(input_cols),
            out_features=len(direct_outputs),
            hidden_widths=architecture["hidden_widths"],
            grid_size=architecture["grid_size"],
            spline_order=architecture["spline_order"],
        )
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:  # torch < 2.6
            state_dict = torch.load(model_path, map_location=device)
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("单 KAN checkpoint 必须是 state_dict")
        # 训练侧 ``SingleKAN`` 将共享 KAN 放在 ``kan`` 子模块中；运行时
        # 仍用同一 builder 构造裸 KAN，去掉这一层序列化包装即可复用权重。
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        if isinstance(state_dict, Mapping) and state_dict and all(
            str(key).startswith("kan.") for key in state_dict
        ):
            state_dict = {
                str(key)[len("kan.") :]: value
                for key, value in state_dict.items()
            }
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(f"单 KAN checkpoint 与列/架构契约不匹配: {exc}") from exc
        model.to(device)
        model.eval()
        return cls(
            profile=profile,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            input_cols=input_cols,
            direct_outputs=direct_outputs,
            metadata=metadata,
            device=device,
        )

    def predict_frame(self, frame: pd.DataFrame) -> Dict[str, float]:
        """对一行或多行原始物理量做六输出推理。"""

        missing = [column for column in self.input_cols if column not in frame.columns]
        if missing:
            raise KeyError(f"单 KAN 在线推理缺少原始输入列: {missing}")
        values = frame[self.input_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("单 KAN 输入包含非有限值")
        x_scaled = np.asarray(self.x_scaler.transform(values), dtype=np.float32)
        with torch.no_grad():
            tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=self.device)
            predicted = self.model(tensor).detach().cpu().numpy()
        if predicted.ndim != 2 or predicted.shape[1] != len(self.direct_outputs):
            raise RuntimeError(
                "单 KAN 输出维度与 direct_outputs 不一致: "
                f"prediction={predicted.shape}, expected=(*,{len(self.direct_outputs)})"
            )
        raw = np.asarray(self.y_scaler.inverse_transform(predicted), dtype=np.float64)
        if raw.shape != predicted.shape or not np.isfinite(raw).all():
            raise RuntimeError("单 KAN 输出包含非有限值或形状错误")
        if len(raw) != 1:
            raise ValueError(
                f"单 KAN 在线推理必须是一行原始输入，实际样本数={len(raw)}"
            )
        return {
            column: float(value)
            for column, value in zip(self.direct_outputs, raw[0])
        }

    def predict(self, frame: pd.DataFrame) -> Dict[str, float]:
        """``predict_frame`` 的简洁别名，供组件适配器调用。"""

        return self.predict_frame(frame)


def _load_saved_routing(artifacts_dir: Path, profile: str) -> Dict[str, str]:
    path = (
        artifacts_dir
        / _ENSEMBLE_DIRS[profile]
        / "reports"
        / "routing.json"
    )
    if not path.exists():
        if profile == "legacy":
            return dict(_DEFAULT_ROUTING)
        raise FileNotFoundError(
            f"缺少主转轮集成路由: {path}\n"
            "请先运行 python archive/training/ensemble_predict.py --profile main。"
        )
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    raw = document.get("routing", document.get("routes", document))
    routing: Dict[str, str] = {}
    for col, value in raw.items():
        if isinstance(value, dict):
            if "best_run" in value:
                routing[str(col)] = str(value["best_run"])
            elif "selected_candidate" in value:
                routing[str(col)] = str(value["selected_candidate"])
            else:
                raise KeyError(f"路由项缺少 best_run/selected_candidate: {col}")
        else:
            routing[str(col)] = str(value)
    return routing


def _load_main_projection_columns(artifacts_dir: Path) -> List[str]:
    path = (
        artifacts_dir
        / _ENSEMBLE_DIRS["main"]
        / "reports"
        / "routing.json"
    )
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    projection = document.get("physical_projection") or {}
    if projection.get("method") != "nonnegative_projection":
        raise RuntimeError("主转轮物理后处理方法不是 nonnegative_projection")
    column_contracts = projection.get("columns") or {}
    columns = set(column_contracts.keys())
    if columns != _MAIN_NONNEGATIVE_OUTPUTS:
        raise RuntimeError(
            "主转轮离线/在线物理后处理契约不一致："
            f"artifact={sorted(columns)}, runtime={sorted(_MAIN_NONNEGATIVE_OUTPUTS)}"
        )
    invalid = {
        col: contract
        for col, contract in column_contracts.items()
        if contract.get("constraint") != "lower_bound"
        or float(contract.get("lower_bound", float("nan"))) != 0.0
    }
    if invalid:
        raise RuntimeError(f"主转轮物理后处理阈值契约无效: {invalid}")
    return sorted(columns)


class RotorEnsemble:
    """按输出列路由的 KAN 集成。"""

    def __init__(
        self,
        bundles: Dict[str, _RunBundle],
        routing: Dict[str, str],
        *,
        profile: str = "legacy",
        physical_projection_columns: Optional[List[str]] = None,
    ) -> None:
        self._bundles = bundles
        self._routing = routing
        self.profile = profile
        self._physical_projection_columns = set(
            physical_projection_columns
            if physical_projection_columns is not None
            else (
                _MAIN_NONNEGATIVE_OUTPUTS
                if profile == "main"
                else _XF_NONNEGATIVE_OUTPUTS
                if profile == "xf"
                else []
            )
        )
        self._col_idx = {
            alias: {col: j for j, col in enumerate(bundle.output_cols)}
            for alias, bundle in bundles.items()
        }

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path,
        routing: Optional[Dict[str, str]] = None,
        device: str = "cpu",
        *,
        profile: str = "legacy",
    ) -> "RotorEnsemble":
        if profile not in _PROFILE_REGISTRIES:
            raise ValueError(
                f"未知转轮模型 profile '{profile}'，可选: {list(_PROFILE_REGISTRIES)}"
            )
        artifacts_dir = Path(artifacts_dir)
        routing = routing or _load_saved_routing(artifacts_dir, profile)
        projection_columns = (
            _load_main_projection_columns(artifacts_dir)
            if profile == "main"
            else sorted(_XF_NONNEGATIVE_OUTPUTS)
            if profile == "xf"
            else []
        )
        registry = _PROFILE_REGISTRIES[profile]

        bundles: Dict[str, _RunBundle] = {}
        for alias in set(routing.values()):
            if alias not in registry:
                raise KeyError(
                    f"{profile} 路由出现未知 run 别名 '{alias}'，"
                    f"已注册别名: {list(registry)}"
                )
            reg = registry[alias]
            model_dir = artifacts_dir / reg["dir"] / "model"
            required = [
                model_dir / "scaler_meta.json",
                model_dir / "scalers.pkl",
                model_dir / "best_model.pt",
            ]
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                raise FileNotFoundError(f"缺少模型文件: {missing}")

            with open(model_dir / "scaler_meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            input_cols = [str(c) for c in meta["input_cols"]]
            output_cols = [str(c) for c in meta["output_cols"]]
            with open(model_dir / "scalers.pkl", "rb") as f:
                _ensure_scaler_pickle_compat()
                scalers = pickle.load(f)

            model = build_kan(
                in_features=len(input_cols),
                out_features=len(output_cols),
                hidden_widths=reg["hidden_widths"],
                grid_size=reg["grid_size"],
                spline_order=reg["spline_order"],
            )
            try:
                state_dict = torch.load(
                    model_dir / "best_model.pt",
                    map_location=device,
                    weights_only=True,
                )
            except TypeError:
                state_dict = torch.load(
                    model_dir / "best_model.pt", map_location=device
                )
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            bundles[alias] = _RunBundle(
                model=model,
                x_scaler=scalers["x_scaler"],
                y_scaler=scalers["y_scaler"],
                input_cols=input_cols,
                output_cols=output_cols,
                device=device,
            )
        return cls(
            bundles,
            routing,
            profile=profile,
            physical_projection_columns=projection_columns,
        )

    @property
    def routing(self) -> Dict[str, str]:
        return dict(self._routing)

    @staticmethod
    def _raw_row(inputs: RotorInputs) -> Dict[str, float]:
        """同时提供 legacy/main 可能需要的原始列名，run 按元数据自行取列。"""

        return {
            "转轮厚度": inputs.rotor_thickness,
            "转轮厚度(mm)": inputs.rotor_thickness,
            "转轮直径": inputs.rotor_diameter,
            "转轮直径(mm)": inputs.rotor_diameter,
            "入口含湿量(g/kg)": inputs.process_inlet_W,
            "转轮转速(rpm)": inputs.rpm,
            "再生温度(℃)": inputs.regen_T,
            "冷吹入口温度(℃)": inputs.cool_blow_inlet_T,
            "再生角度(°)": inputs.regen_angle,
            "风速(m/s)": inputs.airspeed,
            "处理风速(m/s)": inputs.airspeed,
            "冷吹风速(m/s)": inputs.cool_blow_airspeed,
            "冷吹角度(°)": inputs.cool_blow_angle,
            "处理入口温度(℃)": inputs.process_inlet_T,
            # XF 新风转轮输入标签
            "处理/冷吹共享入口温度(℃)": inputs.process_inlet_T,
            "处理/冷吹共享入口含湿量(g/kg)": inputs.process_inlet_W,
            "再生入口温度(℃)": inputs.regen_T,
            "再生入口风量(m³/h)": inputs.regen_flow_m3h,
            "再生入口含湿量(g/kg)": inputs.regen_inlet_W,
            # XF 原始技术表头（训练 scaler_meta 使用）
            "TextBox_Rotor_depth": inputs.rotor_thickness,
            "TextBox_Rotor_diameter": inputs.rotor_diameter,
            "TextBox_Process_inlet_Temp": inputs.process_inlet_T,
            "TextBox_Purge_Inlet_AbsHum": inputs.process_inlet_W,
            "Label_Process_FV": inputs.airspeed,
            "Label_Purge_FV": inputs.cool_blow_airspeed,
            "TextBox_Rotation_Speed": inputs.rpm,
            "TextBox_React_Heater_Temp": inputs.regen_T,
            "Label_React_CMH": inputs.regen_flow_m3h,
            "Label_React_Inlet_AbsHum": inputs.regen_inlet_W,
        }

    def predict(self, inputs: RotorInputs) -> RotorOutputs:
        raw_row = self._raw_row(inputs)
        frame = pd.DataFrame([raw_row])

        needed_derived: List[str] = []
        for bundle in self._bundles.values():
            for col in bundle.input_cols:
                if col not in frame.columns and col in DERIVED_FEATURES:
                    if col not in needed_derived:
                        needed_derived.append(col)
        if needed_derived:
            frame, _ = apply_derived_features(frame, needed_derived)

        run_predictions = {
            alias: bundle.predict(frame)[0]
            for alias, bundle in self._bundles.items()
        }
        values: Dict[str, float] = {}
        for col, alias in self._routing.items():
            if col not in self._col_idx[alias]:
                raise KeyError(f"run '{alias}' 不包含路由输出列 '{col}'")
            values[col] = float(
                run_predictions[alias][self._col_idx[alias][col]]
            )

        # 历史 ensemble 对照路径只对含湿量做非负投影；正式单模型路径在
        # ``Rotor.process`` 中使用同一规则，且不含任何功率输出。
        if self.profile in ("main", "xf"):
            for col in self._physical_projection_columns:
                if col in values:
                    values[col] = max(0.0, values[col])

        if self.profile == "xf":
            required_outputs = [
                "Label_Process_Outlet_Temp",
                "Label_Process_Outlet_AbsHum",
                "Label_Purge_Outlet_Temp",
                "Label_Purge_Outlet_AbsHum",
                "Label_React_Outlet_Temp",
                "Label_React_Outlet_AbsHum",
            ]
            missing = [c for c in required_outputs if c not in values]
            if missing:
                raise KeyError(f"XF 集成路由缺少必要输出列: {missing}")
            process_w = values["Label_Process_Outlet_AbsHum"]
            regen_total = values["Label_React_Outlet_AbsHum"]
            regen_t = values["Label_React_Outlet_Temp"]
            try:
                sat = float(saturation_humidity_ratio_g_per_kg(regen_t, 101325.0))
            except Exception:
                sat = float("inf")
            regen_vapor = min(regen_total, sat) if math.isfinite(sat) else regen_total
            return RotorOutputs(
                process_outlet_T=values["Label_Process_Outlet_Temp"],
                process_outlet_W=process_w,
                process_outlet_Tdp=float(
                    dew_point_water_C(process_w, inputs.pressure_Pa)
                ),
                regen_outlet_T=regen_t,
                regen_outlet_W=regen_vapor,
                cool_blow_outlet_T=values["Label_Purge_Outlet_Temp"],
                cool_blow_outlet_W=values["Label_Purge_Outlet_AbsHum"],
            )

        required_outputs = [
            "处理出口温度",
            "处理出口含湿量",
            "再生出口温度",
            "再生出口含湿量",
            "冷吹出口温度",
            "冷处出口含湿量",
        ]
        missing = [c for c in required_outputs if c not in values]
        if missing:
            raise KeyError(f"集成路由缺少必要输出列: {missing}")
        return RotorOutputs(
            process_outlet_T=values["处理出口温度"],
            process_outlet_W=values["处理出口含湿量"],
            process_outlet_Tdp=float(
                dew_point_water_C(values["处理出口含湿量"], inputs.pressure_Pa)
            ),
            regen_outlet_T=values["再生出口温度"],
            regen_outlet_W=values["再生出口含湿量"],
            cool_blow_outlet_T=values["冷吹出口温度"],
            cool_blow_outlet_W=values["冷处出口含湿量"],
        )


class Rotor:
    """转轮组件适配器。

    正式系统使用 ``model=SingleRotorModel``。``ensemble=`` 仅保留给历史
    单元测试和离线对照调用，``build_system`` 不会再构造它。
    """

    def __init__(
        self,
        model: Any | None = None,
        name: str = "Rotor",
        *,
        ensemble: RotorEnsemble | None = None,
    ) -> None:
        if model is not None and ensemble is not None:
            raise ValueError("model= 与 ensemble= 不能同时提供")
        selected = model if model is not None else ensemble
        if selected is None:
            raise ValueError("必须提供 model=（或历史兼容的 ensemble=）")
        self.model = selected
        # 旧调用方可能检查 .ensemble；单模型路径明确不暴露旧 ensemble。
        self.ensemble = selected if isinstance(selected, RotorEnsemble) else None
        self.name = name

    def process(self, inputs: RotorInputs) -> RotorOutputs:
        is_single_model = isinstance(self.model, SingleRotorModel) or (
            getattr(self.model, "profile", None) in {"xf", "main"}
            and callable(getattr(self.model, "predict_frame", None))
        )
        if is_single_model:
            values = self.model.predict_frame(pd.DataFrame([_raw_row(inputs)]))
            # 训练只输出直接状态；露点在此处统一按系统压力派生。
            process_t = values["处理出口温度"]
            process_w = max(0.0, values["处理出口含湿量"])
            regen_w = max(0.0, values["再生出口含湿量"])
            cool_w = max(0.0, values["冷吹出口含湿量"])
            return RotorOutputs(
                process_outlet_T=process_t,
                process_outlet_W=process_w,
                process_outlet_Tdp=float(
                    dew_point_water_C(process_w, inputs.pressure_Pa)
                ),
                regen_outlet_T=values["再生出口温度"],
                regen_outlet_W=regen_w,
                cool_blow_outlet_T=values["冷吹出口温度"],
                cool_blow_outlet_W=cool_w,
            )
        # 历史 ensemble 仍可被显式离线调用，但不是正式加载路径。
        return self.model.predict(inputs)
