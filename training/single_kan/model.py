"""固定单模型 KAN、物理量预处理和 artifact 序列化。"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from training.src.kan_model import KANConfig, build_kan
from dual_rotor_sim.scalers import TargetScalerAdapter, validate_scaler_columns

from .config import (
    AUDIT_COLUMNS,
    DIRECT_OUTPUT_COLUMNS,
    MAIN_INPUT_COLUMNS,
    PROFILE_CONFIGS,
    RotorProfileConfig,
    XF_INPUT_COLUMNS,
    get_profile,
)

__all__ = [
    "Preprocessor",
    "ReversibleTargetTransform",
    "SingleKAN",
    "TargetScalerAdapter",
    "TargetTransform",
    "build_metadata",
    "build_model",
    "build_single_kan",
    "fit_preprocessor",
    "fit_scalers",
    "load_artifact",
    "save_artifact",
]


class ReversibleTargetTransform:
    """按列执行可逆 ``log1p``，严格拒绝目标契约外的列。"""

    def __init__(
        self,
        *,
        output_columns: Sequence[str],
        log1p_targets: Sequence[str] = (),
    ) -> None:
        self.output_columns = tuple(str(column) for column in output_columns)
        if len(set(self.output_columns)) != len(self.output_columns):
            raise ValueError("目标字段名重复")
        self.log1p_targets = tuple(str(column) for column in log1p_targets)
        unknown = sorted(set(self.log1p_targets) - set(self.output_columns))
        if unknown:
            raise ValueError(f"log1p 目标不在六个直接输出中: {unknown}")
        self.log_indices = tuple(
            self.output_columns.index(column) for column in self.log1p_targets
        )

    @staticmethod
    def _matrix(values: Any, label: str, columns: int) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != columns:
            raise ValueError(f"{label} 必须为 [N,{columns}]，实际 {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{label} 存在非有限值")
        return matrix

    def transform(self, values: Any) -> np.ndarray:
        matrix = self._matrix(values, "log1p 输入", len(self.output_columns)).copy()
        for index in self.log_indices:
            if np.any(matrix[:, index] < 0.0):
                raise ValueError(
                    f"目标列 {self.output_columns[index]} 含负值，无法进行 log1p"
                )
            matrix[:, index] = np.log1p(matrix[:, index])
        if not np.isfinite(matrix).all():
            raise ValueError("log1p 变换产生非有限值")
        return matrix

    def inverse_transform(self, values: Any) -> np.ndarray:
        matrix = self._matrix(values, "expm1 输入", len(self.output_columns)).copy()
        for index in self.log_indices:
            with np.errstate(over="raise", invalid="raise"):
                try:
                    matrix[:, index] = np.expm1(matrix[:, index])
                except FloatingPointError as exc:
                    raise ValueError("expm1 反变换产生非有限值") from exc
        if not np.isfinite(matrix).all():
            raise ValueError("expm1 反变换产生非有限值")
        return matrix

    def state_dict(self) -> dict[str, Any]:
        return {
            "output_columns": list(self.output_columns),
            "log1p_targets": list(self.log1p_targets),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReversibleTargetTransform":
        try:
            return cls(
                output_columns=state["output_columns"],
                log1p_targets=state.get("log1p_targets", ()),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("target transform 状态无效") from exc


def _ensure_matrix(values: Any, *, columns: int, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{label} 必须为 [N,{columns}]，实际 {matrix.shape}")
    if len(matrix) == 0:
        raise ValueError(f"{label} 不得为空")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} 存在非有限值")
    return matrix


@dataclass
class Preprocessor:
    """仅包含模型输入和六个直接目标的可逆缩放器。"""

    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    input_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    target_transform: ReversibleTargetTransform
    fit_ids: tuple[int, ...] = ()

    @property
    def x_scaler(self) -> MinMaxScaler:
        return self.input_scaler

    @property
    def y_scaler(self) -> MinMaxScaler:
        return self.target_scaler

    def transform_features(self, values: Any) -> np.ndarray:
        matrix = _ensure_matrix(values, columns=len(self.input_columns), label="输入")
        transformed = self.input_scaler.transform(matrix)
        if not np.isfinite(transformed).all():
            raise ValueError("输入缩放器产生非有限值")
        return np.asarray(transformed, dtype=np.float32)

    def transform_targets(self, values: Any) -> np.ndarray:
        matrix = _ensure_matrix(values, columns=len(self.target_columns), label="目标")
        transformed = self.target_transform.transform(matrix)
        scaled = self.target_scaler.transform(transformed)
        if not np.isfinite(scaled).all():
            raise ValueError("目标缩放器产生非有限值")
        return np.asarray(scaled, dtype=np.float32)

    def inverse_targets(self, values: Any) -> np.ndarray:
        matrix = _ensure_matrix(values, columns=len(self.target_columns), label="缩放目标")
        restored = self.target_scaler.inverse_transform(matrix)
        restored = self.target_transform.inverse_transform(restored)
        return np.asarray(restored, dtype=np.float64)

    def state_dict(self) -> dict[str, Any]:
        validate_scaler_columns(self.input_scaler, self.input_columns, role="输入")
        validate_scaler_columns(self.target_scaler, self.target_columns, role="输出")
        state = {
            "input_columns": list(self.input_columns),
            "target_columns": list(self.target_columns),
            "input_scaler": self.input_scaler,
            "target_scaler": self.target_scaler,
            "target_transform": self.target_transform.state_dict(),
            "fit_ids": list(self.fit_ids),
        }
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "Preprocessor":
        try:
            input_columns = tuple(str(value) for value in state["input_columns"])
            target_columns = tuple(str(value) for value in state["target_columns"])
            if tuple(target_columns) != tuple(DIRECT_OUTPUT_COLUMNS):
                raise ValueError("目标 scaler 不得包含 power 或处理出口露点")
            input_scaler = state["input_scaler"]
            target_scaler = state["target_scaler"]
            if not isinstance(input_scaler, MinMaxScaler) or not isinstance(
                target_scaler, MinMaxScaler
            ):
                raise TypeError("scaler 类型必须为 MinMaxScaler")
            validate_scaler_columns(input_scaler, input_columns, role="输入")
            validate_scaler_columns(target_scaler, target_columns, role="输出")
            transform = ReversibleTargetTransform.from_state_dict(
                state["target_transform"]
            )
            if tuple(transform.output_columns) != tuple(target_columns):
                raise ValueError("target_transform 列契约与目标 scaler 不一致")
            if set(transform.log1p_targets) & set(AUDIT_COLUMNS):
                raise ValueError("target_transform 不得包含 power 或处理出口露点")
            if set(input_columns) & set(AUDIT_COLUMNS):
                raise ValueError("input scaler 不得包含 power 或处理出口露点")
            if tuple(input_columns) not in (
                tuple(XF_INPUT_COLUMNS),
                tuple(MAIN_INPUT_COLUMNS),
            ):
                raise ValueError("input scaler 输入列不属于原始量契约")
            scaler_output_columns = getattr(target_scaler, "output_cols", None)
            if scaler_output_columns is not None and tuple(scaler_output_columns) != tuple(target_columns):
                raise ValueError("target scaler 列契约与目标列不一致")
            return cls(
                input_columns=input_columns,
                target_columns=target_columns,
                input_scaler=input_scaler,
                target_scaler=target_scaler,
                target_transform=transform,
                fit_ids=tuple(int(value) for value in state.get("fit_ids", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "power" in str(exc):
                raise
            raise ValueError(f"preprocessor 状态无效: {exc}") from exc


def fit_preprocessor(
    features: Any,
    targets: Any,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    log1p_targets: Sequence[str] = ("处理出口含湿量",),
    fit_ids: Sequence[int] | None = None,
) -> Preprocessor:
    """只用训练分区拟合 MinMax 与可逆目标变换。"""

    inputs = tuple(str(value) for value in input_columns)
    outputs = tuple(str(value) for value in target_columns)
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("输入字段为空或重复")
    if outputs != tuple(DIRECT_OUTPUT_COLUMNS):
        raise ValueError("目标必须严格为六个直接状态输出")
    if set(inputs) & set(AUDIT_COLUMNS):
        raise ValueError("power/处理出口露点不得进入输入张量")
    if tuple(inputs) not in (tuple(XF_INPUT_COLUMNS), tuple(MAIN_INPUT_COLUMNS)):
        raise ValueError("输入必须严格属于 XF 10 列或主转轮 9 列原始量契约")
    x = _ensure_matrix(features, columns=len(inputs), label="训练输入")
    y = _ensure_matrix(targets, columns=len(outputs), label="训练目标")
    if len(x) != len(y):
        raise ValueError("训练输入与目标样本数不一致")
    normalized_ids: tuple[int, ...] = ()
    if fit_ids is not None:
        normalized_ids = tuple(int(value) for value in fit_ids)
        if len(normalized_ids) != len(x) or len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("scaler 拟合 ID 数量不符或重复")
    transform = ReversibleTargetTransform(
        output_columns=outputs,
        log1p_targets=log1p_targets,
    )
    x_scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(x)
    x_scaler.input_cols = list(inputs)
    y_transformed = transform.transform(y)
    y_scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(y_transformed)
    y_scaler.output_cols = list(outputs)
    y_scaler.log_columns = list(transform.log1p_targets)
    return Preprocessor(
        input_columns=inputs,
        target_columns=outputs,
        input_scaler=x_scaler,
        target_scaler=y_scaler,
        target_transform=transform,
        fit_ids=normalized_ids,
    )


def build_metadata(
    profile: str | RotorProfileConfig,
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """构造对外 artifact 元数据并显式区分 direct/derived/audit。"""

    cfg = get_profile(profile)
    return {
        "schema_version": 1,
        "profile": cfg.name,
        "input_cols": list(cfg.input_columns),
        "output_cols": list(cfg.direct_outputs),
        "direct_outputs": list(cfg.direct_outputs),
        "derived_outputs": list(cfg.derived_outputs),
        "audit_columns": list(cfg.audit_columns),
        "source_audit_columns": ["Label_PowerConsumption", "Label_Process_Outlet_DP"],
        "derived_features": list(cfg.derived_features),
        "source_input_cols": list(cfg.source_input_columns),
        "source_output_cols": list(cfg.source_output_columns),
        "architecture": {
            "in_features": cfg.in_features,
            "out_features": cfg.out_features,
            "hidden_widths": list(cfg.hidden_widths),
            "grid_size": cfg.grid_size,
            "spline_order": cfg.spline_order,
            "grid_range": [-1.0, 1.0],
        },
        "optimizer": {
            "name": cfg.optimizer,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
        "source_sha256": source_sha256,
    }


class SingleKAN(nn.Module):
    """一个转轮对应一个固定的多输出 KAN。"""

    def __init__(self, profile: str | RotorProfileConfig) -> None:
        super().__init__()
        cfg = get_profile(profile)
        self.profile = cfg.name
        self.contract = cfg
        self.kan = build_kan(
            cfg.in_features,
            cfg.out_features,
            hidden_widths=list(cfg.hidden_widths),
            grid_size=cfg.grid_size,
            spline_order=cfg.spline_order,
            grid_range=(-1.0, 1.0),
        )
        # Existing callers inspect ``model.config`` directly.
        self.config: KANConfig = self.kan.config
        self.metadata = build_metadata(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

    def predict_scaled(self, features: Any, *, device: str | torch.device = "cpu") -> np.ndarray:
        matrix = _ensure_matrix(
            features, columns=self.contract.in_features, label="缩放输入"
        ).astype(np.float32)
        dev = torch.device(device)
        self.to(dev)
        self.eval()
        with torch.no_grad():
            output = self(torch.from_numpy(matrix).to(dev)).detach().cpu().numpy()
        if not np.isfinite(output).all():
            raise ValueError("模型输出存在非有限值")
        return output.astype(np.float64)

    def predict(
        self,
        features: Any,
        preprocessor: Preprocessor | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        """预测六个直接状态；传入 preprocessor 时返回真实物理量。"""

        if preprocessor is None:
            return self.predict_scaled(features, device=device)
        scaled = preprocessor.transform_features(features)
        predicted_scaled = self.predict_scaled(scaled, device=device)
        return preprocessor.inverse_targets(predicted_scaled)


def build_single_kan(profile: str | RotorProfileConfig) -> SingleKAN:
    return SingleKAN(profile)


# Short aliases kept intentionally small so train/integration callers need not
# duplicate profile-specific construction logic.
TargetTransform = ReversibleTargetTransform
build_model = build_single_kan
fit_scalers = fit_preprocessor


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_artifact(
    output_dir: str | Path,
    model: SingleKAN,
    preprocessor: Preprocessor,
    *,
    metadata: Mapping[str, Any] | None = None,
    checkpoint_payload: Mapping[str, Any] | None = None,
) -> Path:
    """保存单模型 artifact；不创建 routing.json。"""

    target = Path(output_dir)
    model_dir = target / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    merged = dict(model.metadata)
    if metadata:
        merged.update(dict(metadata))
    # Reassert hard output contract after any caller metadata merge.
    cfg = get_profile(model.profile)
    if tuple(merged.get("input_cols", ())) != tuple(cfg.input_columns):
        raise ValueError("artifact input_cols 与 profile 契约不一致")
    if tuple(merged.get("output_cols", ())) != tuple(cfg.direct_outputs):
        raise ValueError("artifact output_cols 必须严格等于六个 direct_outputs")
    if tuple(merged.get("direct_outputs", ())) != tuple(DIRECT_OUTPUT_COLUMNS):
        raise ValueError("artifact direct_outputs 必须严格为六个状态输出")
    if tuple(merged.get("output_cols", ())) != tuple(merged.get("direct_outputs", ())):
        raise ValueError("artifact output_cols 与 direct_outputs 列契约不一致")
    if set(merged.get("audit_columns", ())) != set(AUDIT_COLUMNS):
        raise ValueError("artifact audit_columns 必须包含 power 与处理出口露点")
    if merged.get("derived_features", []) not in ([], ()):
        raise ValueError("单 KAN artifact 不允许 derived_features")
    if set(preprocessor.target_columns) != set(DIRECT_OUTPUT_COLUMNS):
        raise ValueError("目标 scaler 不得包含 power 或处理出口露点")

    supplied = dict(checkpoint_payload or {})
    # A caller may pass the best checkpoint while ``model`` has continued to
    # train. Preserve that explicit state; otherwise serialize the current
    # in-memory model.
    model_state = supplied.get("model_state_dict")
    if model_state is None and supplied and all(
        isinstance(value, torch.Tensor) for value in supplied.values()
    ):
        model_state = supplied
    if model_state is None:
        model_state = model.state_dict()
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint_payload.model_state_dict 必须是 mapping")
    # Keep the deployed file weights-only-loadable. Optimizer/preprocessor
    # objects from a training checkpoint (notably sklearn scalers) are not
    # serialized into the runtime model file.
    # Store a bare state_dict, matching the existing runtime artifact contract
    # and allowing callers to use ``load_state_dict(torch.load(...))`` directly.
    torch.save(dict(model_state), model_dir / "best_model.pt")
    with (model_dir / "scalers.pkl").open("wb") as stream:
        pickle.dump(preprocessor.state_dict(), stream)
    scaler_meta = {
        "input_cols": list(preprocessor.input_columns),
        "output_cols": list(preprocessor.target_columns),
        "log1p_targets": list(preprocessor.target_transform.log1p_targets),
        "feature_range": [-1.0, 1.0],
    }
    _write_json(model_dir / "scaler_meta.json", scaler_meta)
    _write_json(model_dir / "metadata.json", merged)
    _write_json(target / "model_metadata.json", merged)
    return target


def _torch_load(path: Path, device: str | torch.device) -> Mapping[str, Any]:
    kwargs = {"map_location": device}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:  # older torch without weights_only
        return torch.load(path, **kwargs)


def load_artifact(
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SingleKAN, Preprocessor, dict[str, Any]]:
    """加载并再次校验单模型 artifact 的列/维度契约。"""

    target = Path(output_dir)
    metadata_path = target / "model_metadata.json"
    if not metadata_path.exists():
        metadata_path = target / "model" / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"artifact 缺少模型元数据: {target}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    profile = str(metadata.get("profile", ""))
    cfg = get_profile(profile)
    if tuple(metadata.get("input_cols", ())) != tuple(cfg.input_columns):
        raise ValueError("artifact input_cols 契约不匹配")
    if tuple(metadata.get("output_cols", ())) != tuple(cfg.direct_outputs):
        raise ValueError("artifact output_cols 必须严格等于 direct_outputs")
    if tuple(metadata.get("direct_outputs", ())) != tuple(DIRECT_OUTPUT_COLUMNS):
        raise ValueError("artifact direct_outputs 必须为六列")
    if tuple(metadata.get("output_cols", ())) != tuple(metadata.get("direct_outputs", ())):
        raise ValueError("artifact output_cols 与 direct_outputs 列契约不一致")
    if tuple(metadata.get("derived_outputs", ())) != ("处理出口露点",):
        raise ValueError("artifact derived_outputs 必须严格为处理出口露点")
    if set(metadata.get("audit_columns", ())) != set(AUDIT_COLUMNS):
        raise ValueError("artifact audit_columns 必须包含 power 与处理出口露点")
    if metadata.get("derived_features", []) not in ([], ()):
        raise ValueError("artifact 含有禁止的 derived_features")
    architecture = metadata.get("architecture")
    expected_architecture = {
        "in_features": cfg.in_features,
        "out_features": cfg.out_features,
        "hidden_widths": list(cfg.hidden_widths),
        "grid_size": cfg.grid_size,
        "spline_order": cfg.spline_order,
    }
    if not isinstance(architecture, Mapping):
        raise ValueError("artifact 缺少 architecture 元数据")
    missing_architecture = [
        key for key in expected_architecture if key not in architecture
    ]
    if missing_architecture:
        raise ValueError(f"artifact architecture 缺少字段: {missing_architecture}")
    for key, expected_value in expected_architecture.items():
        if architecture[key] != expected_value:
            raise ValueError(f"artifact architecture.{key} 契约不匹配")
    model = build_single_kan(cfg.name)
    model_path = target / "best_model.pt"
    if not model_path.exists():
        model_path = target / "model" / "best_model.pt"
    if not model_path.exists():
        raise ValueError(f"artifact 缺少模型 checkpoint: {target}")
    payload = _torch_load(model_path, device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    scaler_path = target / "scalers.pkl"
    if not scaler_path.exists():
        scaler_path = target / "model" / "scalers.pkl"
    if not scaler_path.exists():
        raise ValueError(f"artifact 缺少 scaler: {target}")
    with scaler_path.open("rb") as stream:
        scaler_state = pickle.load(stream)
    preprocessor = Preprocessor.from_state_dict(scaler_state)
    if tuple(preprocessor.input_columns) != tuple(cfg.input_columns):
        raise ValueError("artifact input scaler 输入契约不匹配")
    model.metadata = metadata
    return model, preprocessor, metadata
