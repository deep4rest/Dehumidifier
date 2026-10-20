"""XF 新风转轮的五个固定 KAN 候选。

本模块只声明实验自变量，模型实例始终由
``training.src.kan_model.build_kan`` 构造，不复制 KAN 架构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from training.src.kan_model import KANRegressor, build_kan

from .config import MODEL_INPUT_COLUMNS, MODEL_OUTPUT_COLUMNS

__all__ = [
    "CANDIDATES",
    "OUTPUT_WEIGHTS",
    "CandidateConfig",
    "build_candidate_model",
    "get_candidate",
]


OUTPUT_WEIGHTS = (1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0)
_PROCESS_OUTLET_W = "Label_Process_Outlet_AbsHum"


@dataclass(frozen=True)
class CandidateConfig:
    """不允许运行时漂移的单个候选训练契约。"""

    name: str
    hidden_widths: tuple[int, ...]
    grid_size: int
    loss: str
    log1p_targets: tuple[str, ...]
    learning_rate: float
    weight_decay: float
    input_columns: tuple[str, ...] = MODEL_INPUT_COLUMNS
    output_columns: tuple[str, ...] = MODEL_OUTPUT_COLUMNS
    spline_order: int = 3
    batch_size: int = 64
    max_epochs: int = 300
    grad_clip: float = 1.0
    seed: int = 42
    validation_interval_steps: int = 2_000
    patience_evaluations: int = 10
    huber_delta: float = 1.0
    feature_range: tuple[float, float] = (-1.0, 1.0)


CANDIDATES: Mapping[str, CandidateConfig] = {
    "xf_v1_baseline": CandidateConfig(
        name="xf_v1_baseline",
        hidden_widths=(16,),
        grid_size=5,
        loss="mse",
        log1p_targets=(),
        learning_rate=1e-3,
        weight_decay=0.0,
    ),
    "xf_v2_balanced": CandidateConfig(
        name="xf_v2_balanced",
        hidden_widths=(24,),
        grid_size=6,
        loss="mse",
        log1p_targets=(),
        learning_rate=5e-4,
        weight_decay=1e-4,
    ),
    "xf_v3_capacity": CandidateConfig(
        name="xf_v3_capacity",
        hidden_widths=(32,),
        grid_size=7,
        loss="mse",
        log1p_targets=(),
        learning_rate=5e-4,
        weight_decay=1e-4,
    ),
    "xf_v4_weighted": CandidateConfig(
        name="xf_v4_weighted",
        hidden_widths=(24,),
        grid_size=6,
        loss="weighted_mse",
        log1p_targets=(_PROCESS_OUTLET_W,),
        learning_rate=5e-4,
        weight_decay=1e-4,
    ),
    "xf_v5_robust": CandidateConfig(
        name="xf_v5_robust",
        hidden_widths=(32, 16),
        grid_size=6,
        loss="weighted_huber",
        log1p_targets=(_PROCESS_OUTLET_W,),
        learning_rate=3e-4,
        weight_decay=5e-4,
    ),
}


def get_candidate(name: str) -> CandidateConfig:
    """按固定名称返回候选，未知名称直接失败。"""

    try:
        return CANDIDATES[name]
    except KeyError as exc:
        raise ValueError(f"未知 XF KAN 候选: {name}") from exc


def build_candidate_model(candidate: CandidateConfig) -> KANRegressor:
    """使用项目唯一 KAN builder 构建精确的 10→7 模型。"""

    if candidate.input_columns != MODEL_INPUT_COLUMNS:
        raise ValueError("XF 候选输入必须精确等于锁定的 10 个原始字段")
    if candidate.output_columns != MODEL_OUTPUT_COLUMNS:
        raise ValueError("XF 候选输出必须精确等于锁定的 7 个直接标签")
    return build_kan(
        in_features=len(candidate.input_columns),
        out_features=len(candidate.output_columns),
        hidden_widths=list(candidate.hidden_widths),
        grid_size=candidate.grid_size,
        spline_order=candidate.spline_order,
        grid_range=candidate.feature_range,
    )
