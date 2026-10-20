"""风量候选训练的 KAN 与 MLP 架构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from training.src.kan_model import build_kan

from .config import DIRECT_OUTPUT_COLUMNS, VOLUME_INPUT_COLUMNS_BY_PROFILE


@dataclass(frozen=True)
class CandidateContract:
    profile: str
    model_type: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    hidden_widths: tuple[int, ...]
    grid_size: int | None
    spline_order: int | None
    learning_rate: float
    weight_decay: float
    max_epochs: int

    @property
    def in_features(self) -> int:
        return len(self.input_columns)

    @property
    def out_features(self) -> int:
        return len(self.output_columns)


def candidate_contract(profile: str, model_type: str) -> CandidateContract:
    profile = str(profile).strip().lower()
    model_type = str(model_type).strip().lower()
    try:
        input_columns = tuple(VOLUME_INPUT_COLUMNS_BY_PROFILE[profile])
    except KeyError as exc:
        raise ValueError(f"未知 profile: {profile!r}") from exc
    if model_type not in {"mlp", "kan"}:
        raise ValueError(f"未知 model_type: {model_type!r}")
    if model_type == "kan":
        if profile == "xf":
            return CandidateContract(profile, model_type, input_columns, tuple(DIRECT_OUTPUT_COLUMNS), (24,), 6, 3, 5e-4, 1e-4, 300)
        return CandidateContract(profile, model_type, input_columns, tuple(DIRECT_OUTPUT_COLUMNS), (16,), 5, 3, 1e-3, 0.0, 200)
    return CandidateContract(profile, model_type, input_columns, tuple(DIRECT_OUTPUT_COLUMNS), (128, 128), None, None, 1e-3, 1e-4, 300 if profile == "xf" else 200)


class CandidateModel(nn.Module):
    def __init__(self, contract: CandidateContract) -> None:
        super().__init__()
        self.contract = contract
        self.profile = contract.profile
        self.model_type = contract.model_type
        if contract.model_type == "kan":
            self.network = build_kan(
                contract.in_features,
                contract.out_features,
                hidden_widths=list(contract.hidden_widths),
                grid_size=int(contract.grid_size),
                spline_order=int(contract.spline_order),
                grid_range=(-1.0, 1.0),
            )
        else:
            layers: list[nn.Module] = []
            widths = (contract.in_features, *contract.hidden_widths, contract.out_features)
            for index in range(len(widths) - 1):
                layers.append(nn.Linear(widths[index], widths[index + 1]))
                if index < len(widths) - 2:
                    layers.append(nn.SiLU())
            self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def predict(self, features: Any, preprocessor: Any, *, device: str | torch.device = "cpu", batch_size: int = 4096) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.contract.in_features:
            raise ValueError(f"输入必须为 [N,{self.contract.in_features}]")
        scaled = preprocessor.transform_features(values)
        outputs: list[np.ndarray] = []
        target_device = torch.device(device)
        self.to(target_device)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(scaled), max(1, int(batch_size))):
                    batch = torch.from_numpy(scaled[start:start + batch_size]).to(target_device)
                    outputs.append(self(batch).detach().cpu().numpy())
        finally:
            self.train(was_training)
        prediction = np.concatenate(outputs, axis=0) if outputs else np.empty((0, self.contract.out_features))
        if not np.isfinite(prediction).all():
            raise ValueError("模型输出存在非有限值")
        return preprocessor.inverse_targets(prediction)


def build_candidate_model(profile: str, model_type: str) -> CandidateModel:
    return CandidateModel(candidate_contract(profile, model_type))
