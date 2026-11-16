"""Shared inference policy; training/checkpoint tensors remain unchanged."""
from __future__ import annotations

import copy
import torch
from torch import nn

INFERENCE_NUMERIC_POLICY = "float64_body_float32_io_v1"
ONNX_SESSION_POLICY = "cpu_sequential_no_fusion_no_prepacking_v1"


class StableInferenceModel(nn.Module):
    """Compute the loaded network/head in float64, round once at the output ABI.

    A separate copy preserves the caller's original float32 model. All promoted
    checkpoint values and float32-ceil humidity bounds are represented exactly
    in float64. Physical input/scaler and public output boundaries stay float32.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.inner = copy.deepcopy(model).double()

    @property
    def humidity_head(self):
        return self.inner.humidity_head

    @property
    def network(self):
        return self.inner.network

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.inner(features.to(dtype=torch.float64)).to(dtype=torch.float32)
