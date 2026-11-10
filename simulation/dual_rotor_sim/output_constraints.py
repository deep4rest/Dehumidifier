"""Training and runtime share the identical humidity-only nonnegative output head."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn

class HumidityNonnegativeHead(nn.Module):
    """A smooth, parameter-free lower bound at physical W=0, in scaled coordinates."""

    def __init__(self, output_columns: Sequence[str], target_zero_scaled: Sequence[float], *, beta: float = 50.0) -> None:
        super().__init__()
        zero = np.asarray(target_zero_scaled, dtype=np.float64)
        if zero.shape != (len(output_columns),) or not np.isfinite(zero).all():
            raise ValueError("非负湿度头需要六目标的有限物理零点边界")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("Softplus beta必须为正有限数")
        indices = [i for i,column in enumerate(output_columns) if "含湿量" in column]
        self.columns = tuple(output_columns[i] for i in indices)
        self.beta = float(beta)
        exact = zero[indices]
        rounded = exact.astype(np.float32)
        # Ceil by at most one float32 ULP; avoid a negative physical value when
        # inverse_transform subtracts the original float64 MinMax offset.
        rounded = np.where(rounded.astype(np.float64) < exact,
                           np.nextafter(rounded, np.float32(np.inf)), rounded).astype(np.float32)
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("physical_zero_scaled", torch.from_numpy(exact.copy()))
        self.register_buffer("lower_bound", torch.from_numpy(rounded.copy()))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        humidity = raw.index_select(1, self.indices)
        positive = self.lower_bound + torch.nn.functional.softplus(humidity-self.lower_bound, beta=self.beta)
        return raw.index_copy(1, self.indices, positive)

    def metadata(self) -> dict[str, Any]:
        return {"type":"humidity_softplus_lower_bound", "columns":list(self.columns), "beta":self.beta,
                "physical_zero_scaled":self.physical_zero_scaled.detach().cpu().tolist(),
                "lower_bound_scaled_fp32":self.lower_bound.detach().cpu().tolist(),
                "boundary_rounding":"ceil_float32", "formula":"z0 + softplus(raw-z0, beta)"}


class HumidityConstrainedModel(nn.Module):
    """Runtime wrapper with the same state_dict keys as the trained candidate."""
    def __init__(self, network: nn.Module, output_columns: Sequence[str], target_zero_scaled: Sequence[float], *, beta: float = 50.0):
        super().__init__()
        self.network = network
        self.humidity_head = HumidityNonnegativeHead(output_columns, target_zero_scaled, beta=beta)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.humidity_head(self.network(features))
