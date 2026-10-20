"""KAN（Kolmogorov-Arnold Network）模型实现。

为避免引入第三方依赖（如 pykan），此处提供一个轻量、可训练的 KAN 实现，
核心思想参考 efficient-kan：每条边由 (B-spline 基 + 残差线性) 组合而成，
B-spline 基函数定义在均匀网格上，spline 阶数与网格数可配置。

接口约定：
- 输入张量形状: ``[N, in_features]``（静态回归任务，不含时间维）。
- 输出张量形状: ``[N, out_features]``。
- 因此可直接复用项目中现有的 evaluation / preprocessing 流水线
  （它们对 2D 数组与 2D 输出兼容）。

主要类：
- ``KANLayer``：单层 KAN，输入 ``[B, I]`` -> 输出 ``[B, O]``。
- ``KANRegressor``：堆叠多层 KANLayer 的回归模型。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import nn


def _build_grid(grid_size: int, grid_range: tuple, spline_order: int) -> torch.Tensor:
    """构造扩展的 B-spline 节点。

    实际节点数 = grid_size + 1 + 2*spline_order（左右各扩 spline_order 个，
    用于支持 spline_order 阶 B-spline 的递推计算）。
    """

    lo, hi = grid_range
    h = (hi - lo) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
    grid = grid * h + lo  # [grid_size + 1 + 2*spline_order]
    return grid


class KANLayer(nn.Module):
    """单层 KAN：每个 (输入维, 输出维) 之间由 B-spline + 残差线性组合。

    参数：
        in_features:    输入维度 I
        out_features:   输出维度 O
        grid_size:      B-spline 网格区间数，默认 5
        spline_order:   B-spline 阶数，默认 3
        grid_range:     B-spline 输入定义域（默认 [-1, 1]，配合标准化输入使用）
        base_activation:基底激活，默认 SiLU
        scale_noise:    spline 系数初始化噪声幅度
        scale_base:     残差线性初始化幅度
        scale_spline:   spline 输出权重初始化幅度
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1.0, 1.0),
        base_activation: nn.Module | None = None,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"KANLayer 维度必须为正：in_features={in_features}, out_features={out_features}"
            )
        if grid_size < 1:
            raise ValueError(f"grid_size 必须 >= 1，实际为 {grid_size}")
        if spline_order < 1:
            raise ValueError(f"spline_order 必须 >= 1，实际为 {spline_order}")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = tuple(float(x) for x in grid_range)

        grid = _build_grid(grid_size, self.grid_range, spline_order)
        # grid 在所有输入维度共享，注册为 buffer 以随模型迁移设备
        # 形状: [in_features, grid_size + 1 + 2*spline_order]
        self.register_buffer(
            "grid",
            grid.unsqueeze(0).expand(in_features, -1).contiguous(),
        )

        # 残差线性部分：输出 = base_activation(x) @ base_weight^T
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)

        # spline 系数：每个 (out, in) 对应 grid_size + spline_order 个系数
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        with torch.no_grad():
            noise = (
                (torch.rand_like(self.spline_weight) - 0.5)
                * scale_noise
                / grid_size
            )
            self.spline_weight.copy_(noise * scale_spline)

        self.base_activation = base_activation if base_activation is not None else nn.SiLU()

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """计算 B-spline 基函数值。

        Args:
            x: 输入张量 ``[B, in_features]``。
        Returns:
            B-spline 基函数张量 ``[B, in_features, grid_size + spline_order]``。
        """

        if x.dim() != 2 or x.size(-1) != self.in_features:
            raise ValueError(
                f"KANLayer 期望 [B, {self.in_features}] 输入，实际为 {tuple(x.shape)}"
            )

        grid = self.grid  # [I, G]
        x_exp = x.unsqueeze(-1)  # [B, I, 1]

        # 0 阶 B-spline：示性函数
        bases = ((x_exp >= grid[:, :-1]) & (x_exp < grid[:, 1:])).to(x.dtype)
        # 递推到 spline_order 阶
        for k in range(1, self.spline_order + 1):
            left_num = x_exp - grid[:, : -(k + 1)]
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_num = grid[:, k + 1 :] - x_exp
            right_den = grid[:, k + 1 :] - grid[:, 1:-k]
            # 防止除零
            left = left_num / left_den.clamp(min=1e-8) * bases[:, :, :-1]
            right = right_num / right_den.clamp(min=1e-8) * bases[:, :, 1:]
            bases = left + right

        return bases.contiguous()  # [B, I, G_b]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"KANLayer 仅支持 2D 输入，收到 {tuple(x.shape)}")

        # 残差线性
        base_out = torch.nn.functional.linear(self.base_activation(x), self.base_weight)
        # spline 部分
        bases = self.b_splines(x)  # [B, I, G_b]
        # spline_weight: [O, I, G_b] -> 求和到 [B, O]
        spline_out = torch.einsum("big,oig->bo", bases, self.spline_weight)
        return base_out + spline_out


@dataclass
class KANConfig:
    """KAN 模型构建参数。"""

    in_features: int
    out_features: int
    hidden_widths: List[int] = field(default_factory=lambda: [16])
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple = (-1.0, 1.0)


class KANRegressor(nn.Module):
    """多层 KAN 回归模型。

    形状: ``[N, in_features] -> [N, out_features]``。
    """

    def __init__(self, config: KANConfig) -> None:
        super().__init__()
        if config.in_features <= 0 or config.out_features <= 0:
            raise ValueError(
                f"KAN in/out 维度必须为正: in={config.in_features}, out={config.out_features}"
            )
        widths: List[int] = [config.in_features, *config.hidden_widths, config.out_features]
        layers: List[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(
                KANLayer(
                    in_features=widths[i],
                    out_features=widths[i + 1],
                    grid_size=config.grid_size,
                    spline_order=config.spline_order,
                    grid_range=config.grid_range,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 1:
            # 兼容 [B, 1, F] -> [B, F]
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(
                f"KANRegressor 期望 [B, in_features] 输入，实际为 {tuple(x.shape)}"
            )
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def extra_repr(self) -> str:  # pragma: no cover - 仅日志用
        widths = [self.config.in_features, *self.config.hidden_widths, self.config.out_features]
        return (
            f"widths={widths}, grid_size={self.config.grid_size}, "
            f"spline_order={self.config.spline_order}, grid_range={self.config.grid_range}"
        )


def build_kan(
    in_features: int,
    out_features: int,
    *,
    hidden_widths: Optional[List[int]] = None,
    grid_size: int = 5,
    spline_order: int = 3,
    grid_range: tuple = (-1.0, 1.0),
) -> KANRegressor:
    """根据输入输出维度构建 KAN 网络。"""

    if in_features <= 0:
        raise ValueError(f"in_features 必须为正，实际 {in_features}")
    if out_features <= 0:
        raise ValueError(f"out_features 必须为正，实际 {out_features}")
    cfg = KANConfig(
        in_features=in_features,
        out_features=out_features,
        hidden_widths=list(hidden_widths) if hidden_widths else [max(in_features, 8)],
        grid_size=int(grid_size),
        spline_order=int(spline_order),
        grid_range=tuple(float(x) for x in grid_range),
    )
    return KANRegressor(cfg)
