"""KAN 网络训练循环。

与 ``src.trainer`` 的 LSTM 训练循环类似，但针对 KAN 静态回归场景做了精简：
- 数据形状统一为 ``[N, in_features]`` -> ``[N, out_features]``；
- 训练循环检测 NaN / Inf 并安全停止；
- 支持早停、梯度裁剪、学习率调度，并在最佳验证损失时保存权重。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    """构造 ``[N, F]`` -> ``[N, K]`` 的 DataLoader。"""

    if x.ndim != 2:
        raise ValueError(f"KAN 期望 2D 输入数组，收到 {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"KAN 期望 2D 输出数组，收到 {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"输入输出样本数不匹配: x={x.shape[0]}, y={y.shape[0]}"
        )
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_loss(name: str, weights: Optional[List[float]] = None) -> nn.Module:
    """构造损失函数。

    当 ``weights`` 为 None 时返回原生 mean-reduce 损失；
    否则返回按输出列加权的损失，列权重会在内部归一化为均值=1.0，
    以保证加权后损失量级与原始 mse/huber/mae 可比。
    """

    if weights is None:
        if name == "mse":
            return nn.MSELoss()
        if name == "huber":
            return nn.SmoothL1Loss()
        if name == "mae" or name == "l1":
            return nn.L1Loss()
        raise ValueError(f"未知损失函数: {name}")
    if name not in ("mse", "huber", "mae", "l1"):
        raise ValueError(f"未知损失函数: {name}")
    return WeightedRegressionLoss(name=name, weights=weights)


class WeightedRegressionLoss(nn.Module):
    """按输出列加权的 mse/huber/mae 损失。

    输入 ``pred`` / ``target`` 均为 ``[N, K]``。
    列权重 ``weights`` 会在初始化时归一化为均值=1.0，损失仍组内以 mean 聚合。
    """

    def __init__(self, name: str, weights: List[float]) -> None:
        super().__init__()
        if not weights:
            raise ValueError("WeightedRegressionLoss 接收到空的 weights")
        w = torch.tensor(weights, dtype=torch.float32)
        if torch.any(w < 0):
            raise ValueError(f"输出列权重不允许为负: {weights}")
        if float(w.sum()) <= 0:
            raise ValueError(f"输出列权重总和必须 > 0: {weights}")
        # 归一化使均值=1.0，保证损失量级与原始 mean-reduce 可比
        w = w * (len(w) / float(w.sum()))
        self.register_buffer("weights", w)
        self.name = name

    def _elementwise(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.name == "mse":
            return (pred - target) ** 2
        if self.name == "huber":
            # PyTorch SmoothL1: beta=1.0，此处保持一致
            diff = pred - target
            absd = diff.abs()
            return torch.where(absd < 1.0, 0.5 * diff * diff, absd - 0.5)
        # mae / l1
        return (pred - target).abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"WeightedRegressionLoss 输入形状不匹配: pred={tuple(pred.shape)}, target={tuple(target.shape)}"
            )
        if pred.shape[-1] != self.weights.shape[0]:
            raise ValueError(
                f"WeightedRegressionLoss 输出维与权重长度不一致: pred[-1]={pred.shape[-1]}, weights={self.weights.shape[0]}"
            )
        elem = self._elementwise(pred, target)  # [N, K]
        weighted = elem * self.weights.to(elem.device).view(1, -1)
        return weighted.mean()


def build_optimizer(
    model: nn.Module, name: str, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"未知优化器: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, name: str, epochs: int
) -> Optional[Any]:
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    raise ValueError(f"未知学习率调度器: {name}")


@dataclass
class TrainHistory:
    """KAN 训练历史。"""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = math.inf
    stop_reason: str = "completed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stop_reason": self.stop_reason,
        }


class _NumericalAnomaly(RuntimeError):
    """训练过程中出现 NaN / Inf 时抛出。"""


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            if pred.shape != yb.shape:
                raise _NumericalAnomaly(
                    f"模型输出形状 {tuple(pred.shape)} 与目标 {tuple(yb.shape)} 不匹配"
                )
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise _NumericalAnomaly(
                    f"损失出现 NaN/Inf：value={float(loss)}"
                )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = xb.size(0)
            total += float(loss.detach().cpu()) * bs
            n += bs
    return total / max(n, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    loss_name: str,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    scheduler_name: str,
    early_stopping_patience: int,
    grad_clip: float,
    device: str,
    checkpoint_path: Path,
    loss_weights: Optional[List[float]] = None,
    logger=None,
) -> TrainHistory:
    """KAN 训练主循环，记录损失并保存最佳权重。"""

    dev = _resolve_device(device)
    model = model.to(dev)
    loss_fn = build_loss(loss_name, weights=loss_weights)
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)

    history = TrainHistory()
    no_improve = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        try:
            train_loss = _run_epoch(
                model, train_loader, loss_fn, dev, optimizer, grad_clip
            )
            val_loss = _run_epoch(model, val_loader, loss_fn, dev)
        except _NumericalAnomaly as exc:
            history.stop_reason = f"numerical_anomaly: {exc}"
            if logger is not None:
                logger.error("训练在第 %d 轮因数值异常停止：%s", epoch, exc)
            break
        except RuntimeError as exc:
            # 维度/依赖错误等
            history.stop_reason = f"runtime_error: {exc}"
            if logger is not None:
                logger.error("训练在第 %d 轮因运行时错误停止：%s", epoch, exc)
            break

        cur_lr = float(optimizer.param_groups[0]["lr"])
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.learning_rate.append(cur_lr)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        if val_loss < history.best_val_loss - 1e-8:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1

        if logger is not None:
            logger.info(
                "epoch=%d train_loss=%.6f val_loss=%.6f lr=%.6g best_val=%.6f",
                epoch,
                train_loss,
                val_loss,
                cur_lr,
                history.best_val_loss,
            )

        if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
            history.stop_reason = f"early_stopping@{epoch}"
            if logger is not None:
                logger.info(
                    "早停触发于第 %d 轮（连续无提升 %d 轮）", epoch, no_improve
                )
            break

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
    return history


def save_history(history: TrainHistory, target_dir: Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "train_history.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(history.to_dict(), f, ensure_ascii=False, indent=2)
    return target
