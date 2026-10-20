"""模型评估、反归一化与可视化输出。

提供：
- 在归一化与真实物理尺度上同时计算 MAE、RMSE、MAPE、R²；
- 误差阈值检查；
- 训练曲线、预测对比图、残差分布图。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# 在受限账户或无主目录写权限的 Windows 环境中，Matplotlib 默认缓存目录
# （~/.matplotlib）可能不可写。训练产物之外的缓存统一放到项目可写的 .tmp 下。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MPL_CACHE_DIR = _PROJECT_ROOT / ".tmp" / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")  # 非交互后端，适合脚本运行
import matplotlib.pyplot as plt

# 配置中文字体回退，避免在中文标题上出现缺字警告。
# 若系统未安装相应字体，则回退到默认字体。
matplotlib.rcParams["font.sans-serif"] = [
    "Heiti TC", "PingFang SC", "Hiragino Sans GB", "Songti SC",
    "Microsoft YaHei", "WenQuanYi Zen Hei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


# ---- 指标计算 -------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, output_cols: List[str]
) -> Dict[str, Dict[str, float]]:
    """对每个输出变量分别计算指标。输入形状 [N, multi_step, n_outputs]。"""

    # 把多步维度展开到样本维度，再按列计算
    if y_true.ndim == 3:
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred

    metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(output_cols):
        yt = y_true_flat[:, i]
        yp = y_pred_flat[:, i]
        metrics[name] = {
            "mae": _mae(yt, yp),
            "rmse": _rmse(yt, yp),
            "mape": _mape(yt, yp),
            "r2": _r2(yt, yp),
        }
    return metrics


# ---- 推理 ----------------------------------------------------

def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """对 DataLoader 进行推理，返回 (y_true, y_pred) 张量。"""

    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


# ---- 反归一化 -------------------------------------------------

def inverse_transform_outputs(
    y: np.ndarray, y_scaler: Any
) -> np.ndarray:
    """对形状 [N, multi_step, n_outputs] 的张量应用 y_scaler 反归一化。"""

    if y.ndim == 3:
        n, t, k = y.shape
        flat = y.reshape(-1, k)
        inv = y_scaler.inverse_transform(flat)
        return inv.reshape(n, t, k)
    return y_scaler.inverse_transform(y)


# ---- 可视化 ---------------------------------------------------

def plot_loss_curve(history: Dict[str, Any], target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=150)
    plt.close()
    return target


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_cols: List[str],
    target_dir: Path,
    *,
    max_points: int = 2000,
) -> List[Path]:
    """对每个输出绘制散点对比图（真实 vs 预测）。"""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if y_true.ndim == 3:
        yt = y_true.reshape(-1, y_true.shape[-1])
        yp = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        yt, yp = y_true, y_pred

    n = yt.shape[0]
    if n > max_points:
        idx = np.random.RandomState(0).choice(n, max_points, replace=False)
        yt = yt[idx]
        yp = yp[idx]

    paths: List[Path] = []
    for i, name in enumerate(output_cols):
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        path = target_dir / f"pred_vs_true_{i:02d}_{safe_name}.png"
        plt.figure(figsize=(6, 6))
        plt.scatter(yt[:, i], yp[:, i], s=8, alpha=0.5)
        lo = float(min(yt[:, i].min(), yp[:, i].min()))
        hi = float(max(yt[:, i].max(), yp[:, i].max()))
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        plt.xlabel(f"true {name}")
        plt.ylabel(f"pred {name}")
        plt.title(f"Prediction vs Ground Truth: {name}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_cols: List[str],
    target_dir: Path,
) -> List[Path]:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if y_true.ndim == 3:
        yt = y_true.reshape(-1, y_true.shape[-1])
        yp = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        yt, yp = y_true, y_pred

    paths: List[Path] = []
    for i, name in enumerate(output_cols):
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        path = target_dir / f"residual_{i:02d}_{safe_name}.png"
        plt.figure(figsize=(7, 4))
        plt.hist(yp[:, i] - yt[:, i], bins=50, alpha=0.8)
        plt.axvline(0, color="r", linestyle="--", linewidth=1)
        plt.xlabel("residual (pred - true)")
        plt.ylabel("count")
        plt.title(f"Residual Distribution: {name}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


# ---- 评估封装 -------------------------------------------------

@dataclass
class EvaluationResult:
    """整体评估结果。"""

    metrics_scaled: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metrics_real: Dict[str, Dict[str, float]] = field(default_factory=dict)
    threshold_violations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "metrics_scaled": self.metrics_scaled,
            "metrics_real": self.metrics_real,
            "threshold_violations": self.threshold_violations,
        }


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    output_cols: List[str],
    y_scaler: Any,
    error_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[EvaluationResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """评估单个数据划分，返回指标与张量（归一化与真实尺度）。"""

    y_true_s, y_pred_s = predict(model, loader, device)
    y_true_r = inverse_transform_outputs(y_true_s, y_scaler)
    y_pred_r = inverse_transform_outputs(y_pred_s, y_scaler)

    result = EvaluationResult(n_samples=int(y_true_s.shape[0]))
    result.metrics_scaled = compute_metrics(y_true_s, y_pred_s, output_cols)
    result.metrics_real = compute_metrics(y_true_r, y_pred_r, output_cols)

    error_thresholds = error_thresholds or {}
    for name, threshold in error_thresholds.items():
        if name in result.metrics_real:
            mae = result.metrics_real[name]["mae"]
            if mae > threshold:
                result.threshold_violations[name] = {
                    "mae": mae,
                    "threshold": float(threshold),
                }
    return result, y_true_s, y_pred_s, y_true_r, y_pred_r


def save_evaluation(result: EvaluationResult, target_dir: Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "evaluation.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return target
