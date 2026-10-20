"""配置加载与运行目录管理。"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# 历史配置工具位于 <项目根目录>/archive/training/src/。
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RunPaths:
    """单次训练运行涉及的目录与文件路径。"""

    run_dir: Path
    field_dir: Path
    model_dir: Path
    report_dir: Path
    figure_dir: Path
    config_snapshot: Path

    @classmethod
    def create(cls, artifacts_dir: Path, run_name: Optional[str]) -> "RunPaths":
        if not run_name:
            run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = artifacts_dir / run_name
        field_dir = run_dir / "fields"
        model_dir = run_dir / "model"
        report_dir = run_dir / "reports"
        figure_dir = run_dir / "figures"
        for path in (run_dir, field_dir, model_dir, report_dir, figure_dir):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            field_dir=field_dir,
            model_dir=model_dir,
            report_dir=report_dir,
            figure_dir=figure_dir,
            config_snapshot=run_dir / "config_snapshot.yaml",
        )


@dataclass
class Config:
    """训练配置容器。原始 YAML 字典存放在 raw 中以便快照保存。"""

    raw: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None

    # ---- 数据 ----
    excel_path: Path = Path("转轮实验数据收集.xlsx")
    sheet_name: Optional[str] = None
    header_row: int = 0
    column_aliases: Dict[str, str] = field(default_factory=dict)
    time_column: Optional[str] = None
    group_column: Optional[str] = None
    group_columns: List[str] = field(default_factory=list)
    input_columns: Optional[List[str]] = None
    output_columns: Optional[List[str]] = None
    input_keywords: List[str] = field(default_factory=list)
    output_keywords: List[str] = field(default_factory=list)
    # 派生特征名称列表（来自 src.feature_engineering.DERIVED_FEATURES），
    # 训练入口会在数据清洗前生成这些列并追加到 selection.inputs。
    derived_features: List[str] = field(default_factory=list)

    # ---- 预处理 ----
    missing_threshold: float = 0.3
    missing_strategy: str = "linear"
    outlier_method: str = "iqr"
    outlier_factor: float = 3.0
    scaler: str = "standard"
    # MinMax 目标区间，仅 scaler=minmax 时生效
    scale_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    # 在拟合输出缩放器前需要先做 log1p 变换的输出列名
    log_transform_outputs: List[str] = field(default_factory=list)

    # ---- 窗口 ----
    window_size: int = 16
    horizon: int = 1
    multi_step: int = 1
    stride: int = 1

    # ---- 划分 ----
    split_method: str = "time"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    test_groups: List[Any] = field(default_factory=list)

    # ---- 模型 (LSTM) ----
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # ---- 模型 (KAN) ----
    hidden_widths: List[int] = field(default_factory=lambda: [16])
    grid_size: int = 5
    spline_order: int = 3
    grid_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])

    # ---- 训练 ----
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    # 按输出列名指定加权（仅在 loss=mse/huber/mae 时生效），为 None 时所有列权重均为 1.0
    loss_weights: Optional[Dict[str, float]] = None
    optimizer: str = "adam"
    scheduler: str = "plateau"
    early_stopping_patience: int = 20
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42

    # ---- 评估 ----
    error_thresholds: Dict[str, float] = field(default_factory=dict)

    # ---- 输出 ----
    artifacts_dir: Path = PROJECT_ROOT / "results" / "training" / "legacy"
    run_name: Optional[str] = None


def _resolve_path(p: str | os.PathLike) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(config_path: str | os.PathLike) -> Config:
    """读取 YAML 配置并填充 Config 数据类。"""

    config_path = _resolve_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg = Config(raw=raw, config_path=config_path)

    data = raw.get("data", {}) or {}
    cfg.excel_path = _resolve_path(data.get("excel_path", cfg.excel_path))
    cfg.sheet_name = data.get("sheet_name")
    cfg.header_row = int(data.get("header_row", cfg.header_row))
    cfg.column_aliases = {
        str(k): str(v) for k, v in dict(data.get("column_aliases") or {}).items()
    }
    cfg.time_column = data.get("time_column")
    cfg.group_column = data.get("group_column")
    cfg.group_columns = [str(v) for v in (data.get("group_columns") or [])]
    cfg.input_columns = data.get("input_columns")
    cfg.output_columns = data.get("output_columns")
    cfg.input_keywords = list(data.get("input_keywords") or [])
    cfg.output_keywords = list(data.get("output_keywords") or [])
    cfg.derived_features = list(data.get("derived_features") or [])

    pre = raw.get("preprocess", {}) or {}
    cfg.missing_threshold = float(pre.get("missing_threshold", cfg.missing_threshold))
    cfg.missing_strategy = pre.get("missing_strategy", cfg.missing_strategy)
    cfg.outlier_method = pre.get("outlier_method", cfg.outlier_method)
    cfg.outlier_factor = float(pre.get("outlier_factor", cfg.outlier_factor))
    cfg.scaler = pre.get("scaler", cfg.scaler)
    raw_scale_range = pre.get("scale_range")
    if raw_scale_range is not None:
        cfg.scale_range = [float(v) for v in raw_scale_range]
    cfg.log_transform_outputs = list(pre.get("log_transform_outputs") or [])

    win = raw.get("windowing", {}) or {}
    cfg.window_size = int(win.get("window_size", cfg.window_size))
    cfg.horizon = int(win.get("horizon", cfg.horizon))
    cfg.multi_step = int(win.get("multi_step", cfg.multi_step))
    cfg.stride = int(win.get("stride", cfg.stride))

    split = raw.get("split", {}) or {}
    cfg.split_method = split.get("method", cfg.split_method)
    cfg.train_ratio = float(split.get("train_ratio", cfg.train_ratio))
    cfg.val_ratio = float(split.get("val_ratio", cfg.val_ratio))
    cfg.test_ratio = float(split.get("test_ratio", cfg.test_ratio))
    cfg.test_groups = list(split.get("test_groups") or [])

    model = raw.get("model", {}) or {}
    cfg.hidden_size = int(model.get("hidden_size", cfg.hidden_size))
    cfg.num_layers = int(model.get("num_layers", cfg.num_layers))
    cfg.dropout = float(model.get("dropout", cfg.dropout))
    cfg.bidirectional = bool(model.get("bidirectional", cfg.bidirectional))
    raw_hidden_widths = model.get("hidden_widths")
    if raw_hidden_widths is not None:
        cfg.hidden_widths = [int(v) for v in raw_hidden_widths]
    cfg.grid_size = int(model.get("grid_size", cfg.grid_size))
    cfg.spline_order = int(model.get("spline_order", cfg.spline_order))
    raw_grid_range = model.get("grid_range")
    if raw_grid_range is not None:
        cfg.grid_range = [float(v) for v in raw_grid_range]

    train = raw.get("train", {}) or {}
    cfg.batch_size = int(train.get("batch_size", cfg.batch_size))
    cfg.epochs = int(train.get("epochs", cfg.epochs))
    cfg.learning_rate = float(train.get("learning_rate", cfg.learning_rate))
    cfg.weight_decay = float(train.get("weight_decay", cfg.weight_decay))
    cfg.loss = train.get("loss", cfg.loss)
    raw_loss_weights = train.get("loss_weights")
    if raw_loss_weights is None:
        cfg.loss_weights = None
    else:
        if not isinstance(raw_loss_weights, dict):
            raise ValueError(
                f"train.loss_weights 必须是 'dict[列名->权重]' 形式，收到 {type(raw_loss_weights).__name__}"
            )
        cfg.loss_weights = {str(k): float(v) for k, v in raw_loss_weights.items()}
    cfg.optimizer = train.get("optimizer", cfg.optimizer)
    cfg.scheduler = train.get("scheduler", cfg.scheduler)
    cfg.early_stopping_patience = int(
        train.get("early_stopping_patience", cfg.early_stopping_patience)
    )
    cfg.grad_clip = float(train.get("grad_clip", cfg.grad_clip))
    cfg.device = train.get("device", cfg.device)
    cfg.seed = int(train.get("seed", cfg.seed))

    evalc = raw.get("evaluate", {}) or {}
    cfg.error_thresholds = dict(evalc.get("error_thresholds") or {})

    out = raw.get("output", {}) or {}
    cfg.artifacts_dir = _resolve_path(out.get("artifacts_dir", cfg.artifacts_dir))
    cfg.run_name = out.get("run_name")

    return cfg


def set_seed(seed: int) -> None:
    """设定全局随机种子，保证可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def snapshot_config(cfg: Config, target: Path) -> None:
    """将原始 YAML 配置快照写入运行目录。"""

    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.raw, f, allow_unicode=True, sort_keys=False)
