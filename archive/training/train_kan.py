"""转轮 KAN 建模训练入口。

执行顺序（覆盖 task-item.md 全部 10 项）：

1. 解析命令行 / YAML 配置，创建运行目录、固定随机种子、打印关键参数；
2. 扫描 Excel 工作表并加载目标 sheet，识别可用字段；
3. 字段画像 + 输入输出字段校验（含字段不存在 / 为空的硬阻断）；
4. 数据数值化、缺失/异常值处理、清洗报告写入；
5. 数据划分 + 训练集拟合缩放器 + 划分级别的样本张量构造；
6. 根据输入输出维度构建 KAN 网络；
7. 训练循环：每轮记录损失、监控 NaN/Inf、保存最佳权重；
8. 在测试集上评估：MAE/RMSE/MAPE/R2，归一化与真实尺度均输出；
9. 保存模型权重、配置快照、字段选择、训练日志、评估指标与图表；
10. ``--smoke-test`` 模式执行端到端最小化验证，打印产物清单。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from archive.training.src.config import Config, RunPaths, load_config, set_seed, snapshot_config
from archive.training.src.data_loader import (
    coerce_numeric_columns,
    load_sheet,
    scan_workbook,
    select_sheet,
)
from archive.training.src.dataset_manifest import build_dataset_manifest, save_dataset_manifest
from archive.training.src.evaluation import (
    EvaluationResult,
    evaluate_split,
    plot_loss_curve,
    plot_prediction_scatter,
    plot_residual_distribution,
    save_evaluation,
)
from archive.training.src.feature_selector import save_feature_selection, select_features
from training.src.feature_engineering import apply_derived_features, list_available_features
from archive.training.src.field_profiler import build_field_dictionary, save_field_dictionary
from training.src.kan_model import build_kan
from archive.training.src.kan_trainer import (
    _resolve_device,
    make_loader,
    save_history,
    train_model,
)
from training.src.preprocessing import (
    clean_dataframe,
    fit_scalers,
    save_cleaning_report,
    save_scalers,
    transform_split,
)
from archive.training.src.splitting import build_split_report, save_split_report, split_dataframe


def _build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("rotary_kan")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="转轮 KAN 建模训练入口")
    parser.add_argument(
        "--config", "-c", type=str, default="archive/training/configs/kan.yaml",
        help="训练配置 YAML 文件路径",
    )
    parser.add_argument("--run-name", type=str, default=None, help="覆盖运行名称")
    parser.add_argument(
        "--inputs", type=str, default=None,
        help="逗号分隔的输入字段，覆盖 YAML 的 data.input_columns",
    )
    parser.add_argument(
        "--outputs", type=str, default=None,
        help="逗号分隔的输出字段，覆盖 YAML 的 data.output_columns",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖批大小")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="最小可运行模式：限制训练轮数、批量与模型规模，用于端到端验证",
    )
    return parser.parse_args()


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.run_name:
        cfg.run_name = args.run_name
    if args.inputs:
        cfg.input_columns = [c.strip() for c in args.inputs.split(",") if c.strip()]
    if args.outputs:
        cfg.output_columns = [c.strip() for c in args.outputs.split(",") if c.strip()]
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.learning_rate = float(args.lr)
    if args.seed is not None:
        cfg.seed = int(args.seed)


def _apply_smoke_test(cfg: Config) -> None:
    cfg.epochs = min(cfg.epochs, 3)
    cfg.batch_size = max(8, min(cfg.batch_size, 64))
    cfg.early_stopping_patience = max(cfg.early_stopping_patience, 0)
    cfg.hidden_widths = [max(8, min(cfg.hidden_widths[0] if cfg.hidden_widths else 8, 16))]


def _validate_columns(
    df,
    inputs: Optional[List[str]],
    outputs: Optional[List[str]],
    logger: logging.Logger,
) -> bool:
    """硬性校验显式输入/输出字段是否全部存在；只要有 1 个缺失就阻止训练。"""

    missing_inputs = [c for c in (inputs or []) if c not in df.columns]
    missing_outputs = [c for c in (outputs or []) if c not in df.columns]
    if missing_inputs or missing_outputs:
        if missing_inputs:
            logger.error("输入字段不存在于数据中: %s", missing_inputs)
        if missing_outputs:
            logger.error("输出字段不存在于数据中: %s", missing_outputs)
        return False
    return True


def _build_static_arrays(
    df,
    input_cols: List[str],
    output_cols: List[str],
    x_scaler,
    y_scaler,
):
    """将一个划分转为 KAN 所需的 ``[N, in]`` / ``[N, out]`` 数组（删除残余 NaN 行）。"""

    x = x_scaler.transform(df[input_cols].to_numpy(dtype=np.float32))
    y = y_scaler.transform(df[output_cols].to_numpy(dtype=np.float32))
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    return x[mask].astype(np.float32), y[mask].astype(np.float32), int((~mask).sum())


def _write_training_plan(
    cfg: Config,
    paths: RunPaths,
    sheet_name: str,
    inputs: List[str],
    outputs: List[str],
    eval_result: EvaluationResult,
    history_dict: Dict[str, Any],
) -> Path:
    """将本次训练的关键配置与指标写入训练方案说明，便于复现。"""

    target = paths.report_dir / "training_plan.md"
    lines: List[str] = []
    lines.append("# 转轮 KAN 训练方案说明\n")
    lines.append("## 数据来源\n")
    lines.append(f"- Excel 文件: `{cfg.excel_path}`")
    lines.append(f"- 工作表: `{sheet_name}`\n")
    lines.append("## 输入字段\n")
    for c in inputs:
        lines.append(f"- {c}")
    lines.append("\n## 输出字段\n")
    for c in outputs:
        lines.append(f"- {c}")
    lines.append("\n## 预处理\n")
    lines.append(f"- 缺失值策略: {cfg.missing_strategy}（阈值 {cfg.missing_threshold}）")
    lines.append(f"- 异常值方法: {cfg.outlier_method}（factor={cfg.outlier_factor}）")
    lines.append(
        f"- 缩放器: {cfg.scaler}"
        + (f"（区间 {tuple(cfg.scale_range)}）" if cfg.scaler == "minmax" else "")
    )
    lines.append("\n## 划分\n")
    lines.append(
        f"- 方式: {cfg.split_method}，比例 train/val/test = "
        f"{cfg.train_ratio}/{cfg.val_ratio}/{cfg.test_ratio}"
    )
    lines.append("\n## 模型结构（KAN）\n")
    lines.append(
        f"- hidden_widths={cfg.hidden_widths}, grid_size={cfg.grid_size}, "
        f"spline_order={cfg.spline_order}, grid_range={tuple(cfg.grid_range)}"
    )
    lines.append("\n## 训练\n")
    lines.append(
        f"- batch_size={cfg.batch_size}, epochs={cfg.epochs}, lr={cfg.learning_rate}, "
        f"loss={cfg.loss}, optimizer={cfg.optimizer}, scheduler={cfg.scheduler}, "
        f"early_stopping_patience={cfg.early_stopping_patience}, grad_clip={cfg.grad_clip}, "
        f"seed={cfg.seed}"
    )
    if cfg.loss_weights:
        lines.append(
            "- loss_weights="
            + ", ".join(f"{k}={v:g}" for k, v in cfg.loss_weights.items())
        )
    lines.append(
        f"- 训练终止原因: {history_dict.get('stop_reason', 'completed')}; "
        f"best_epoch={history_dict.get('best_epoch')}, "
        f"best_val_loss={history_dict.get('best_val_loss')}"
    )
    lines.append("\n## 测试集指标（真实物理尺度）\n")
    for col, m in eval_result.metrics_real.items():
        lines.append(
            f"- {col}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
            f"MAPE={m['mape']:.2f}%, R2={m['r2']:.4f}"
        )
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main() -> int:
    args = parse_args()
    cfg: Config = load_config(args.config)
    _apply_overrides(cfg, args)
    if args.smoke_test:
        _apply_smoke_test(cfg)

    paths = RunPaths.create(cfg.artifacts_dir, cfg.run_name)
    snapshot_config(cfg, paths.config_snapshot)
    set_seed(cfg.seed)

    logger = _build_logger(paths.run_dir / "train.log")
    logger.info("==== 转轮 KAN 训练 ====")
    logger.info("运行目录: %s", paths.run_dir)
    logger.info("配置文件: %s", cfg.config_path)
    logger.info("数据文件: %s", cfg.excel_path)
    logger.info(
        "关键超参: epochs=%d, batch_size=%d, lr=%g, seed=%d, scaler=%s, "
        "hidden_widths=%s, grid_size=%d, spline_order=%d",
        cfg.epochs, cfg.batch_size, cfg.learning_rate, cfg.seed,
        cfg.scaler, cfg.hidden_widths, cfg.grid_size, cfg.spline_order,
    )

    # ---------- 1. 工作表扫描 ----------
    try:
        summaries = scan_workbook(
            cfg.excel_path,
            header_row=cfg.header_row,
            column_aliases=cfg.column_aliases,
        )
    except FileNotFoundError as exc:
        logger.error("数据文件不存在: %s", exc)
        return 2
    except Exception as exc:
        logger.error("Excel 读取失败: %s", exc)
        return 2

    workbook_summary = [s.to_dict() for s in summaries]
    with open(paths.field_dir / "workbook_summary.json", "w", encoding="utf-8") as f:
        json.dump(workbook_summary, f, ensure_ascii=False, indent=2)

    target_sheet = select_sheet(summaries, cfg.sheet_name)
    logger.info(
        "选用工作表: %s（%d 行 × %d 列）",
        target_sheet.sheet_name, target_sheet.n_rows, target_sheet.n_cols,
    )
    df = load_sheet(
        cfg.excel_path,
        target_sheet.sheet_name,
        header_row=cfg.header_row,
        column_aliases=cfg.column_aliases,
    )
    logger.info("可用字段: %s", list(df.columns))

    # ---------- 2. 字段硬校验与数值化 ----------
    if not _validate_columns(df, cfg.input_columns, cfg.output_columns, logger):
        logger.error("显式字段不存在，已阻止训练")
        return 3
    numeric_columns = list(
        dict.fromkeys(
            list(cfg.input_columns or [])
            + list(cfg.output_columns or [])
            + list(cfg.group_columns)
        )
    )
    df = coerce_numeric_columns(df, numeric_columns)

    manifest = build_dataset_manifest(
        cfg.excel_path,
        sheet_name=target_sheet.sheet_name,
        header_row=cfg.header_row,
        column_aliases=cfg.column_aliases,
        df=df,
        input_columns=list(cfg.input_columns or []),
        output_columns=list(cfg.output_columns or []),
        group_columns=cfg.group_columns,
    )
    manifest_path = save_dataset_manifest(manifest, paths.field_dir)
    logger.info(
        "数据清单已保存: %s（rows=%d, groups=%d, sha256=%s…）",
        manifest_path,
        manifest["rows"],
        manifest["group_count"],
        manifest["source_sha256"][:12],
    )
    # 保留源表行号用于审计 train/val/test 的逐样本归属；不参与模型输入。
    # header_row 为零基索引，Excel 数据首行 = header_row + 2（再加当前零基行索引）。
    df["__source_excel_row__"] = (
        np.arange(len(df), dtype=np.int64) + cfg.header_row + 2
    )

    # ---------- 3. 字段画像 ----------
    profiles = build_field_dictionary(
        df, cfg.input_keywords, cfg.output_keywords, cfg.missing_threshold,
    )
    save_field_dictionary(profiles, paths.field_dir)
    logger.info("字段字典已保存到 %s", paths.field_dir)

    # ---------- 4. 输入输出字段确认 ----------
    selection = select_features(
        profiles,
        df,
        explicit_inputs=cfg.input_columns,
        explicit_outputs=cfg.output_columns,
        explicit_time=cfg.time_column,
        explicit_group=cfg.group_column,
        missing_threshold=cfg.missing_threshold,
    )
    save_feature_selection(selection, paths.field_dir)
    for warn in selection.warnings:
        logger.warning(warn)

    if not selection.inputs:
        logger.error("未识别到输入字段，请通过 --inputs 或配置 data.input_columns 指定")
        return 3
    if not selection.outputs:
        logger.error("未识别到输出字段，请通过 --outputs 或配置 data.output_columns 指定")
        return 3
    logger.info("输入字段(%d): %s", len(selection.inputs), selection.inputs)
    logger.info("输出字段(%d): %s", len(selection.outputs), selection.outputs)

    # ---------- 4.5 派生特征（可选） ----------
    if cfg.derived_features:
        unknown = [n for n in cfg.derived_features if n not in list_available_features()]
        if unknown:
            logger.error(
                "配置中的派生特征未注册: %s；可用列表: %s",
                unknown, list_available_features(),
            )
            return 3
        try:
            df, added = apply_derived_features(df, cfg.derived_features)
        except (KeyError, ValueError) as exc:
            logger.error("派生特征生成失败: %s", exc)
            return 3
        # 避免重复追加；保持 selection.inputs 中出现顺序
        for name in added:
            if name not in selection.inputs:
                selection.inputs.append(name)
        logger.info("启用派生特征(%d): %s", len(added), added)
        logger.info(
            "扩展后输入字段(%d): %s", len(selection.inputs), selection.inputs,
        )
        # 重新保存 feature_selection，便于后续复现
        save_feature_selection(selection, paths.field_dir)

    # ---------- 5. 数据清洗 ----------
    cols_to_clean = list(dict.fromkeys(selection.inputs + selection.outputs))
    cleaned, clean_report = clean_dataframe(
        df,
        cols_to_clean,
        missing_strategy=cfg.missing_strategy,
        outlier_method=cfg.outlier_method,
        outlier_factor=cfg.outlier_factor,
    )
    save_cleaning_report(clean_report, paths.report_dir)
    logger.info("清洗后样本数: %d/%d", clean_report.rows_after, clean_report.rows_before)
    if clean_report.rows_after < 10:
        logger.error(
            "清洗后样本不足（%d 行），无法进行有意义的训练",
            clean_report.rows_after,
        )
        return 4

    # ---------- 6. 数据划分 ----------
    split = split_dataframe(
        cleaned,
        method=cfg.split_method,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        group_column=selection.group_column,
        group_columns=cfg.group_columns,
        test_groups=cfg.test_groups,
        seed=cfg.seed,
    )
    split_report = build_split_report(
        split,
        output_cols=selection.outputs,
        time_column=selection.time_column,
        group_column=selection.group_column,
        group_columns=cfg.group_columns,
    )
    if split_report.get("group_leakage"):
        logger.error("设备几何组跨 train/val/test 泄漏，已阻止训练")
        return 4
    split_report["source_rows_total"] = int(
        len(split.train_df) + len(split.val_df) + len(split.test_df)
    )
    save_split_report(split_report, paths.report_dir)
    assignment_parts = []
    assignment_columns = ["__source_excel_row__"]
    if "实验ID" in cleaned.columns:
        assignment_columns.append("实验ID")
    assignment_columns.extend(
        c for c in cfg.group_columns if c not in assignment_columns
    )
    for partition, part in (
        ("train", split.train_df),
        ("val", split.val_df),
        ("test", split.test_df),
    ):
        assignment = part[assignment_columns].copy()
        assignment.insert(0, "partition", partition)
        assignment_parts.append(assignment)
    pd.concat(assignment_parts, ignore_index=True).to_csv(
        paths.report_dir / "split_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    logger.info(
        "划分: train=%d, val=%d, test=%d (method=%s)",
        len(split.train_df), len(split.val_df), len(split.test_df), split.method,
    )
    if len(split.train_df) == 0 or len(split.val_df) == 0 or len(split.test_df) == 0:
        logger.error("数据量不足以完成 train/val/test 划分，请调整划分比例或扩充数据")
        return 4

    # ---------- 7. 缩放器 ----------
    feature_range = None
    if cfg.scaler == "minmax":
        feature_range = (float(cfg.scale_range[0]), float(cfg.scale_range[1]))
    x_scaler, y_scaler = fit_scalers(
        split.train_df, selection.inputs, selection.outputs, cfg.scaler,
        feature_range=feature_range,
        log_transform_outputs=cfg.log_transform_outputs,
    )
    input_domain: Dict[str, Dict[str, float]] = {}
    for col in selection.inputs:
        series = np.asarray(split.train_df[col], dtype=np.float64)
        finite = series[np.isfinite(series)]
        if finite.size:
            input_domain[col] = {
                "min": float(finite.min()),
                "max": float(finite.max()),
            }
    save_scalers(
        x_scaler,
        y_scaler,
        selection.inputs,
        selection.outputs,
        paths.model_dir,
        input_domain=input_domain,
    )

    # ---------- 8. 构造 KAN 训练数组 ----------
    X_tr, Y_tr, drop_tr = _build_static_arrays(
        split.train_df, selection.inputs, selection.outputs, x_scaler, y_scaler,
    )
    X_va, Y_va, drop_va = _build_static_arrays(
        split.val_df, selection.inputs, selection.outputs, x_scaler, y_scaler,
    )
    X_te, Y_te, drop_te = _build_static_arrays(
        split.test_df, selection.inputs, selection.outputs, x_scaler, y_scaler,
    )
    if drop_tr or drop_va or drop_te:
        logger.warning(
            "因残余 NaN/Inf 删除样本: train=%d, val=%d, test=%d",
            drop_tr, drop_va, drop_te,
        )
    logger.info(
        "样本张量: train=%s, val=%s, test=%s",
        X_tr.shape, X_va.shape, X_te.shape,
    )
    if X_tr.shape[0] == 0 or X_va.shape[0] == 0 or X_te.shape[0] == 0:
        logger.error("某个划分的有效样本为空，无法训练")
        return 4

    train_loader = make_loader(X_tr, Y_tr, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_va, Y_va, cfg.batch_size, shuffle=False)
    test_loader = make_loader(X_te, Y_te, cfg.batch_size, shuffle=False)

    # ---------- 8. 构建模型 ----------
    try:
        model = build_kan(
            in_features=len(selection.inputs),
            out_features=len(selection.outputs),
            hidden_widths=cfg.hidden_widths,
            grid_size=cfg.grid_size,
            spline_order=cfg.spline_order,
            grid_range=tuple(cfg.grid_range),
        )
    except ValueError as exc:
        logger.error("KAN 构建失败: %s", exc)
        return 5
    logger.info("模型: %s", model)

    # ---------- 9. 训练 ----------
    # 列加权 loss：将 cfg.loss_weights（列名→权重）按 selection.outputs 顺序映射为向量；
    # 未在配置中出现的列默认为 1.0，所有列都为 1.0 时传 None 以走原生 mean-reduce 路径。
    loss_weights_list: Optional[List[float]] = None
    if cfg.loss_weights:
        unknown = [k for k in cfg.loss_weights if k not in selection.outputs]
        if unknown:
            logger.warning(
                "train.loss_weights 中存在未在输出列中的名称，将被忽略: %s", unknown,
            )
        weight_vec = [
            float(cfg.loss_weights.get(col, 1.0)) for col in selection.outputs
        ]
        # 若全部为 1.0 则不启用加权路径，保证与 v1/v3 完全一致
        if any(abs(w - 1.0) > 1e-9 for w in weight_vec):
            loss_weights_list = weight_vec
            logger.info(
                "启用列加权 loss：%s",
                ", ".join(
                    f"{col}={w:g}" for col, w in zip(selection.outputs, weight_vec)
                ),
            )
        else:
            logger.info("train.loss_weights 全为 1.0，未启用加权路径")

    history = train_model(
        model, train_loader, val_loader,
        epochs=cfg.epochs,
        loss_name=cfg.loss,
        optimizer_name=cfg.optimizer,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        scheduler_name=cfg.scheduler,
        early_stopping_patience=cfg.early_stopping_patience,
        grad_clip=cfg.grad_clip,
        device=cfg.device,
        checkpoint_path=paths.model_dir / "best_model.pt",
        loss_weights=loss_weights_list,
        logger=logger,
    )
    save_history(history, paths.report_dir)
    if history.train_loss:
        plot_loss_curve(history.to_dict(), paths.figure_dir / "loss_curve.png")
    if history.best_epoch < 0:
        logger.error("训练未产出有效轮次，原因：%s", history.stop_reason)
        return 6
    logger.info(
        "最佳验证损失: %.6f (epoch=%d, stop_reason=%s)",
        history.best_val_loss, history.best_epoch, history.stop_reason,
    )

    # ---------- 10. 评估 ----------
    device = _resolve_device(cfg.device)
    result, _, _, y_true_r, y_pred_r = evaluate_split(
        model, test_loader,
        device=device,
        output_cols=selection.outputs,
        y_scaler=y_scaler,
        error_thresholds=cfg.error_thresholds,
    )
    save_evaluation(result, paths.report_dir)
    plot_prediction_scatter(y_true_r, y_pred_r, selection.outputs, paths.figure_dir)
    plot_residual_distribution(y_true_r, y_pred_r, selection.outputs, paths.figure_dir)

    # 保存预测对比表（CSV，便于人工查看）
    np.savez(
        paths.report_dir / "predictions.npz",
        y_true=y_true_r, y_pred=y_pred_r, output_cols=np.array(selection.outputs),
    )
    for col, m in result.metrics_real.items():
        logger.info(
            "[test/real] %s: MAE=%.4f RMSE=%.4f MAPE=%.2f%% R2=%.4f",
            col, m["mae"], m["rmse"], m["mape"], m["r2"],
        )
    if result.threshold_violations:
        logger.warning("下列输出未达到误差阈值: %s", result.threshold_violations)

    # ---------- 11. 训练方案说明 ----------
    plan_path = _write_training_plan(
        cfg, paths,
        sheet_name=target_sheet.sheet_name,
        inputs=selection.inputs,
        outputs=selection.outputs,
        eval_result=result,
        history_dict=history.to_dict(),
    )
    logger.info("训练方案说明已保存: %s", plan_path)

    # ---------- 12. 结束总结 ----------
    summary = {
        "success": True,
        "run_dir": str(paths.run_dir),
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "metrics_real": result.metrics_real,
        "stop_reason": history.stop_reason,
    }
    with open(paths.run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("训练成功，主要产物保存于: %s", paths.run_dir)

    # ---------- 13. 最小可运行验证 ----------
    if args.smoke_test:
        required = [
            paths.config_snapshot,
            paths.field_dir / "field_dictionary.csv",
            paths.field_dir / "feature_selection.json",
            paths.field_dir / "dataset_manifest.json",
            paths.report_dir / "split_report.json",
            paths.report_dir / "split_assignments.csv",
            paths.report_dir / "cleaning_report.json",
            paths.model_dir / "best_model.pt",
            paths.model_dir / "scalers.pkl",
            paths.model_dir / "scaler_meta.json",
            paths.report_dir / "evaluation.json",
            paths.report_dir / "train_history.json",
            paths.figure_dir / "loss_curve.png",
            plan_path,
            paths.run_dir / "summary.json",
        ]
        missing = [str(p) for p in required if not Path(p).exists()]
        if missing:
            logger.error("最小可运行验证失败，缺失产物: %s", missing)
            return 7
        logger.info("最小可运行验证通过，全部关键产物均已生成")

    return 0


if __name__ == "__main__":
    sys.exit(main())
