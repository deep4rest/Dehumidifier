"""KAN 训练前字段与数据质量评估入口脚本。

用法示例：

    python3 assess_fields.py --config configs/kan.yaml \
        --output-dir artifacts/field_quality_assessment

该脚本只读取原始数据，不会修改训练配置或原始数据文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import List, Optional

# 默认输入字段：用户在需求文档中固化的 8 个建模输入
DEFAULT_INPUT_FIELDS: List[str] = [
    "入口含湿量",
    "转轮转速",
    "再生温度",
    "冷吹入口温度",
    "再生角度",
    "风速",
    "冷吹角度",
    "处理入口温度",
]

# 默认输出字段：用户在需求文档中固化的 8 个建模输出
DEFAULT_OUTPUT_FIELDS: List[str] = [
    "冷吹出口温度",
    "冷吹出口含湿量",
    "处理出口含湿量",
    "处理出口温度",
    "处理出口露点",
    "再生出口含湿量",
    "再生出口温度",
    "power",
]


def _ensure_src_on_path() -> None:
    """允许从项目根目录直接以 `python3 assess_fields.py` 启动。"""

    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KAN 训练前字段与数据质量评估"
    )
    parser.add_argument(
        "--config",
        default="archive/training/configs/kan.yaml",
        help="训练配置文件路径，用于读取 excel_path / sheet_name 等",
    )
    parser.add_argument(
        "--excel-path",
        default=None,
        help="覆盖 Excel 数据路径；优先级高于配置文件",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="指定要评估的工作表，默认读取行数最多的工作表",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="覆盖默认输入字段列表（空格分隔）",
    )
    parser.add_argument(
        "--outputs",
        nargs="*",
        default=None,
        help="覆盖默认输出字段列表（空格分隔）",
    )
    parser.add_argument(
        "--output-dir",
        default="results/training/legacy/field_quality_assessment",
        help="评估报告输出目录",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.3,
        help="缺失率高风险阈值，默认 0.3",
    )
    parser.add_argument(
        "--outlier-ratio-threshold",
        type=float,
        default=0.1,
        help="IQR 异常率高风险阈值，默认 0.1",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="IQR 异常检测倍数，默认 1.5",
    )
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=3.0,
        help="Z-score 异常检测阈值，默认 3.0",
    )
    return parser.parse_args()


def main() -> int:
    _ensure_src_on_path()

    from archive.training.src.config import load_config, PROJECT_ROOT  # noqa: WPS433
    from archive.training.src.data_loader import (  # noqa: WPS433
        list_sheets,
        load_sheet,
        scan_workbook,
        select_sheet,
    )
    from archive.training.src.field_quality import run_assessment  # noqa: WPS433

    args = parse_args()

    # ---- 读取配置以确定数据路径 ----
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}", file=sys.stderr)
        return 2

    cfg = load_config(config_path)

    excel_path = Path(args.excel_path) if args.excel_path else cfg.excel_path
    if not excel_path.is_absolute():
        excel_path = PROJECT_ROOT / excel_path
    if not excel_path.exists():
        print(f"❌ Excel 数据文件不存在: {excel_path}", file=sys.stderr)
        return 2

    # ---- 选择目标 sheet ----
    try:
        summaries = scan_workbook(excel_path)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 读取 Excel 失败: {exc}", file=sys.stderr)
        return 3

    requested_sheet: Optional[str] = args.sheet or cfg.sheet_name
    target = select_sheet(summaries, requested_sheet)
    print(
        f"📄 使用工作表: {target.sheet_name}（{target.n_rows} 行 x {target.n_cols} 列）"
    )

    # ---- 加载 DataFrame ----
    try:
        df = load_sheet(excel_path, target.sheet_name)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 加载工作表失败: {exc}", file=sys.stderr)
        return 3

    # ---- 字段配置 ----
    inputs = args.inputs if args.inputs else DEFAULT_INPUT_FIELDS
    outputs = args.outputs if args.outputs else DEFAULT_OUTPUT_FIELDS
    print(f"🔎 输入字段（{len(inputs)}）: {inputs}")
    print(f"🎯 输出字段（{len(outputs)}）: {outputs}")
    print(f"📊 数据列总数: {len(df.columns)}")

    # ---- 输出目录 ----
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # ---- 执行评估 ----
    machine, artifacts = run_assessment(
        df=df,
        excel_path=excel_path,
        sheet_name=target.sheet_name,
        requested_inputs=inputs,
        requested_outputs=outputs,
        output_dir=output_dir,
        missing_threshold=args.missing_threshold,
        outlier_ratio_threshold=args.outlier_ratio_threshold,
        iqr_factor=args.iqr_factor,
        zscore_threshold=args.zscore_threshold,
        grid_range=tuple(cfg.grid_range) if cfg.grid_range else (-1.0, 1.0),
    )

    # ---- 控制台摘要 ----
    summary = machine["summary"]
    print("=" * 60)
    print(
        f"未匹配字段数: {summary['n_unmatched_fields']}，"
        f"高风险字段数: {summary['n_high_risk_fields']}"
    )
    if summary["high_risk_fields"]:
        print("⚠️ 高风险字段:")
        for n in summary["high_risk_fields"]:
            print(f"  - {n}")
    if summary["can_proceed_to_training"]:
        print("✅ 数据检查通过，可以进入 KAN 训练阶段。")
    else:
        print("❌ 数据检查未通过，请阅读 Markdown 报告处理高风险问题。")
    print(f"Markdown 报告: {artifacts.markdown_path}")
    print(f"JSON 报告:     {artifacts.json_path}")
    print(f"CSV 统计:      {artifacts.csv_path}")
    return 0 if summary["can_proceed_to_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
