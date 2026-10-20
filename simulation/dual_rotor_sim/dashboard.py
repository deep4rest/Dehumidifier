"""基于既有 SystemResult 生成单工况结果仪表板。"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .core.system.system import SystemResult, convergence_label


_PROCESS_PATH_NODES = [
    ("OA", "新风入口"),
    ("AfterFan1", "风机1后"),
    ("AfterCC1", "表冷1后"),
    ("AfterR1_P", "转轮1处理出"),
    ("BeforeFan2", "混风后"),
    ("AfterFan2", "风机2后"),
    ("AfterCC2", "表冷2后"),
    ("Supply", "送风出口"),
]

_ENERGY_NAMES = {
    "CC1_kW": "前表冷器",
    "CC2_kW": "中表冷器",
    "Heater_Low_kW": "主转轮再生加热",
    "Heater_OA_kW": "新风转轮再生加热",
}
_PUBLIC_ENERGY_KEYS = tuple(_ENERGY_NAMES)


@dataclass(frozen=True)
class _DashboardAirState:
    """图片展示所需的空气状态，不携带干空气流量。"""

    T: float
    W: float
    Tdp: float
    RH: float
    h: float


@dataclass(frozen=True)
class _DashboardPayload:
    """实时结果与历史记录共用的最小绘图数据。"""

    supply_state: _DashboardAirState
    cooling_kW: float
    heating_kW: float
    energy_breakdown: dict[str, float]
    intermediate_states: dict[str, _DashboardAirState]
    solver_label: str = "未评估收敛性"


def _dashboard_air_state_from_result(state: Any) -> _DashboardAirState:
    return _DashboardAirState(
        T=float(state.T),
        W=float(state.W),
        Tdp=float(state.Tdp),
        RH=float(state.RH),
        h=float(state.h),
    )


def _dashboard_air_state_from_record(
    state: Mapping[str, Any],
) -> _DashboardAirState:
    return _DashboardAirState(
        T=float(state["T_degC"]),
        W=float(state["W_g_per_kg"]),
        Tdp=(float("-inf") if state["Tdp_degC"] is None and state["W_g_per_kg"] == 0 else float(state["Tdp_degC"])),
        RH=float(state["RH_fraction"]),
        h=float(state["h_kJ_per_kg_da"]),
    )


def _public_energy_breakdown(
    breakdown: Mapping[str, Any],
) -> dict[str, float]:
    """固定顺序提取四个热工负荷，拒绝旧风机和转轮电机字段。"""
    return {
        key: float(breakdown[key])
        for key in _PUBLIC_ENERGY_KEYS
        if key in breakdown
    }


def _dashboard_payload_from_result(result: SystemResult) -> _DashboardPayload:
    return _DashboardPayload(
        solver_label=convergence_label(result.convergence),
        supply_state=_dashboard_air_state_from_result(result.supply_state),
        cooling_kW=float(result.cooling_kW),
        heating_kW=float(result.heating_kW),
        energy_breakdown=_public_energy_breakdown(result.energy_breakdown),
        intermediate_states={
            str(name): _dashboard_air_state_from_result(state)
            for name, state in result.intermediate_states.items()
        },
    )


def _dashboard_payload_from_record(
    document: Mapping[str, Any],
) -> _DashboardPayload:
    """从历史 ``result.json`` 提取绘图数据，不重新运行模型。"""
    result = document["result"]
    summary = result["energy_summary_kW"]
    return _DashboardPayload(
        solver_label=convergence_label(result.get("convergence", {})),
        supply_state=_dashboard_air_state_from_record(result["supply_state"]),
        cooling_kW=float(summary["cooling_load"]),
        heating_kW=float(summary["heating_load"]),
        energy_breakdown=_public_energy_breakdown(result["energy_breakdown_kW"]),
        intermediate_states={
            str(name): _dashboard_air_state_from_record(state)
            for name, state in result["intermediate_states"].items()
        },
    )


def _state_table_rows(
    states: Mapping[str, _DashboardAirState],
) -> list[list[str]]:
    """生成六列表格行；流量不属于图片展示契约。"""
    rows: list[list[str]] = []
    for key, label in _PROCESS_PATH_NODES:
        state = states.get(key)
        if state is None:
            rows.append([label, "—", "—", "—", "—", "—"])
        else:
            rows.append(
                [
                    label,
                    f"{state.T:.2f}",
                    f"{state.W:.4f}",
                    f"{state.Tdp:.2f}",
                    f"{state.RH * 100:.2f}",
                    f"{state.h:.2f}",
                ]
            )
    return rows


def _configure_matplotlib():
    cache_dir = Path(__file__).resolve().parents[2] / ".tmp" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    chosen = None
    for font_path, companion_paths in (
        (
            Path("/mnt/c/Windows/Fonts/msyh.ttc"),
            (Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),),
        ),
        (Path("/mnt/c/Windows/Fonts/simhei.ttf"), ()),
    ):
        if font_path.is_file():
            fm.fontManager.addfont(font_path)
            for companion_path in companion_paths:
                if companion_path.is_file():
                    fm.fontManager.addfont(companion_path)
            chosen = fm.FontProperties(fname=font_path).get_name()
            break
    if chosen is None:
        candidates = (
            "PingFang HK",
            "Hiragino Sans GB",
            "Microsoft YaHei",
            "SimHei",
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
        )
        available = {font.name for font in fm.fontManager.ttflist}
        chosen = next((name for name in candidates if name in available), None)
    if chosen:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        print("[告警] 未找到中文字体，结果图中文字可能显示异常。", file=sys.stderr)
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _config_value(
    config: dict[str, Any],
    section: str,
    key: str,
    default: Any,
) -> Any:
    values = config.get(section)
    return values.get(key, default) if isinstance(values, dict) else default


def _style_table(table: Any, *, header_color: str = "#4c72b0") -> None:
    """统一表头与交替行样式，提升高分辨率图片的可读性。"""
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#b7c1cc")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef3f8")
        else:
            cell.set_facecolor("#ffffff")


def _save_dashboard_payload(
    *,
    payload: _DashboardPayload,
    config: dict[str, Any],
    target: Path,
) -> Path:
    """把已归一化的数据保存为紧凑的三段式仪表盘。"""
    plt = _configure_matplotlib()
    import numpy as np
    import matplotlib.gridspec as gridspec

    target.parent.mkdir(parents=True, exist_ok=True)
    supply = payload.supply_state
    states = payload.intermediate_states
    breakdown = payload.energy_breakdown

    oa_t = _config_value(config, "outdoor_air", "T_degC", "—")
    oa_w = _config_value(config, "outdoor_air", "W_g_per_kg", "—")
    heater_low = _config_value(
        config, "control_setpoints", "heater_low_T_degC", "—"
    )
    heater_oa = _config_value(
        config, "control_setpoints", "heater_oa_T_degC", "—"
    )

    fig = plt.figure(figsize=(18, 13.5), facecolor="white")
    fig.suptitle(
        f"双转轮除湿机单工况仿真结果 · {payload.solver_label}\n"
        f"新风 {oa_t}℃ / {oa_w} g/kg；"
        f"主转轮/新风转轮再生加热 {heater_low}/{heater_oa}℃",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(top=0.90, bottom=0.055, left=0.055, right=0.965)
    grid = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[0.90, 1.20, 0.85],
        hspace=0.34,
        wspace=0.30,
    )

    ax_kpi = fig.add_subplot(grid[0, 0])
    ax_kpi.axis("off")
    kpis = [
        ["送风温度", f"{supply.T:.2f} ℃"],
        ["送风含湿量", f"{supply.W:.4f} g/kg"],
        ["送风露点", f"{supply.Tdp:.2f} ℃"],
        ["送风相对湿度", f"{supply.RH * 100:.2f} %"],
        ["送风比焓", f"{supply.h:.2f} kJ/kg"],
    ]
    table = ax_kpi.table(
        cellText=kpis,
        colLabels=["关键指标", "预测值"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)
    _style_table(table)
    ax_kpi.set_title("送风预测", fontsize=12, pad=8)

    ax_group = fig.add_subplot(grid[0, 1])
    group_labels = ["制冷负荷", "加热负荷"]
    group_values = [
        payload.cooling_kW,
        payload.heating_kW,
    ]
    bars = ax_group.bar(
        group_labels,
        group_values,
        color=["#55a868", "#c44e52"],
        width=0.62,
    )
    ax_group.bar_label(bars, fmt="%.2f", fontsize=9)
    ax_group.set_ylabel("kW")
    ax_group.set_title("冷/热负荷", fontsize=12)
    ax_group.tick_params(axis="x", labelrotation=15)
    ax_group.grid(axis="y", alpha=0.25)

    ax_energy = fig.add_subplot(grid[0, 2])
    labels = [_ENERGY_NAMES[key] for key in breakdown]
    values = list(breakdown.values())
    bars = ax_energy.barh(
        labels,
        values,
        color=["#c44e52" if value >= 0 else "#4c72b0" for value in values],
        height=0.62,
    )
    ax_energy.axvline(0, color="black", linewidth=0.8)
    ax_energy.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax_energy.set_xlabel("kW")
    ax_energy.set_title("热工组件负荷", fontsize=12)
    ax_energy.tick_params(axis="y", labelsize=8)
    ax_energy.grid(axis="x", alpha=0.25)
    ax_energy.margins(x=0.12)

    path_states = [states.get(key) for key, _label in _PROCESS_PATH_NODES]
    x_values = np.arange(len(_PROCESS_PATH_NODES))
    temperatures = [state.T if state else np.nan for state in path_states]
    dew_points = [state.Tdp if state else np.nan for state in path_states]
    humidities = [state.W if state else np.nan for state in path_states]

    # 中部空气状态路径占整行。
    ax_path = fig.add_subplot(grid[1, :])
    ax_path.plot(x_values, temperatures, "o-", label="干球温度 (℃)")
    ax_path.plot(x_values, dew_points, "s--", label="露点温度 (℃)")
    ax_path.set_ylabel("温度 (℃)")
    ax_path.set_xticks(x_values)
    ax_path.set_xticklabels(
        [label for _key, label in _PROCESS_PATH_NODES],
        rotation=22,
        ha="right",
        fontsize=8,
    )
    ax_path.grid(axis="y", alpha=0.25)
    ax_humidity = ax_path.twinx()
    ax_humidity.plot(
        x_values,
        humidities,
        "^:",
        color="#dd8452",
        label="含湿量 (g/kg)",
    )
    ax_humidity.set_ylabel("含湿量 (g/kg)", color="#dd8452")
    lines_1, names_1 = ax_path.get_legend_handles_labels()
    lines_2, names_2 = ax_humidity.get_legend_handles_labels()
    ax_path.legend(lines_1 + lines_2, names_1 + names_2, loc="best", fontsize=9)
    ax_path.set_title("处理侧空气状态沿程", fontsize=12)

    ax_table = fig.add_subplot(grid[2, :])
    ax_table.axis("off")
    state_table = ax_table.table(
        cellText=_state_table_rows(states),
        colLabels=[
            "处理侧节点",
            "干球温度 (℃)",
            "含湿量 (g/kg)",
            "露点 (℃)",
            "相对湿度 (%)",
            "比焓 (kJ/kg)",
        ],
        colWidths=[0.17, 0.17, 0.17, 0.15, 0.17, 0.17],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    state_table.auto_set_font_size(False)
    state_table.set_fontsize(9)
    state_table.scale(1.0, 1.38)
    _style_table(state_table)
    ax_table.set_title("处理侧空气状态沿程数据表", fontsize=12, pad=10)

    fig.savefig(target, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def save_dashboard(
    *,
    result: SystemResult,
    config: dict[str, Any],
    target: Path,
) -> Path:
    """使用本次已完成的仿真结果保存 ``fig1_dashboard.png``。"""
    return _save_dashboard_payload(
        payload=_dashboard_payload_from_result(result),
        config=config,
        target=target,
    )


def save_dashboard_from_record(*, record_path: Path, target: Path) -> Path:
    """从历史 ``result.json`` 原样读取数值并重绘仪表盘。"""
    with record_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    return _save_dashboard_payload(
        payload=_dashboard_payload_from_record(document),
        config=document["config"],
        target=target,
    )
