"""双转轮单 KAN 实验的数据与模型契约。

这里的列名分成两层：``source_*`` 是 XLS 第一行的技术字段，作为源文件
的稳定键；``input_columns``/``output_columns`` 是数据适配器统一暴露给训练
和运行时的中文物理字段。这样可以处理两份 XLS 的历史命名差异，同时不
把 ``power`` 或 Excel 露点列偷偷变成模型输入/输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = REPOSITORY_ROOT / "data"
ARTIFACTS_DIR = REPOSITORY_ROOT / "artifacts"
TRAINING_RESULTS_DIR = REPOSITORY_ROOT / "results" / "training"

XF_SOURCE_PATH = DATA_DIR / "新风转轮数据集.xls"
MAIN_SOURCE_PATH = DATA_DIR / "主转轮数据集.xls"
SOURCE_PATHS = {"xf": XF_SOURCE_PATH, "main": MAIN_SOURCE_PATH}

SHEET_NAMES = {"xf": "CollectRecordXF", "main": "CollectRecord"}

# 哈希是原始 bytes 的 SHA-256；读取前必须重新计算，任何变化均硬失败。
EXPECTED_SOURCE_SHA256 = {
    "xf": "116ea18458bf831225765472440a21da8045cb8e9bfadd898f677b004ce026cb",
    "main": "a1173dee9789832e54087bd2a0f7d8865b8371fa7789d4a191449a45731f704f",
}

# 有效数据行数（技术表头之外还会跳过一行中文说明）。这些值用于源审计，
# 小样本/单元测试的 DataFrame 不要求必须达到全量行数。
EXPECTED_DATA_ROWS = {"xf": 57_186, "main": 60_610}
EXPECTED_RAW_ROWS = EXPECTED_DATA_ROWS  # 兼容审计脚本的旧命名

ID_COLUMN = "ID"
POWER_COLUMN = "power"
DEW_POINT_COLUMN = "处理出口露点"
AUDIT_COLUMNS = (POWER_COLUMN, DEW_POINT_COLUMN)

# 统一的六个直接空气状态输出。处理出口露点是运行时由处理出口含湿量
# 派生出的结果，绝不能出现在这个列表中。
DIRECT_OUTPUT_COLUMNS = (
    "处理出口温度",
    "处理出口含湿量",
    "冷吹出口温度",
    "冷吹出口含湿量",
    "再生出口温度",
    "再生出口含湿量",
)

# 新风沿用旧实验已经确认的十个原始量；这些都是采集字段，不是 feature
# engineering 产生的派生量。
XF_INPUT_COLUMNS = (
    "转轮厚度",
    "转轮直径",
    "处理入口温度(℃)",
    "入口含湿量(g/kg)",
    "处理风速(m/s)",
    "冷吹风速(m/s)",
    "转轮转速(rpm)",
    "再生温度(℃)",
    "再生风量",
    "再生入口含湿量",
)

# 主转轮严格固定为计划中的九个原始量；不包括风量、再生入口湿度、
# power、露点或任何 derived_features。
MAIN_INPUT_COLUMNS = (
    "转轮厚度",
    "转轮直径",
    "入口含湿量(g/kg)",
    "转轮转速(rpm)",
    "再生温度(℃)",
    "冷吹入口温度(℃)",
    "处理入口温度(℃)",
    "处理风速(m/s)",
    "冷吹风速(m/s)",
)

# 当前尺寸隔离 CSV 使用处理/冷吹风量（m³/h），而不是旧 XLS 中的风速。
# 这些列只供候选模型训练使用；旧 XLS 兼容入口仍保留上面的历史字段契约。
VOLUME_XF_INPUT_COLUMNS = (
    "转轮厚度",
    "转轮直径",
    "处理入口温度(℃)",
    "入口含湿量(g/kg)",
    "处理风量",
    "冷吹风量",
    "转轮转速(rpm)",
    "再生温度(℃)",
    "再生风量",
    "再生入口含湿量",
)
VOLUME_MAIN_INPUT_COLUMNS = (
    "转轮厚度",
    "转轮直径",
    "入口含湿量(g/kg)",
    "转轮转速(rpm)",
    "再生温度(℃)",
    "冷吹入口温度(℃)",
    "处理入口温度(℃)",
    "处理风量",
    "冷吹风量",
)
VOLUME_INPUT_COLUMNS_BY_PROFILE = {
    "xf": VOLUME_XF_INPUT_COLUMNS,
    "main": VOLUME_MAIN_INPUT_COLUMNS,
}
ALL_MODEL_INPUT_COLUMNS = tuple(
    dict.fromkeys((*XF_INPUT_COLUMNS, *MAIN_INPUT_COLUMNS,
                   *VOLUME_XF_INPUT_COLUMNS, *VOLUME_MAIN_INPUT_COLUMNS))
)

INPUT_COLUMNS_BY_PROFILE = {"xf": XF_INPUT_COLUMNS, "main": MAIN_INPUT_COLUMNS}
OUTPUT_COLUMNS_BY_PROFILE = {"xf": DIRECT_OUTPUT_COLUMNS, "main": DIRECT_OUTPUT_COLUMNS}
XF_OUTPUT_COLUMNS = DIRECT_OUTPUT_COLUMNS
MAIN_OUTPUT_COLUMNS = DIRECT_OUTPUT_COLUMNS
SOURCE_HASHES = EXPECTED_SOURCE_SHA256
SOURCE_FILES = SOURCE_PATHS

# 源文件技术字段。main 的 ``Label_React_Inlet_AbsHum`` 在中文说明行中
# 标为“冷处出口含湿量”，故在 canonical 层映射成冷吹出口含湿量；技术键
# 仍完整保留在元数据中，避免修改原始 XLS。
XF_SOURCE_INPUT_COLUMNS = (
    "TextBox_Rotor_depth",
    "TextBox_Rotor_diameter",
    "TextBox_Process_inlet_Temp",
    "TextBox_Purge_Inlet_AbsHum",
    "Label_Process_FV",
    "Label_Purge_FV",
    "TextBox_Rotation_Speed",
    "TextBox_React_Heater_Temp",
    "Label_React_CMH",
    "Label_React_Inlet_AbsHum",
)
MAIN_SOURCE_INPUT_COLUMNS = (
    "TextBox_Rotor_depth",
    "TextBox_Rotor_diameter",
    "TextBox_Purge_Inlet_AbsHum",
    "TextBox_Rotation_Speed",
    "TextBox_React_Heater_Temp",
    "TextBox_Purge_Inlet_Temp",
    "TextBox_Process_inlet_Temp",
    "Label_Process_FV",
    "Label_Purge_FV",
)

XF_SOURCE_OUTPUT_COLUMNS = (
    "Label_Process_Outlet_Temp",
    "Label_Process_Outlet_AbsHum",
    "Label_Purge_Outlet_Temp",
    "Label_Purge_Outlet_AbsHum",
    "Label_React_Outlet_Temp",
    "Label_React_Outlet_AbsHum",
)
MAIN_SOURCE_OUTPUT_COLUMNS = (
    "Label_Process_Outlet_Temp",
    "Label_Process_Outlet_AbsHum",
    "Label_Purge_Outlet_Temp",
    "Label_React_Inlet_AbsHum",
    "Label_React_Outlet_Temp",
    "Label_React_Outlet_AbsHum",
)
SOURCE_INPUT_COLUMNS_BY_PROFILE = {
    "xf": XF_SOURCE_INPUT_COLUMNS,
    "main": MAIN_SOURCE_INPUT_COLUMNS,
}
SOURCE_OUTPUT_COLUMNS_BY_PROFILE = {
    "xf": XF_SOURCE_OUTPUT_COLUMNS,
    "main": MAIN_SOURCE_OUTPUT_COLUMNS,
}
SOURCE_AUDIT_COLUMNS = ("Label_PowerConsumption", "Label_Process_Outlet_DP")

XF_INPUT_SOURCE_TO_CANONICAL = dict(zip(XF_SOURCE_INPUT_COLUMNS, XF_INPUT_COLUMNS))
MAIN_INPUT_SOURCE_TO_CANONICAL = dict(zip(MAIN_SOURCE_INPUT_COLUMNS, MAIN_INPUT_COLUMNS))
XF_OUTPUT_SOURCE_TO_CANONICAL = dict(zip(XF_SOURCE_OUTPUT_COLUMNS, DIRECT_OUTPUT_COLUMNS))
MAIN_OUTPUT_SOURCE_TO_CANONICAL = dict(zip(MAIN_SOURCE_OUTPUT_COLUMNS, DIRECT_OUTPUT_COLUMNS))

COMMON_SOURCE_TO_CANONICAL = {
    "ID": "ID",
    "Label_PowerConsumption": POWER_COLUMN,
    "Label_Process_Outlet_DP": DEW_POINT_COLUMN,
}
SOURCE_TO_CANONICAL_BY_PROFILE = {
    "xf": {
        **COMMON_SOURCE_TO_CANONICAL,
        **XF_INPUT_SOURCE_TO_CANONICAL,
        **XF_OUTPUT_SOURCE_TO_CANONICAL,
    },
    "main": {
        **COMMON_SOURCE_TO_CANONICAL,
        **MAIN_INPUT_SOURCE_TO_CANONICAL,
        **MAIN_OUTPUT_SOURCE_TO_CANONICAL,
    },
}

CANONICAL_TO_SOURCE_BY_PROFILE = {
    profile: {canonical: source for source, canonical in mapping.items()}
    for profile, mapping in SOURCE_TO_CANONICAL_BY_PROFILE.items()
}

# 说明行中常见的中文字段。适配器优先保留实际读取到的说明行，这个表仅
# 作为测试 DataFrame/缺少说明行时的稳定 fallback。
CANONICAL_LABELS = {
    "ID": "实验ID",
    "转轮厚度": "转轮厚度",
    "转轮直径": "转轮直径",
    "处理入口温度": "处理入口温度(℃)",
    "处理入口温度(℃)": "处理入口温度(℃)",
    "入口含湿量(g/kg)": "入口含湿量(g/kg)",
    "处理风速(m/s)": "处理风速(m/s)",
    "冷吹风速(m/s)": "冷吹风速(m/s)",
    "转轮转速(rpm)": "转轮转速(rpm)",
    "再生温度(℃)": "再生温度(℃)",
    "再生风量": "再生风量",
    "再生入口含湿量": "再生入口含湿量",
    "冷吹入口温度(℃)": "冷吹入口温度(℃)",
    "处理出口温度": "处理出口温度",
    "处理出口含湿量": "处理出口含湿量",
    "冷吹出口温度": "冷吹出口温度",
    "冷吹出口含湿量": "冷吹出口含湿量",
    "再生出口温度": "再生出口温度",
    "再生出口含湿量": "再生出口含湿量",
    POWER_COLUMN: "power",
    DEW_POINT_COLUMN: DEW_POINT_COLUMN,
}

FIXED_XF_OUTER_GROUP = (300, 1070)
FIXED_XF_FOLD_1_GROUPS = ((200, 1070), (400, 1070), (400, 965))

ALLOWED_XF_GEOMETRY_GROUPS = frozenset(
    {
        (200, 550), (300, 550), (400, 550),
        (200, 650), (300, 650), (400, 650),
        (200, 700), (300, 700),
        (200, 965), (300, 965), (400, 965),
        (200, 1070), (300, 1070), (400, 1070),
    }
)
ALLOWED_MAIN_GEOMETRY_GROUPS = frozenset(
    (depth, diameter)
    for diameter in (
        550, 650, 700, 870, 965, 1070, 1220, 1370, 1525, 1650,
        1730, 1850, 1940, 2070, 2190, 2300, 2450, 2650, 2743,
    )
    for depth in (200, 300, 400)
)
ALLOWED_GEOMETRY_GROUPS = {
    "xf": ALLOWED_XF_GEOMETRY_GROUPS,
    "main": ALLOWED_MAIN_GEOMETRY_GROUPS,
}


@dataclass(frozen=True)
class RotorProfileConfig:
    """一个转轮的不可变训练契约。"""

    name: str
    source_path: Path
    sheet_name: str
    expected_sha256: str
    input_columns: tuple[str, ...]
    direct_outputs: tuple[str, ...]
    source_input_columns: tuple[str, ...]
    source_output_columns: tuple[str, ...]
    hidden_widths: tuple[int, ...]
    grid_size: int
    spline_order: int = 3
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 200
    patience: int = 20
    validation_interval_steps: int = 2_000
    checkpoint_interval_steps: int = 2_000
    derived_features: tuple[str, ...] = ()
    derived_outputs: tuple[str, ...] = (DEW_POINT_COLUMN,)
    audit_columns: tuple[str, ...] = AUDIT_COLUMNS
    split_policy: str = "main_geometry_70_15_15"

    @property
    def in_features(self) -> int:
        return len(self.input_columns)

    @property
    def out_features(self) -> int:
        return len(self.direct_outputs)

    @property
    def source_to_canonical(self) -> Mapping[str, str]:
        return SOURCE_TO_CANONICAL_BY_PROFILE[self.name]


PROFILE_CONFIGS = {
    "xf": RotorProfileConfig(
        name="xf",
        source_path=XF_SOURCE_PATH,
        sheet_name=SHEET_NAMES["xf"],
        expected_sha256=EXPECTED_SOURCE_SHA256["xf"],
        input_columns=XF_INPUT_COLUMNS,
        direct_outputs=DIRECT_OUTPUT_COLUMNS,
        source_input_columns=XF_SOURCE_INPUT_COLUMNS,
        source_output_columns=XF_SOURCE_OUTPUT_COLUMNS,
        hidden_widths=(24,),
        grid_size=6,
        learning_rate=5e-4,
        weight_decay=1e-4,
        max_epochs=300,
        patience=20,
        split_policy="xf_outer_and_fold_1",
    ),
    "main": RotorProfileConfig(
        name="main",
        source_path=MAIN_SOURCE_PATH,
        sheet_name=SHEET_NAMES["main"],
        expected_sha256=EXPECTED_SOURCE_SHA256["main"],
        input_columns=MAIN_INPUT_COLUMNS,
        direct_outputs=DIRECT_OUTPUT_COLUMNS,
        source_input_columns=MAIN_SOURCE_INPUT_COLUMNS,
        source_output_columns=MAIN_SOURCE_OUTPUT_COLUMNS,
        hidden_widths=(16,),
        grid_size=5,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=200,
        patience=20,
        split_policy="main_geometry_70_15_15",
    ),
}


def get_profile(profile: str | RotorProfileConfig) -> RotorProfileConfig:
    """解析 profile；未知名称用中文错误硬失败。"""

    if isinstance(profile, RotorProfileConfig):
        return profile
    key = str(profile).strip().lower()
    try:
        return PROFILE_CONFIGS[key]
    except KeyError as exc:
        raise ValueError(f"未知转轮 profile: {profile!r}，仅支持 xf/main") from exc


# 与旧训练脚本类似的便捷别名，避免调用方为选择 profile 重新复制契约。
MODEL_INPUT_COLUMNS = XF_INPUT_COLUMNS
MODEL_OUTPUT_COLUMNS = DIRECT_OUTPUT_COLUMNS
XF_MODEL_INPUT_COLUMNS = XF_INPUT_COLUMNS
MAIN_MODEL_INPUT_COLUMNS = MAIN_INPUT_COLUMNS
XF_MODEL_OUTPUT_COLUMNS = DIRECT_OUTPUT_COLUMNS
MAIN_MODEL_OUTPUT_COLUMNS = DIRECT_OUTPUT_COLUMNS
DERIVED_OUTPUT_COLUMNS = (DEW_POINT_COLUMN,)
