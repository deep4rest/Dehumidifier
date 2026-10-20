"""XF 数据契约中不允许运行时改写的常量。"""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_PATH = REPOSITORY_ROOT / "data" / "新风转轮数据集.xls"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "training" / "legacy_xf"
REPORTS_DIR = RESULTS_DIR / "reports"

SHEET_NAME = "CollectRecordXF"
EXPECTED_SOURCE_SHA256 = (
    "116ea18458bf831225765472440a21da8045cb8e9bfadd898f677b004ce026cb"
)
EXPECTED_RAW_ROWS = 57_186
EXPECTED_CLEAN_ROWS = 57_185
EXCLUDED_ID = 1407
OUTER_GROUP = (300, 1070)

MODEL_INPUT_COLUMNS = (
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

MODEL_OUTPUT_COLUMNS = (
    "Label_Process_Outlet_Temp",
    "Label_Process_Outlet_AbsHum",
    "Label_Purge_Outlet_Temp",
    "Label_Purge_Outlet_AbsHum",
    "Label_React_Outlet_Temp",
    "Label_React_Outlet_AbsHum",
    "Label_PowerConsumption",
)

MODEL_INPUT_LABELS = {
    "TextBox_Rotor_depth": "转轮厚度(mm)",
    "TextBox_Rotor_diameter": "转轮直径(mm)",
    "TextBox_Process_inlet_Temp": "处理/冷吹共享入口温度(℃)",
    "TextBox_Purge_Inlet_AbsHum": "处理/冷吹共享入口含湿量(g/kg)",
    "Label_Process_FV": "处理风速(m/s)",
    "Label_Purge_FV": "冷吹风速(m/s)",
    "TextBox_Rotation_Speed": "转轮转速(rpm)",
    "TextBox_React_Heater_Temp": "再生入口温度(℃)",
    "Label_React_CMH": "再生入口风量(m³/h)",
    "Label_React_Inlet_AbsHum": "再生入口含湿量(g/kg)",
}

MODEL_INPUT_RUNTIME_FIELDS = {
    "TextBox_Rotor_depth": "rotor_thickness",
    "TextBox_Rotor_diameter": "rotor_diameter",
    "TextBox_Process_inlet_Temp": "process_inlet_T",
    "TextBox_Purge_Inlet_AbsHum": "process_inlet_W",
    "Label_Process_FV": "airspeed",
    "Label_Purge_FV": "cool_blow_airspeed",
    "TextBox_Rotation_Speed": "rpm",
    "TextBox_React_Heater_Temp": "regen_T",
    "Label_React_CMH": "regen_flow_m3h",
    "Label_React_Inlet_AbsHum": "regen_inlet_W",
}

MODEL_OUTPUT_LABELS = {
    "Label_Process_Outlet_Temp": "处理出口温度",
    "Label_Process_Outlet_AbsHum": "处理出口含湿量",
    "Label_Purge_Outlet_Temp": "冷吹出口温度",
    "Label_Purge_Outlet_AbsHum": "冷吹出口含湿量",
    "Label_React_Outlet_Temp": "再生出口温度",
    "Label_React_Outlet_AbsHum": "再生出口总含水量",
    "Label_PowerConsumption": "转轮平均功率(kW)",
}

MODEL_OUTPUT_RUNTIME_FIELDS = {
    "Label_Process_Outlet_Temp": "process_outlet_T",
    "Label_Process_Outlet_AbsHum": "process_outlet_W",
    "Label_Purge_Outlet_Temp": "cool_blow_outlet_T",
    "Label_Purge_Outlet_AbsHum": "cool_blow_outlet_W",
    "Label_React_Outlet_Temp": "regen_outlet_T",
    "Label_React_Outlet_AbsHum": "regen_outlet_total_W",
    "Label_PowerConsumption": "power_kW",
}

AUDIT_NUMERIC_COLUMNS = (
    "ID",
    "TextBox_Purge_CMH",
    "TextBox_Process_CMH",
    "TextBox_Purge_Inlet_Temp",
    "Label_Purge_Zone_Angle",
    "Label_React_Zone_Angle",
    "TextBox_OA_CMH",
    "TextBox_OA_AbsHum",
)

FIXED_FOLD_GROUPS = {
    "fold_1": ((200, 1070), (400, 1070), (400, 965)),
    "fold_2": ((300, 550), (300, 700), (400, 650)),
    "fold_3": ((200, 965), (400, 550)),
    "fold_4": ((200, 650), (300, 965)),
    "fold_5": ((200, 550), (200, 700), (300, 650)),
}

