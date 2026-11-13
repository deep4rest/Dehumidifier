"""仪表盘必须能从历史记录重绘，并隔离已废弃的电功率字段。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from dual_rotor_sim.dashboard import (
    _dashboard_payload_from_record,
    _process_path_nodes,
    _state_table_rows,
    _terminal_summary,
    save_dashboard_from_record,
)


def _legacy_run_document() -> dict:
    """返回包含旧电功率字段的完整历史记录形状。"""
    state = {
        "T_degC": 17.25,
        "W_g_per_kg": 0.1234,
        "mdot_kg_da_s": 9.8765,
        "Tdp_degC": -40.5,
        "RH_fraction": 0.0123,
        "h_kJ_per_kg_da": 17.75,
    }
    nodes = (
        "OA",
        "AfterFan1",
        "AfterCC1",
        "AfterR1_P",
        "BeforeFan2",
        "AfterFan2",
        "AfterCC2",
        "Supply",
    )
    return {
        "run_id": "legacy-qy1070",
        "config": {
            "outdoor_air": {"T_degC": 35.0, "W_g_per_kg": 27.02},
            "control_setpoints": {
                "heater_low_T_degC": 130.0,
                "heater_oa_T_degC": 115.0,
            },
        },
        "result": {
            "supply_state": dict(state),
            "energy_summary_kW": {
                "electrical": 112.368,
                "cooling_load": 104.418,
                "heating_load": 31.905,
            },
            "energy_breakdown_kW": {
                "Fan1_kW": 4.0,
                "CC1_kW": -77.803,
                "Rotor1Motor_kW": 22.633,
                "Fan2_kW": 18.5,
                "CC2_kW": -26.614,
                "Rotor2Motor_kW": 67.235,
                "Heater_Low_kW": 13.533,
                "Heater_OA_kW": 18.372,
            },
            "intermediate_states": {name: dict(state) for name in nodes},
            "rotor1_outputs": {"power_kW": 22.633},
            "rotor2_outputs": {"power_kW": 67.235},
        },
    }


def test_legacy_record_adapter_excludes_all_obsolete_power_categories():
    """旧 JSON 即使含电功率字段，仪表盘负荷数据也只能保留热工项。"""
    payload = _dashboard_payload_from_record(_legacy_run_document())

    assert payload.cooling_kW == 104.418
    assert payload.heating_kW == 31.905
    assert payload.energy_breakdown == {
        "CC1_kW": -77.803,
        "CC2_kW": -26.614,
        "Heater_Low_kW": 13.533,
        "Heater_OA_kW": 18.372,
    }


def test_state_table_rows_have_six_columns_without_dry_air_flow():
    """若表格重新读取 mdot 或恢复第七列，本测试必须失败。"""
    payload = _dashboard_payload_from_record(_legacy_run_document())

    rows = _state_table_rows(payload.intermediate_states)

    assert len(rows) == 8
    assert all(len(row) == 6 for row in rows)
    assert all("9.8765" not in row for row in rows)


def test_historical_record_redraw_writes_a_readable_png(tmp_path: Path):
    """历史记录适配入口应直接生成可读取的 PNG，不加载或重跑模型。"""
    import json

    record_path = tmp_path / "result.json"
    record_path.write_text(
        json.dumps(_legacy_run_document(), ensure_ascii=False),
        encoding="utf-8",
    )
    target = tmp_path / "fig1_dashboard.png"

    saved = save_dashboard_from_record(record_path=record_path, target=target)

    assert saved == target
    with Image.open(target) as image:
        assert image.format == "PNG"
        assert image.width >= 2000
        assert image.height >= 1500


def _terminal_run_document(*, target: float | None = 20., cooling: float = -5., heating: float = 0.) -> dict:
    document = _legacy_run_document()
    control = document["config"]["control_setpoints"]
    control["supply_T_target_degC"] = target
    result = document["result"]
    result["energy_breakdown_kW"].update({"CC3_kW": cooling, "Heater_Post_kW": heating})
    result["energy_summary_kW"]["cooling_load"] += abs(cooling)
    result["energy_summary_kW"]["heating_load"] += heating
    inlet = {**result["supply_state"], "T_degC": 25.}
    after_cc3 = {**inlet, "T_degC": 20. if cooling else 25.}
    result["intermediate_states"]["AfterR2_P"] = inlet
    result["intermediate_states"]["AfterCC3"] = after_cc3
    result["supply_state"]["T_degC"] = 28. if heating else after_cc3["T_degC"]
    result["intermediate_states"]["Supply"] = dict(result["supply_state"])
    return document


def test_terminal_record_keeps_six_loads_and_actual_ten_node_path():
    payload = _dashboard_payload_from_record(_terminal_run_document())
    assert tuple(payload.energy_breakdown) == (
        "CC1_kW", "CC2_kW", "CC3_kW", "Heater_Low_kW", "Heater_OA_kW", "Heater_Post_kW",
    )
    assert payload.energy_breakdown["CC3_kW"] == -5.
    assert payload.energy_breakdown["Heater_Post_kW"] == 0.
    nodes = _process_path_nodes(payload.intermediate_states)
    assert [name for name, _ in nodes][-3:] == ["AfterR2_P", "AfterCC3", "Supply"]
    rows = _state_table_rows(payload.intermediate_states)
    assert len(rows) == 10
    assert [row[1] for row in rows[-3:]] == ["25.00", "20.00", "20.00"]


def test_terminal_summary_distinguishes_auto_cooling_heating_and_legacy():
    cool = _terminal_run_document(target=20., cooling=-5., heating=0.)
    assert "送风目标 20.0℃" in _terminal_summary(_dashboard_payload_from_record(cool), cool["config"])
    assert "后表冷：开，后加热：关" in _terminal_summary(_dashboard_payload_from_record(cool), cool["config"])
    heat = _terminal_run_document(target=28., cooling=0., heating=4.)
    assert "后表冷：关，后加热：开" in _terminal_summary(_dashboard_payload_from_record(heat), heat["config"])
    bypass = _terminal_run_document(target=None, cooling=0., heating=0.)
    assert "旁通" in _terminal_summary(_dashboard_payload_from_record(bypass), bypass["config"])
    legacy = _legacy_run_document()
    assert "历史记录" in _terminal_summary(_dashboard_payload_from_record(legacy), legacy["config"])
    assert len(_process_path_nodes(_dashboard_payload_from_record(legacy).intermediate_states)) == 8


def test_terminal_record_redraw_writes_a_readable_png(tmp_path: Path):
    import json
    record_path = tmp_path / "terminal.json"
    record_path.write_text(json.dumps(_terminal_run_document(), ensure_ascii=False), encoding="utf-8")
    target = tmp_path / "terminal_dashboard.png"
    save_dashboard_from_record(record_path=record_path, target=target)
    with Image.open(target) as image:
        assert image.format == "PNG"
        assert image.width >= 2000
        assert image.height >= 1800
