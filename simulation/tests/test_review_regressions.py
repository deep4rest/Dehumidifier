"""仿真审查回归：系统压力与未收敛留档。"""
import json
from types import SimpleNamespace

import pytest
from dual_rotor_sim import AirState, SystemParams, ControlSetpoints
from dual_rotor_sim.core.system.components.cooling_coil import CoolingCoil
from dual_rotor_sim.core.system.components.heater import Heater
from dual_rotor_sim.core.system.components.mixer import Mixer
from dual_rotor_sim.core.system.psychrometrics import saturation_humidity_ratio_g_per_kg, dew_point_water_C
from dual_rotor_sim.core.system.system import DualRotorSystem
from dual_rotor_sim.run_records import result_to_dict
from simulation.tests.test_cli import _sample_system_result
from dual_rotor_sim import cli


def test_pressure_survives_coil_heater_split_and_mix():
    inlet = AirState(30, 15, 1, pressure_Pa=80000)
    outlet, _ = CoolingCoil(10).process(inlet)
    assert outlet.W == pytest.approx(saturation_humidity_ratio_g_per_kg(10, 80000))
    assert outlet.Tdp == pytest.approx(10)
    assert outlet.RH == pytest.approx(1)
    warm, _ = Heater(20).process(outlet.at_mdot(2))
    mixed = Mixer().mix(warm, inlet)
    assert mixed.pressure_Pa == 80000
    assert mixed.Tdp == pytest.approx(dew_point_water_C(mixed.W, 80000))


def test_mixer_rejects_different_pressures():
    with pytest.raises(ValueError, match='压力'):
        Mixer().mix(AirState(20, 5, pressure_Pa=80000), AirState(20, 5))


def test_coupled_solver_persists_last_evaluated_residuals():
    class Divergent:
        def simulate(self, *args, **kwargs):
            result = _sample_system_result()
            result.intermediate_states['AfterMix3'] = AirState(kwargs['r1_regen_T_override']+10, kwargs['r1_regen_W_override']+1)
            result.intermediate_states['AfterR2_CB'] = AirState(kwargs['r2_regen_T_override']+10, 2)
            return result
    with pytest.warns(RuntimeWarning, match='未收敛'):
        result = DualRotorSystem.simulate_coupled_xf(Divergent(), AirState(35,18), AirState(25,10), AirState(35,18), SystemParams(), ControlSetpoints(), max_iter=2)
    record = result_to_dict(result)
    assert record['convergence']['converged'] is False
    assert record['convergence']['iterations'] == 2
    assert record['convergence']['residuals'] == {'r1_T_degC': 10.0, 'r1_W_g_per_kg': 1.0, 'r2_T_degC': 10.0}


def test_cli_keeps_nonconverged_run_but_exits_nonzero(monkeypatch, tmp_path, capsys):
    result = _sample_system_result()
    result.convergence = {'status': 'not_converged', 'converged': False, 'iterations': 60, 'residuals': {}}
    monkeypatch.setattr(cli, '_build_from_config', lambda cfg: result)
    monkeypatch.setattr(cli, 'RESULTS_DIR', tmp_path / 'runs')
    path = tmp_path / 'case.yaml'; path.write_text('schema_version: 2')
    with pytest.raises(SystemExit) as exc:
        cli.cmd_simulate(SimpleNamespace(config=str(path), quiet=True, json=True))
    assert exc.value.code == 4
    records = list((tmp_path / 'runs').glob('*/result.json'))
    assert len(records) == 1
    assert json.loads(records[0].read_text())['result']['convergence']['converged'] is False
    assert '未收敛' in capsys.readouterr().err


def test_complete_system_propagates_configured_pressure_to_every_node():
    from simulation.tests.system.test_independent_rotor_params import RecordingRotor
    r1, r2 = RecordingRotor(), RecordingRotor()
    result = DualRotorSystem(rotor1=r1, rotor2=r2).simulate(
        AirState(35,18,1), AirState(25,10,.5), AirState(35,18,.3),
        SystemParams(pressure_Pa=80000), ControlSetpoints())
    assert {s.pressure_Pa for s in result.intermediate_states.values()} == {80000}
    assert r1.calls[0].pressure_Pa == r2.calls[0].pressure_Pa == 80000
    assert result.supply_state.Tdp == pytest.approx(dew_point_water_C(result.supply_state.W, 80000))


def test_converged_solver_records_success_and_actual_iterations():
    class FixedPoint:
        def simulate(self, *args, **kwargs):
            result = _sample_system_result()
            result.intermediate_states['AfterMix3'] = AirState(kwargs['r1_regen_T_override'], kwargs['r1_regen_W_override'])
            result.intermediate_states['AfterR2_CB'] = AirState(kwargs['r2_regen_T_override'], 2)
            return result
    result = DualRotorSystem.simulate_coupled_xf(FixedPoint(), AirState(35,18), AirState(25,10), AirState(35,18), SystemParams(), ControlSetpoints())
    assert result.convergence['converged'] is True
    assert result.convergence['iterations'] == 1
    assert all(value == 0 for value in result.convergence['residuals'].values())


def test_record_redraw_retains_unconverged_label_and_dry_limit():
    from dual_rotor_sim.dashboard import _dashboard_payload_from_record
    result = _sample_system_result()
    result.supply_state.W = 0
    result.convergence = {'converged': False}
    record = result_to_dict(result)
    assert record['supply_state']['Tdp_degC'] is None
    payload = _dashboard_payload_from_record({'result': record})
    assert '未收敛' in payload.solver_label
    assert payload.supply_state.Tdp == float('-inf')


def test_cli_dry_limit_json_is_standard_json(capsys):
    result = _sample_system_result(); result.supply_state.W = 0
    cli._print_result(result, True)
    text = capsys.readouterr().out
    value = json.loads(text, parse_constant=lambda x: pytest.fail('非标准JSON常量: '+x))
    assert value['supply_Tdp'] is None
