"""Terminal switching, condensate and six-load integration."""
import copy
from pathlib import Path
import math
import pytest
import yaml
from dual_rotor_sim.core.system.air_state import AirState
from dual_rotor_sim.core.system.components.terminal import condition_supply
from dual_rotor_sim.core.system import psychrometrics as psy
from dual_rotor_sim.config import parse_simulation_config,build_runtime_inputs,ConfigError


@pytest.mark.parametrize('T,target,mode',[(25.,15.,'cool'),(10.,20.,'heat'),(20.,20.,'off'),(20.,None,'off')])
def test_terminal_switches_one_component_and_preserves_flow_pressure(T,target,mode):
    inlet=AirState(T=T,W=.04,mdot=1.,pressure_Pa=90000.)
    cc3,out,e=condition_supply(inlet,target)
    assert out.T==(T if target is None else target)
    assert out.W==inlet.W and out.mdot==inlet.mdot and out.pressure_Pa==inlet.pressure_Pa
    assert e['CC3_kW']<=0 and e['Heater_Post_kW']>=0
    assert (e['CC3_kW']<0)==(mode=='cool')
    assert (e['Heater_Post_kW']>0)==(mode=='heat')
    assert sum(e.values())==pytest.approx(out.mdot*(out.h-inlet.h),abs=1e-12)


def test_terminal_cooling_condenses_and_returns_true_humidity():
    inlet=AirState(T=30.,W=20.,mdot=1.)
    cc3,out,e=condition_supply(inlet,10.)
    assert out.W==pytest.approx(psy.saturation_humidity_ratio_g_per_kg(10.))
    assert out.W<inlet.W and out.T==10.
    assert e['Heater_Post_kW']==0 and e['CC3_kW']<0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-273.15,True])
def test_terminal_rejects_invalid_target(bad):
    with pytest.raises(ValueError):condition_supply(AirState(20.,.04,1.),bad)


def test_optional_terminal_target_maps_and_rejects_invalid_config():
    root=Path(__file__).resolve().parents[3]
    raw=yaml.safe_load((root/'simulation/configs/default_case.yaml').read_text(encoding='utf-8'))
    assert build_runtime_inputs(parse_simulation_config(raw))['setpoints'].supply_T_target is None
    raw['control_setpoints']['supply_T_target_degC']=-35.
    assert build_runtime_inputs(parse_simulation_config(raw))['setpoints'].supply_T_target==-35.
    for bad in [float('nan'),-273.15,True]:
        raw['control_setpoints']['supply_T_target_degC']=bad
        with pytest.raises(ConfigError):parse_simulation_config(raw)
