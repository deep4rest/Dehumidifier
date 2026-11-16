"""Control-search feasibility, unchanged boundaries, and independent input goals."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dual_rotor_sim.optimization as opt
from dual_rotor_sim.cli import _make_parser
from dual_rotor_sim.core.system.air_state import AirState


def result(*, W=.0444, T=20., cooling=100., heating=20., converged=True):
    state=AirState(T=T,W=W,mdot=1.)
    return SimpleNamespace(supply_state=state,intermediate_states={'Supply':state},
        cooling_kW=cooling,heating_kW=heating,
        energy_breakdown={'CC1_kW':-cooling,'CC2_kW':0.,'CC3_kW':0.,
                          'Heater_Low_kW':heating,'Heater_OA_kW':0.,'Heater_Post_kW':0.},
        convergence={'converged':converged,'iterations':10,'residuals':{}})


def test_low_load_failure_and_wet_output_cannot_beat_feasible_trial():
    settings=opt.SearchSettings(-45.,20.,energy_reference_kW=120.)
    good=opt.assess_result(result(cooling=200.),settings)
    failed=opt.assess_result(result(cooling=0.,heating=0.,converged=False),settings)
    wet=opt.assess_result(result(W=.1,cooling=0.,heating=0.),settings)
    assert good['feasible']
    assert not failed['feasible'] and failed['score']>good['score']
    assert not wet['feasible'] and wet['score']>good['score']


def test_independent_replay_rejects_equal_loss_with_a_different_dew_point():
    from dual_rotor_sim.core.system.psychrometrics import humidity_ratio_from_dew_point_C
    settings=opt.SearchSettings(-45.,20.,energy_reference_kW=120.)
    before=opt.assess_result(result(W=float(humidity_ratio_from_dew_point_C(-45.1))),settings)
    after=opt.assess_result(result(W=float(humidity_ratio_from_dew_point_C(-44.9))),settings)
    assert before['optimization_loss']==pytest.approx(after['optimization_loss'],abs=1e-12)
    assert opt.independent_assessments_match(before,copy.deepcopy(before))
    assert not opt.independent_assessments_match(after,before)


def test_independent_replay_rejects_changed_load_distribution_with_equal_total():
    assessment=opt.assess_result(result(),opt.SearchSettings(-45.,20.,energy_reference_kW=120.))
    altered=copy.deepcopy(assessment)
    altered['energy_breakdown_kW']['CC1_kW']+=1.
    altered['energy_breakdown_kW']['CC2_kW']-=1.
    assert not opt.independent_assessments_match(altered,assessment)


def test_unclipped_relative_humidity_rejects_hidden_supersaturation():
    sample=result(T=20.,W=30.)
    assert sample.supply_state.RH==1.
    assessment=opt.assess_result(sample,opt.SearchSettings(40.,20.,energy_reference_kW=120.))
    assert 'supersaturated' in assessment['reasons']
    assert assessment['max_unclipped_RH']>1.


def test_zero_water_drying_limit_is_legal_but_not_in_finite_target_band():
    assessment=opt.assess_result(result(W=0.),opt.SearchSettings(-45.,20.,energy_reference_kW=120.))
    assert not assessment['feasible'] and assessment['dry_limit_dew_point']
    assert 'dew_point_below_band' in assessment['reasons']
    assert assessment['supply_Tdp_degC'] is None
    assert assessment['dew_point_error_degC'] is None


@pytest.mark.parametrize('dp,feasible',[
    (-46.000001,False),(-46.,True),(-45.,True),(-44.,True),(-43.999999,False),(-80.,False),
])
def test_both_dew_point_limits_are_inclusive_and_excess_drying_is_rejected(dp,feasible):
    from dual_rotor_sim.core.system.psychrometrics import humidity_ratio_from_dew_point_C
    assessment=opt.assess_result(result(W=float(humidity_ratio_from_dew_point_C(dp))),opt.SearchSettings(-45.,20.,energy_reference_kW=120.))
    assert assessment['feasible'] is feasible
    assert assessment['dew_point_band_degC']==[-46.,-44.]


def test_previous_wrong_optimum_near_minus_92_is_not_accepted():
    assessment=opt.assess_result(result(W=4.8637490243770733e-08),opt.SearchSettings(-45.,20.,energy_reference_kW=120.))
    assert not assessment['feasible']
    assert 'dew_point_below_band' in assessment['reasons']
    assert abs(assessment['dew_point_error_degC'])>46.


def test_supply_temperature_target_is_required_and_enforced():
    with pytest.raises(TypeError):opt.SearchSettings(-45.)
    settings=opt.SearchSettings(-45.,20.,energy_reference_kW=120.)
    assert opt.assess_result(result(T=20.00005),settings)['feasible']
    missed=opt.assess_result(result(T=20.0002),settings)
    assert not missed['feasible'] and 'supply_temperature_target' in missed['reasons']


@pytest.mark.parametrize('kwargs',[
    {'dew_point_target_degC':float('nan')}, {'step_degC':0},
    {'dew_point_tolerance_degC':0.}, {'dew_point_tolerance_degC':-1.},
    {'dew_point_tolerance_degC':1.1}, {'dew_point_tolerance_degC':float('nan')},
    {'supply_T_min_degC':25.,'supply_T_max_degC':15.},
    {'heating_weight':0.}, {'seeds':()}, {'maxiter':-1},
    {'cooling_weight':2.}, {'heating_weight':.5},
    {'supply_T_target_degC':float('nan')}, {'energy_reference_kW':0.},
    {'energy_reference_kW':float('nan')}, {'dew_point_penalty_coefficient':0.},
    {'dew_point_penalty_coefficient':-1.},
    {'bounds':((9.,13.),)},
    {'initial_controls':(10.,15.)}, {'initial_controls':(9.,15.,49.,90.)},
])
def test_invalid_search_contract_fails_before_evaluation(kwargs):
    values={'dew_point_target_degC':-45.,'supply_T_target_degC':20.,**kwargs}
    with pytest.raises(ValueError):opt.SearchSettings(**values)


def config():
    root=Path(__file__).resolve().parents[2]
    return yaml.safe_load((root/'simulation/configs/default_case.yaml').read_text(encoding='utf-8'))


def test_search_keeps_flows_geometry_rpm_and_finds_discrete_known_minimum(monkeypatch):
    raw=config();original=copy.deepcopy(raw);observed=[];calls=[]
    desired=(11.,15.,101.,111.)
    def forward(system,oa,ra,regen,params,setpoints):
        observed.append((oa,ra,regen,params))
        values=tuple(getattr(setpoints,k.removesuffix('_degC')) for k in opt.CONTROL_NAMES)
        calls.append(values)
        return result(cooling=100.+sum((a-b)**2 for a,b in zip(values,desired)),heating=0.)
    monkeypatch.setattr(opt,'forward_simulate',forward)
    settings=opt.SearchSettings(-45.,20.,bounds=((10.,12.),(14.,16.),(100.,102.),(110.,112.)),
                                step_degC=1.,maxiter=1,popsize=2,seeds=(7,))
    records=[]
    report=opt.optimize_controls(raw,settings,system=object(),evaluation_callback=records.append)
    assert raw==original
    assert tuple(report['best']['controls'].values())==desired
    assert report['best']['objective_weighted_thermal_kW']==100.
    assert report['energy_reference_kW']==657.
    assert all(row['energy_reference_kW']==657. for row in records)
    assert report['unique_forward_evaluations']==len(calls)==len(records)
    assert len(calls)==len(set(calls))
    updated=copy.deepcopy(report['best_config'])
    for k in opt.CONTROL_NAMES:updated['control_setpoints'][k]=original['control_setpoints'][k]
    assert updated['control_setpoints'].pop('supply_T_target_degC')==20.
    assert updated==original
    assert all(x==observed[0] for x in observed)
    assert not report['global_minimum_proven'] and not report['actual_electricity_optimized']


def test_all_failed_trials_return_no_best_configuration(monkeypatch):
    monkeypatch.setattr(opt,'forward_simulate',lambda *args:result(converged=False))
    settings=opt.SearchSettings(-45.,20.,coarse_grid=False,maxiter=0,popsize=2,seeds=(1,))
    report=opt.optimize_controls(config(),settings,system=object())
    assert report['status']=='no_feasible_candidate_found'
    assert report['best'] is None and report['best_config'] is None
    assert report['feasible_evaluations']==0


def test_initial_controls_are_recomputed_with_current_model(monkeypatch):
    calls=[]
    def forward(*args):
        point=tuple(getattr(args[-1],name.removesuffix('_degC')) for name in opt.CONTROL_NAMES)
        calls.append(point)
        return result(cooling=100.+sum((v-t)**2 for v,t in zip(point,(11.,15.,101.,111.))))
    monkeypatch.setattr(opt,'forward_simulate',forward)
    settings=opt.SearchSettings(-45.,20.,initial_controls=(11.,15.,101.,111.),coarse_grid=False,maxiter=0,popsize=2,seeds=(1,))
    report=opt.optimize_controls(config(),settings,system=object())
    assert (11.,15.,101.,111.) in calls
    assert report['best']['controls']==dict(zip(opt.CONTROL_NAMES,(11.,15.,101.,111.)))
    assert report['dew_point_constraint']=='absolute_error_to_target'
    assert report['schema_version']==3


def test_cli_requires_explicit_dew_point_and_temperature_targets():
    with pytest.raises(SystemExit):_make_parser().parse_args(['optimize','--config','case.yaml'])
    with pytest.raises(SystemExit):_make_parser().parse_args(['optimize','--config','case.yaml','--dew-point-max','-45'])
    with pytest.raises(SystemExit):_make_parser().parse_args(['optimize','--config','case.yaml','--dew-point-target','-45'])
    args=_make_parser().parse_args(['optimize','--config','case.yaml','--dew-point-target','-45','--supply-temp-target','-35'])
    assert args.dew_point_target==-45.
    assert args.dew_point_tolerance==1.
    assert args.supply_temp_target==-35.
    assert args.energy_reference is None
    assert args.dew_point_penalty_coefficient==.1
    assert args.supply_temp_min is None and args.supply_temp_max is None


@pytest.mark.parametrize('error,weight',[
    (-1.,.35),(-.500001,.35),(-.5,.1),(0.,.1),(.3,.1),(.300001,.5),(1.,.5),
])
def test_piecewise_weight_boundaries_follow_user_contract(error,weight):
    assert opt.dew_point_region_weight(error)==weight


@pytest.mark.parametrize('error',[-1.000001,1.000001,float('inf'),float('nan')])
def test_piecewise_weights_do_not_replace_dew_point_hard_limits(error):
    assert opt.dew_point_region_weight(error) is None


@pytest.mark.parametrize('error,weight',[(-.8,.35),(-.4,.1),(0.,.1),(.2,.1),(.8,.5)])
def test_normalized_objective_and_physical_energy_are_separate(error,weight):
    from dual_rotor_sim.core.system.psychrometrics import humidity_ratio_from_dew_point_C
    settings=opt.SearchSettings(-45.,20.,energy_reference_kW=60.,dew_point_penalty_coefficient=.1)
    record=opt.assess_result(result(W=float(humidity_ratio_from_dew_point_C(-45.+error))),settings)
    assert record['thermal_load_sum_kW']==120.
    assert record['normalized_energy']==2.
    assert record['dew_point_region_weight']==weight
    assert record['dew_point_distance_penalty']==pytest.approx(.1*weight*abs(error))
    assert record['optimization_loss']==pytest.approx(2.+.1*weight*abs(error))
    assert record['score']<1.


def test_all_six_loads_count_and_inconsistent_or_missing_totals_fail():
    sample=result(cooling=60.,heating=6.)
    sample.energy_breakdown={'CC1_kW':-10.,'CC2_kW':-20.,'CC3_kW':-30.,
                             'Heater_Low_kW':1.,'Heater_OA_kW':2.,'Heater_Post_kW':3.}
    settings=opt.SearchSettings(-45.,20.,energy_reference_kW=10.)
    accepted=opt.assess_result(sample,settings)
    assert accepted['feasible'] and accepted['thermal_load_sum_kW']==66.
    assert accepted['normalized_energy']==6.6
    sample.energy_breakdown.pop('CC3_kW')
    assert 'invalid_load_contract' in opt.assess_result(sample,settings)['reasons']
    sample=result();sample.energy_breakdown['CC3_kW']=-1.
    assert 'invalid_load_contract' in opt.assess_result(sample,settings)['reasons']


def test_no_reference_assessment_does_not_invent_normalization():
    record=opt.assess_result(result(),opt.SearchSettings(-45.,20.))
    assert record['feasible']
    assert record['energy_reference_kW'] is None
    assert record['normalized_energy'] is None
    assert record['optimization_loss'] is None and record['score'] is None


def test_zero_baseline_energy_requires_explicit_reference(monkeypatch):
    monkeypatch.setattr(opt,'forward_simulate',lambda *args:result(cooling=0.,heating=0.))
    settings=opt.SearchSettings(-45.,20.,coarse_grid=False,maxiter=0,popsize=2,seeds=(1,))
    with pytest.raises(RuntimeError,match='显式提供energy_reference_kW'):
        opt.optimize_controls(config(),settings,system=object())
    explicit=opt.SearchSettings(-45.,20.,energy_reference_kW=100.,coarse_grid=False,maxiter=0,popsize=2,seeds=(1,))
    report=opt.optimize_controls(config(),explicit,system=object())
    assert report['best']['normalized_energy']==0.
    assert report['energy_reference_kW']==100.
    assert report['energy_reference_source']=='explicit'


def test_baseline_numerical_failure_cannot_silently_calibrate_from_next_trial(monkeypatch):
    calls=[]
    def forward(*args):
        calls.append(1)
        if len(calls)==1:raise ValueError('baseline numerical failure')
        return result()
    monkeypatch.setattr(opt,'forward_simulate',forward)
    settings=opt.SearchSettings(-45.,20.,coarse_grid=False,maxiter=0,popsize=2,seeds=(1,))
    with pytest.raises(RuntimeError,match='首次baseline.*显式提供energy_reference_kW'):
        opt.optimize_controls(config(),settings,system=object())
    assert len(calls)==1


def test_nonfinite_normalized_loss_is_rejected_without_epsilon():
    import json
    record=opt.assess_result(result(),opt.SearchSettings(-45.,20.,energy_reference_kW=1e-320))
    assert not record['feasible'] and 'invalid_objective' in record['reasons']
    assert record['optimization_loss'] is None and record['normalized_energy'] is None
    json.dumps(record,allow_nan=False)


def test_search_ranks_composite_loss_instead_of_lowest_raw_heat(monkeypatch):
    from dual_rotor_sim.core.system.psychrometrics import humidity_ratio_from_dew_point_C
    def forward(*args):
        x=args[-1].cc1_T_outlet
        other=(args[-1].cc2_T_outlet,args[-1].heater_oa_T,args[-1].heater_low_T)
        heat=100.+2.*(x-11.)+sum((a-b)**2 for a,b in zip(other,(15.,101.,111.)))
        error=0. if x==12. else .9
        return result(cooling=heat,heating=0.,W=float(humidity_ratio_from_dew_point_C(-45.+error)))
    monkeypatch.setattr(opt,'forward_simulate',forward)
    settings=opt.SearchSettings(-45.,20.,energy_reference_kW=100.,bounds=((11.,12.),(14.,16.),(100.,102.),(110.,112.)),
                                step_degC=.5,maxiter=0,popsize=2,seeds=(1,))
    report=opt.optimize_controls(config(),settings,system=object())
    assert report['best']['controls']==dict(zip(opt.CONTROL_NAMES,(12.,15.,101.,111.)))
    assert report['best']['thermal_load_sum_kW']==102.
    assert report['best']['optimization_loss']==1.02
    assert min(r['thermal_load_sum_kW'] for r in report['evaluations'] if r['feasible'])==100.
    assert report['minimum_thermal_load_candidate']['thermal_load_sum_kW']==100.
    assert report['minimum_thermal_load_candidate']['optimization_loss']>report['best']['optimization_loss']


def test_minimum_thermal_candidate_excludes_baseline_outside_search_bounds(monkeypatch):
    def forward(*args):
        controls=tuple(getattr(args[-1],k.removesuffix('_degC')) for k in opt.CONTROL_NAMES)
        heat=1. if controls==(11.,15.,115.,130.) else 10.+sum((a-b)**2 for a,b in zip(controls,(11.,15.,101.,111.)))
        return result(cooling=heat,heating=0.)
    monkeypatch.setattr(opt,'forward_simulate',forward)
    settings=opt.SearchSettings(-45.,20.,bounds=((10.,12.),(14.,16.),(100.,102.),(110.,112.)),
                                step_degC=1.,maxiter=0,popsize=2,seeds=(1,))
    report=opt.optimize_controls(config(),settings,system=object())
    assert report['baseline']['thermal_load_sum_kW']==1.
    assert not report['baseline']['eligible_for_selection']
    assert report['minimum_thermal_load_candidate']['thermal_load_sum_kW']==10.
    assert report['minimum_thermal_load_candidate']['eligible_for_selection']
