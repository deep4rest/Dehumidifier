"""Physical zero, temperature identity, useful gradients and checkpoint isolation."""
import numpy as np
import pytest
import torch

from training.single_kan.candidate_models import HumidityNonnegativeHead, build_candidate_model, candidate_contract
from training.single_kan.model import fit_preprocessor


def preprocessor(profile="xf"):
    c=candidate_contract(profile,"kan")
    x=np.array([[200,550,9,3.9,1000,50,50,333,.2], [400,850,12,5.1,4000,200,140,1200,.4]],np.float32) if profile=="xf" else np.array([
        [200,550,.3,6,100,27,15,1000,50], [400,850,1,6,140,27,15,4000,200]],np.float32)
    y=np.array([[15,.00371,40,.431,25,7.52], [25,7.0461,100,15,50,60]],np.float32)
    return fit_preprocessor(x,y,c.input_columns,c.output_columns,fit_ids=[1,2])


@pytest.mark.parametrize("profile", ["xf", "main"])
def test_head_preserves_temperatures_and_physical_nonnegative_after_fp32_inverse(profile):
    pre=preprocessor(profile)
    head=HumidityNonnegativeHead(pre.target_columns,pre.target_scaler.min_)
    raw=torch.tensor([[-2.,-100.,-.7,-100.,3.,-100.], [.4,-.98,-1.4,.2,-.9,.5]],dtype=torch.float32)
    constrained=head(raw)
    torch.testing.assert_close(constrained[:,[0,2,4]],raw[:,[0,2,4]],rtol=0,atol=0)
    physical=pre.inverse_targets(constrained.detach().numpy())
    assert (physical[:,[1,3,5]]>=0).all()
    assert (head.lower_bound.double()>=head.physical_zero_scaled).all()
    assert not any(True for _ in head.parameters())
    assert head.metadata()["beta"]==50.
    # A valid negative scaled value must remain representable, not clamp to z=0.
    assert constrained[1,1]<0


@pytest.mark.parametrize("profile", ["xf", "main"])
def test_softplus_has_gradient_below_boundary_and_is_identity_far_above_it(profile):
    pre=preprocessor(profile)
    head=HumidityNonnegativeHead(pre.target_columns,pre.target_scaler.min_)
    raw=torch.zeros((1,6),requires_grad=True)
    with torch.no_grad():
        raw[:,[1,3,5]]=head.lower_bound-.02
    head(raw).sum().backward()
    assert ((raw.grad[:,[1,3,5]]>0)&(raw.grad[:,[1,3,5]]<1)).all()
    torch.testing.assert_close(raw.grad[:,[0,2,4]],torch.ones((1,3)),rtol=0,atol=0)
    far=torch.ones((2,6))
    torch.testing.assert_close(head(far),far,rtol=0,atol=1e-6)


@pytest.mark.parametrize("profile,parameter_count", [("xf",3600),("main",2160)])
def test_same_network_initialization_parameter_count_and_strict_package_boundary(profile,parameter_count):
    pre=preprocessor(profile)
    torch.manual_seed(20260908)
    baseline=build_candidate_model(profile,"kan")
    torch.manual_seed(20260908)
    constrained=build_candidate_model(profile,"kan",humidity_output_constraint="softplus",target_zero_scaled=pre.target_scaler.min_)
    assert baseline.parameter_count==constrained.parameter_count==parameter_count
    for name,value in baseline.network.state_dict().items():
        torch.testing.assert_close(value,constrained.network.state_dict()[name],rtol=0,atol=0)
    restored=build_candidate_model(profile,"kan",humidity_output_constraint="softplus",target_zero_scaled=pre.target_scaler.min_)
    restored.load_state_dict(constrained.state_dict())
    assert restored.output_constraint_metadata()==constrained.output_constraint_metadata()
    with pytest.raises(RuntimeError,match="Unexpected key"):
        baseline.load_state_dict(constrained.state_dict())


def test_cli_rejects_legacy_and_log_experiment_with_nonnegative_head():
    from training.single_kan.train import main
    assert main(["--profile","xf","--humidity-output-constraint","softplus","--smoke-test"])==2
    assert main(["--profile","xf","--dataset-dir","unused","--humidity-output-constraint","softplus",
                 "--process-humidity-transform","log","--smoke-test"])==2


@pytest.mark.parametrize("profile", ["xf", "main", "all"])
def test_cli_dispatches_both_rotors_with_softplus(profile,monkeypatch):
    from training.single_kan import candidate_train
    from training.single_kan.train import main
    calls=[]
    def record(**kwargs):
        calls.append(kwargs)
        return {"status":"completed"}
    monkeypatch.setattr(candidate_train,"run_candidate_suite",record)
    assert main(["--profile",profile,"--dataset-dir","unused","--humidity-output-constraint","softplus","--smoke-test"])==0
    assert calls[0]["profile"]==profile
    assert calls[0]["humidity_output_constraint"]=="softplus"


def test_auto_enables_softplus_for_both_profiles_and_explicit_none_is_preserved(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from training.single_kan import candidate_train
    calls=[]
    monkeypatch.setattr(candidate_train,"load_candidate_dataset",lambda root,profile,**kw:SimpleNamespace(profile=profile))
    def train(dataset,model_type,output,**kwargs):
        calls.append((dataset.profile,kwargs["humidity_output_constraint"]))
        return {"status":"completed"}
    monkeypatch.setattr(candidate_train,"train_candidate",train)
    candidate_train.run_candidate_suite(profile="all",model_type="kan",dataset_dir=tmp_path,output_root=tmp_path/"auto",smoke_test=True)
    assert calls==[("xf","softplus"),("main","softplus")]
    calls.clear()
    candidate_train.run_candidate_suite(profile="main",model_type="kan",dataset_dir=tmp_path,output_root=tmp_path/"none",smoke_test=True,humidity_output_constraint="none")
    assert calls==[("main","none")]
