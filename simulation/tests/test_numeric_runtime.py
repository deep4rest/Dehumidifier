"""Stable inference changes arithmetic precision without altering trained values."""
import torch
from dual_rotor_sim.numeric_runtime import StableInferenceModel


def test_float64_body_preserves_original_weights_and_float32_io():
    original=torch.nn.Linear(3,2,bias=False)
    before={key:value.clone() for key,value in original.state_dict().items()}
    model=StableInferenceModel(original)
    x=torch.tensor([[1.0000001,1e-7,-.9999999]],dtype=torch.float32)
    expected=torch.nn.functional.linear(x.double(),before['weight'].double()).float()
    actual=model(x)
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert next(model.parameters()).dtype==torch.float64
    assert next(original.parameters()).dtype==torch.float32
    for key,value in original.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
