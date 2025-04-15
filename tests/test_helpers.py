import torch
from eval_env import helpers

def test_set_seed():
    helpers.set_seed(123)
    a = torch.rand(1).item()
    helpers.set_seed(123)
    b = torch.rand(1).item()
    assert a == b

def test_check_metadata_serializable():
    meta = {'eval_0': {'a': torch.tensor([1.0])}}
    res = helpers.check_metadata_serializable(meta)
    assert isinstance(res, dict)

def test_check_metadata_serializable_all_types():
    meta = {'tensor': torch.tensor([1.0])}
    res = helpers.check_metadata_serializable_all_types(meta)
    assert isinstance(res, dict)
