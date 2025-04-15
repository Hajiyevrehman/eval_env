import pytest
import torch
from eval_env import evaluator

# Simple linear model code as string (PyTorch reference)
PYTORCH_LINEAR = '''
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(4, 2)
    def forward(self, x):
        return self.linear(x)
def get_init_inputs():
    return [torch.randn(1, 4)]
def get_inputs():
    return [torch.randn(1, 4)]
'''

# For testing, pretend this is a CUDA version (actually same as above)
CUDA_LINEAR = PYTORCH_LINEAR.replace('Model', 'ModelNew')

def test_evaluate_and_reward_linear():
    # Use a fixed baseline time and tolerance
    baseline_time = 10.0  # ms
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    result = evaluator.evaluate_and_reward(
        original_model_src=PYTORCH_LINEAR,
        custom_model_src=CUDA_LINEAR,
        problem_id=0,
        level=0,
        baseline_time=baseline_time,
        tolerance=1e-2,
        num_correct_trials=1,
        num_perf_trials=2,
        device=device,
        verbose=False,
    )
    assert 'reward' in result
    assert result['accuracy_score'] == 1.0  # Should be correct
    assert result['linear_time_score'] <= 1.0
    assert result['runtime'] > 0 or result['runtime'] == -1.0

# Additional tests for edge cases can be added here
