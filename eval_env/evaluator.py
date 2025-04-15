"""
Evaluation Environment Core Logic
"""
import torch
import os
from . import helpers


def evaluate_and_reward(
    original_model_src: str,
    custom_model_src: str,
    problem_id: int,
    level: int,
    baseline_time: float,
    tolerance: float = 1e-2,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    device: torch.device = None,
    verbose: bool = False,
):
    """
    Evaluate a candidate CUDA kernel against the reference PyTorch model.
    Returns a reward based on correctness and runtime performance.
    """
    result = helpers.eval_kernel_against_ref(
        original_model_src=original_model_src,
        custom_model_src=custom_model_src,
        seed_num=42,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        verbose=verbose,
        measure_performance=True,
        build_dir=None,
        device=device or (torch.cuda.current_device() if torch.cuda.is_available() else None),
    )

    # Accuracy score: 1.0 if correct, else 0.0
    accuracy_score = 1.0 if result and getattr(result, 'correctness', False) else 0.0

    # Time score: linear scaling, clipped to [0,1]
    runtime = getattr(result, 'runtime', -1.0)
    if runtime > 0 and baseline_time > 0:
        linear_time_score = max(0.0, 1.0 - (runtime / baseline_time))
    else:
        linear_time_score = 0.0

    reward = accuracy_score * 0.5 + linear_time_score * 0.5

    return {
        "reward": reward,
        "accuracy_score": accuracy_score,
        "linear_time_score": linear_time_score,
        "runtime": runtime,
        "baseline_time": baseline_time,
        "metadata": getattr(result, 'metadata', {}),
    }
