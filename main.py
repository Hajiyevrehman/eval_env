from fastapi import FastAPI
from pydantic import BaseModel
from eval_env import evaluator
import torch

app = FastAPI()

class EvalRequest(BaseModel):
    original_model_src: str
    custom_model_src: str
    problem_id: int
    level: int
    baseline_time: float
    tolerance: float = 1e-2
    num_correct_trials: int = 1
    num_perf_trials: int = 10
    verbose: bool = False

@app.post("/evaluate")
def evaluate_kernel(req: EvalRequest):
    result = evaluator.evaluate_and_reward(
        original_model_src=req.original_model_src,
        custom_model_src=req.custom_model_src,
        problem_id=req.problem_id,
        level=req.level,
        baseline_time=req.baseline_time,
        tolerance=req.tolerance,
        num_correct_trials=req.num_correct_trials,
        num_perf_trials=req.num_perf_trials,
        device=torch.cuda.current_device() if torch.cuda.is_available() else None,
        verbose=req.verbose,
    )
    return result
