# Evaluation Environment for PyTorch-to-CUDA RL

This project provides an API for evaluating CUDA kernel code generated from PyTorch models. It computes a reward based on correctness and runtime performance, suitable for reinforcement learning with LLMs.

## Features
- Accepts PyTorch and CUDA model code via REST API
- Evaluates correctness and runtime
- Returns a reward: `reward = accuracy * 0.5 + linear_time * 0.5`

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```
3. Send POST requests to `/evaluate` with the required fields.

## API Example

POST `/evaluate`
```json
{
  "original_model_src": "...",
  "custom_model_src": "...",
  "problem_id": 1,
  "level": 2,
  "baseline_time": 123.4
}
```

## Project Structure
- `eval_env/` - Core logic and helpers
- `main.py` - FastAPI server
- `requirements.txt` - Dependencies

---

## TODO
- [ ] Support for custom CUDA kernel testing in the evaluation pipeline (see docs for example usage)
- [ ] Expose and document all evaluation parameters (e.g., measure_performance, build_dir, etc.) in the API and README
- [ ] Some evaluation parameters are not yet settable via the API or CLI (e.g., build_dir, measure_performance, etc.)
- [ ] Add more real-world CUDA kernel tests and examples

Replace `helpers.py` with your actual evaluation helpers code.
