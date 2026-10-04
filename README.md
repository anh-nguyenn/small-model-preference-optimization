# Small-Model Preference Optimization Research

A small, reproducible research codebase for experimenting with **Direct Preference
Optimization (DPO)** and **Group Relative Policy Optimization (GRPO)** on a small
language model, using GSM8K-style reasoning tasks as the test bed.

This is a Week 1 learning project: the goal is to go from understanding DPO/GRPO
conceptually to being able to run both methods, end-to-end, on a small model —
not to produce a novel research result. See [`CLAUDE.md`](./CLAUDE.md) for the
full project spec and day-by-day plan.

## Model and dataset

- Model: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Dataset: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (`main` config)

## Hardware

Developed primarily on a MacBook (Apple Silicon) using PyTorch **MPS**. No CUDA
or paid cloud GPU is required for Week 1. LoRA/PEFT is used to keep training
cheap; smoke tests use ~10–50 examples and 1–5 optimization steps — the goal is
to verify the pipeline works, not to produce a well-trained model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify PyTorch + MPS:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

## Repository structure

```text
small-model-preference-optimization/
├── configs/            # experiment configs
├── data/               # local data (not committed)
├── notebooks/          # exploratory notebooks
├── scripts/            # thin entry points (test_model.py, smoke tests, ...)
├── src/
│   ├── datasets/       # dataset loading/formatting
│   ├── evaluation/     # answer extraction, metrics
│   ├── rewards/        # reward functions (e.g. binary correctness)
│   └── training/       # DPO/GRPO training setup
├── tests/              # pytest unit tests
├── results/            # experiment outputs (config.json, metrics.json, notes.md)
├── notes/              # conceptual notes (policy gradient, PPO)
│   └── papers/         # paper summaries (DPO, GRPO)
└── requirements.txt
```

## Commands

```bash
python scripts/test_model.py       # load Qwen, run a sample generation
python scripts/test_dataset.py     # load GSM8K, print sample Q/A pairs

pytest tests/                      # run unit tests (e.g. answer extraction)

python scripts/smoke_test_dpo.py   # tiny DPO smoke test (1-5 steps)
python scripts/smoke_test_grpo.py  # tiny GRPO smoke test (1-5 steps)
```

## Principles

- **Clarity over abstraction.** This is a research project; avoid premature
  architecture.
- **Reproducibility.** Seeds, model/dataset names, and hyperparameters are
  explicit and stored alongside results.
- **No silent methodology changes.** Changes to reward functions, answer
  extraction, prompt format, etc. are documented, not silently swapped in.
- **Cheap experiments.** Small datasets, few steps, LoRA/PEFT — the point of
  Week 1 is a working pipeline, not model quality.

## Status

Week 1, Day 1 — environment and repository setup. See `CLAUDE.md` for the
full task breakdown and success criteria.
