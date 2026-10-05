# Small-Model Preference Optimization Research

A small, reproducible research codebase for experimenting with **Direct Preference
Optimization (DPO)** and **Group Relative Policy Optimization (GRPO)** on a small
language model, using GSM8K-style reasoning tasks as the test bed.

This is a Week 1 learning project: the goal is to go from understanding DPO/GRPO
conceptually to being able to run both methods, end-to-end, on a small model —
not to produce a novel research result. See [`CLAUDE.md`](./CLAUDE.md) for the
full project spec and day-by-day plan, [`PLAN.md`](./PLAN.md) for a
live, checkbox-tracked progress log, and [`PLAYGROUND.md`](./PLAYGROUND.md)
for a hands-on, run-this-yourself guide to every file written so far.

**Working agreement:** before implementing any day's code, we first write a
background-knowledge + implementation-plan + relevant-papers document, then
implement against it. See `notes/` for the docs produced so far.

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
├── PLAN.md             # day-by-day progress checklist
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

## Background reading

Written before implementing the corresponding training code — background
knowledge, an implementation plan, and relevant papers for each topic:

- [`notes/policy_gradient.md`](./notes/policy_gradient.md) — RL vocabulary
  (policy/action/reward/advantage), the policy gradient theorem, and why
  baselines reduce variance
- [`notes/ppo.md`](./notes/ppo.md) — the PPO clipped objective, the full
  PPO-for-RLHF architecture, and what DPO/GRPO each remove from it
- [`notes/papers/dpo.md`](./notes/papers/dpo.md) — the DPO loss derivation
  and why it needs no reward model or RL loop
- [`notes/papers/grpo.md`](./notes/papers/grpo.md) — the group-relative
  advantage derivation, and how GRPO compares to both PPO and DPO
- [`notes/evaluation_pipeline.md`](./notes/evaluation_pipeline.md) — GSM8K's
  actual format, the answer-extraction design, and the binary reward
- [`notes/dpo_smoke_test.md`](./notes/dpo_smoke_test.md) — the synthetic
  preference-pair construction and TRL `DPOTrainer`/LoRA setup, verified
  against the installed TRL source

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

Week 1, Day 6 of 7 complete:

- [x] Day 1 — environment set up, dependencies installed, MPS verified, Qwen loads
- [x] Day 2 — policy gradient / PPO background notes
- [x] Day 3 — DPO paper notes
- [x] Day 4 — GRPO paper notes
- [x] Day 5 — evaluation pipeline (model/dataset smoke scripts, answer extraction, binary reward — 12/12 tests passing)
- [x] Day 6 — tiny DPO smoke test (LoRA + TRL `DPOTrainer`, 3 steps on MPS, checkpoint saved — 17/17 tests passing)
- [ ] Day 7 — tiny GRPO smoke test

See [`PLAN.md`](./PLAN.md) for the detailed, per-day checklist and file log,
and `CLAUDE.md` for the full task spec and success criteria.
