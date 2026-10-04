# Project Plan & Progress

Tracks progress against the Week 1 plan in [`CLAUDE.md`](./CLAUDE.md). Updated
as each day's work lands — check off items as they're completed, and log
every file created/edited under that day so the history stays easy to audit.

**Working agreement (added after Day 2):** before writing any implementation
code for a day, first write a detailed documentation pass — background
knowledge, a coding implementation plan, and relevant papers where
applicable — then implement. Each day's section below links to that doc.

---

## Day 1 — Environment and GitHub Repository ✅ Complete

- [x] Repository exists — [`small-model-preference-optimization`](https://github.com/anh-nguyenn/small-model-preference-optimization) (private)
- [x] Virtual environment works (`.venv`, Python 3.12.6)
- [x] Dependencies install (torch, transformers, datasets, accelerate, trl, peft, evaluate, wandb, matplotlib, pandas, pytest)
- [x] PyTorch imports (torch 2.14.1)
- [x] MPS status is printed (`torch.backends.mps.is_available()` → `True`)
- [x] Qwen model can be loaded (`Qwen/Qwen2.5-0.5B-Instruct`, 494M params loaded successfully)

**Files created:**
- [x] `.gitignore`
- [x] `README.md`
- [x] `CLAUDE.md` (copied from spec)
- [x] `requirements.txt`
- [x] Directory scaffold: `configs/`, `data/`, `notebooks/`, `scripts/`, `src/{datasets,evaluation,rewards,training}/` (with `__init__.py`), `tests/` (with `__init__.py`), `results/`, `notes/papers/`
- [x] `.venv/` created locally (not committed, per `.gitignore`)

---

## Day 2 — Policy Gradient / PPO Notes ✅ Complete

No implementation required this day (per spec) — notes only.

- [x] `notes/policy_gradient.md` — RL vocabulary from scratch, the prompt/
      token/LM/reward mapping, policy gradient theorem (intuitive),
      variance problem, advantage = reward − baseline, worked numeric
      example, papers table, glossary, open questions
- [x] `notes/ppo.md` — why PPO exists (trust regions, clipped objective
      explained term-by-term), full PPO-for-RLHF architecture (policy /
      reference / reward / value models), explicit table of what DPO and
      GRPO each strip out of that stack, papers table, glossary, open
      questions

**Files created:**
- [x] `notes/policy_gradient.md`
- [x] `notes/ppo.md`

---

## Day 3 — DPO ✅ Complete

Day 3's only deliverable per spec *is* documentation — no separate code step.

- [x] `notes/papers/dpo.md` using the required structure (Problem / Existing
      approach / Main contribution / How training works / Inputs required /
      Why it is cheaper than PPO / Limitations / Relation to this project /
      Open questions), plus a papers table. Covers the reward
      reparameterization trick, the closed-form DPO loss derivation, how it
      maps onto `policy_gradient.md`/`ppo.md`, and the Day 6 plan for
      building a synthetic GSM8K preference dataset.

**Files created:**
- [x] `notes/papers/dpo.md`

---

## Day 4 — GRPO ⬜ Not started

- [ ] Documentation pass: background knowledge + implementation plan +
      relevant papers (before any code)
- [ ] `notes/papers/grpo.md` covering the group-relative advantage
      derivation and "GRPO replaces the learned critic with group
      statistics"

**Files created:** _(none yet)_

---

## Day 5 — Build Evaluation Pipeline ⬜ Not started

- [ ] Documentation pass: background knowledge + implementation plan +
      relevant papers (before any code)
- [ ] `scripts/test_model.py` — load Qwen, send a prompt, print generation
- [ ] `scripts/test_dataset.py` — load GSM8K, print 10 sample Q/A pairs
- [ ] `src/evaluation/answer_extraction.py` — `extract_answer(text) -> str | None`
- [ ] Unit tests for answer extraction
- [ ] `src/rewards/correctness.py` — deterministic binary reward

**Files created:** _(none yet)_

---

## Day 6 — Tiny DPO Smoke Test ⬜ Not started

- [ ] Documentation pass: background knowledge + implementation plan +
      relevant papers (before any code)
- [ ] `scripts/smoke_test_dpo.py` — TRL `DPOTrainer`, 10–50 examples,
      1–5 optimization steps, checkpoint saved

**Files created:** _(none yet)_

---

## Day 7 — Tiny GRPO Smoke Test ⬜ Not started

- [ ] Documentation pass: background knowledge + implementation plan +
      relevant papers (before any code)
- [ ] `scripts/smoke_test_grpo.py` — TRL `GRPOTrainer`, small group size,
      binary correctness reward, very few steps

**Files created:** _(none yet)_

---

## Week 1 Final Success Criteria (from `CLAUDE.md`)

- [x] Research repository is clean and reproducible
- [x] Qwen2.5-0.5B runs locally
- [ ] GSM8K loads correctly
- [ ] Automatic answer extraction works
- [ ] Binary correctness reward works
- [x] DPO concepts are documented
- [x] PPO concepts are documented
- [ ] GRPO concepts are documented
- [ ] Tiny DPO training reaches at least one optimization step
- [ ] Tiny GRPO training reaches at least one optimization step, if local hardware permits
- [ ] Configs/results are saved reproducibly
- [x] Code is committed to Git

---

## Full file log (chronological)

| Day | File | Status |
|---|---|---|
| 1 | `.gitignore` | ✅ created |
| 1 | `README.md` | ✅ created |
| 1 | `CLAUDE.md` | ✅ created (spec copy) |
| 1 | `requirements.txt` | ✅ created |
| 1 | `configs/.gitkeep`, `data/.gitkeep`, `notebooks/.gitkeep`, `results/.gitkeep` | ✅ created |
| 1 | `src/__init__.py`, `src/{datasets,evaluation,rewards,training}/__init__.py` | ✅ created |
| 1 | `tests/__init__.py` | ✅ created |
| 2 | `notes/policy_gradient.md` | ✅ created |
| 2 | `notes/ppo.md` | ✅ created |
| 2 | `PLAN.md` (this file) | ✅ created |
| 3 | `notes/papers/dpo.md` | ✅ created |
