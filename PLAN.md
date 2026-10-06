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

## Day 4 — GRPO ✅ Complete

Day 4's only deliverable per spec *is* documentation — no separate code step.

- [x] `notes/papers/grpo.md` — the group-relative advantage derivation, the
      full GRPO objective (PPO's clipped surrogate + group-relative
      advantage + explicit KL penalty), a 3-way comparison table against
      PPO and DPO (RL loop? value network? reward model?), the GRPO-vs-DPO
      generation-cost trade-off, limitations (degenerate zero-variance
      groups, no within-response credit assignment), the Day 7 plan, and
      a papers table (DeepSeekMath, DeepSeek-R1, PPO, Schulman's KL note)

**Files created:**
- [x] `notes/papers/grpo.md`

---

## Day 5 — Build Evaluation Pipeline ✅ Complete

- [x] Documentation pass: `notes/evaluation_pipeline.md` — GSM8K format
      background (incl. the `#### N` marker and `<<...>>` calculator
      annotations), the "one `extract_answer` for both ground truth and
      model output" design decision, numeric-normalization and
      float-equality comparison rules, a file-by-file implementation plan
      with draft code and an 8-case test table, and a references section
      (GSM8K paper, HF `transformers`/`datasets` docs)
- [x] `scripts/test_model.py` — load Qwen, apply chat template, generate
      greedily on MPS, print response. Verified: correctly solves the
      box-of-balls example from `CLAUDE.md` ("20 balls").
- [x] `scripts/test_dataset.py` — load GSM8K (`main`, configurable split/n),
      print question + raw answer + extracted ground truth for each sample.
      Verified against real examples (e.g. "Natalia" question → `72`).
- [x] `src/evaluation/answer_extraction.py` — `extract_answer(text) -> str | None`,
      built TDD-style (RED confirmed via failing import, then implemented)
- [x] Unit tests for answer extraction — 8 cases in `tests/test_answer_extraction.py`
- [x] `src/rewards/correctness.py` — deterministic binary reward,
      TDD-style, 4 cases in `tests/test_correctness.py`
- [x] All 12 tests pass (`pytest tests/`)
- [x] Fixed a real bug found during implementation: `apply_chat_template`
      returns a `BatchEncoding` (not a bare tensor) on transformers 5.18.0
      — documented in `notes/evaluation_pipeline.md` §8, not silently patched

**Files created:**
- [x] `notes/evaluation_pipeline.md` (doc, then updated post-implementation with §8 notes)
- [x] `src/evaluation/answer_extraction.py`
- [x] `src/rewards/correctness.py`
- [x] `tests/test_answer_extraction.py`
- [x] `tests/test_correctness.py`
- [x] `scripts/test_model.py`
- [x] `scripts/test_dataset.py`

---

## Day 6 — Tiny DPO Smoke Test ✅ Complete

- [x] Documentation pass: `notes/dpo_smoke_test.md` — synthetic GSM8K
      preference-pair construction (chosen/rejected differ only in the
      final number), TRL `DPOTrainer`/`DPOConfig` API verified directly
      against the installed `trl==1.14.1` source (not guessed), the LoRA
      paper, and all 3 pre-implementation open questions resolved
- [x] `src/datasets/dpo_preference.py` — `build_dpo_preference_dataset()`,
      TDD-style, 5 unit tests in `tests/test_dpo_preference.py`
- [x] `scripts/smoke_test_dpo.py` — TRL `DPOTrainer` + LoRA (`peft_config`
      passed directly, `ref_model=None` → adapter-disabling, no second
      model copy), 20 preference examples, 3 optimization steps
- [x] Run verified: all 8 Day 6 success criteria met (model/dataset load,
      trainer initializes, forward/backward pass work, loss finite,
      3 steps completed, checkpoint saved) — ran on MPS in ~29s, no
      hardware issues
- [x] `results/dpo_smoke_001/{config.json,metrics.json,notes.md}` written
      per the `CLAUDE.md` Result Storage convention; `.gitignore` extended
      so the LoRA adapter / tokenizer copy / training_args.bin stay local
- [x] All 17 project tests pass (`pytest tests/`)

**Files created:**
- [x] `notes/dpo_smoke_test.md` (doc, then updated with §7 post-implementation notes)
- [x] `src/datasets/dpo_preference.py`
- [x] `tests/test_dpo_preference.py`
- [x] `scripts/smoke_test_dpo.py`
- [x] `results/dpo_smoke_001/config.json`
- [x] `results/dpo_smoke_001/metrics.json`
- [x] `results/dpo_smoke_001/notes.md`

**Files modified:**
- [x] `.gitignore` — exclude generated checkpoint artifacts under `results/*/`

---

## Day 7 — Tiny GRPO Smoke Test ✅ Complete

- [x] Documentation pass: `notes/grpo_smoke_test.md` — the `(prompt,
      answer)` dataset design, TRL `GRPOTrainer`/`GRPOConfig` API verified
      directly against the installed `trl==1.14.1` source (reward
      function signature, no `ref_model` param, two library defaults that
      silently diverge from the paper's formulation, the zero-variance
      epsilon, the batch-size/group-size divisibility constraint), and
      all open questions resolved
- [x] `src/datasets/grpo_prompts.py` — `build_grpo_prompt_dataset()`,
      TDD-style, 3 unit tests
- [x] `src/rewards/correctness.py` extended with `gsm8k_grpo_reward()` —
      a thin TRL-compatible adapter over the existing `compute_reward`,
      2 unit tests
- [x] `scripts/smoke_test_grpo.py` — TRL `GRPOTrainer` + LoRA, group size
      4, 8 prompts, 3 steps
- [x] **Real bug found and fixed, not just a clean first run:** the first
      actual run completed without crashing but with a dead reward signal
      (`reward=0.0` for every group) — root-caused by direct token-id
      inspection to `GRPOTrainer`'s `eos_token_id` being a single token
      the model doesn't naturally produce in non-chat prompting mode.
      Fixed via `generation_kwargs` override. A second, unrelated fix
      (`padding_side`) was tried first, found to be correct-but-not-
      causal, and kept anyway rather than silently dropped. Full
      post-mortem in `notes/grpo_smoke_test.md` §7 and
      `results/grpo_smoke_001/notes.md`.
- [x] Run verified after the fix: rewards `0.75, 0.75, 1.0` with genuine
      nonzero `reward_std` (`0.5, 0.5, 0.0`) — the group-relative
      advantage mechanism demonstrably exercised with real signal, not
      just "ran without error." ~71s on MPS.
- [x] `results/grpo_smoke_001/{config.json,metrics.json,notes.md}` written
- [x] All 22 project tests pass (`pytest tests/`)

**Files created:**
- [x] `notes/grpo_smoke_test.md` (doc, then updated with §7 post-implementation notes)
- [x] `src/datasets/grpo_prompts.py`
- [x] `tests/test_grpo_prompts.py`
- [x] `tests/test_grpo_reward.py`
- [x] `scripts/smoke_test_grpo.py`
- [x] `results/grpo_smoke_001/config.json`
- [x] `results/grpo_smoke_001/metrics.json`
- [x] `results/grpo_smoke_001/notes.md`

**Files modified:**
- [x] `src/rewards/correctness.py` — added `gsm8k_grpo_reward()`

---

## Week 1 Final Success Criteria (from `CLAUDE.md`)

- [x] Research repository is clean and reproducible
- [x] Qwen2.5-0.5B runs locally
- [x] GSM8K loads correctly
- [x] Automatic answer extraction works
- [x] Binary correctness reward works
- [x] DPO concepts are documented
- [x] PPO concepts are documented
- [x] GRPO concepts are documented
- [x] Tiny DPO training reaches at least one optimization step
- [x] Tiny GRPO training reaches at least one optimization step, if local hardware permits
- [x] Configs/results are saved reproducibly
- [x] Code is committed to Git

**All Week 1 success criteria complete.**

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
| 3 | `README.md` | ✅ updated (progress status, notes links) |
| 4 | `notes/papers/grpo.md` | ✅ created |
| 5 | `notes/evaluation_pipeline.md` | ✅ created, then updated with post-implementation notes |
| 5 | `src/evaluation/answer_extraction.py` | ✅ created |
| 5 | `src/rewards/correctness.py` | ✅ created |
| 5 | `tests/test_answer_extraction.py` | ✅ created (8 tests) |
| 5 | `tests/test_correctness.py` | ✅ created (4 tests) |
| 5 | `scripts/test_model.py` | ✅ created |
| 5 | `scripts/test_dataset.py` | ✅ created |
| 6 | `notes/dpo_smoke_test.md` | ✅ created, then updated with post-implementation notes |
| 6 | `src/datasets/dpo_preference.py` | ✅ created |
| 6 | `tests/test_dpo_preference.py` | ✅ created (5 tests) |
| 6 | `scripts/smoke_test_dpo.py` | ✅ created |
| 6 | `results/dpo_smoke_001/config.json`, `metrics.json`, `notes.md` | ✅ created |
| 6 | `.gitignore` | ✅ updated (exclude `results/*/` checkpoint artifacts) |
| 6 | `PLAYGROUND.md` | ✅ created (hands-on guide to run/tweak every file so far) |
| 7 | `notes/grpo_smoke_test.md` | ✅ created, then updated with §7 post-implementation notes |
| 7 | `src/datasets/grpo_prompts.py` | ✅ created |
| 7 | `tests/test_grpo_prompts.py` | ✅ created (3 tests) |
| 7 | `tests/test_grpo_reward.py` | ✅ created (2 tests) |
| 7 | `src/rewards/correctness.py` | ✅ updated (added `gsm8k_grpo_reward()`) |
| 7 | `scripts/smoke_test_grpo.py` | ✅ created |
| 7 | `results/grpo_smoke_001/config.json`, `metrics.json`, `notes.md` | ✅ created |
