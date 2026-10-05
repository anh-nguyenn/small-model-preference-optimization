# Playground Guide — Explore the Code Yourself

A hands-on companion to `PLAN.md` (what's done) and `notes/` (why it's built
this way). This doc is for actually **running and poking at** every file
that's been written so far, in the order that builds understanding
fastest — each step uses only what the previous steps already showed you.

Every command below assumes you're in the repo root with the venv active:

```bash
cd small-model-preference-optimization
source .venv/bin/activate
```

## Quick map

| Step | File | What it is | Needs model/network? |
|---|---|---|---|
| 1 | `scripts/test_model.py` | Load Qwen, generate a response | Yes (first run downloads/loads the model) |
| 2 | `scripts/test_dataset.py` | Load GSM8K, print real examples | Yes (downloads dataset, cached after) |
| 3 | `src/evaluation/answer_extraction.py` | Pull the final number out of text | No — pure Python |
| 4 | `src/rewards/correctness.py` | Binary correct/incorrect reward | No — pure Python |
| 5 | `src/datasets/dpo_preference.py` | Build (chosen, rejected) pairs | Yes (downloads dataset) |
| 6 | `scripts/smoke_test_dpo.py` | Full DPO training loop | Yes (heaviest — ~30s) |

Steps 3–4 have zero network/model dependency and run in milliseconds —
they're the best files to actually edit and re-run while you're learning,
since the feedback loop is instant.

---

## Step 1 — `scripts/test_model.py`

**What it does:** loads `Qwen/Qwen2.5-0.5B-Instruct`, applies its chat
template to a prompt, generates greedily, prints the response. No custom
project logic — just the `transformers` API.

```bash
python scripts/test_model.py
```

Expected output ends with something like:

```text
Response:
To determine the total number of balls in the box, you simply add the
number of red balls to the number of blue balls.
...
Therefore, there are 20 balls in the box.
```

**Try tweaking:**
- `--prompt "What is 17 * 23?"` — see how it does on a different problem
- `--max-new-tokens 32` — truncate the response, see it cut off mid-thought
- Open the file and change `do_sample=False` to `True` (add `temperature=0.8`)
  — run it twice, see the response change between runs (this is exactly the
  kind of sampling GRPO will need in Step-7-to-come, generating multiple
  *different* responses to the same prompt)

**Background:** `notes/evaluation_pipeline.md` §4.4 explains the chat
template detail and a real bug this script hit during development
(§8) — worth reading if you're curious why `apply_chat_template` is called
the specific way it is here.

---

## Step 2 — `scripts/test_dataset.py`

**What it does:** loads the real GSM8K dataset and prints `n` examples —
question, raw answer field, and what `extract_answer` pulls from it.

```bash
python scripts/test_dataset.py --n 5
```

Expected output, one block per example:

```text
--- Example 1 ---
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Raw answer field: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
Extracted ground-truth answer: 72
```

**Try tweaking:**
- `--split test --n 10` — look at the held-out split instead of train
- `--n 50` and skim for any example where "Extracted ground-truth answer"
  looks wrong — if you find one, that's a real edge case worth adding to
  `tests/test_answer_extraction.py` (see Step 3)

**Background:** `notes/evaluation_pipeline.md` §2 walks through exactly
why the raw answer field looks like this (the `####` marker, the
`<<12+8=20>>` calculator annotations).

---

## Step 3 — `src/evaluation/answer_extraction.py`

**What it does:** one function, `extract_answer(text) -> str | None`. No
dependencies, no network — read the whole file, it's ~20 lines of regex.

Run its tests to see every case it's built to handle:

```bash
python -m pytest tests/test_answer_extraction.py -v
```

You should see 8 tests pass, each named after the behavior it checks
(e.g. `test_falls_back_to_last_number_when_no_keyword_present`).

**Play with it directly** — no script needed, just a one-liner:

```bash
python3 -c "
from src.evaluation.answer_extraction import extract_answer
print(extract_answer('After some work, I get 42 as my final count.'))
"
```

**Try tweaking:**
- Open `tests/test_answer_extraction.py`, add a new test with a phrasing
  you think might break it (e.g. `\"The results were: 5, 10, and 15.\"` —
  what should it extract? what *does* it extract?), run pytest, see if
  your intuition matches the code
- Open `answer_extraction.py` itself and comment out the `_ANSWER_PHRASE`
  pattern — rerun the tests, see exactly which ones fail and why (this
  shows you which test cases depend on which regex)

**Background:** `notes/evaluation_pipeline.md` §3 explains the design
decisions (why one function handles both ground truth and model output,
why the fallback picks the *last* number, why comparison is numeric not
string-based).

---

## Step 4 — `src/rewards/correctness.py`

**What it does:** `compute_reward(generated, ground_truth) -> float`,
calling `extract_answer` on both sides and comparing. This is the
project's entire reward signal — deterministic, no learned model, no LLM
judge.

```bash
python -m pytest tests/test_correctness.py -v
```

**Play with it directly:**

```bash
python3 -c "
from src.rewards.correctness import compute_reward
print(compute_reward('The answer is 20.', '#### 20'))   # expect 1.0
print(compute_reward('The answer is 19.', '#### 20'))   # expect 0.0
print(compute_reward('I have no idea.', '#### 20'))     # expect 0.0
"
```

**Try tweaking:** this is literally the function GRPO (Day 7) will call
once per sampled response to compute `reward_i` before the group-relative
advantage formula — try writing a tiny loop that computes the reward for
5 made-up "model responses" to the same ground truth, and compute the
mean/std yourself. That's the exact calculation `notes/papers/grpo.md` §4
describes, just done by hand instead of inside TRL's trainer.

---

## Step 5 — `src/datasets/dpo_preference.py`

**What it does:** builds a small, deterministic (prompt, chosen, rejected)
dataset from real GSM8K rows — chosen and rejected share identical
reasoning text, only the final number differs.

```bash
python -m pytest tests/test_dpo_preference.py -v
```

**Play with it directly** — print a real row and compare chosen vs.
rejected side by side:

```bash
python3 -c "
from src.datasets.dpo_preference import build_dpo_preference_dataset
row = build_dpo_preference_dataset(n_examples=1)[0]
print('PROMPT:  ', row['prompt'])
print('CHOSEN:  ', row['chosen'])
print('REJECTED:', row['rejected'])
"
```

**Try tweaking:**
- `build_dpo_preference_dataset(n_examples=3, wrong_offset=10)` — make the
  "wrong" answer further off, see how `rejected` changes
- Pick a row and run both `chosen` and `rejected` through
  `extract_answer()` yourself to confirm they really do differ (this is
  exactly what `tests/test_dpo_preference.py` asserts automatically)

**Background:** `notes/dpo_smoke_test.md` §2 explains why this specific
construction (same reasoning, different final number) was chosen over
alternatives.

---

## Step 6 — `scripts/smoke_test_dpo.py`

**What it does:** the full pipeline — builds the preference dataset (Step
5), loads the model (Step 1), wraps it in a LoRA adapter, and runs TRL's
`DPOTrainer` for a few real optimization steps.

```bash
python scripts/smoke_test_dpo.py
```

Takes about 30 seconds on Apple Silicon MPS. Watch for:

```text
Trainable (LoRA) parameters: 540,672
...
{'loss': '0.6729', 'grad_norm': '5.477', ...}
...
Completed global steps: 3
...
Day 6 success criteria:
  [x] model loads
  ...
```

**Try tweaking:**
- `--n-examples 40 --max-steps 5` — push to the top of the "10–50
  examples, 1–5 steps" smoke-test range from `CLAUDE.md`
- `--learning-rate 1e-3` — a much bigger learning rate; watch `grad_norm`
  and `loss` in the printed logs — does training look more or less stable?
- `--lora-r 16 --lora-alpha 32` — bigger LoRA rank, more trainable
  parameters (check the printed count) — re-run and compare
- After a run, open `results/dpo_smoke_001/metrics.json` and
  `results/dpo_smoke_001/notes.md` to see exactly what got recorded and
  why (see `CLAUDE.md`'s "Result Storage" convention)

**Background:** `notes/dpo_smoke_test.md` is the full design doc for this
script — read §3 if you want to understand exactly how TRL's `DPOTrainer`
handles the reference model and LoRA under the hood (verified against the
installed library's actual source, not assumed).

---

## If something looks wrong

Check the relevant note's "Implementation notes" / "§8" / "§7"
post-implementation section first — several real issues hit during
development are documented there rather than silently patched
(`notes/evaluation_pipeline.md` §8 has the `apply_chat_template` surprise,
`notes/dpo_smoke_test.md` §7 has the DPO-specific findings). If what
you're seeing isn't covered there, it's a new finding worth adding.
