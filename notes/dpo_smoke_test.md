# Day 6 — Tiny DPO Smoke Test: Implementation Plan

> Written before `scripts/smoke_test_dpo.py`. Read `notes/papers/dpo.md`
> first for the theory (reward reparameterization, the DPO loss). This note
> is purely about *this project's* implementation: how the preference
> dataset is built, how TRL's `DPOTrainer` is actually configured (verified
> against the installed `trl==1.14.1` source, not guessed), and what
> "success" means for a smoke test per `CLAUDE.md`.

---

## 1. What "success" means here

Per `CLAUDE.md` Day 6, this is explicitly **not** about model quality. The
test succeeds if:

```text
model loads
dataset loads
trainer initializes
forward pass works
loss is produced
backward pass works
1–5 optimization steps complete
checkpoint can be saved
```

Scope guardrails already set by `CLAUDE.md`: 10–50 preference examples,
LoRA/PEFT (avoid full fine-tuning), no full-model training.

---

## 2. Background: building the preference dataset

`notes/papers/dpo.md` (§Relation to this project) already flagged that this
project needs a **synthetic** GSM8K preference dataset — there's no human
preference data lying around for GSM8K. The design choice, made concrete
here:

**Chosen and rejected share identical reasoning text; only the final
number differs.** Concretely, for a GSM8K example:

```
reasoning = everything in the "answer" field before "####", with
            calculator annotations ("<<12+8=20>>") stripped
correct_number = extract_answer(example["answer"])   -- reuses Day 5's
                  extract_answer() on the "#### N" marker, no new parser

chosen   = f"{reasoning}\nTherefore, the answer is {correct_number}."
rejected = f"{reasoning}\nTherefore, the answer is {wrong_number}."
           where wrong_number = correct_number + 1   (deterministic offset)
```

Why this construction, specifically:

- **Deterministic and reproducible** — no model sampling needed to build
  the dataset, no randomness beyond dataset shuffling. A rerun produces
  the exact same preference pairs.
- **Isolates the signal DPO is supposed to learn.** Since chosen/rejected
  differ *only* in the final number, the preference pair cleanly encodes
  "say the correct number," with no confound from differences in
  reasoning style, length, or phrasing between the two responses.
- **Reuses Day 5's `extract_answer`**, per `CLAUDE.md`'s reuse principle —
  no second ground-truth parser.
- **Explicitly documented as a simplification**, consistent with the
  limitation already flagged in `notes/papers/dpo.md` ("Limitations"):
  real DPO datasets encode genuine preference judgments; this one encodes
  "verified-correct vs. verified-wrong," which is the project's chosen
  proxy for "preference" throughout (same proxy GRPO's reward function
  uses in Day 7). This is **not** a new methodology decision introduced
  silently — it's the same binary-correctness notion from `src/rewards/
  correctness.py` (Day 5), just used to label pairs instead of score single
  responses.

`wrong_number = correct_number + 1` is deliberately the simplest possible
"wrong" — a smoke test doesn't need a *plausible* wrong answer, just an
unambiguously incorrect one that `extract_answer`/float-comparison agrees
is different from the correct one.

---

## 3. Background: TRL's actual `DPOTrainer` API (verified, not guessed)

Before writing the script, the installed `trl==1.14.1` was inspected
directly (constructor signature, docstrings, and source) rather than
assumed from the paper or older TRL versions, since this exact question
was flagged as open in `notes/papers/dpo.md` §Open questions.

**Findings:**

1. **Dataset format:** a `datasets.Dataset` with string columns `"prompt"`,
   `"chosen"`, `"rejected"`. In the non-conversational (plain string) case
   — what this project uses, for simplicity — the trainer tokenizes
   `example["prompt"] + example["chosen"]` as one sequence, meaning
   **`chosen`/`rejected` must be completions only, not the prompt repeated
   inside them.** This confirms the construction in §2 is already in the
   right shape (reasoning text is the completion; the question is the
   prompt).
2. **Reference model, resolved:** passing `ref_model=None` together with
   a `peft_config` (see next point) makes the trainer automatically use
   "the initial policy state before training" as the reference — internally
   implemented via **adapter-disabling** (`use_adapter(model,
   adapter_name=...)`) when it detects a PEFT model. **No second full model
   copy is ever loaded.** This directly resolves the open question from
   `notes/papers/dpo.md` §7 ("How does TRL's `DPOTrainer` handle the
   reference model when using LoRA?").
3. **LoRA, resolved:** pass a `peft.LoraConfig` as `peft_config=...`
   directly to `DPOTrainer(...)` — the trainer itself calls
   `get_peft_model(model, peft_config)` internally. **The script should
   pass the plain base model, not a pre-wrapped `PeftModel`**, and let the
   trainer do the wrapping; wrapping it manually first and also passing
   `peft_config` raises an explicit error in the trainer's own validation.
4. **Key `DPOConfig` defaults confirmed by introspection:** `beta=0.1`
   (matches `notes/papers/dpo.md`'s assumption), `max_steps=-1` (must be
   set explicitly), `per_device_train_batch_size=8` (too large for a
   10–50 example smoke test — will be set smaller), `learning_rate=1e-6`
   (very conservative default; bumped up for this smoke test since the
   goal is "see the loss/gradients move," not production fine-tuning
   stability — documented as a deliberate deviation from the default, not
   a silent one).

---

## 4. Implementation plan

### 4.1 `src/datasets/dpo_preference.py` (new, reusable)

```python
import re
from datasets import Dataset, load_dataset
from src.evaluation.answer_extraction import extract_answer

_CALCULATOR_ANNOTATION = re.compile(r"<<[^>]*>>")

def _strip_calculator_annotations(text: str) -> str:
    return _CALCULATOR_ANNOTATION.sub("", text)

def build_dpo_preference_dataset(n_examples: int, split: str = "train", wrong_offset: int = 1) -> Dataset:
    """Build a synthetic (prompt, chosen, rejected) preference dataset from
    GSM8K. chosen/rejected share identical reasoning text; only the final
    number differs (see notes/dpo_smoke_test.md section 2 for rationale).
    """
    raw = load_dataset("openai/gsm8k", "main", split=split).select(range(n_examples))
    rows = []
    for example in raw:
        reasoning, _, _ = example["answer"].partition("####")
        reasoning = _strip_calculator_annotations(reasoning).strip()
        correct = extract_answer(example["answer"])
        wrong = str(int(float(correct)) + wrong_offset)
        rows.append({
            "prompt": example["question"] + "\n",
            "chosen": f"{reasoning}\nTherefore, the answer is {correct}.",
            "rejected": f"{reasoning}\nTherefore, the answer is {wrong}.",
        })
    return Dataset.from_list(rows)
```

(Structure-level plan; exact formatting/whitespace finalized at
implementation time, tested against real GSM8K rows.)

### 4.2 `tests/test_dpo_preference.py`

- A built example's `chosen` and `rejected` differ (not identical strings).
- `extract_answer(chosen)` matches the GSM8K ground truth; `extract_answer(rejected)` does not.
- `prompt` does not itself contain the word "Therefore" (sanity check that reasoning lives in the completion, not the prompt — matters per §3 point 1).

### 4.3 `scripts/smoke_test_dpo.py`

- CLI args: `--model` (default `Qwen/Qwen2.5-0.5B-Instruct`), `--n-examples`
  (default `20`, within the 10–50 range), `--max-steps` (default `3`,
  within 1–5), `--learning-rate` (default `1e-4` — a typical LoRA
  fine-tuning magnitude, explicitly overriding `DPOConfig`'s conservative
  `1e-6` default per §3 point 4), `--lora-r` (default `8`), `--lora-alpha`
  (default `16`), `--output-dir` (default `results/dpo_smoke_001`), `--seed`
  (default `42`, matching `CLAUDE.md`'s reproducibility principle).
- Build the dataset via `build_dpo_preference_dataset` (§4.1).
- Load the **plain** base model (no manual PEFT wrapping — §3 point 3)
  and tokenizer.
- `peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
  target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")` — minimal
  target modules (attention query/value projections only) is enough to
  prove the training loop works; not trying to maximize LoRA coverage for
  a smoke test.
- `DPOConfig(output_dir=..., per_device_train_batch_size=2, max_steps=...,
  learning_rate=..., beta=0.1, seed=..., report_to="none", logging_steps=1)`
  — `report_to="none"` keeps this fully local, no wandb dependency for the
  smoke test itself (wandb stays in `requirements.txt` for later,
  non-smoke-test experiments where tracking matters more).
- `DPOTrainer(model=model, ref_model=None, args=config,
  train_dataset=dataset, peft_config=peft_config)` — `ref_model=None` per
  §3 point 2.
- `trainer.train()`, then explicitly check and print each success
  criterion from §1 (e.g. assert the final logged loss is a finite float,
  not NaN).
- `trainer.save_model(output_dir)` — saves the LoRA adapter only (small),
  not a merged full checkpoint, consistent with `CLAUDE.md`'s "do not
  commit large checkpoints" rule (the adapter itself is also excluded from
  git via a `results/*/checkpoint*/` gitignore rule if its size turns out
  non-trivial — confirmed at implementation time).

### 4.4 Result storage (per `CLAUDE.md`'s "Result Storage" convention)

```text
results/dpo_smoke_001/
├── config.json   # model name, seed, n_examples, max_steps, beta, lora_r,
│                 # lora_alpha, learning_rate, batch_size -- every
│                 # CLAUDE.md-mandated reproducibility setting
├── metrics.json  # per-step training loss from trainer.state.log_history
└── notes.md      # what happened, any hardware limitation hit verbatim
                  # (per the Day 7 "document the exact limitation" clause,
                  # applied here too in case LoRA/MPS has surprises)
```

---

## 5. Relevant references

| Source | Why it matters here |
|---|---|
| Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen, 2021 — *LoRA: Low-Rank Adaptation of Large Language Models* | This is the project's first use of LoRA/PEFT — the paper behind `peft.LoraConfig`, freezing the base weights and training only low-rank adapter matrices on the attention projections. |
| TRL `DPOTrainer`/`DPOConfig` source (`trl==1.14.1`, inspected directly, §3) | Ground truth for the exact API this project calls — preferred over the paper or older blog posts, which may describe an older TRL API. |
| `notes/papers/dpo.md` | The loss/theory this script's training loop is executing. |

---

## 6. Open questions (going into implementation)

- **Does `per_device_train_batch_size=2` with `max_steps=3` over 20
  examples actually produce 3 *gradient* steps, or fewer due to how TRL
  computes `max_steps` vs. dataset length?** Will check the actual
  `trainer.state.global_step` after training rather than assuming.
- **Target modules `["q_proj", "v_proj"]`** — are these the right module
  names for Qwen2's architecture (vs. e.g. Llama's naming)? Qwen2 is
  believed to use the same `q_proj`/`k_proj`/`v_proj`/`o_proj` naming as
  Llama-family models, but this will be confirmed by checking that
  `get_peft_model` actually finds and wraps nonzero trainable parameters
  (if it silently finds zero matching modules, that's a bug to catch, not
  something to discover later from a loss that mysteriously never moves).
- **MPS + LoRA + DPO memory/stability** — untested combination on this
  specific hardware. If training fails or is unstable, document the exact
  error per `CLAUDE.md`'s hardware-honesty clause rather than silently
  switching to CPU or shrinking parameters without saying so.

---

## 7. Implementation notes (written after implementing)

All three open questions from §6 resolved cleanly, and no deviation from
the plan was needed:

- **`max_steps=3` with `per_device_train_batch_size=2` over 20 examples
  did produce exactly 3 gradient steps** (`trainer.state.global_step ==
  3`), confirmed by checking the actual trainer state rather than
  assuming it from the config.
- **`target_modules=["q_proj", "v_proj"]` correctly matched Qwen2.5's
  attention projections** — confirmed by asserting trainable-parameter
  count `> 0` in the script itself (540,672 trainable LoRA params), not
  just hoping silently. If it had matched zero modules, that assertion
  would have caught it immediately instead of surfacing as a mysteriously
  flat loss later.
- **MPS + LoRA + DPO worked without any issue** — 3 steps over 20 examples
  completed in ~29 seconds, no OOM, no fallback to CPU needed. Full run
  details and metrics are in `results/dpo_smoke_001/notes.md`.
- **One new, minor finding not anticipated in the plan:** `trainer.
  save_model()` also writes a full copy of the tokenizer
  (`tokenizer.json`, ~11MB) plus `training_args.bin`, a `README.md`, and
  `adapter_config.json`/`chat_template.jinja` alongside the ~2.1MB LoRA
  adapter itself. None of this is "a large model checkpoint" in the sense
  `CLAUDE.md` means to forbid, but it's also not something worth
  committing (the tokenizer is just a copy of a public one). Handled by
  extending `.gitignore` with `results/*/` patterns for exactly these
  generated-artifact filenames, while still tracking `config.json`,
  `metrics.json`, and `notes.md` per run — keeping the "Result Storage"
  convention in `CLAUDE.md` intact rather than committing checkpoint
  blobs by accident.

See `results/dpo_smoke_001/notes.md` for the actual run's metrics and
observations (loss values, reward accuracies, and an explicit statement of
what the run does and does not demonstrate).
