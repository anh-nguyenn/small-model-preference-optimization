# Day 7 — Tiny GRPO Smoke Test: Implementation Plan

> Written before `scripts/smoke_test_grpo.py`. Read `notes/papers/grpo.md`
> first for the theory. Like Day 6, the installed `trl==1.14.1` was
> inspected directly (constructor signatures, source, docstrings) before
> writing this plan — several things below turned out to differ from what
> `notes/papers/grpo.md` assumed, and those differences are the most
> important content of this note.

---

## 1. What "success" means here

Per `CLAUDE.md` Day 7:

```text
Qwen2.5-0.5B, very small dataset, very small group size, very few steps,
binary correctness reward. Goal: prove the full GRPO loop works
(generate -> reward -> group-relative advantage -> backprop -> update).
If Mac memory prevents backward training, document the exact limitation
rather than silently changing the experiment.
```

---

## 2. Background: what's reused vs. what's new

**Reused, unchanged:**
- `src/rewards/correctness.py`'s `compute_reward` — the exact same binary
  reward used to label DPO's preference pairs (Day 6) scores each sampled
  GRPO response here too, per `CLAUDE.md`'s "never silently change
  experimental methodology."
- `src/evaluation/answer_extraction.py` — used inside the reward, as always.
- The plain-text (non-conversational) prompt/completion format from Day 6's
  DPO script, for the same reason: fewer moving parts in a smoke test.

**New for Day 7:**
- A `(prompt, answer)` dataset — much simpler than DPO's preference pairs,
  since GRPO doesn't need pre-built chosen/rejected text at all. It needs
  only the question and the raw ground truth, because the *model itself*
  generates the responses to be scored, live, during training (this is
  the online/on-policy nature of GRPO documented in `notes/papers/grpo.md`).
- A reward function wrapper matching TRL's specific expected signature
  (§3 below) — a thin adapter over `compute_reward`, not new reward logic.

---

## 3. Background: TRL's actual `GRPOTrainer` API (verified, not guessed)

### 3.1 Reward function signature

Inspecting `trl.rewards.accuracy_reward` (TRL's own built-in example) and
the trainer's call site confirms a custom reward function is called as:

```python
reward_func(prompts=prompts, completions=completions, completion_ids=...,
            **reward_kwargs)
```

where `reward_kwargs` contains **every other column present in the
dataset** (e.g. if the dataset has an `"answer"` column, the function
receives `answer=[...]` automatically), plus `trainer_state`, `log_extra`,
`log_metric`. So the reward function this project needs is:

```python
def gsm8k_grpo_reward(prompts, completions, answer, **kwargs):
    return [compute_reward(completion, gt) for completion, gt in zip(completions, answer)]
```

Since prompts are plain strings (not chat-message lists) in this project's
non-conversational setup, `completions` are plain generated strings too —
directly compatible with `compute_reward`'s existing signature, no
reformatting needed.

### 3.2 There is no `ref_model` parameter at all

Unlike `DPOTrainer`, `GRPOTrainer.__init__` has **no `ref_model` argument**.
Reading the source directly: the reference model is created automatically,
*only if* `beta != 0`:

- **If `beta == 0.0`** (TRL's own default — see §3.3): `self.ref_model =
  None` and the KL-penalty/reference-logprob machinery is **skipped
  entirely** — cheaper by default, at the cost of never exercising the
  mechanism `notes/papers/grpo.md` describes.
- **If `beta != 0.0` and the model is PEFT-wrapped**: the trainer uses
  `use_adapter(model, adapter_name=...)` to temporarily disable the LoRA
  adapter and compute reference log-probs from the frozen base weights —
  **the exact same mechanism DPO uses** (`notes/dpo_smoke_test.md` §3,
  point 2). No second full model copy is loaded either way.
- `sync_ref_model=True` (periodically refreshing the reference) is
  explicitly **not supported with PEFT models** (the trainer raises if you
  try) and is a no-op when `beta == 0.0`. Default is `False` — the
  reference stays fixed at the pre-training state for the whole run, same
  as DPO.

This resolves two of `notes/papers/grpo.md`'s open questions directly:
*"is π_ref re-synced periodically"* (no, by default, and not even
supported with PEFT) and implicitly confirms the adapter-disabling
mechanism from DPO carries over unchanged to GRPO.

### 3.3 Two defaults that silently diverge from the paper's formulation

Introspecting `GRPOConfig`'s actual field defaults surfaced two real
findings worth flagging explicitly — exactly the kind of thing
`CLAUDE.md` principle #3 ("never silently change experimental
methodology") is about, except here it's the *library* that changed
defaults away from the paper, not us:

| `GRPOConfig` field | TRL's default | What `notes/papers/grpo.md` assumed |
|---|---|---|
| `beta` (KL coefficient) | **`0.0`** (no KL penalty, no reference model created at all) | A nonzero KL penalty, per the DeepSeekMath objective (`notes/papers/grpo.md` §Main contribution) |
| `loss_type` | **`"dapo"`** | The paper's per-sequence-length-normalized loss, i.e. TRL's own `"grpo"` option |

TRL's own field docstring is explicit about why: `"grpo"` *"is not
recommended due to length bias — this approach tends to prefer shorter
completions with positive advantages and longer ones with negative
advantages"* — a known issue fixed by the DAPO paper's normalization
(Yu et al., 2025), which is now TRL's default loss formulation for *every*
GRPO run, not just DAPO-branded ones.

**Decision for this project:** explicitly set `beta=0.04` (a small,
commonly-cited nonzero value) and `loss_type="grpo"` — overriding both
library defaults — specifically so this smoke test actually exercises the
KL-penalty/reference-model code path and the per-sequence-normalized loss
documented in `notes/papers/grpo.md`, rather than silently running a
different (even if arguably better-engineered) variant the library
defaults to. This is a deliberate, documented divergence from TRL's
*library* defaults in favor of fidelity to what this project's own notes
derived and committed to in writing — not a correctness claim that
`"grpo"`/`beta=0.04` is better than `"dapo"`/`beta=0.0` for real training
(TRL's own docs suggest the opposite). Flagged again in §6 as something a
longer, non-smoke-test run should reconsider.

### 3.4 Zero-variance groups: resolved

`notes/papers/grpo.md` §7 asked how TRL handles a group where every
response gets the same reward (`group_std = 0`). Confirmed directly in
source: `advantages = (rewards - mean) / (std + 1e-4)` — a small epsilon
is added to the denominator unconditionally. No group is skipped; a
zero-std group just produces a near-zero advantage for everyone in it
(division by ~`1e-4` of a ~0 numerator), which is logged (`is_std_zero`)
but doesn't error or exclude the group.

### 3.5 Batch size / group size divisibility constraint

`generation_batch_size` (which must be divisible by `num_generations`,
i.e. the group size `G`) defaults to `per_device_train_batch_size ×
num_processes × steps_per_generation`. With a single device and
`steps_per_generation=1` (both defaults), this means **`per_device_train_
batch_size` must be a multiple of `num_generations`**. Simplest choice for
a smoke test: set `per_device_train_batch_size = num_generations` exactly
— each training step processes exactly one prompt's full group of `G`
generated responses.

### 3.6 Sampling must stay stochastic

`temperature` defaults to `1.0`. This is deliberately **not** overridden
to `0.0`/greedy the way `scripts/test_model.py` does — if every one of the
`G` responses in a group were generated greedily (deterministically), they
would all be identical, `group_std` would always be `0`, and the
group-relative advantage would carry no signal at all. Diversity across
the group is a requirement for GRPO, not a quality-of-generation nicety —
worth stating explicitly since it's the opposite choice from every
previous script in this project.

---

## 4. Implementation plan

### 4.1 `src/datasets/grpo_prompts.py` (new, reusable)

```python
from datasets import Dataset, load_dataset

def build_grpo_prompt_dataset(n_examples: int, split: str = "train") -> Dataset:
    """Build a (prompt, answer) dataset from GSM8K for GRPO. `answer` is
    the raw GSM8K answer field (with the "#### N" marker) -- the reward
    function extracts the ground truth from it via the same
    extract_answer() used everywhere else in this project.
    """
    raw = load_dataset("openai/gsm8k", "main", split=split).select(range(n_examples))
    return Dataset.from_dict({
        "prompt": [example["question"] + "\n" for example in raw],
        "answer": [example["answer"] for example in raw],
    })
```

Much simpler than Day 6's `dpo_preference.py` — no chosen/rejected
construction needed, since GRPO scores the policy's *own* generations
rather than needing pre-built comparison pairs.

### 4.2 `src/rewards/correctness.py` (extended, not duplicated)

Add one thin adapter function alongside the existing `compute_reward`,
matching TRL's exact expected signature (§3.1):

```python
def gsm8k_grpo_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """TRL GRPOTrainer-compatible reward function. Thin wrapper around
    compute_reward -- no new reward logic, per CLAUDE.md's reuse principle.
    """
    return [compute_reward(completion, gt) for completion, gt in zip(completions, answer)]
```

### 4.3 `tests/test_grpo_prompts.py` and a reward-wrapper test

- `build_grpo_prompt_dataset` produces the requested row count, with
  `"prompt"`/`"answer"` columns, and `answer` contains a real `"#### N"`
  marker (sanity check it's the raw field, not pre-extracted).
- `gsm8k_grpo_reward` given a mix of correct/incorrect completions for the
  same `answer` returns the expected `[1.0, 0.0, ...]` pattern — this is
  really just re-confirming `compute_reward`'s existing tests still apply
  through the new wrapper's exact call shape (`prompts=`, `completions=`,
  `answer=` as keyword args, matching how the trainer actually calls it).

### 4.4 `scripts/smoke_test_grpo.py`

- CLI args: `--model` (default `Qwen/Qwen2.5-0.5B-Instruct`), `--n-examples`
  (default `8` — "very small dataset" per `CLAUDE.md`), `--num-generations`
  (default `4` — "very small group size"), `--max-steps` (default `3`),
  `--learning-rate` (default `1e-5` — one order of magnitude more
  conservative than Day 6's DPO default, since on-policy RL updates are
  typically more sensitive than DPO's supervised-style loss), `--beta`
  (default `0.04`, per §3.3), `--lora-r`/`--lora-alpha` (same defaults as
  Day 6: `8`/`16`), `--output-dir` (default `results/grpo_smoke_001`),
  `--seed` (default `42`).
- Build the dataset via `build_grpo_prompt_dataset` (§4.1).
- Load the plain base model (same pattern as Day 6 — the trainer wraps it
  in PEFT itself via `peft_config`, not pre-wrapped).
- `peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
  target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")` — identical
  target modules to Day 6, already confirmed to match Qwen2's architecture
  there.
- `GRPOConfig(output_dir=..., per_device_train_batch_size=num_generations,
  num_generations=num_generations, max_steps=..., learning_rate=...,
  beta=..., loss_type="grpo", max_completion_length=128, seed=...,
  report_to="none", logging_steps=1)` — `per_device_train_batch_size =
  num_generations` to satisfy the divisibility constraint from §3.5 with
  the simplest possible setup (one prompt's group per step);
  `max_completion_length=128` keeps generation fast, matching
  `scripts/test_model.py`'s smoke-test-scale default.
- `GRPOTrainer(model=model, reward_funcs=gsm8k_grpo_reward, args=config,
  train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config)`.
- `trainer.train()`, then check and print the same style of success
  criteria as Day 6 (finite loss, nonzero global steps, trainable params
  received gradients), plus GRPO-specific ones: the per-step mean reward
  and reward std actually logged (proving the group-relative advantage
  computation ran, not just a generic forward/backward pass).
- `trainer.save_model(output_dir)`.

### 4.5 Result storage

```text
results/grpo_smoke_001/
├── config.json   # model, seed, n_examples, num_generations, max_steps,
│                 # beta, loss_type, lora_r, lora_alpha, learning_rate
├── metrics.json  # per-step loss, reward mean/std from log_history
└── notes.md      # what happened; hardware limitations if any, verbatim
                  # (CLAUDE.md's explicit Day 7 instruction)
```

---

## 5. Relevant references

| Source | Why it matters here |
|---|---|
| Shao et al., 2024 — DeepSeekMath (GRPO) | Already the primary reference in `notes/papers/grpo.md`; the `loss_type="grpo"` choice (§3.3) is chosen specifically to match this paper's formulation. |
| Yu et al., 2025 — *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* | The paper behind TRL's actual default `loss_type="dapo"` — explains *why* the library moved away from the original per-sequence-normalized GRPO loss (length bias). |
| Liu et al., 2025 — *Understanding R1-Zero-Like Training: A Critical Perspective* ("Dr. GRPO") | Referenced in TRL's own `scale_rewards` docstring as the source of the "don't scale by std, it introduces a question-difficulty bias" critique — not used in this project (`scale_rewards="group"`, matching our documented formula), but worth knowing this is an active, unresolved debate in the literature, not a settled question. |
| TRL `GRPOTrainer`/`GRPOConfig` source (`trl==1.14.1`, inspected directly, §3) | Ground truth for the exact API this project calls. |

---

## 6. Open questions going into implementation

- **Does `beta=0.04` with a PEFT model and `num_generations=4` actually
  fit comfortably in 16GB unified memory, generating 4 completions per
  prompt plus a reference forward pass?** This is the actual hardware
  question `CLAUDE.md` asks to be answered honestly either way.
- **Is `loss_type="grpo"`'s known length bias visible at all in a 3-step
  smoke test?** Almost certainly not observable at this scale — noted here
  so it isn't mistaken for a non-issue if this project is ever extended
  beyond Week 1's smoke-test scope.
- **Should a future, longer-than-smoke-test GRPO run switch to TRL's
  `"dapo"` default instead of `"grpo"`?** Flagged for later per §3.3 —
  not a Week 1 decision.

---

## 7. Implementation notes (written after implementing)

The plan in §1–6 turned out to need one real correction, found by direct
investigation rather than guessing. Full narrative in
`results/grpo_smoke_001/notes.md`; summarized here because it's a finding
about the *pipeline*, not just this one run.

**Symptom:** the first real run completed without crashing — finite loss,
checkpoint saved, every mechanical success criterion from §1 nominally
met — but `reward=0.0` and `reward_std=0.0` for every single group
(`frac_reward_zero_std: 1`). This is the dangerous kind of failure: it
would pass a check that only looks for "did it crash," while the actual
RL signal was completely dead.

**Root cause, confirmed by inspecting actual generated token ids (not
assumed):** `GRPOTrainer` builds its generation config with
`eos_token_id=self._tokenizer.eos_token_id` — a single id (`<|im_end|>`,
`151645`). But `model.generation_config.eos_token_id` is a **list**,
`[151645, 151643]`. This project's prompts are plain-text/non-chat (same
choice as Day 6's DPO script, for simplicity) — and in that prompting
style, Qwen2.5-0.5B-Instruct naturally stops on `151643` (`<|endoftext|>`),
not `151645`. A direct generation test proved this: one completion read
`...\boxed{72}.` (the correct answer) immediately followed by
`<|endoftext|>`. The trainer's restricted `eos_token_id` never recognized
that token as a stop signal, so generation ran straight past the correct
answer to `max_completion_length` every time, and the token soup that
followed broke `extract_answer`'s parse for every sample.

**Fix:** `GRPOConfig(..., generation_kwargs={"eos_token_id": model.
generation_config.eos_token_id})` — the trainer explicitly supports this
override (`args.generation_kwargs` is merged into its internal generation
kwargs). After this fix, rewards became `0.75, 0.75, 1.0` with genuine
nonzero `reward_std` on the first two steps.

**A red herring worth naming as one:** `tokenizer.padding_side` defaulting
to `"right"` instead of the `"left"` the trainer's own docstring requires
for batched generation was *also* fixed (it's still objectively correct
per that docstring), but re-running with **only** that fix produced
byte-for-byte identical losses to the unfixed run — proof it wasn't what
caused this particular symptom. Both fixes are kept in the script: one
because it demonstrably fixed the real bug, the other because it's
correct regardless and documented as "fixed but not causally
responsible," rather than silently dropped once it looked unnecessary.

**`max_completion_length=320` (§6, raised from an initial 128) was
necessary but not sufficient on its own** — the first broken run already
used 320 and still produced reward=0 everywhere, because the real
blocker was the `eos_token_id` mismatch, not token budget. Both fixes
were required together.

This is exactly the kind of failure `CLAUDE.md`'s "fix errors before
declaring the task complete" and "never silently change experimental
methodology" principles are meant to catch: a run that completes without
error is not the same as a run that actually exercises the mechanism it's
supposed to demonstrate, and the fix that happened to be tried first
(padding) isn't necessarily the fix that mattered.
