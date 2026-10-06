# GRPO Smoke Test — Run Notes

Run via `python scripts/smoke_test_grpo.py` (default args — see
`config.json`). Purpose: prove the GRPO training pipeline runs end to end,
per `CLAUDE.md` Day 7 — **not** to produce a well-trained model. Design
rationale: `notes/grpo_smoke_test.md`.

## Outcome

All Day 7 success criteria met, on the **second real attempt** — the first
attempt ran mechanically (no crash, finite loss, checkpoint saved) but with
a corrupted reward signal that would have looked like quiet success if not
checked closely. Full story below.

```text
[x] model loads        — Qwen/Qwen2.5-0.5B-Instruct, float32
[x] dataset loads       — 8 GSM8K prompts
[x] trainer initializes — TRL GRPOTrainer, LoRA via peft_config
                           (540,672 trainable params), group size 4
[x] G=4 responses generated per prompt, each scored
[x] group-relative advantage computed
[x] loss is produced and finite
[x] backward pass works — grad_norm nonzero every step (65.4, 18.8, 0.027)
[x] 3 optimization steps completed
[x] checkpoint saved — LoRA adapter only, ~2.1MB
```

Ran on Apple Silicon MPS, 3 steps, group size 4 (12 completions total),
**~71 seconds total**.

## The real finding: two bugs, only one of which mattered

**First run** (not kept — superseded by the fix below): every single one
of the 12 sampled completions hit `max_completion_length` with
`reward=0.0` and `reward_std=0.0` for every group (`frac_reward_zero_std:
1`). This is a textbook "ran fine, learned nothing, and nothing crashed to
tell you" failure — worth describing precisely rather than just saying "it
didn't work at first":

1. **Suspected, fixed, but NOT the actual cause:** `tokenizer.padding_side`
   defaults to `"right"`, but `GRPOTrainer`'s own docstring requires
   `"left"` for batched generation (it only auto-sets this when it builds
   its own tokenizer, which we don't let it do). Fixed by setting
   `tokenizer.padding_side = "left"` explicitly. **Rerunning with only
   this fix produced byte-for-byte identical losses to the unfixed run** —
   proof this wasn't the actual cause here, even though it's still the
   objectively correct thing to do per the trainer's own documented
   requirement, and could matter with differently-shaped prompts in the
   same batch.
2. **The actual cause, found by direct investigation, not guessing:**
   `GRPOTrainer` builds its internal `GenerationConfig` with
   `eos_token_id=self._tokenizer.eos_token_id` — a **single id**
   (`<|im_end|>`, `151645`). But `model.generation_config.eos_token_id` is
   a **list**, `[151645, 151643]`. In this project's raw-completion
   (non-chat) prompting style, Qwen2.5-0.5B-Instruct naturally stops on
   `151643` (`<|endoftext|>`), not `151645`. Confirmed by inspecting the
   actual generated token ids directly: one completion read `...\boxed{72}.`
   — the correct answer! — immediately followed by `<|endoftext|>`. The
   trainer's restricted `eos_token_id` never recognized that stop signal,
   so generation ran past the correct answer all the way to
   `max_completion_length`, and whatever token soup followed corrupted
   `extract_answer`'s fallback parse, producing a wrong (or no) answer for
   *every single sample*, hence reward=0 everywhere.
   **Fix:** pass `generation_kwargs={"eos_token_id": model.
   generation_config.eos_token_id}` through `GRPOConfig` (a mechanism
   the trainer explicitly supports for exactly this kind of override).

A separate, earlier, correct-but-insufficient-alone finding:
`max_completion_length=128` (the original plan's value) was also too
short — a direct generation test showed this model needs ~215 tokens of
verbose step-by-step reasoning to reach a stated answer in non-chat mode.
Fixed by raising it to `320`. **Neither fix alone was sufficient** — both
were required together (enough token budget to reach the answer, AND an
`eos_token_id` list that actually recognizes where the model naturally
stops).

## Observations from the successful run

- `completions/clipped_ratio` dropped from `1.0` (every completion
  truncated) to `0`, `0.25`, `0` across the 3 steps — most completions now
  terminate naturally; `completions/mean_terminated_length` matches
  `completions/mean_length` whenever `clipped_ratio` is 0, confirming real
  termination rather than truncation.
- Per-step mean reward: `0.75, 0.75, 1.0` — i.e. 3/4, 3/4, then 4/4 of the
  sampled completions were correct. `reward_std`: `0.5, 0.5, 0.0` — the
  first two steps have genuine within-group variance (exactly what
  group-relative advantage needs); the third step's `reward_std=0` is a
  legitimate degenerate case (all 4 completions correct), handled by
  TRL's `+1e-4` epsilon in the denominator rather than erroring
  (`notes/papers/grpo.md` §7's zero-variance open question, now directly
  observed, not just read from source).
- `grad_norm` (65.4 → 18.8 → 0.027) is large and clearly responding to
  real reward signal, in sharp contrast to the broken first run's
  `grad_norm` of ~0.001–0.003 (near-zero, consistent with near-zero
  advantages when every group has `reward_std=0`).

## What this run does NOT show

Per `CLAUDE.md`: "Week 1 is successful even if model accuracy does not
improve." 3 steps over 8 prompts with group size 4 is nowhere near enough
to claim GRPO improved this model's GSM8K accuracy — it demonstrates the
pipeline mechanics (generate → reward → group-relative advantage →
backprop → update) run correctly end to end, nothing more. The high
observed reward (0.75–1.0) likely reflects that these particular easy,
early-GSM8K-train-split problems are already within this model's
capability zero-shot, not that 3 GRPO steps taught it anything.
