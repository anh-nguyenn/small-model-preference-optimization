# DPO Smoke Test — Run Notes

Run via `python scripts/smoke_test_dpo.py` (default args — see `config.json`
for exact hyperparameters). Purpose: prove the DPO training pipeline runs
end to end, per `CLAUDE.md` Day 6 — **not** to produce a well-trained
model. Design rationale: `notes/dpo_smoke_test.md`.

## Outcome

All Day 6 success criteria met:

```text
[x] model loads        — Qwen/Qwen2.5-0.5B-Instruct, float32
[x] dataset loads       — 20 synthetic GSM8K preference pairs
[x] trainer initializes — TRL DPOTrainer, LoRA via peft_config (540,672
                           trainable params), ref_model=None (adapter-
                           disabling, no second model copy loaded)
[x] forward pass works
[x] loss is produced    — see per_step_loss in metrics.json
[x] backward pass works — grad_norm was nonzero every step (~4.4-5.5)
[x] 3 optimization steps completed
[x] checkpoint saved    — LoRA adapter only, ~2.1MB (adapter_model.safetensors)
```

Ran on Apple Silicon MPS, 3 steps over 20 examples (batch size 2),
**~29 seconds total** — no hardware limitation encountered; this
combination (LoRA + DPOTrainer + MPS) just worked without needing a
fallback to CPU or any parameter reduction.

## Observations (not conclusions — 3 steps, 20 examples, no claim of learning)

- Loss stayed close to `ln(2) ≈ 0.693` across all 3 steps (`0.673`,
  `0.687`, `0.674`) — expected at this scale: DPO's loss starts near
  `ln(2)` whenever the policy and reference agree on both responses
  (`logits/chosen` and `logits/rejected` were -2.2ish for both), which is
  exactly true at step 0 since the "reference" *is* the pre-training
  policy state.
- `rewards/accuracies` (fraction of the batch where the implicit reward
  for `chosen` exceeds `rejected`) hit `1.0` at step 1, dropped to `0.5` at
  step 3 — noisy, as expected from a 2-example batch size over only 3
  steps; not a meaningful trend at this scale.
- `mean_token_accuracy` (~0.82–0.86) reflects that Qwen2.5-0.5B-Instruct
  already predicts GSM8K-style reasoning tokens reasonably well before any
  DPO training — unsurprising for an instruct-tuned model on a well-known
  benchmark's style.

## What this run does NOT show

Per `CLAUDE.md`: "Week 1 is successful even if model accuracy does not
improve." This run makes no claim about whether DPO improved GSM8K
accuracy — it only demonstrates the pipeline mechanics listed above. Any
accuracy claim would need a held-out eval before/after, which is out of
scope for a 3-step, 20-example smoke test.
