# DPO (Direct Preference Optimization) — Paper Notes

> **Day 3 deliverable.** This is the full documentation deliverable for Day 3
> — per `CLAUDE.md`, Day 3 has no separate code step; this note *is* the
> implementation of the day's task. Read `notes/policy_gradient.md` and
> `notes/ppo.md` first — this note constantly refers back to both.
>
> Paper: Rafailov, Sharma, Mitchell, Ermon, Manning, Finn, 2023 — *Direct
> Preference Optimization: Your Language Model is Secretly a Reward Model*
> (NeurIPS 2023).

---

## Problem

RLHF (reinforcement learning from human feedback) wants language models to
behave according to human preferences — e.g. "prefer the correct, well-
explained answer over an incorrect or sloppy one." The standard way to do
this, as detailed in `ppo.md` §3, is:

1. Collect pairwise human preferences: for the same prompt, a human picks
   which of two responses is better.
2. Train a separate **reward model** on those preferences (a classifier
   that outputs a scalar score for any (prompt, response) pair).
3. Run **PPO** to fine-tune the policy against that reward model, with a KL
   penalty keeping the policy near a reference model.

This pipeline works (it's how InstructGPT and early ChatGPT were trained),
but it is a lot of machinery: a whole extra model to train (the reward
model), and then a famously finicky RL optimization stage on top of it,
needing four models in memory simultaneously (`ppo.md` §3.1).

**The question DPO asks:** can we skip the reward model and the RL loop
entirely, and train the policy directly on preference pairs, while still
provably optimizing the same underlying objective that RLHF-via-PPO
optimizes?

---

## Existing approach

RLHF-via-PPO, concretely:

```
Human preference data: (prompt, response_A, response_B, which one is better)
        ↓
Train reward model r_φ(prompt, response) with a Bradley-Terry loss:
   P(A preferred over B) = σ( r_φ(prompt, A) - r_φ(prompt, B) )
        ↓
Run PPO to maximize:  E[ r_φ(prompt, response) ] - β · KL(π_θ || π_ref)
   (KL penalty prevents "reward hacking" — overfitting to the reward
   model's blind spots instead of genuinely improving; see ppo.md §6,
   Christiano et al.)
```

The **Bradley-Terry model** is the standard statistical model for pairwise
comparisons: it says the probability that response A beats response B is a
sigmoid of the *difference* in their underlying scalar scores. This detail
matters — DPO reuses this exact assumption, just swaps out what the "score"
is a function of.

---

## Main contribution

DPO's key observation: for the *exact* KL-constrained reward-maximization
objective that PPO is trying to approximately solve, the **optimal policy
has a closed-form relationship to the reward function**:

```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

Where `x` = prompt, `y` = response, `π*` = the optimal policy for that
reward, `π_ref` = reference policy, `β` = the KL penalty strength, and
`Z(x)` is a normalizing constant that depends only on the prompt (not on
`y`).

This says: **any reward function implies a specific optimal policy, and
that relationship can be algebraically inverted** — given a policy, you can
read off "the reward function it is implicitly optimal for," up to an
additive, response-independent constant.

### The trick: substitute this into the Bradley-Terry loss

Because the Bradley-Terry preference probability only depends on a
*difference* of two rewards, `r(x, y_w) - r(x, y_l)`, the pesky `Z(x)`
term — which depends only on the prompt, not on which response — **cancels
out**:

```
r(x, y_w) - r(x, y_l)
  = β log(π*(y_w|x)/π_ref(y_w|x)) - β log(π*(y_l|x)/π_ref(y_l|x))
```

Plugging this directly into the Bradley-Terry preference probability gives
a loss that is a function of **the policy alone** — no reward model object
ever needs to exist:

```
L_DPO(θ) = - E_(x, y_w, y_l)~D [
    log σ( β · log(π_θ(y_w|x)/π_ref(y_w|x))
         − β · log(π_θ(y_l|x)/π_ref(y_l|x)) )
]
```

- `y_w` = the **w**inning / chosen response
- `y_l` = the **l**osing / rejected response
- `σ` = sigmoid
- `β` = same role as the KL coefficient in PPO — controls how far `π_θ` is
  allowed to drift from `π_ref` (small `β` → weaker constraint → bigger,
  riskier updates; large `β` → policy stays closer to `π_ref`)

**This is literally a logistic regression loss** over the margin between
two "implicit reward" terms. No sampling, no reward model, no RL — just a
forward pass on `(prompt, chosen)` and `(prompt, rejected)` through both the
policy and the (frozen) reference model, and a sigmoid cross-entropy loss.
This is the paper's titular claim: *your language model is secretly a
reward model* — `β log(π_θ/π_ref)` **is** the implicit reward, you just
never need to materialize it as a separate network.

### Connecting this to `policy_gradient.md`

It's worth being explicit about what got eliminated relative to
`policy_gradient.md` §3–4: there is no `∇θ log π_θ(action) · Advantage`
anywhere in the DPO loss. There is no sampling of trajectories, no reward
collected on a generated response, no baseline to subtract. DPO sidesteps
the entire policy-gradient machinery by working in closed form on a fixed
dataset instead. (Contrast this with GRPO in `notes/papers/grpo.md`, Day 4,
which *keeps* the policy-gradient machinery and only removes the value
network.)

---

## How training works

1. Start from an SFT (supervised fine-tuned) checkpoint — DPO is a
   fine-tuning step, not a from-scratch training method. In this project,
   `Qwen/Qwen2.5-0.5B-Instruct` already plays this role (it's instruction-
   tuned out of the box).
2. Set `π_ref` = a frozen copy of that same starting checkpoint. It never
   receives gradient updates.
3. Set `π_θ` = the trainable copy (in this project, wrapped with a LoRA
   adapter — only the adapter weights train, per `CLAUDE.md`'s "avoid full
   fine-tuning" rule).
4. For each `(prompt, chosen, rejected)` triple in a batch:
   - Forward pass `(prompt, chosen)` through `π_θ` → sum token log-probs
     → `log π_θ(y_w|x)`
   - Forward pass `(prompt, chosen)` through `π_ref` (no grad) →
     `log π_ref(y_w|x)`
   - Repeat both for `(prompt, rejected)` → `log π_θ(y_l|x)`,
     `log π_ref(y_l|x)`
   - Compute `L_DPO` from the formula above
5. Backward pass updates only `π_θ` (via the LoRA adapter). `π_ref` never
   moves.
6. No generation happens during training at all — this is an **offline**
   method: the dataset of `(chosen, rejected)` pairs is fixed up front,
   unlike GRPO/PPO which generate fresh rollouts from the *current* policy
   every step (**online** methods).

---

## Inputs required

```text
prompt            — the input text
chosen response   — the preferred completion
rejected response — the dispreferred completion
reference model   — frozen, typically = the starting checkpoint
β (beta)          — KL-strength-like hyperparameter (TRL default: 0.1)
```

No reward model. No value/critic model. No online sampling.

---

## Why it is cheaper than PPO

Directly extending the table from `ppo.md` §3.1:

| Cost in PPO-for-RLHF | DPO's answer |
|---|---|
| Train a separate reward model first | **Not needed** — the policy itself is reparameterized as an implicit reward (Main contribution, above) |
| Value/critic network, trained online | **Not needed** — no advantage estimation at all; it's a classification loss |
| Online rollout generation every training step | **Not needed** — trains on a static, pre-built dataset of (chosen, rejected) pairs |
| 4 models resident in memory (policy, ref, reward, value) | **2 models** (policy + frozen reference) |
| RL instability (reward hacking, hyperparameter sensitivity, variance) | Ordinary supervised-style optimization — same stability profile as fine-tuning with cross-entropy |

This is exactly why Day 6 (tiny DPO smoke test) is tractable on a MacBook
with no paid GPU: it's "a slightly unusual supervised loss," not an RL loop.

---

## Limitations

- **Needs pre-built preference pairs.** Someone (or some heuristic) has to
  decide what counts as "chosen" vs "rejected" ahead of time. For GSM8K in
  this project, that will be constructed heuristically (correct final
  answer = chosen, incorrect = rejected) rather than from genuine human
  preference judgments — see "Relation to this project" below.
- **Offline / no exploration.** Because training data isn't sampled from
  the current policy, DPO can't correct the policy's *current* specific
  mistakes the way an online method (GRPO, PPO) naturally does — it can
  only teach it to prefer the writen pairs it's given.
- **Distribution shift over many epochs.** As `π_θ` moves away from
  `π_ref` during training, the original chosen/rejected pairs (produced
  relative to the old policy) may stop reflecting what the *current* policy
  would actually generate. For the short smoke test in this project (1–5
  steps) this is a non-issue; it matters more for longer training runs.
- **No explicit, reusable reward model.** The implicit reward
  `β log(π_θ/π_ref)` exists mathematically but isn't typically used
  downstream (e.g. for best-of-N re-ranking) without extra plumbing.
- **Quality bottleneck is entirely data quality.** Since there's no RL
  exploration to compensate, how good and how representative the
  `(chosen, rejected)` pairs are directly caps how good the result can be.

---

## Relation to this project

- **Data construction (Day 6):** this project will build a small, synthetic
  GSM8K preference dataset: for a handful of GSM8K questions, `chosen` =
  a response whose extracted final answer matches the ground truth (via
  `src/evaluation/answer_extraction.py`, Day 5), `rejected` = a response
  whose extracted answer is wrong. This reuses Day 5's answer-extraction
  and correctness-reward logic as the *labeling* mechanism for preference
  pairs, even though DPO training itself never calls the reward function
  directly (reward is implicit, per Main contribution).
- **Training setup (Day 6, `scripts/smoke_test_dpo.py`):** TRL's
  `DPOTrainer`, `Qwen/Qwen2.5-0.5B-Instruct` as the base model, a LoRA
  adapter for the trainable policy (per `CLAUDE.md` — avoid full
  fine-tuning), 10–50 preference examples, 1–5 optimization steps. Goal is
  "the pipeline runs and a loss/backward pass occurs," not model quality
  (per spec, explicitly).
- **Relation to GRPO (Day 4):** DPO and GRPO are this project's two actual
  subjects. This note, together with `notes/papers/grpo.md`, should make
  clear they are *not* competing implementations of the same idea — DPO is
  offline/pairwise, GRPO is online/group-relative. Both are legitimate
  "cheaper than PPO" answers, just via different cuts (see the table in
  `ppo.md` §3.1).

---

## Open questions

- **How does TRL's `DPOTrainer` handle the reference model when using
  LoRA?** Likely it can reuse the same base weights and just disable the
  adapter to get reference log-probs, instead of loading a second full
  copy of the model — need to confirm this when Day 6 is implemented, since
  it directly affects memory footprint on the MacBook.
- **What `β` to use for the smoke test?** Not tuning anything this project
  (explicitly out of scope per `CLAUDE.md` — "do not optimize
  hyperparameters"), so TRL's default (`0.1`) is the starting point unless
  it visibly prevents the loss from moving at all in the smoke test.
- **How to construct the 10–50 preference pairs cheaply?** Options to
  decide at Day 6 implementation time: (a) hand-write a handful of GSM8K
  `(correct, incorrect)` response pairs, or (b) sample the base model's own
  generations for a few GSM8K prompts and sort them into chosen/rejected
  via the Day 5 answer-extraction + correctness reward. Option (b) is more
  faithful to how real DPO datasets are usually built, but option (a) is
  simpler and sufficient for a smoke test whose only goal is proving the
  training loop runs.

---

## Relevant papers

| Paper | Why it matters here |
|---|---|
| Rafailov, Sharma, Mitchell, Ermon, Manning, Finn, 2023 — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* | The paper this entire note documents. |
| Ouyang et al., 2022 — *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)* | The RLHF-via-PPO baseline DPO is positioned against (also cited in `ppo.md`). |
| Bradley & Terry, 1952 — *Rank Analysis of Incomplete Block Designs* | The pairwise-comparison statistical model (`P(A≻B) = σ(score_A − score_B)`) that both the reward-model stage of RLHF and DPO's loss are built on. |
| Christiano et al., 2017 — *Deep Reinforcement Learning from Human Preferences* | Earlier reward-model-from-preferences work that DPO's reparameterization trick eliminates the need for (also cited in `ppo.md`). |
