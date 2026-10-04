# GRPO (Group Relative Policy Optimization) — Paper Notes

> **Day 4 deliverable.** Like Day 3, this is the entire spec deliverable for
> the day — no separate code step. Read `notes/policy_gradient.md` (especially
> §4.3, which previews this exact method) and `notes/ppo.md` first — this note
> assumes both.
>
> Paper: Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo, 2024 —
> *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
> Language Models* (introduces GRPO in §4). Later scaled up in Guo et al.,
> 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
> Reinforcement Learning*.

---

## Problem

PPO-for-LLMs (`ppo.md` §3) estimates the advantage/baseline using a learned
**value network** (critic) roughly the size of the policy itself. For
reasoning tasks like GSM8K, this is a particularly bad fit:

- Rewards are typically **sparse and outcome-level** — a single scalar
  (correct/incorrect) at the very end of the response, not per-token. A
  value network trained to predict "expected future reward from a partial,
  half-written solution" has a genuinely hard regression target under this
  reward structure, and is itself expensive and often inaccurate.
- Training a value network concurrently with the policy is itself a source
  of instability (`ppo.md` §3.1) — it's a moving-target regression problem,
  and if its estimates are bad, so are the resulting advantages.
- On top of the generation cost shared by any online RL method, you now
  also pay for a second trained model's forward+backward pass, every step.

**The question GRPO asks:** for tasks where you can cheaply sample *multiple*
responses to the same prompt, can the group of samples itself supply a good
enough baseline — with zero extra trained parameters?

---

## Existing approach

PPO-for-LLMs, recapped and specialized to the reasoning-task setting:

```
prompt
  ↓
policy generates ONE response (token by token)
  ↓
reward model / verifier scores it  →  reward (often just at the final token)
  ↓
value network estimates a per-step baseline  →  advantage (often via GAE)
  ↓
PPO clipped objective updates BOTH policy and value network
```

As detailed in `ppo.md` §3, this needs up to 4 models in memory for full
RLHF; even in a reasoning-RL setting with a rule-based reward instead of a
learned reward model, the value network is still there, still expensive,
and still a second thing to get right.

---

## Main contribution

GRPO replaces the value network with **group statistics**. Concretely:

```
Prompt
  ↓
Generate G responses from the CURRENT policy (not 1 — a whole group)
  ↓
Reward each response independently  →  reward_1 ... reward_G
  ↓
Compute group-relative advantage for each response:

      advantage_i = (reward_i - group_mean) / group_std

  where group_mean = mean(reward_1 ... reward_G)
        group_std  = std(reward_1 ... reward_G)
  ↓
Broadcast advantage_i to every token in response i
  ↓
Update the policy with a PPO-style clipped objective, using these
advantages instead of a value network's estimate
```

This is a **Monte Carlo baseline**: instead of training a model to *predict*
"what reward should I expect here," GRPO just *measures* it directly, from
G actual samples of the current policy answering the same prompt. The group
mean plays exactly the role of `Baseline` in `policy_gradient.md` §4.2–4.3;
dividing by `group_std` additionally normalizes the scale of the advantage
across prompts of different difficulty (a prompt where rewards vary a lot
gets rescaled differently than one where they barely vary).

### The GRPO objective, precisely

Following the paper, for prompt `q` with sampled responses `{o_1 ... o_G}`:

```
J_GRPO(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t
                 min( ratio_{i,t}·A_i,  clip(ratio_{i,t}, 1-ε, 1+ε)·A_i )
             ]  −  β · D_KL(π_θ ‖ π_ref)
```

where `ratio_{i,t} = π_θ(token_t | q, o_i,<t) / π_θ_old(token_t | q, o_i,<t)`
— exactly the PPO probability ratio from `ppo.md` §2, and `A_i` is the
group-relative advantage above, the **same value for every token** in
response `i` (since the reward is outcome-level, there's no finer-grained
per-token signal to assign).

Two details worth flagging explicitly:

1. **The clipped surrogate is untouched from PPO** (`ppo.md` §2) — GRPO's
   innovation is entirely in *where the advantage comes from*, not in how
   the policy update itself is clipped/stabilized.
2. **The KL penalty is a direct term in the loss**, not folded into the
   reward the way classic PPO-for-RLHF does it (`ppo.md` §3). The DeepSeekMath
   paper uses an unbiased, always-non-negative KL estimator (Schulman's "k3"
   estimator, see §Relevant papers) rather than the naive
   `log π_θ - log π_ref` difference.

### Connecting this to `policy_gradient.md` and `ppo.md`

GRPO sits in a very specific spot relative to the other two methods already
documented:

| | Keeps the RL/sampling loop? | Needs a value network? | Needs a reward model? |
|---|---|---|---|
| PPO (`ppo.md`) | Yes | Yes | Yes (RLHF setting) |
| **GRPO** | **Yes** | **No** | No (rule-based reward in this project) |
| DPO (`notes/papers/dpo.md`) | No | No | No |

GRPO is the "remove just the value network, keep everything else about PPO"
point in this design space — the mirror image of DPO, which removes the
*entire* RL loop instead. Both remove the reward model in this project's
usage (we use the deterministic binary reward from Day 5, not a learned
one), but that's a project-specific choice, not inherent to either method
(the original DeepSeekMath/R1 papers do use GRPO with both rule-based *and*
learned reward signals in different experiments).

---

## How training works

1. Start from a policy `π_θ` (here, `Qwen/Qwen2.5-0.5B-Instruct`, optionally
   LoRA-wrapped per `CLAUDE.md`'s "avoid full fine-tuning" rule) and a frozen
   reference `π_ref` (a snapshot of the starting checkpoint, used only for
   the KL penalty — same role as `π_ref` in DPO, but here it never
   contributes to the gradient except through that penalty term).
2. For each prompt `q` in a batch:
   - Sample `G` responses `{o_1, ..., o_G}` from the current policy (with
     some temperature, so the G samples actually differ from each other).
   - Score each response with the reward function — in this project,
     `src/rewards/correctness.py` (Day 5): did `extract_answer(o_i)` match
     the GSM8K ground truth? `reward_i ∈ {0.0, 1.0}`.
3. Compute `group_mean` and `group_std` over `{reward_1 ... reward_G}` for
   that prompt, then `advantage_i = (reward_i - group_mean) / group_std`
   for each response (see §5 worked example).
4. Compute the per-token probability ratio `ratio_{i,t}` between the
   current policy and the policy that generated the rollouts, and apply the
   PPO clipped objective from §Main contribution, using `A_i` broadcast
   across all tokens of response `i`.
5. Subtract the KL penalty against `π_ref`.
6. Backward pass updates `π_θ` only. No value network exists to update.
7. Repeat for the next batch of prompts, re-sampling fresh rollouts from
   the (now updated) policy each time — this is what makes GRPO an
   **online** method, unlike DPO.

---

## Inputs required

```text
prompt
group size G                — how many responses to sample per prompt
reward function              — here: deterministic binary correctness (Day 5)
reference model π_ref        — frozen, for the KL penalty only
clip epsilon (ε)             — same role as in PPO
KL coefficient (β)
sampling temperature          — needed so the G responses aren't identical
```

No value network. No reward model (in this project's usage — rule-based
reward instead).

---

## Why it is cheaper than PPO

| Cost in PPO-for-RLHF (`ppo.md` §3.1) | GRPO's answer |
|---|---|
| Value network, trained online, as a second model | **Not needed** — group mean/std computed directly from sampled rewards, zero extra parameters |
| GAE / per-step baseline estimation | Replaced by a single group-level normalization, computed once per prompt |
| Reward model (RLHF setting) | Not used in this project either way (rule-based reward) — orthogonal to GRPO itself |
| 4 models resident in memory | **2 models** (policy + frozen reference) — same count as DPO |

**Important trade-off to note explicitly:** GRPO is *not* free compared to
DPO. It still requires **generating `G` responses per prompt, every
training step** — generation is the slowest part of working with an LLM.
DPO trains on a fixed, pre-built dataset and never generates anything during
training. So while GRPO is cheaper than PPO (removes the value network),
it is *more* expensive than DPO (keeps the sampling loop). This is the
central trade-off to keep straight across all three methods documented in
`notes/`.

---

## Limitations

- **Generation cost scales with `G`.** Larger groups give better-estimated
  `group_mean`/`group_std` (less noisy baselines) but cost proportionally
  more inference compute per prompt, per step.
- **Degenerate groups.** If every response in a group gets the *same*
  reward (all correct, or all wrong — plausible with a 0.5B model on easy
  or very hard GSM8K problems), `group_std = 0` and the advantage formula
  divides by zero. In practice this needs an epsilon in the denominator or
  the group needs to be skipped — flagged as an open question below since
  it needs confirming against TRL's actual implementation.
- **No within-response credit assignment.** Because the reward is
  outcome-level (one scalar for the whole response), every token in a
  response gets the *same* advantage, correct reasoning steps and sloppy
  ones alike. (The DeepSeekMath paper also explores a process-supervised
  variant with per-step rewards; out of scope for this project.)
- **Still an on-policy RL method.** It inherits RL's general fragility
  relative to DPO's supervised-style training: sampling cost, more moving
  pieces, more ways for a smoke test to fail for infrastructure reasons
  unrelated to the method itself (e.g. memory pressure from holding `G`
  generations' activations for backprop).

---

## Relation to this project

- **Reward reuse (Day 5 → Day 7):** the exact same
  `src/rewards/correctness.py` and `src/evaluation/answer_extraction.py`
  built on Day 5 for DPO's preference-pair labeling will be reused here as
  the per-response reward function — this project deliberately uses one
  reward definition everywhere, per `CLAUDE.md`'s "never silently change
  experimental methodology" rule.
- **Day 7 (`scripts/smoke_test_grpo.py`):** TRL's `GRPOTrainer`,
  `Qwen/Qwen2.5-0.5B-Instruct`, a **small** group size, a **very small**
  GSM8K subset, and very few steps — the goal is only to prove the full
  loop (generate → reward → group-advantage → backprop → optimizer step)
  executes, not to improve accuracy.
- **Hardware honesty clause:** per `CLAUDE.md`, *"If Mac memory prevents
  backward training, document the exact limitation rather than silently
  changing the experiment."* Concretely: if group size or batch size has to
  be reduced to fit in 16GB unified memory, that reduction and the exact
  error it was avoiding must be written into `results/<run>/notes.md` when
  Day 7 happens — not silently baked into the script with no trace.

---

## Open questions

- **Does TRL's `GRPOTrainer` add an epsilon to `group_std`, or skip
  zero-variance groups entirely?** Needs confirming against the installed
  TRL version's source when Day 7 is implemented — directly affects
  whether an all-correct or all-wrong group silently contributes zero
  gradient or raises an error.
- **What group size `G` is actually feasible** for Qwen2.5-0.5B with
  backward passes on a 16GB MacBook? Not pre-committing a number now —
  this is exactly the kind of empirical finding that should be measured
  and documented at Day 7, not guessed in advance.
- **Which KL estimator does TRL use** for the `D_KL(π_θ ‖ π_ref)` term —
  the paper's k3 estimator, or a simpler approximation? Affects how to
  interpret the KL coefficient `β` if it's ever touched.
- **Is `π_ref` re-synced periodically** (as some GRPO/PPO variants do,
  updating the reference every N steps) or held fixed for the entire smoke
  test? For a 1–5 step smoke test this is moot, but worth knowing before
  any longer run beyond Week 1.

---

## Relevant papers

| Paper | Why it matters here |
|---|---|
| Shao et al., 2024 — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* | Introduces GRPO (§4 of the paper); the `(reward − group_mean)/group_std` formula and the full objective in "Main contribution" above are drawn directly from here. |
| Guo et al., 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* | Large-scale application of GRPO to reasoning RL (including a pure-RL "R1-Zero" setting with no SFT stage) — the clearest evidence GRPO scales on exactly the kind of math/reasoning tasks this project's GSM8K setup is a miniature of. |
| Schulman, Wolski, Dhariwal, Radford, Klimov, 2017 — *Proximal Policy Optimization Algorithms* | The clipped surrogate objective GRPO reuses unmodified (`ppo.md` §2). |
| Schulman, 2020 — *Approximating KL Divergence* (blog note) | The unbiased, non-negative KL estimator ("k3") used by GRPO's KL penalty term. |

See also: `notes/policy_gradient.md` §4.3 (which previews this exact
advantage formula before this note derives it in full) and `notes/ppo.md`
§3.1 (the cost table this note's "Why it is cheaper than PPO" section
extends).
