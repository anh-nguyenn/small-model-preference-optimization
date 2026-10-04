# PPO (Proximal Policy Optimization) — Background Notes

> **Day 2 deliverable, part 2.** Read `policy_gradient.md` first — this note
> assumes you already know what "advantage" and "baseline" mean.
>
> **This project does not implement PPO.** Per `CLAUDE.md`: *"Do not
> implement PPO from scratch."* This note exists purely so that DPO (Day 3)
> and GRPO (Day 4) make sense as *reactions to* PPO's specific pain points,
> most of which are about cost and stability, not correctness.

## 1. The problem PPO solves

Vanilla policy gradient (REINFORCE, `policy_gradient.md` §3) has a practical
failure mode: if you take too large a step in parameter space based on one
noisy batch of rollouts, you can wreck the policy — and because the *next*
batch of data is sampled from that now-wrecked policy, there's no way to
recover. Small steps are safe but slow; large steps are fast but can
destroy training irrecoverably. This instability is the single biggest
practical obstacle to using plain policy gradients at scale.

**TRPO** (Trust Region Policy Optimization, Schulman et al., 2015) was the
first serious fix: constrain each update so the new policy cannot move too
far from the old one, measured via KL divergence. It works, but the
constrained optimization (conjugate gradient + line search) is complex and
expensive to implement correctly.

**PPO** (Schulman et al., 2017) approximates the same trust-region idea with
something you can implement using plain stochastic gradient descent — no
second-order optimization required. This is why PPO became the default
"RL for deep learning" algorithm, including for RLHF on LLMs (InstructGPT,
Ouyang et al., 2022).

---

## 2. The PPO clipped objective

Define the **probability ratio** between the new and old policy for a given
action:

```
r(θ) = π_θ(action | state) / π_θ_old(action | state)
```

`r(θ) = 1` means the new policy assigns the same probability as the old one
did at the time the data was sampled. `r(θ) > 1` means the new policy likes
this action more; `r(θ) < 1` means it likes it less.

The vanilla (unclipped) surrogate objective would be:

```
L(θ) = E[ r(θ) · Advantage ]
```

This is maximized by pushing `r(θ)` as high as possible whenever `Advantage
> 0` — with no limit. That unbounded push is exactly the instability
described in §1. PPO's fix is the **clipped surrogate objective**:

```
L_CLIP(θ) = E[ min( r(θ)·Advantage,  clip(r(θ), 1-ε, 1+ε)·Advantage ) ]
```

Intuition, case by case:

- **`Advantage > 0`** (good action): increasing `r(θ)` increases the
  objective — but only up to `1+ε`. Beyond that, the `clip(...)` term caps
  it, and `min(...)` picks the capped (smaller) value. The policy is not
  rewarded for moving the ratio further than `1+ε` in one update.
- **`Advantage < 0`** (bad action): decreasing `r(θ)` increases the
  objective — but only down to `1-ε`. Same capping logic, mirrored.

`ε` (epsilon) is typically `0.1`–`0.2`. The net effect: **each update can
only move the policy's probability on any given action by a bounded
fraction**, which is a soft, first-order-friendly approximation of TRPO's
hard trust-region constraint. This is why PPO can be trained with ordinary
SGD/Adam and no constrained optimization machinery.

---

## 3. Full PPO architecture (for LLM RLHF specifically)

Classic PPO-for-RLHF (as in InstructGPT) needs **up to four models loaded at
once**:

```
1. Policy model     — the LLM being trained (generates responses)
2. Reference model  — a frozen copy of the policy *before* RL started,
                       used only to compute a KL penalty so the policy
                       doesn't drift too far from its starting behavior
3. Reward model     — a separately trained model that scores a (prompt,
                       response) pair with a scalar reward
4. Value model (critic) — predicts expected future reward from a partial
                       state, used to compute the baseline/advantage
                       (this is the "Baseline" from policy_gradient.md §4.3)
```

Flow:

```
prompt
  ↓
policy generates response (token by token)
  ↓
reward model scores the full response  →  reward
  ↓
value model estimates baseline at each step
  ↓
advantage = reward - baseline   (often smoothed via GAE, see §4)
  ↓
KL penalty subtracted: reward_adjusted = reward - β · KL(policy || reference)
  ↓
PPO clipped objective update (policy model AND value model both updated)
```

### 3.1 Why this is expensive

- **Four models in memory at once** (policy, reference, reward, value) — on
  an 0.5B model that's ~2B parameters worth of memory just for the
  forward passes, before considering optimizer states for the two models
  being trained.
- **A reward model must be trained first**, as its own supervised learning
  problem, on human (or synthetic) preference data — an entire separate
  pipeline before any RL happens.
- **The value model is also being trained online**, concurrently with the
  policy, which is itself a source of instability (a moving-target
  regression problem).

This cost is precisely what both DPO and GRPO independently cut down,
in two different ways:

| Method | What it removes from the PPO stack |
|---|---|
| **DPO** (Day 3) | Removes the reward model *and* the value model *and* the RL loop itself — reframes preference learning as a closed-form classification loss on (chosen, rejected) pairs. Only needs policy + reference model. |
| **GRPO** (Day 4) | Removes just the value model (replaces it with group statistics, `policy_gradient.md` §4.3) — but keeps the RL loop, the reward signal, and (per DeepSeekMath) the reference-model KL penalty. |

Seeing this table is the main point of Day 2: DPO and GRPO are not
unrelated alternatives invented independently — they are both specific,
different answers to "which expensive piece of PPO can we cut, and how."

---

## 4. Generalized Advantage Estimation (GAE) — one paragraph

Real PPO implementations rarely use a raw one-step advantage. **GAE**
(Schulman et al., 2016) blends multi-step advantage estimates with an
exponential decay factor `λ`, trading off bias and variance similarly to
how `λ` works in TD(λ). This project does not implement GAE — it's
mentioned here only because you'll see `gae_lambda` as a hyperparameter name
if you ever read TRL's `PPOTrainer` source, and it's good to recognize what
it refers to.

---

## 5. Why this project skips implementing PPO

Per `CLAUDE.md` engineering principles ("keep experiments cheap") and
explicit instruction ("Do not implement PPO from scratch"):

1. **Four-model memory footprint** is a poor fit for a 16GB MacBook running
   everything locally with no paid GPU (Week 1 hardware constraint).
2. **Training a reward model is itself a multi-day project** — out of scope
   for "go from understanding DPO/GRPO conceptually to running both on a
   small model" in one week.
3. **DPO and GRPO are the actual subjects of this research codebase.** PPO
   is background context to understand *why* they exist, not a method this
   project needs working code for.

So nothing in `scripts/` or `src/training/` will ever instantiate a PPO
trainer. If you see `trl.PPOTrainer` imported anywhere later in this repo
without an explicit, documented reason, that's a deviation from the spec —
flag it.

---

## 6. Relevant papers

| Paper | Why it matters here |
|---|---|
| Schulman, Levine, Abbeel, Jordan, Moritz, 2015 — *Trust Region Policy Optimization (TRPO)* | The trust-region idea PPO approximates; explains *why* unconstrained policy gradient updates are dangerous. |
| Schulman, Wolski, Dhariwal, Radford, Klimov, 2017 — *Proximal Policy Optimization Algorithms* | The clipped surrogate objective in §2, in full. |
| Schulman, Moritz, Levine, Jordan, Abbeel, 2016 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation* | GAE, §4. |
| Ouyang et al., 2022 (OpenAI) — *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)* | The canonical "PPO for LLM RLHF" pipeline described in §3 — reward model + policy + reference + PPO. |
| Christiano et al., 2017 — *Deep Reinforcement Learning from Human Preferences* | Earlier work establishing the "learn a reward model from pairwise human preferences, then optimize with RL" pattern that InstructGPT scales up, and that DPO later bypasses. |

---

## 7. Implementation plan (what this enables later, not today)

**No code ships today.** This note's purpose is entirely to set up the
contrast table in §3.1 so that:

- **Day 3 (`notes/papers/dpo.md`)** can explain DPO's closed-form loss as
  "optimizing the same underlying objective as PPO-for-RLHF, but with the
  reward model and RL loop algebraically eliminated" — that derivation only
  makes sense with the PPO pipeline from §3 in hand.
- **Day 4 (`notes/papers/grpo.md`)** can explain the GRPO loop as "§3's
  pipeline, minus the value model box, plus §4.3-of-`policy_gradient.md`'s
  group-relative advantage in its place."
- **Day 6 (`scripts/smoke_test_dpo.py`)** and **Day 7
  (`scripts/smoke_test_grpo.py`)** will use TRL's `DPOTrainer` and
  `GRPOTrainer` respectively — never `PPOTrainer` — consistent with §5.

## 8. Glossary

| Term | Meaning |
|---|---|
| Trust region | A constraint keeping the new policy "close" to the old one per update, to prevent destructive large steps |
| Probability ratio `r(θ)` | `π_new / π_old` for a given action; PPO's core quantity |
| Clipping (`ε`) | Bounding how far `r(θ)` is allowed to move the objective in one update |
| Reference model | A frozen snapshot of the policy, used to penalize drift (KL penalty) — reused by both PPO and GRPO, and conceptually by DPO's "implicit reward" |
| Reward model | A separately trained model scoring (prompt, response) pairs — **not used anywhere in this project** (we use the deterministic binary reward from Day 5 instead) |
| KL penalty | A term subtracted from reward proportional to how far the policy has drifted from the reference model, preventing reward over-optimization / "reward hacking" |

## 9. Open questions (to revisit later in the project)

- GRPO (per DeepSeekMath) keeps a KL penalty against a reference model, the
  same way PPO does. Need to confirm in Day 4 notes exactly how TRL's
  `GRPOTrainer` implements that term, and with what default coefficient.
- Since we never train a reward model, every reward signal in this project
  is deterministic and rule-based (Day 5). This sidesteps "reward hacking"
  concerns entirely for Week 1, but it's worth explicitly noting as a
  scope limitation — real RLHF reward models have failure modes (reward
  hacking, see Christiano et al.) that this project's binary-correctness
  setup does not exercise at all.
