# Policy Gradient Methods — Background Notes

> **Day 2 deliverable.** No code is implemented today. This note exists so that
> Day 3 (DPO), Day 4 (GRPO), and later the smoke tests make sense conceptually
> instead of being "follow the TRL API and hope." Read this before `ppo.md`.

## 1. Why this note exists

DPO and GRPO (the two methods this project actually implements) are both
**reactions to policy gradient / PPO-style RL**. DPO replaces the RL loop
entirely with a classification-style loss. GRPO keeps the RL loop but removes
one expensive piece of it (the value network). You cannot appreciate *why*
either method is designed the way it is without first understanding the
vanilla policy gradient — what it optimizes, where its variance comes from,
and why that variance is a problem in practice.

So this note builds the minimum RL vocabulary needed, in plain terms, with a
worked numeric example. It does not derive every theorem rigorously — see
§6 for the papers if you want the full math.

---

## 2. Reinforcement learning in one picture

```
Policy
  ↓
Action probability
  ↓
Reward
  ↓
Advantage
  ↓
Policy update
```

An agent (the **policy**) is in some **state**, picks an **action** according
to a probability distribution it controls, the environment returns a
**reward**, and the agent nudges its action probabilities so that
better-than-expected actions become more likely and worse-than-expected
actions become less likely. "Better than expected" is what **advantage**
means — see §4.

### 2.1 Mapping this onto a language model

This project's entire use of RL concepts is through this mapping (same table
as `CLAUDE.md`, repeated here because it's the single most important idea in
this note):

| RL concept | LLM equivalent |
|---|---|
| State  | prompt + tokens generated so far |
| Action | the next token to generate |
| Policy | the language model itself (its output distribution over the vocabulary) |
| Reward | a scalar score for the *completed* response (e.g. 1.0 if the final numeric answer matches GSM8K's ground truth, else 0.0 — see `src/rewards/correctness.py`, built on Day 5) |

A full episode = generating one complete response token-by-token until EOS.
The reward in our setup is **sparse and terminal**: it only arrives once, at
the end of the response, not per-token. This matters later (§4.3).

---

## 3. The policy gradient theorem (intuitively)

We want to find model parameters `θ` that maximize expected reward:

```
J(θ) = E_{response ~ policy_θ}[ reward(response) ]
```

We can't differentiate `reward(response)` directly (it's often a non-
differentiable function — e.g. "does the extracted number match?"). The
policy gradient theorem gives us a way to estimate `∇θ J(θ)` anyway, using
only the *log-probability* of the actions the policy actually took:

```
∇θ J(θ) ≈ E[ reward(response) · ∇θ log π_θ(response) ]
```

In words: **push up the log-probability of a response, scaled by how much
reward it got.** High-reward responses get reinforced; low-reward responses
get suppressed. This is often written per-token for an autoregressive model:

```
∇θ J(θ) ≈ E[ Σ_t  reward(response) · ∇θ log π_θ(token_t | state_t) ]
```

This is the **REINFORCE** algorithm (Williams, 1992 — see §6). It is simple
and unbiased, but has a well-known problem: **high variance**.

---

## 4. Why variance is a problem, and how "advantage" fixes it

### 4.1 The problem

Suppose every response in a batch gets a reward between 8 and 10 (all pretty
good, just by different amounts). Raw REINFORCE still pushes up the log-prob
of *every single one* of them, proportionally to its raw reward. The
gradient signal is dominated by the scale of the reward, not by whether a
response was good *relative to what was expected*. This makes training noisy
and sample-inefficient — you need a lot of samples to average out the noise.

### 4.2 The fix: subtract a baseline

```
Advantage = Reward - Baseline
```

If `Baseline` is some estimate of "the reward I'd expect anyway" (e.g. the
average reward across a group of responses to the same prompt), then:

- A response that beats the baseline → positive advantage → reinforced.
- A response that's worse than the baseline → negative advantage → suppressed.
- A response that's exactly average → ~zero advantage → little gradient
  signal wasted on it.

Crucially, **subtracting any baseline that doesn't depend on the action
doesn't bias the gradient estimate** (a standard RL result), but it can
massively reduce its variance. This is why basically every practical policy
gradient method uses *some* form of baseline.

### 4.3 Where the baseline comes from — this is the key design fork

This is precisely the decision point that separates PPO from GRPO:

| Method | Baseline source |
|---|---|
| **PPO** (classic actor-critic) | A learned **value network** (critic) that predicts expected reward for a given state, trained alongside the policy |
| **GRPO** (Day 4) | The **mean reward of a group** of responses sampled for the *same prompt* — no separate network needed |

```
GRPO advantage_i = (reward_i - group_mean) / group_std
```

GRPO's insight (previewed here, detailed in `notes/papers/grpo.md` on Day 4)
is: if you already have to generate multiple responses per prompt to get a
reliable reward signal, you can use *their own statistics* as the baseline,
instead of training and maintaining a separate critic model. That's one
whole model you don't have to hold in memory — a big deal on a 16GB MacBook.

---

## 5. Worked toy example

Say a prompt has 4 sampled responses with binary correctness rewards:

```
response_1: reward = 1.0  (correct)
response_2: reward = 0.0  (wrong)
response_3: reward = 1.0  (correct)
response_4: reward = 0.0  (wrong)

group_mean = 0.5
group_std  ≈ 0.577   (population std of [1,0,1,0])

advantage_1 = (1.0 - 0.5) / 0.577 ≈ +0.87   → reinforce this response
advantage_2 = (0.0 - 0.5) / 0.577 ≈ -0.87   → suppress this response
advantage_3 = (1.0 - 0.5) / 0.577 ≈ +0.87   → reinforce
advantage_4 = (0.0 - 0.5) / 0.577 ≈ -0.87   → suppress
```

Without a baseline, both correct responses would get a flat `+1.0` nudge and
both incorrect ones would get literally **zero** gradient (reward=0 → no
signal at all, regardless of how "close" they were). With the group-relative
advantage, correct answers are pushed up and incorrect answers are
*actively* pushed down — this is what gives GRPO (and policy gradient
methods generally) their learning signal, not just a magnitude but a
direction for every sample.

---

## 6. Relevant papers

| Paper | Why it matters here |
|---|---|
| Williams, 1992 — *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* | Introduces REINFORCE — the basic policy gradient estimator this whole note explains. |
| Sutton, McAllester, Singh, Mansour, 2000 — *Policy Gradient Methods for Reinforcement Learning with Function Approximation* | The formal policy gradient theorem; also introduces baselines rigorously. |
| Schulman, Moritz, Levine, Jordan, Abbeel, 2016 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)* | How to compute advantages well in practice when rewards aren't purely terminal/sparse; referenced again in `ppo.md`. |
| Shao et al., 2024 (DeepSeekMath) — *GRPO* | Full derivation of the group-relative advantage formula used in §4.3 and Day 4. |

We are **not** re-deriving these papers' math from scratch here — the goal
is the intuition needed to read TRL's trainer code and this project's reward
functions without the RL vocabulary being a black box.

---

## 7. Implementation plan (what this enables later, not today)

**Day 2 itself ships no code** — this file and `ppo.md` are the entire
deliverable, per `CLAUDE.md`.

What this note unblocks downstream:

- **Day 4 (`notes/papers/grpo.md`)** will reuse the `Advantage = Reward -
  Baseline` framing directly, specializing `Baseline = group_mean` and
  `Advantage = (reward - group_mean) / group_std` as derived in §4.3–5.
- **Day 5 (`src/rewards/correctness.py`)** implements exactly the
  `reward(response)` function this note treats as a black box — binary,
  deterministic, no learned reward model (see `ppo.md` §4 for why we avoid
  a learned reward model entirely).
- **Day 7 (`scripts/smoke_test_grpo.py`)** is the first place any of this
  math actually executes: TRL's `GRPOTrainer` computes the group-relative
  advantage from §4.3 internally, using the reward function from Day 5.

No REINFORCE or PPO loop is hand-implemented anywhere in this project (see
`ppo.md` §5 for why). This note's job is purely conceptual grounding.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Policy | The model that chooses actions; here, the language model's token distribution |
| Trajectory / rollout | One full generated response, prompt to EOS |
| Reward | Scalar score for a completed trajectory |
| Baseline | A reference value subtracted from reward to reduce variance; must not depend on the action taken |
| Advantage | `Reward - Baseline`; the actual learning signal |
| On-policy | Training data generated by the *current* policy (as opposed to a frozen/older one) |
| Critic / value network | A learned model that predicts expected reward from a state — one way to produce a baseline (used in PPO, not in GRPO) |

## 9. Open questions (to revisit later in the project)

- How much does group size (`G` in GRPO) affect advantage estimate quality
  on an 0.5B model? (Not tested until Day 7 smoke test, and only
  qualitatively — this is a smoke test, not a sweep.)
- Binary 0/1 rewards make `group_std = 0` when every response in a group
  gets the same reward (e.g. all correct or all wrong) — division by zero.
  Need to check how TRL's `GRPOTrainer` handles this (likely an epsilon in
  the denominator) when we actually implement Day 7.
