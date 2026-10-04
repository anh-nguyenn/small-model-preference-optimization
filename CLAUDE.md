# Small-Model Preference Optimization Research

## Project Goal

Build a small, reproducible research codebase for experimenting with:

- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)
- Small language models
- GSM8K-style reasoning tasks

The immediate goal is **not research novelty**.

The Week 1 goal is:

> Go from understanding DPO/GRPO conceptually to being able to run both methods on a small model.

---

## Hardware Constraints

Primary development machine:

- MacBook
- Apple Silicon preferred
- Use PyTorch MPS when available
- Do not assume CUDA is available
- Do not require a paid cloud GPU for Week 1

Avoid CUDA-only dependencies unless absolutely necessary.

Do **not** install or depend on `bitsandbytes` for the initial Mac setup.

Cloud GPU support may be added later, but all basic development should work locally first.

---

## Initial Model and Dataset

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Dataset:

```text
openai/gsm8k
```

Use the `main` GSM8K configuration unless there is a specific reason not to.

---

## Core Python Stack

Use:

```text
torch
transformers
datasets
accelerate
trl
peft
evaluate
wandb
matplotlib
pandas
pytest
```

Prefer simple, standard Hugging Face / TRL APIs.

Do not introduce unnecessary frameworks.

---

## Repository Structure

Create and maintain:

```text
small-model-preference-optimization/
├── configs/
├── data/
├── notebooks/
├── scripts/
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── smoke_test_dpo.py
│   └── smoke_test_grpo.py
├── src/
│   ├── datasets/
│   ├── evaluation/
│   ├── rewards/
│   └── training/
├── tests/
├── results/
├── notes/
│   └── papers/
├── requirements.txt
├── README.md
├── .gitignore
└── CLAUDE.md
```

Keep reusable logic in `src/`.

Keep scripts thin.

---

# Engineering Principles

## 1. Prefer clarity over abstraction

This is a research project.

Do not create complicated architecture prematurely.

Good:

```python
def extract_answer(text: str)
...
```

Bad:

```text
AbstractEvaluationPipelineFactoryManager
```

---

## 2. Make experiments reproducible

Always expose important experiment settings explicitly.

Examples:

```text
seed
model_name
dataset_name
learning_rate
batch_size
max_steps
max_new_tokens
num_generations
LoRA rank
LoRA alpha
```

Set random seeds where possible.

Store experiment configs with results.

---

## 3. Never silently change experimental methodology

If changing:

- reward function
- answer extraction
- dataset filtering
- model checkpoint
- prompt format
- DPO preference construction
- GRPO group size
- evaluation metric

document the change clearly.

Scientific comparability is more important than making a run succeed.

---

## 4. Keep experiments cheap

For Week 1:

- use very small datasets
- use only a few training steps
- use Qwen2.5-0.5B
- use LoRA/PEFT where appropriate
- avoid full fine-tuning
- avoid large model downloads
- do not start long training runs

Smoke tests should normally use roughly:

```text
10–50 examples
1–5 training steps
```

The purpose is only to verify the pipeline works.

---

# Week 1 Tasks

## Day 1 — Environment and Github Repository

Set up:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install:

```bash
pip install --upgrade pip
pip install torch transformers datasets accelerate trl peft evaluate wandb matplotlib pandas pytest
```

Verify MPS:

```python
import torch

print(torch.backends.mps.is_available())
```

Create the repository structure, push to github

Do not start DPO or GRPO training yet.

### Day 1 success criteria

```text
[ ] Repository exists
[ ] Virtual environment works
[ ] Dependencies install
[ ] PyTorch imports
[ ] MPS status is printed
[ ] Qwen model can be loaded if memory permits
```

---

## Day 2 — Policy Gradient / PPO Notes

No significant implementation is required.

The researcher should understand:

```text
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

For LLMs:

```text
State  = prompt + previous tokens
Action = next token
Policy = language model
Reward = score of generated response
```

Key conceptual relationship:

```text
Advantage = Reward - Baseline
```

Create:

```text
notes/policy_gradient.md
notes/ppo.md
```

Do not implement PPO from scratch.

---

## Day 3 — DPO

Understand preference data:

```text
prompt
├── chosen response
└── rejected response
```

DPO should increase preference for the chosen response relative to the rejected response.

Create:

```text
notes/papers/dpo.md
```

Use this structure:

```text
Problem
Existing approach
Main contribution
How training works
Inputs required
Why it is cheaper than PPO
Limitations
Relation to this project
Open questions
```

---

## Day 4 — GRPO

Understand:

```text
Prompt
  ↓
Generate G responses
  ↓
Reward each response
  ↓
Compute group-relative advantages
  ↓
Update policy
```

Core idea:

```text
advantage_i =
(reward_i - group_mean) / group_std
```

The important conceptual point:

> GRPO replaces the learned critic/value model with statistics from the generated response group.

Create:

```text
notes/papers/grpo.md
```

---

# Day 5 — Build Evaluation Pipeline

## Task 1 — Load Qwen

Create:

```text
scripts/test_model.py
```

It should:

1. Load the tokenizer.
2. Load `Qwen/Qwen2.5-0.5B-Instruct`.
3. Use MPS if available.
4. Send a simple prompt.
5. Generate a response.
6. Print the decoded answer.

Example prompt:

```text
Solve this problem carefully:

A box contains 12 red balls and 8 blue balls.
How many balls are there?
```

Keep generation settings simple.

Do not optimize performance yet.

---

## Task 2 — Load GSM8K

Create:

```text
scripts/test_dataset.py
```

Load GSM8K and print ten samples containing:

```text
question
ground-truth answer
```

Do not transform the entire dataset yet.

---

## Task 3 — Answer Extraction

Create reusable logic such as:

```text
src/evaluation/answer_extraction.py
```

Implement:

```python
def extract_answer(text: str) -> str | None:
    ...
```

It should extract the final numeric answer from typical GSM8K-style responses.

Example:

```text
We compute 12 + 8 = 20.
Therefore the answer is 20.
```

Expected:

```text
20
```

Write unit tests.

Examples:

```python
assert extract_answer("Therefore the answer is 20.") == "20"
assert extract_answer("The final answer is -3.") == "-3"
```

Do not overengineer unusual edge cases initially.

---

## Task 4 — Binary Reward

Create:

```text
src/rewards/correctness.py
```

Implement a simple verifiable reward:

```python
reward = 1.0 if predicted_answer == correct_answer else 0.0
```

No LLM judge.

No learned reward model.

This should be deterministic.

---

# Day 6 — Tiny DPO Smoke Test

Create:

```text
scripts/smoke_test_dpo.py
```

Use TRL's DPO trainer.

The goal is **not model quality**.

The test succeeds if:

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

Use only roughly 10–50 preference examples.

If needed, manually construct a tiny preference dataset with fields equivalent to:

```python
{
    "prompt": "...",
    "chosen": "...",
    "rejected": "..."
}
```

Use LoRA/PEFT if it makes local training easier.

Avoid full-model training.

---

# Day 7 — Tiny GRPO Smoke Test

Create:

```text
scripts/smoke_test_grpo.py
```

Use TRL's GRPO trainer.

Use:

- Qwen2.5-0.5B
- very small dataset
- very small group size
- very few steps
- binary correctness reward

The goal is simply to prove that the full GRPO loop works.

Expected flow:

```text
GSM8K prompt
↓
Qwen generates multiple responses
↓
extract final answer
↓
compare to ground truth
↓
reward ∈ {0, 1}
↓
GRPO computes relative advantages
↓
backpropagation
↓
optimizer update
```

If Mac memory prevents backward training, document the exact limitation rather than silently changing the experiment.

The inference/evaluation pipeline must still remain runnable locally.

---

# Result Storage

For every experiment, create a result directory such as:

```text
results/
└── dpo_smoke_001/
    ├── config.json
    ├── metrics.json
    └── notes.md
```

For later experiments:

```text
results/
├── dpo/
├── grpo/
└── grpo_to_dpo/
```

Do not commit large checkpoints to Git.

---

# Git Rules

Commit:

- source code
- scripts
- configs
- small result files
- documentation
- experiment metadata

Do not commit:

```text
.venv/
__pycache__/
*.pyc
wandb/
large model checkpoints
Hugging Face cache
large generated datasets
```

Add appropriate entries to `.gitignore`.

---

# Coding Agent Behavior

When implementing tasks:

1. Inspect the existing repository before creating files.
2. Reuse existing code instead of duplicating logic.
3. Make the smallest correct change.
4. Run the relevant script/test after modification.
5. Fix errors before declaring the task complete.
6. Explain important technical decisions briefly.
7. Do not start expensive or long-running training automatically.
8. Do not download multi-billion-parameter models without explicit approval.
9. Do not delete experiment results without explicit approval.
10. Do not modify the research question merely to make implementation easier.

---

# Commands the Agent Should Support

The following should eventually work:

```bash
python scripts/test_model.py
python scripts/test_dataset.py

pytest tests/

python scripts/smoke_test_dpo.py
python scripts/smoke_test_grpo.py
```

Prefer adding command-line arguments where useful, for example:

```bash
python scripts/test_model.py --model Qwen/Qwen2.5-0.5B-Instruct
```

But do not overbuild the CLI during Week 1.

---

# Week 1 Final Success Criteria

By the end of Week 1:

```text
[ ] Research repository is clean and reproducible
[ ] Qwen2.5-0.5B runs locally
[ ] GSM8K loads correctly
[ ] Automatic answer extraction works
[ ] Binary correctness reward works
[ ] DPO concepts are documented
[ ] GRPO concepts are documented
[ ] PPO concepts are documented
[ ] Tiny DPO training reaches at least one optimization step
[ ] Tiny GRPO training reaches at least one optimization step, if local hardware permits
[ ] Configs/results are saved reproducibly
[ ] Code is committed to Git
```

Week 1 is successful even if model accuracy does not improve.

The important outcome is:

> The researcher understands the training pipeline and has a clean codebase capable of running controlled DPO and GRPO experiments.

---

# What NOT to Do Yet

Do not:

- search for research novelty
- run large benchmark sweeps
- train models larger than necessary
- rent cloud GPUs unless local limitations actually block progress
- implement PPO from scratch
- build a custom training framework
- use an LLM-as-a-judge reward
- optimize hyperparameters
- compare dozens of models
- publish conclusions from smoke-test results

Those belong to later stages of the project.
