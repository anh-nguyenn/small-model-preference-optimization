"""Build a (prompt, answer) dataset from GSM8K for GRPO.

Simpler than Day 6's dpo_preference.py -- GRPO scores the policy's own
live generations rather than needing pre-built chosen/rejected pairs.
Design rationale: notes/grpo_smoke_test.md section 2.
"""

from datasets import Dataset, load_dataset


def build_grpo_prompt_dataset(n_examples: int, split: str = "train") -> Dataset:
    """Return a Dataset with "prompt" and "answer" columns.

    "answer" is the raw GSM8K answer field (including the "#### N"
    marker) -- the reward function extracts the ground truth from it via
    the same extract_answer() used everywhere else in this project.
    """
    raw = load_dataset("openai/gsm8k", "main", split=split).select(range(n_examples))
    return Dataset.from_dict(
        {
            "prompt": [example["question"] + "\n" for example in raw],
            "answer": [example["answer"] for example in raw],
        }
    )
