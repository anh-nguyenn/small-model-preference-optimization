"""Deterministic binary correctness reward for GSM8K-style responses.

No learned reward model, no LLM judge — per CLAUDE.md's reward principles.
Design rationale lives in notes/evaluation_pipeline.md (sections 3.4-3.5).
"""

from src.evaluation.answer_extraction import extract_answer


def compute_reward(generated_text: str, ground_truth: str) -> float:
    """Return 1.0 if the final numeric answer in `generated_text` matches
    the one in `ground_truth`, else 0.0.

    `ground_truth` may be a raw GSM8K `answer` field (containing a
    "#### N" marker) or an already-bare number string -- extract_answer
    handles both, so there is no separate ground-truth parser here.

    A response whose answer can't be extracted at all is treated the same
    as a wrong answer (reward 0.0), not specially.
    """
    predicted = extract_answer(generated_text)
    target = extract_answer(ground_truth)

    if predicted is None or target is None:
        return 0.0

    try:
        return 1.0 if float(predicted) == float(target) else 0.0
    except ValueError:
        return 0.0


def gsm8k_grpo_reward(
    prompts: list[str], completions: list[str], answer: list[str], **kwargs
) -> list[float]:
    """TRL `GRPOTrainer`-compatible reward function.

    A thin adapter over `compute_reward` -- no new reward logic, per
    CLAUDE.md's reuse principle. The trainer calls reward functions with
    `prompts=`, `completions=`, every other dataset column as a keyword
    (here, `answer=`), plus extras like `trainer_state`/`log_extra` this
    function doesn't need (absorbed by `**kwargs`). See
    notes/grpo_smoke_test.md section 3.1 for how this signature was
    confirmed against the installed TRL version's source.
    """
    return [
        compute_reward(completion, ground_truth)
        for completion, ground_truth in zip(completions, answer)
    ]
