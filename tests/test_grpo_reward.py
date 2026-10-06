from src.rewards.correctness import gsm8k_grpo_reward


def test_scores_each_completion_against_its_own_answer():
    prompts = ["Q1\n", "Q1\n", "Q2\n"]
    completions = [
        "Therefore, the answer is 20.",
        "Therefore, the answer is 19.",
        "Therefore, the answer is 72.",
    ]
    answers = ["#### 20", "#### 20", "#### 72"]

    rewards = gsm8k_grpo_reward(prompts=prompts, completions=completions, answer=answers)

    assert rewards == [1.0, 0.0, 1.0]


def test_ignores_extra_trainer_kwargs():
    rewards = gsm8k_grpo_reward(
        prompts=["Q\n"],
        completions=["Therefore, the answer is 5."],
        answer=["#### 5"],
        trainer_state=None,
        log_extra=None,
        completion_ids=[[1, 2, 3]],
    )
    assert rewards == [1.0]
