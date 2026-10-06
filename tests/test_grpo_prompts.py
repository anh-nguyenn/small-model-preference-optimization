from src.datasets.grpo_prompts import build_grpo_prompt_dataset


def test_builds_requested_number_of_examples():
    dataset = build_grpo_prompt_dataset(n_examples=5)
    assert len(dataset) == 5


def test_has_prompt_and_answer_columns():
    dataset = build_grpo_prompt_dataset(n_examples=3)
    assert set(dataset.column_names) == {"prompt", "answer"}


def test_answer_is_the_raw_gsm8k_field_with_marker():
    dataset = build_grpo_prompt_dataset(n_examples=3)
    for row in dataset:
        assert "####" in row["answer"]
