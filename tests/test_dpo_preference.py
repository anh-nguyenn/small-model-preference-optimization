from src.datasets.dpo_preference import build_dpo_preference_dataset
from src.evaluation.answer_extraction import extract_answer


def test_builds_requested_number_of_examples():
    dataset = build_dpo_preference_dataset(n_examples=5)
    assert len(dataset) == 5


def test_has_prompt_chosen_rejected_columns():
    dataset = build_dpo_preference_dataset(n_examples=3)
    assert set(dataset.column_names) == {"prompt", "chosen", "rejected"}


def test_chosen_and_rejected_differ():
    dataset = build_dpo_preference_dataset(n_examples=5)
    for row in dataset:
        assert row["chosen"] != row["rejected"]


def test_chosen_extracts_to_correct_answer_rejected_does_not():
    dataset = build_dpo_preference_dataset(n_examples=5)
    for row in dataset:
        chosen_answer = extract_answer(row["chosen"])
        rejected_answer = extract_answer(row["rejected"])
        assert chosen_answer is not None
        assert rejected_answer is not None
        assert float(chosen_answer) != float(rejected_answer)


def test_prompt_does_not_contain_the_reasoning():
    dataset = build_dpo_preference_dataset(n_examples=5)
    for row in dataset:
        assert "Therefore" not in row["prompt"]
