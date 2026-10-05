from src.rewards.correctness import compute_reward


def test_matching_answer_gets_full_reward():
    generated = "We compute 12 + 8 = 20.\nTherefore the answer is 20."
    ground_truth = "There are 12 + 8 = <<12+8=20>>20 balls in total.\n#### 20"
    assert compute_reward(generated, ground_truth) == 1.0


def test_wrong_answer_gets_zero_reward():
    generated = "Therefore the answer is 19."
    ground_truth = "#### 20"
    assert compute_reward(generated, ground_truth) == 0.0


def test_unparsable_generation_gets_zero_reward():
    generated = "I'm not sure how to solve this."
    ground_truth = "#### 20"
    assert compute_reward(generated, ground_truth) == 0.0


def test_matches_regardless_of_decimal_formatting():
    generated = "The answer is 20.0."
    ground_truth = "#### 20"
    assert compute_reward(generated, ground_truth) == 1.0
