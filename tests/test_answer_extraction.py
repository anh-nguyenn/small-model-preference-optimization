from src.evaluation.answer_extraction import extract_answer


def test_extracts_answer_from_simple_phrase():
    assert extract_answer("Therefore the answer is 20.") == "20"


def test_extracts_negative_answer_from_final_answer_phrase():
    assert extract_answer("The final answer is -3.") == "-3"


def test_extracts_from_real_gsm8k_ground_truth_format():
    text = "There are 12 + 8 = <<12+8=20>>20 balls in total.\n#### 20"
    assert extract_answer(text) == "20"


def test_normalizes_comma_thousands_separator():
    assert extract_answer("We have 1,200 apples. The answer is 1,200.") == "1200"


def test_extracts_decimal_answer():
    assert extract_answer("The answer is 2.5 meters.") == "2.5"


def test_falls_back_to_last_number_when_no_keyword_present():
    assert extract_answer("We start with 12, add 8, so there are 20 balls.") == "20"


def test_returns_none_when_no_number_present():
    assert extract_answer("I'm not sure how to solve this.") is None


def test_prefers_keyword_tagged_number_over_earlier_numbers():
    text = "There are 12 red and 8 blue, so the answer is 20."
    assert extract_answer(text) == "20"
