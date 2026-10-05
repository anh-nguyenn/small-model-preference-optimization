"""Build a synthetic DPO preference dataset from GSM8K.

Design rationale lives in notes/dpo_smoke_test.md (section 2). chosen and
rejected share identical reasoning text; only the final number differs --
this isolates "say the correct number" as the only signal the preference
pair encodes, with no confound from reasoning style/length differences.
"""

import re

from datasets import Dataset, load_dataset

from src.evaluation.answer_extraction import extract_answer

_CALCULATOR_ANNOTATION = re.compile(r"<<[^>]*>>")


def _strip_calculator_annotations(text: str) -> str:
    return _CALCULATOR_ANNOTATION.sub("", text)


def build_dpo_preference_dataset(
    n_examples: int, split: str = "train", wrong_offset: int = 1
) -> Dataset:
    """Build an (prompt, chosen, rejected) preference dataset from GSM8K.

    `chosen` is the dataset's own reasoning followed by the correct final
    answer; `rejected` is the same reasoning followed by `correct +
    wrong_offset`. No model sampling is involved -- this is fully
    deterministic, so reruns produce identical data.
    """
    raw = load_dataset("openai/gsm8k", "main", split=split).select(range(n_examples))

    rows = []
    for example in raw:
        reasoning, _, _ = example["answer"].partition("####")
        reasoning = _strip_calculator_annotations(reasoning).strip()

        correct = extract_answer(example["answer"])
        wrong = str(int(float(correct)) + wrong_offset)

        rows.append(
            {
                "prompt": example["question"] + "\n",
                "chosen": f"{reasoning}\nTherefore, the answer is {correct}.",
                "rejected": f"{reasoning}\nTherefore, the answer is {wrong}.",
            }
        )

    return Dataset.from_list(rows)
