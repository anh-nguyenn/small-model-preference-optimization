"""Day 5, Task 2: load GSM8K, print sample (question, ground-truth answer) pairs.

Only slices the first `--n` examples -- does not transform or materialize
the whole dataset. See notes/evaluation_pipeline.md section 4.5.

Usage:
    python scripts/test_dataset.py
    python scripts/test_dataset.py --split train --n 10
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/test_dataset.py` (not just `python -m`)
# by putting the repo root on sys.path so `src` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.evaluation.answer_extraction import extract_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading openai/gsm8k (config=main, split={args.split})")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    sample = dataset.select(range(args.n))

    for i, example in enumerate(sample):
        extracted = extract_answer(example["answer"])
        print(f"--- Example {i + 1} ---")
        print(f"Question: {example['question']}")
        print(f"Raw answer field: {example['answer']}")
        print(f"Extracted ground-truth answer: {extracted}")
        print()


if __name__ == "__main__":
    main()
