"""Day 5, Task 1: load Qwen, send a prompt, print the generated response.

This is a smoke script, not a benchmark: greedy decoding, a small
max-new-tokens default, no performance optimization. See
notes/evaluation_pipeline.md section 4.4 for the design rationale.

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 128
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = """Solve this problem carefully:

A box contains 12 red balls and 8 blue balls.
How many balls are there?"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device()
    print(f"Device: {device}")

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()

    # Qwen2.5-Instruct is a chat model: apply its chat template rather than
    # feeding raw text, otherwise the model has no instruction-following
    # context (see notes/evaluation_pipeline.md section 4.4).
    # NOTE: apply_chat_template(..., return_tensors="pt") returns a
    # BatchEncoding (dict of input_ids + attention_mask) on transformers
    # 5.x, not a bare tensor -- so it's unpacked with **inputs below.
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    print("Prompt:")
    print(args.prompt)
    print()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # greedy decoding for a reproducible smoke test
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
