"""Day 6: tiny DPO smoke test.

Goal is NOT model quality -- it's proving the pipeline runs end to end:
model loads, dataset loads, trainer initializes, forward/backward pass
work, a few optimization steps complete, a checkpoint saves. See
notes/dpo_smoke_test.md for the full design rationale, including why
TRL's DPOTrainer is called the way it is below (verified against the
installed trl==1.14.1 source, not guessed).

Usage:
    python scripts/smoke_test_dpo.py
    python scripts/smoke_test_dpo.py --n-examples 20 --max-steps 3
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.datasets.dpo_preference import build_dpo_preference_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--output-dir", default="results/dpo_smoke_001")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building preference dataset ({args.n_examples} examples)...")
    dataset = build_dpo_preference_dataset(n_examples=args.n_examples)
    print(f"Dataset loaded: {len(dataset)} examples")
    print("Example row:", dataset[0])

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    print("Model loaded")

    # peft_config is passed directly to DPOTrainer, which wraps the model
    # itself via get_peft_model(). Passing an already-PEFT-wrapped model
    # here as well would raise an error in the trainer's own validation
    # (notes/dpo_smoke_test.md section 3, point 3).
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    config = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        seed=args.seed,
        report_to="none",
        logging_steps=1,
        save_strategy="no",  # we save explicitly at the end, see below
    )

    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None + peft_config -> trainer uses the
        # pre-training policy state as reference via adapter-disabling
        # (notes/dpo_smoke_test.md section 3, point 2) -- no second full
        # model copy is ever loaded.
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    print("Trainer initialized")

    trainable_params = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    print(f"Trainable (LoRA) parameters: {trainable_params:,}")
    assert trainable_params > 0, "LoRA target_modules matched zero parameters"

    print(f"Training for {args.max_steps} steps...")
    train_result = trainer.train()

    print(f"Completed global steps: {trainer.state.global_step}")
    assert trainer.state.global_step > 0, "No optimization steps were taken"

    losses = [
        entry["loss"] for entry in trainer.state.log_history if "loss" in entry
    ]
    print(f"Per-step losses: {losses}")
    assert losses, "No loss was ever logged"
    assert all(math.isfinite(loss) for loss in losses), "Loss was NaN/inf"

    print(f"Saving checkpoint to {output_dir}...")
    trainer.save_model(str(output_dir))
    print("Checkpoint saved")

    config_record = {
        "model_name": args.model,
        "seed": args.seed,
        "n_examples": args.n_examples,
        "max_steps": args.max_steps,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": ["q_proj", "v_proj"],
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": 2,
    }
    (output_dir / "config.json").write_text(json.dumps(config_record, indent=2))

    metrics_record = {
        "global_step": trainer.state.global_step,
        "per_step_loss": losses,
        "final_loss": losses[-1],
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_record, indent=2))

    print("\nDay 6 success criteria:")
    print("  [x] model loads")
    print("  [x] dataset loads")
    print("  [x] trainer initializes")
    print("  [x] forward pass works (loss was computed)")
    print("  [x] loss is produced")
    print("  [x] backward pass works (trainable params received gradients)")
    print(f"  [x] {trainer.state.global_step} optimization step(s) completed")
    print("  [x] checkpoint saved")


if __name__ == "__main__":
    main()
