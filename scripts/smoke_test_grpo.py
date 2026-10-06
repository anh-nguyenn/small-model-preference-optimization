"""Day 7: tiny GRPO smoke test.

Goal is NOT model quality -- it's proving the full loop runs: generate G
responses per prompt -> reward each -> compute group-relative advantage ->
backprop -> optimizer update. See notes/grpo_smoke_test.md for the full
design rationale, including several places TRL's actual defaults diverge
from the paper's formulation (verified against the installed
trl==1.14.1 source, not guessed).

Usage:
    python scripts/smoke_test_grpo.py
    python scripts/smoke_test_grpo.py --n-examples 8 --num-generations 4 --max-steps 3
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
from trl import GRPOConfig, GRPOTrainer

from src.datasets.grpo_prompts import build_grpo_prompt_dataset
from src.rewards.correctness import gsm8k_grpo_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--output-dir", default="results/grpo_smoke_001")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building prompt dataset ({args.n_examples} examples)...")
    dataset = build_grpo_prompt_dataset(n_examples=args.n_examples)
    print(f"Dataset loaded: {len(dataset)} examples")
    print("Example row:", dataset[0])

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # GRPOTrainer's own docstring requires left-padding for batched
    # generation; it only sets this automatically when it constructs its
    # own tokenizer (processing_class=None). Since we pass our own
    # tokenizer explicitly, we must set this ourselves -- omitting it was
    # a real bug caught during implementation (notes/grpo_smoke_test.md
    # section 7): with right-padding, every one of the 12 sampled
    # completions in the first run hit max_completion_length without ever
    # producing a coherent, EOS-terminated answer, giving reward=0 across
    # every single group.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    print("Model loaded")

    # The real root cause of the reward=0/zero-variance bug found during
    # implementation (notes/grpo_smoke_test.md section 7): GRPOTrainer
    # builds its GenerationConfig with eos_token_id=tokenizer.eos_token_id,
    # a SINGLE id (<|im_end|>, 151645). But model.generation_config.eos_
    # token_id is a LIST, [151645, 151643] -- and in raw-completion
    # (non-chat) prompting, Qwen2.5-Instruct naturally stops on 151643
    # (<|endoftext|>), not 151645. Confirmed by inspecting the actual
    # generated token ids: a correct "...\\boxed{72}." was immediately
    # followed by <|endoftext|>, which the trainer's restricted
    # eos_token_id never recognized, so generation ran to
    # max_completion_length every time, past the real answer, corrupting
    # the reward signal. Fixed by passing the model's full eos id list
    # through GRPOConfig's own generation_kwargs override mechanism.
    eos_token_ids = model.generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    # peft_config is passed directly to GRPOTrainer, which wraps the model
    # itself, same as DPOTrainer in Day 6 (notes/grpo_smoke_test.md section 3.2).
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.num_generations,  # satisfies the
        # generation_batch_size % num_generations == 0 constraint with the
        # simplest possible setup: one prompt's full group per step
        # (notes/grpo_smoke_test.md section 3.5).
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,  # TRL defaults to 0.0 (no KL penalty, no reference
        # model at all); overridden here to actually exercise that code
        # path (notes/grpo_smoke_test.md section 3.3).
        loss_type="grpo",  # TRL defaults to "dapo"; overridden here to
        # match the per-sequence-normalized objective this project
        # documented in notes/papers/grpo.md (same section 3.3).
        max_completion_length=320,  # measured, not guessed: a direct
        # generation test (notes/grpo_smoke_test.md section 7) showed this
        # model needs ~215 tokens of verbose step-by-step reasoning to
        # reach a stated final answer in raw-completion (non-chat) mode.
        # An earlier value of 128 truncated every completion before any
        # answer was ever stated, producing reward=0 for 100% of samples.
        # temperature is left at GRPOConfig's default (1.0) deliberately --
        # greedy decoding would make every response in a group identical,
        # collapsing group_std to 0 (notes/grpo_smoke_test.md section 3.6).
        generation_kwargs={"eos_token_id": eos_token_ids},  # see the fix
        # explanation above -- without this, TRL's own default restricts
        # eos_token_id to a single token that this prompting mode doesn't
        # naturally produce.
        seed=args.seed,
        report_to="none",
        logging_steps=1,
        save_strategy="no",  # we save explicitly at the end, see below
    )

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=gsm8k_grpo_reward,
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

    print(
        f"Training for {args.max_steps} steps "
        f"(group size {args.num_generations})..."
    )
    train_result = trainer.train()

    print(f"Completed global steps: {trainer.state.global_step}")
    assert trainer.state.global_step > 0, "No optimization steps were taken"

    losses = [
        entry["loss"] for entry in trainer.state.log_history if "loss" in entry
    ]
    rewards = [
        entry["reward"] for entry in trainer.state.log_history if "reward" in entry
    ]
    reward_stds = [
        entry["reward_std"]
        for entry in trainer.state.log_history
        if "reward_std" in entry
    ]
    print(f"Per-step losses: {losses}")
    print(f"Per-step mean rewards: {rewards}")
    print(f"Per-step reward std (group-relative spread): {reward_stds}")
    assert losses, "No loss was ever logged"
    assert all(math.isfinite(loss) for loss in losses), "Loss was NaN/inf"
    assert rewards, "No reward was ever logged -- the reward function may not have run"

    print(f"Saving checkpoint to {output_dir}...")
    trainer.save_model(str(output_dir))
    print("Checkpoint saved")

    config_record = {
        "model_name": args.model,
        "seed": args.seed,
        "n_examples": args.n_examples,
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
        "beta": args.beta,
        "loss_type": "grpo",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": ["q_proj", "v_proj"],
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.num_generations,
        "max_completion_length": 320,
    }
    (output_dir / "config.json").write_text(json.dumps(config_record, indent=2))

    metrics_record = {
        "global_step": trainer.state.global_step,
        "per_step_loss": losses,
        "per_step_reward": rewards,
        "per_step_reward_std": reward_stds,
        "final_loss": losses[-1],
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_record, indent=2))

    print("\nDay 7 success criteria:")
    print("  [x] model loads")
    print("  [x] dataset loads")
    print("  [x] trainer initializes")
    print("  [x] G responses generated per prompt, each scored (reward logged)")
    print("  [x] group-relative advantage computed (reward_std logged)")
    print("  [x] loss is produced and finite")
    print("  [x] backward pass works (trainable params received gradients)")
    print(f"  [x] {trainer.state.global_step} optimization step(s) completed")
    print("  [x] checkpoint saved")


if __name__ == "__main__":
    main()
