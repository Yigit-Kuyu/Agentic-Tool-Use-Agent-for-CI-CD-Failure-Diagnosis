"""Command-line entry point for DPO-style preference fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the parent directory to sys.path so the 'agi_tool' package can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi_tool.training.dpo import (
    DEFAULT_DPO_INIT_POLICY_PATH,
    DEFAULT_DPO_PREFERENCES_PATH,
    DPOTrainConfig,
    train_dpo_policy,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the BC policy with DPO-style trajectory preferences.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional dataset directory override.")
    parser.add_argument(
        "--preferences",
        type=str,
        default=str(DEFAULT_DPO_PREFERENCES_PATH),
        help="Path to the preference JSONL file, usually dpo_preferences_v2.jsonl.",
    )
    parser.add_argument(
        "--init-policy",
        type=str,
        default=str(DEFAULT_DPO_INIT_POLICY_PATH),
        help="Path to the saved BC policy JSON used as both init and frozen reference.",
    )
    parser.add_argument("--train-split", type=str, default="train", help="Preference split used for training.")
    parser.add_argument("--validation-split", type=str, default="test", help="Preference split used for validation.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of DPO optimization epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="DPO learning rate.")
    parser.add_argument("--beta", type=float, default=0.5, help="DPO temperature / preference sharpness.")
    parser.add_argument("--l2-weight", type=float, default=1e-5, help="L2 regularization coefficient.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for example ordering.")
    parser.add_argument("--max-steps", type=int, default=6, help="Episode step limit when reconstructing states.")
    parser.add_argument(
        "--output",
        type=str,
        default="agi_tool/dpo_policy.json",
        help="Path to save the DPO-fine-tuned policy.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = DPOTrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        l2_weight=args.l2_weight,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    policy, result = train_dpo_policy(
        init_policy_path=args.init_policy,
        preference_path=args.preferences,
        dataset_root=args.dataset_root,
        train_split=args.train_split,
        validation_split=args.validation_split,
        config=config,
    )

    output_path = policy.save(Path(args.output))
    summary = {
        "output_path": str(output_path),
        "train_examples": result.train_examples,
        "val_examples": result.val_examples,
        "train_preference_accuracy": result.train_preference_accuracy,
        "val_preference_accuracy": result.val_preference_accuracy,
        "epochs_completed": result.epochs_completed,
        "final_loss": result.loss_history[-1] if result.loss_history else None,
        "build_summary": {
            "total_records": result.build_summary.total_records,
            "usable_examples": result.build_summary.usable_examples,
            "skipped_examples": result.build_summary.skipped_examples,
            "skipped_by_reason": result.build_summary.skipped_by_reason,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
