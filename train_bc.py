"""Command-line entry point for behavior-cloning training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the parent directory to sys.path so the 'agi_tool' package can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi_tool.training.bc import train_behavior_cloning_policy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CI/CD diagnosis behavior-cloning policy.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional dataset directory override.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to train on.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Holdout fraction of cases.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for case splitting.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of optimization epochs. Tuned default for this small deterministic dataset.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Gradient descent learning rate.")
    parser.add_argument("--l2-weight", type=float, default=1e-4, help="L2 regularization coefficient.")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden width of the BC residual MLP.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="agi_tool/bc_policy.json",
        help="Path to save the trained policy.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    policy, result = train_behavior_cloning_policy(
        dataset_root=args.dataset_root,
        split=args.split,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_weight=args.l2_weight,
        hidden_dim=args.hidden_dim,
    )

    output_path = policy.save(Path(args.output))
    summary = {
        "output_path": str(output_path),
        "train_examples": result.train_examples,
        "val_examples": result.val_examples,
        "train_accuracy": result.train_accuracy,
        "val_accuracy": result.val_accuracy,
        "epochs_completed": result.epochs_completed,
        "final_loss": result.loss_history[-1] if result.loss_history else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
