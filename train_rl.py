"""Command-line entry point for DQN-style RL fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the parent directory to sys.path so the 'agi_tool' package can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi_tool.training.rl import DEFAULT_BC_POLICY_PATH, RLTrainConfig, train_rl_fine_tuning


DATASET_DEFAULT_BC_POLICIES = {
    "combined_dataset": Path("agi_tool/bc_policy_combined.json"),
    "github_real_dataset_expanded": Path("agi_tool/bc_policy_github_real_expanded.json"),
    "github_real_dataset": Path("agi_tool/bc_policy_github_real.json"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the CI/CD diagnosis policy with a small DQN-style setup."
    )
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional dataset directory override.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to fine-tune on.")
    parser.add_argument(
        "--bc-policy",
        type=str,
        default=str(DEFAULT_BC_POLICY_PATH),
        help="Path to the saved behavior-cloning policy JSON. If omitted, dataset-specific defaults are used when available.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=80,
        help="Number of RL episodes. Tuned default for this small deterministic dataset.",
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Online Q-network learning rate.")
    parser.add_argument("--epsilon-start", type=float, default=0.2, help="Initial epsilon-greedy rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.02, help="Minimum epsilon-greedy rate.")
    parser.add_argument("--epsilon-decay", type=float, default=0.98, help="Per-episode epsilon decay.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 decay on Q-network weights.")
    parser.add_argument("--td-clip", type=float, default=5.0, help="Clip range for TD target errors.")
    parser.add_argument("--evaluation-interval", type=int, default=20, help="Episodes between evaluations.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Holdout fraction of cases.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("--max-steps", type=int, default=5, help="Episode step limit in the environment.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width of the small Q-network MLP.")
    parser.add_argument("--replay-capacity", type=int, default=4096, help="Maximum replay-buffer size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Replay minibatch size.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=32,
        help="Environment steps to collect before replay-based gradient updates begin.",
    )
    parser.add_argument(
        "--target-sync-steps",
        type=int,
        default=50,
        help="Environment-step interval for copying the online network into the target network.",
    )
    parser.add_argument(
        "--gradient-steps-per-env-step",
        type=int,
        default=1,
        help="How many replay minibatch updates to run after each environment step.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Huber-loss transition point for TD regression.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="agi_tool/rl_policy.json",
        help="Path to save the fine-tuned RL policy.",
    )
    return parser


def resolve_bc_policy_path(dataset_root: str | None, bc_policy: str) -> str:
    provided = Path(bc_policy)
    if str(provided) != str(DEFAULT_BC_POLICY_PATH):
        return bc_policy
    if dataset_root is None:
        return bc_policy

    dataset_name = Path(dataset_root).name
    candidate = DATASET_DEFAULT_BC_POLICIES.get(dataset_name)
    if candidate is not None and candidate.exists():
        return str(candidate)
    return bc_policy


def main() -> None:
    args = build_arg_parser().parse_args()
    resolved_bc_policy = resolve_bc_policy_path(args.dataset_root, args.bc_policy)
    config = RLTrainConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        weight_decay=args.weight_decay,
        td_clip=args.td_clip,
        evaluation_interval=args.evaluation_interval,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        target_sync_steps=args.target_sync_steps,
        gradient_steps_per_env_step=args.gradient_steps_per_env_step,
        huber_delta=args.huber_delta,
    )
    policy, result = train_rl_fine_tuning(
        dataset_root=args.dataset_root,
        bc_policy_path=resolved_bc_policy,
        split=args.split,
        config=config,
    )

    output_path = policy.save(Path(args.output))
    summary = {
        "output_path": str(output_path),
        "bc_policy_path": resolved_bc_policy,
        "episodes_completed": result.episodes_completed,
        "final_epsilon": result.final_epsilon,
        "train_case_count": result.train_case_count,
        "val_case_count": result.val_case_count,
        "total_env_steps": result.total_env_steps,
        "replay_size": result.replay_size,
        "final_train_average_reward": result.final_train_eval.average_reward,
        "final_train_success_rate": result.final_train_eval.success_rate,
        "final_val_average_reward": result.final_val_eval.average_reward,
        "final_val_success_rate": result.final_val_eval.success_rate,
        "last_episode_reward": result.reward_history[-1] if result.reward_history else None,
        "evaluation_history": result.evaluation_history,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
