"""CLI for side-by-side BC vs RL evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agi_tool.training.eval import compare_policies, write_comparison_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare BC and RL policies on CI/CD diagnosis rollouts.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional dataset directory override.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument("--bc-policy", type=str, default="agi_tool/bc_policy.json", help="BC policy JSON path.")
    parser.add_argument("--rl-policy", type=str, default="agi_tool/rl_policy.json", help="RL policy JSON path.")
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Optional specific case id to evaluate. Repeat for multiple cases.",
    )
    parser.add_argument("--max-steps", type=int, default=5, help="Episode step limit during evaluation.")
    parser.add_argument(
        "--output",
        type=str,
        default="agi_tool/eval_report.json",
        help="Path to save the JSON evaluation report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = compare_policies(
        bc_policy_path=args.bc_policy,
        rl_policy_path=args.rl_policy,
        dataset_root=args.dataset_root,
        split=args.split,
        case_ids=args.case_id,
        max_steps=args.max_steps,
    )
    output_path = write_comparison_report(report, Path(args.output))

    summary = {
        "output_path": str(output_path),
        "split": report.split,
        "bc_average_reward": report.bc_report.aggregate.average_reward,
        "bc_success_rate": report.bc_report.aggregate.success_rate,
        "bc_action_accuracy": report.bc_report.aggregate.action_accuracy,
        "bc_diagnosis_accuracy": report.bc_report.aggregate.diagnosis_accuracy,
        "bc_fix_accuracy": report.bc_report.aggregate.fix_accuracy,
        "rl_average_reward": report.rl_report.aggregate.average_reward,
        "rl_success_rate": report.rl_report.aggregate.success_rate,
        "rl_action_accuracy": report.rl_report.aggregate.action_accuracy,
        "rl_diagnosis_accuracy": report.rl_report.aggregate.diagnosis_accuracy,
        "rl_fix_accuracy": report.rl_report.aggregate.fix_accuracy,
        "reward_gap_rl_minus_bc": report.reward_gap_rl_minus_bc,
        "success_gap_rl_minus_bc": report.success_gap_rl_minus_bc,
        "trajectory_gap_rl_minus_bc": report.trajectory_gap_rl_minus_bc,
        "num_cases": report.bc_report.aggregate.num_cases,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
