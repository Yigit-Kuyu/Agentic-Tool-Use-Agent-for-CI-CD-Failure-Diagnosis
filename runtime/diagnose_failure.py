"""CLI for running policy inference on a failure case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agi_tool.runtime.inference import render_final_answer, run_diagnosis_from_path, save_diagnosis_result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a saved BC or RL policy on a diagnosis case.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional dataset directory override.")
    parser.add_argument("--policy", type=str, default="agi_tool/rl_policy.json", help="Path to BC or RL policy JSON.")
    parser.add_argument("--case-id", type=str, default=None, help="Specific case id to diagnose.")
    parser.add_argument("--split", type=str, default="test", help="Split to sample from when case-id is omitted.")
    parser.add_argument("--max-steps", type=int, default=5, help="Episode step limit during inference.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for split sampling.")
    parser.add_argument(
        "--output",
        type=str,
        default="agi_tool/diagnosis_result.json",
        help="Path to save the structured diagnosis result.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_diagnosis_from_path(
        policy_path=args.policy,
        dataset_root=args.dataset_root,
        case_id=args.case_id,
        split=args.split,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    output_path = save_diagnosis_result(result, Path(args.output))
    summary = {
        "output_path": str(output_path),
        "case_id": result.case_id,
        "category": result.category,
        "split": result.split,
        "tool_history": result.tool_history,
        "total_reward": result.total_reward,
        "final_answer_scores": result.final_answer_scores,
        "web_augmentation_query": result.web_augmentation_query,
        "rendered_answer": render_final_answer(result),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
