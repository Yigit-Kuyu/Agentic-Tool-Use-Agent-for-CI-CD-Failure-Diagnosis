"""CLI for post-diagnosis web augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agi_tool.runtime.web_aug import (
    augment_diagnosis_from_result_path,
    render_web_augmented_answer,
    run_and_augment_diagnosis,
    save_web_augmentation_result,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich a diagnosis result with test-time web search.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional dataset directory override.")
    parser.add_argument(
        "--diagnosis-result",
        type=str,
        default=None,
        help="Existing diagnosis JSON to augment. If omitted, a fresh diagnosis run is executed first.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="agi_tool/rl_policy.json",
        help="Path to BC or RL policy JSON when --diagnosis-result is not provided.",
    )
    parser.add_argument("--case-id", type=str, default=None, help="Specific case id to diagnose.")
    parser.add_argument("--split", type=str, default="test", help="Split to sample from when case-id is omitted.")
    parser.add_argument("--max-steps", type=int, default=5, help="Episode step limit during inference.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for split sampling.")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of web search results to keep.")
    parser.add_argument(
        "--output",
        type=str,
        default="agi_tool/augmented_diagnosis_result.json",
        help="Path to save the augmented diagnosis result.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.diagnosis_result:
        result = augment_diagnosis_from_result_path(
            args.diagnosis_result,
            max_results=args.max_results,
        )
    else:
        result = run_and_augment_diagnosis(
            policy_path=args.policy,
            dataset_root=args.dataset_root,
            case_id=args.case_id,
            split=args.split,
            max_steps=args.max_steps,
            seed=args.seed,
            max_results=args.max_results,
        )

    output_path = save_web_augmentation_result(result, Path(args.output))
    summary = {
        "output_path": str(output_path),
        "case_id": result.diagnosis_result.case_id,
        "category": result.diagnosis_result.category,
        "search_provider": result.search_provider,
        "search_status": result.search_status,
        "search_error": result.search_error,
        "web_hit_count": len(result.web_hits),
        "rendered_answer": render_web_augmented_answer(result),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
