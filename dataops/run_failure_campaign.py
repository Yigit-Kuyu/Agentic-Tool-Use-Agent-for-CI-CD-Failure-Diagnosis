#!/usr/bin/env python3

"""Run a multi-template GitHub failure campaign.

This is the missing automation layer above `run_failure_loop.py`:

- choose many distinct case templates
- run them one by one
- optionally push each one
- optionally collect new GitHub failures
- optionally rebuild the dataset after each case

It is designed for batch collection such as 10, 20, or 50 controlled failures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_CAMPAIGN_ROOT = Path("agi_tool/generated_failure_campaigns")
PREPARE_SCRIPT_PATH = Path("agi_tool/dataops/prepare_failure_mutation.py")
RUN_LOOP_SCRIPT_PATH = Path("agi_tool/dataops/run_failure_loop.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch of controlled failure templates.")
    parser.add_argument(
        "--templates",
        type=str,
        default="all",
        help="Comma-separated template ids, or 'all'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit after template selection, useful for first batches like 5 or 20.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push each generated branch to GitHub.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect GitHub failures after each pushed case.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Rebuild the GitHub-derived dataset after each case.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help="Wait time before collection for each case.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the campaign when one template fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the selected campaign without modifying anything.",
    )
    parser.add_argument(
        "--campaign-root",
        type=str,
        default=str(DEFAULT_CAMPAIGN_ROOT),
        help="Directory where campaign reports are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    available_templates = list_available_templates()
    selected_templates = select_templates(
        available_templates=available_templates,
        selector=args.templates,
        limit=args.limit,
    )
    campaign_root = Path(args.campaign_root)
    campaign_root.mkdir(parents=True, exist_ok=True)
    campaign_id = datetime.now(UTC).strftime("campaign_%Y%m%dT%H%M%SZ")
    report_path = campaign_root / f"{campaign_id}.json"

    if args.dry_run:
        payload = {
            "campaign_id": campaign_id,
            "dry_run": True,
            "selected_templates": selected_templates,
            "count": len(selected_templates),
            "would_run": {
                "push": args.push,
                "collect": args.collect,
                "convert": args.convert,
                "wait_seconds": args.wait_seconds,
            },
            "report_path": str(report_path),
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    results: list[dict[str, Any]] = []
    for index, template_id in enumerate(selected_templates, start=1):
        command = build_loop_command(
            template_id=template_id,
            push=args.push,
            collect=args.collect,
            convert=args.convert,
            wait_seconds=args.wait_seconds,
            name_suffix=campaign_id,
        )
        try:
            output = run_command(command)
            result_payload = parse_last_json(output)
            results.append(
                {
                    "index": index,
                    "template_id": template_id,
                    "status": "success",
                    "result": result_payload,
                }
            )
        except subprocess.CalledProcessError as exc:
            error_payload = {
                "index": index,
                "template_id": template_id,
                "status": "failed",
                "returncode": exc.returncode,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            }
            results.append(error_payload)
            if not args.continue_on_error:
                break

        report = {
            "campaign_id": campaign_id,
            "dry_run": False,
            "selected_templates": selected_templates,
            "completed": len(results),
            "results": results,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    final_report = {
        "campaign_id": campaign_id,
        "dry_run": False,
        "selected_templates": selected_templates,
        "completed": len(results),
        "successes": sum(1 for item in results if item["status"] == "success"),
        "failures": sum(1 for item in results if item["status"] == "failed"),
        "results": results,
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(json.dumps(final_report, indent=2))


def list_available_templates() -> list[str]:
    spec = importlib.util.spec_from_file_location("prepare_failure_mutation", PREPARE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load template definitions from {PREPARE_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return [item.mutation_id for item in module.MUTATIONS]


def select_templates(*, available_templates: list[str], selector: str, limit: int | None) -> list[str]:
    if selector.strip().lower() == "all":
        selected = list(available_templates)
    else:
        requested = [part.strip() for part in selector.split(",") if part.strip()]
        unknown = [item for item in requested if item not in available_templates]
        if unknown:
            raise SystemExit(f"Unknown template ids: {', '.join(unknown)}")
        selected = requested

    if limit is not None:
        if limit <= 0:
            raise SystemExit("--limit must be positive when provided.")
        selected = selected[:limit]

    if not selected:
        raise SystemExit("No templates selected for the campaign.")
    return selected


def build_loop_command(
    *,
    template_id: str,
    push: bool,
    collect: bool,
    convert: bool,
    wait_seconds: int,
    name_suffix: str,
) -> list[str]:
    command = [
        sys.executable,
        str(RUN_LOOP_SCRIPT_PATH),
        "--template",
        template_id,
        "--name-suffix",
        name_suffix,
    ]
    if push:
        command.append("--push")
    if collect:
        command.append("--collect")
    if convert:
        command.append("--convert")
    if wait_seconds != 120:
        command.extend(["--wait-seconds", str(wait_seconds)])
    return command


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    return completed.stdout


def parse_last_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise SystemExit("Expected JSON output from helper script but received nothing.")
    return json.loads(text)


if __name__ == "__main__":
    main()
