#!/usr/bin/env python3

"""Run the controlled failure-generation loop as one command.

Stages:

1. Apply one controlled failure template to CI_CD_Workflow.
2. Create branch / commit / optionally push to GitHub.
3. Optionally collect failed workflow logs from GitHub.
4. Optionally convert collected failures into dataset rows.

This is intentionally resumable because GitHub Actions runs are asynchronous and
large campaigns should be executed in batches rather than as one fragile shot.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_REPO_ROOT = Path("CI_CD_Workflow")
DEFAULT_OUTPUT_ROOT = Path("agi_tool/github_real_dataset")
PREPARE_SCRIPT_PATH = Path("agi_tool/dataops/prepare_failure_mutation.py")
COLLECT_SCRIPT_PATH = Path("agi_tool/dataops/github_repo_take_failure.py")
CONVERT_SCRIPT_PATH = Path("agi_tool/dataops/github_failures_to_dataset.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the controlled GitHub failure loop.")
    parser.add_argument(
        "--template",
        required=True,
        help="Case template id, for example import_path_error_1.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Target Git repository.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Dataset output directory for the conversion stage.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the generated branch to origin after commit.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Run github_repo_take_failure.py after push. Requires GITHUB_TOKEN.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Run github_failures_to_dataset.py after collection.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help="How long to wait before collection, to give GitHub Actions time to run.",
    )
    parser.add_argument(
        "--branch-prefix",
        default="generated-failure",
        help="Prefix for generated branches.",
    )
    parser.add_argument(
        "--commit-prefix",
        default="generated_failure",
        help="Prefix for generated commit messages.",
    )
    parser.add_argument(
        "--name-suffix",
        default=None,
        help="Optional suffix appended to generated branch and commit names.",
    )
    parser.add_argument(
        "--keep-branch",
        action="store_true",
        help="Do not switch back to the original branch after the commit stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the full loop without modifying the repo, committing, pushing, or collecting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root)
    output_root = Path(args.output_root)

    original_branch = git_output(repo_root, "branch", "--show-current").strip() or "main"
    clean_worktree_or_die(repo_root)

    prepare_cmd = [str(PREPARE_SCRIPT_PATH), "--template", args.template]
    if args.dry_run:
        prepare_cmd.append("--dry-run")
    prepare_summary = run_python(prepare_cmd)
    prepare_payload = parse_last_json(prepare_summary)
    suffix = args.name_suffix or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    branch_name = f"{args.branch_prefix}/{args.template}-{suffix}"
    commit_message = f"{args.commit_prefix}:{args.template}:{suffix}"

    if args.dry_run:
        summary = {
            "template_id": args.template,
            "branch_name": branch_name,
            "commit_message": commit_message,
            "dry_run": True,
            "repo_root": str(repo_root),
            "output_root": str(output_root),
            "original_branch": original_branch,
            "changed_files": prepare_payload["changed_files"],
            "would_run": {
                "push": args.push,
                "collect": args.collect,
                "convert": args.convert,
            },
            "next_step_hint": "Remove --dry-run to actually apply the loop.",
        }
        print(json.dumps(summary, indent=2))
        return

    run_git(repo_root, "checkout", "-b", branch_name)
    for path in prepare_payload["changed_files"]:
        run_git(repo_root, "add", path)
    run_git(repo_root, "commit", "-m", commit_message)

    pushed = False
    if args.push:
        run_git(repo_root, "push", "-u", "origin", branch_name)
        pushed = True

    collected = False
    if args.collect:
        if not pushed:
            raise SystemExit("--collect requires --push because GitHub must see the commit first.")
        if not os.environ.get("GITHUB_TOKEN"):
            raise SystemExit("--collect requires GITHUB_TOKEN in the environment.")
        if args.wait_seconds > 0:
            time.sleep(args.wait_seconds)
        run_python([str(COLLECT_SCRIPT_PATH)])
        collected = True

    converted = False
    if args.convert:
        if not collected and args.collect:
            raise SystemExit("Collection failed before conversion.")
        run_python(
            [
                str(CONVERT_SCRIPT_PATH),
                "--repo-root",
                str(repo_root),
                "--output-root",
                str(output_root),
            ]
        )
        converted = True

    if not args.keep_branch:
        run_git(repo_root, "checkout", original_branch)

    summary = {
        "template_id": args.template,
        "branch_name": branch_name,
        "commit_message": commit_message,
        "dry_run": False,
        "pushed": pushed,
        "collected": collected,
        "converted": converted,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "original_branch": original_branch,
        "next_step_hint": next_step_hint(args.push, args.collect, args.convert, branch_name),
    }
    print(json.dumps(summary, indent=2))


def clean_worktree_or_die(repo_root: Path) -> None:
    status = git_output(repo_root, "status", "--short").strip()
    if status:
        raise SystemExit(
            "CI_CD_Workflow has uncommitted changes. Commit or stash them before running the loop."
        )


def git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def run_git(repo_root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo_root), *args], check=True, stdout=sys.stderr)


def run_python(args: list[str]) -> str:
    completed = subprocess.run([sys.executable, *args], text=True, capture_output=True, check=True)
    return completed.stdout


def parse_last_json(text: str) -> dict[str, object]:
    text = text.strip()
    if not text:
        raise SystemExit("Expected JSON output from helper script but received nothing.")
    return json.loads(text)


def next_step_hint(pushed: bool, collected: bool, converted: bool, branch_name: str) -> str:
    if not pushed:
        return f"Push branch {branch_name} when you are ready to trigger GitHub Actions."
    if pushed and not collected:
        return "Wait for the GitHub Actions run to fail, then rerun with --collect."
    if collected and not converted:
        return "Failure logs were collected; rerun with --convert to refresh the dataset bundle."
    return "Loop completed. Review the generated dataset rows before starting a large campaign."


if __name__ == "__main__":
    main()
