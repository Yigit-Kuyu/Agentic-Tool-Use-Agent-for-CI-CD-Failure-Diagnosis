#!/usr/bin/env python3

"""Convert collected GitHub Actions failures into a draft dataset bundle.

This script is the bridge between:

1. `github_repo_take_failure.py` / `github_failure_summary.md`
2. the JSONL dataset schema used by `agi_tool`

It keeps the output separate from the checked-in synthetic dataset so the real-data
pipeline can evolve without overwriting the baseline assets.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_PATH = Path("agi_tool/github_failure_summary.md")
DEFAULT_REPO_PATH = Path("CI_CD_Workflow")
DEFAULT_OUTPUT_ROOT = Path("agi_tool/github_real_dataset")
DEFAULT_DOCS_SOURCE = Path("agi_tool/dataset/docs_kb.jsonl")

TRAINING_ACTIONS = [
    "retrieve_logs",
    "inspect_repo",
    "search_docs",
    "run_static_check",
    "final_answer",
]

STATIC_CHECK_BY_CATEGORY = {
    "missing_dependency": "dependency_or_requirements_path_problem_detected",
    "wrong_python_version": "python_version_mismatch_detected",
    "import_path_error": "internal_import_path_missing",
    "file_not_found": "referenced_file_missing",
    "bad_github_actions_yaml": "github_actions_yaml_invalid",
    "docker_build_failure": "dockerfile_build_context_or_dependency_problem",
    "failed_unit_test": "unit_test_logic_or_shape_failure_detected",
    "missing_env_variable": "required_environment_variable_missing",
}

EXPERT_ACTIONS_BY_CATEGORY = {
    "missing_dependency": ["retrieve_logs", "inspect_repo", "search_docs", "final_answer"],
    "wrong_python_version": ["retrieve_logs", "inspect_repo", "search_docs", "final_answer"],
    "docker_build_failure": ["retrieve_logs", "inspect_repo", "search_docs", "final_answer"],
    "missing_env_variable": ["retrieve_logs", "inspect_repo", "search_docs", "final_answer"],
    "import_path_error": ["retrieve_logs", "inspect_repo", "run_static_check", "final_answer"],
    "file_not_found": ["retrieve_logs", "inspect_repo", "run_static_check", "final_answer"],
    "bad_github_actions_yaml": ["retrieve_logs", "inspect_repo", "run_static_check", "final_answer"],
    "failed_unit_test": ["retrieve_logs", "inspect_repo", "run_static_check", "final_answer"],
}

KNOWN_CATEGORIES = tuple(EXPERT_ACTIONS_BY_CATEGORY)

SUMMARY_SECTION_RE = re.compile(
    r"^## Run #(?P<run_number>\d+)\s+—\s+(?P<title>.+?)\n\n"
    r"- Run ID: `(?P<run_id>\d+)`\n"
    r"- Commit: `(?P<commit>[0-9a-f]+)`\n"
    r"- Created: `(?P<created_at>[^`]+)`\n"
    r"- Reason type: `(?P<reason_type>[^`]+)`\n"
    r"- Failure reason: `(?P<reason>[^`]+)`\n"
    r"- Log file: `(?P<log_file>[^`]+)`\n"
    r"- URL: (?P<url>\S+)\n\n"
    r"```text\n(?P<context>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)

ENV_VAR_RE = re.compile(r"Missing required environment variable:\s*([A-Za-z_][A-Za-z0-9_]*)", re.I)
MODULE_RE = re.compile(r"(?:ModuleNotFoundError|ImportError):.*No module named ['\"]?([A-Za-z0-9_\.]+)['\"]?", re.I)
FILE_RE = re.compile(r"(?:FileNotFoundError|No such file or directory):\s*['\"]?([^'\"\n]+)['\"]?", re.I)
PYTHON_VERSION_RE = re.compile(r"Python\s+(3\.\d+)", re.I)


@dataclass(frozen=True)
class FailureRecord:
    run_number: int
    title: str
    run_id: str
    commit: str
    created_at: str
    reason_type: str
    reason: str
    log_file: str
    url: str
    context: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert GitHub Actions failure summaries into an agi_tool dataset bundle."
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to github_failure_summary.md.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(DEFAULT_REPO_PATH),
        help="Path to the local git repo that contains the failing commits.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where the generated dataset files will be written.",
    )
    parser.add_argument(
        "--docs-source",
        type=str,
        default=str(DEFAULT_DOCS_SOURCE),
        help="Existing docs_kb.jsonl to reuse for search_docs observations.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary_path = Path(args.summary)
    repo_root = Path(args.repo_root)
    output_root = Path(args.output_root)
    docs_source = Path(args.docs_source)

    records = parse_failure_summary(summary_path)
    docs_by_category = load_docs_by_category(docs_source)
    dataset = build_dataset(records, repo_root=repo_root, docs_by_category=docs_by_category)
    write_dataset(output_root, dataset, docs_source=docs_source)

    summary = {
        "output_root": str(output_root),
        "num_cases": len(dataset["failure_cases"]),
        "categories": sorted({record["category"] for record in dataset["failure_cases"]}),
        "splits": summarize_splits(dataset["failure_cases"]),
        "note": "Draft real-data dataset generated from GitHub failure summaries plus repo commit snapshots.",
    }
    print(json.dumps(summary, indent=2))


def parse_failure_summary(summary_path: Path) -> list[FailureRecord]:
    text = summary_path.read_text(encoding="utf-8")
    deduped: dict[str, FailureRecord] = {}

    for match in SUMMARY_SECTION_RE.finditer(text):
        record = FailureRecord(
            run_number=int(match.group("run_number")),
            title=match.group("title").strip(),
            run_id=match.group("run_id"),
            commit=match.group("commit"),
            created_at=match.group("created_at"),
            reason_type=match.group("reason_type"),
            reason=match.group("reason").strip(),
            log_file=match.group("log_file").strip(),
            url=match.group("url").strip(),
            context=match.group("context").strip(),
        )
        deduped.setdefault(record.run_id, record)

    records = sorted(deduped.values(), key=lambda item: item.run_number)
    if not records:
        raise ValueError(f"No failure records could be parsed from {summary_path}.")
    return records


def load_docs_by_category(path: Path) -> dict[str, str]:
    docs_by_category: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docs_by_category[str(record["category"])] = str(record["content"])
    return docs_by_category


def build_dataset(
    records: list[FailureRecord],
    *,
    repo_root: Path,
    docs_by_category: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    available_case_ids = [build_case_id(record) for record in records]
    test_count = max(1, round(len(records) * 0.2))
    test_case_ids = set(available_case_ids[-test_count:])

    failure_cases: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    repo_snapshots: list[dict[str, Any]] = []
    tool_observations: list[dict[str, Any]] = []
    expert_trajectories: list[dict[str, Any]] = []
    dpo_preferences: list[dict[str, Any]] = []

    for record in records:
        case_id = build_case_id(record)
        category = infer_category(record)
        split = "test" if case_id in test_case_ids else "train"
        repo_tree = git_ls_tree(repo_root, record.commit)
        relevant_files = infer_relevant_files(category, record)
        snippets = build_file_snippets(repo_root, record.commit, relevant_files, record)
        diagnosis = infer_root_cause(category, record, snippets)
        fix = infer_expected_fix(category, record, snippets)
        evidence = build_evidence(record, relevant_files)
        user_query = (
            f"My GitHub Actions workflow failed in repository {repo_root.name}. "
            "Diagnose the root cause and suggest a fix."
        )
        docs_text = docs_by_category.get(category, docs_by_category.get("failed_unit_test", ""))

        failure_cases.append(
            {
                "case_id": case_id,
                "category": category,
                "run_index": record.run_number,
                "source": "github_real_failure",
                "repo_name": repo_root.name,
                "project_dir": "ml_cicd_demo",
                "user_query": user_query,
                "failure_type": category,
                "failure_signal": record.reason,
                "root_cause": diagnosis,
                "expected_fix": fix,
                "related_files": list(relevant_files),
                "log_id": f"log__{case_id}",
                "split": split,
                "web_augmentation_query": f"{record.reason} GitHub Actions fix",
            }
        )

        logs.append(
            {
                "case_id": case_id,
                "log_id": f"log__{case_id}",
                "source": "github_real_failure",
                "raw_log": record.context,
            }
        )

        repo_snapshots.append(
            {
                "case_id": case_id,
                "source": "github_real_failure",
                "snapshot_status": "reconstructed_from_local_git_commit",
                "repo_tree": repo_tree,
                "available_paths_from_document": git_list_paths(repo_root, record.commit),
                "relevant_files": list(relevant_files),
                "file_snippets": snippets,
                "note": (
                    "Snapshot reconstructed automatically from the local git history at the failing commit. "
                    "These snippets are heuristic and should be reviewed before treating them as gold labels."
                ),
            }
        )

        tool_observations.append(
            {
                "case_id": case_id,
                "observations": {
                    "retrieve_logs": record.context,
                    "inspect_repo": {
                        "relevant_files": list(relevant_files),
                        "file_snippets": snippets,
                    },
                    "search_docs": docs_text,
                    "run_static_check": STATIC_CHECK_BY_CATEGORY[category],
                },
                "target_final_answer": {
                    "diagnosis": diagnosis,
                    "fix": fix,
                    "evidence": evidence,
                },
            }
        )

        expert_actions = list(EXPERT_ACTIONS_BY_CATEGORY[category])
        expert_trajectories.append(
            {
                "case_id": case_id,
                "source": "github_real_failure_heuristic",
                "expert_actions": expert_actions,
                "steps": build_expert_steps(expert_actions),
                "supervised_labels": [
                    {"input_state_index": index, "target_action": action}
                    for index, action in enumerate(expert_actions)
                ],
            }
        )

        dpo_preferences.append(
            {
                "case_id": case_id,
                "source": "github_real_failure_heuristic",
                "prompt": (
                    f"User query: {user_query}\n"
                    "Available actions: retrieve_logs, inspect_repo, search_docs, run_static_check, final_answer.\n"
                    "Choose a diagnostic trajectory and final answer."
                ),
                "chosen": (
                    f"{' -> '.join(expert_actions)}\n"
                    f"FINAL: {diagnosis} Fix: {fix}"
                ),
                "rejected": (
                    "final_answer\n"
                    "FINAL: Give generic CI/CD advice without checking the actual log and repository evidence."
                ),
                "preference_reason": (
                    "Chosen trajectory inspects case-specific evidence before answering; "
                    "rejected trajectory is premature and generic."
                ),
            }
        )

    return {
        "failure_cases": failure_cases,
        "logs": logs,
        "repo_snapshots": repo_snapshots,
        "tool_observations": tool_observations,
        "expert_trajectories": expert_trajectories,
        "dpo_preferences": dpo_preferences,
    }


def build_case_id(record: FailureRecord) -> str:
    slug = slugify(record.title)
    return f"github__{slug}__run_{record.run_number:02d}"


def infer_category(record: FailureRecord) -> str:
    title_prefix = slugify(record.title)
    for category in KNOWN_CATEGORIES:
        if title_prefix.startswith(category):
            return category

    lower_reason = record.reason.lower()
    lower_context = record.context.lower()
    if record.reason_type == "missing_required_env" or "missing required environment variable" in lower_reason:
        return "missing_env_variable"
    if record.reason_type.startswith("docker_"):
        return "docker_build_failure"
    if "invalid workflow" in lower_reason or ".github/workflows" in lower_context:
        return "bad_github_actions_yaml"
    if "modulenotfounderror" in lower_context or "importerror" in lower_context:
        module = MODULE_RE.search(record.context)
        if module and module.group(1).split(".")[0] in {"app", "src", "ml_cicd_demo"}:
            return "import_path_error"
        return "missing_dependency"
    if "filenotfounderror" in lower_context or "no such file or directory" in lower_context:
        return "file_not_found"
    if "requires python" in lower_reason or "python-version" in lower_context or "syntaxerror" in lower_context:
        return "wrong_python_version"
    return "failed_unit_test"


def infer_relevant_files(category: str, record: FailureRecord) -> list[str]:
    relevant = [".github/workflows/ci.yml"]

    if category in {"missing_env_variable", "docker_build_failure"}:
        relevant.extend(["ml_cicd_demo/Dockerfile", "ml_cicd_demo/requirements.txt"])
    if category in {"failed_unit_test", "missing_env_variable", "wrong_python_version", "missing_dependency"}:
        relevant.extend(["ml_cicd_demo/tests/test_model.py", "ml_cicd_demo/app/model.py"])
    if category in {"missing_dependency", "wrong_python_version"}:
        relevant.append("ml_cicd_demo/requirements.txt")
    if category == "file_not_found":
        relevant.append("ml_cicd_demo/tests/test_api.py")

    return dedupe_preserve(relevant)


def build_file_snippets(
    repo_root: Path,
    commit: str,
    relevant_files: list[str],
    record: FailureRecord,
) -> dict[str, str]:
    snippets: dict[str, str] = {}
    for path in relevant_files:
        content = git_show_file(repo_root, commit, path)
        if content is None:
            continue
        snippet = extract_relevant_snippet(content, record)
        if snippet:
            snippets[normalize_dataset_path(path)] = snippet
    return snippets


def infer_root_cause(category: str, record: FailureRecord, snippets: dict[str, str]) -> str:
    if category == "missing_env_variable":
        variable = extract_env_var(record) or "the required environment variable"
        if "Dockerfile" in record.context:
            return (
                f"The Docker build requires environment variable {variable}, but it is not provided in CI."
            )
        return (
            f"The application or tests require environment variable {variable}, but it is not defined in CI."
        )
    if category == "failed_unit_test":
        if "assert false" in record.context.lower():
            return "The unit test intentionally asserts False, so the workflow fails even though the code executes."
        return "A unit test assertion does not match the current implementation behavior."
    if category == "docker_build_failure":
        return "The Docker image build fails because a Dockerfile step or dependency install step is invalid."
    if category == "wrong_python_version":
        return "The workflow Python runtime does not match the code or dependency requirements."
    if category == "missing_dependency":
        module = extract_missing_module(record) or "a required package"
        return f"{module} is required by the code or tests but is not available in the CI environment."
    if category == "import_path_error":
        module = extract_missing_module(record) or "an internal module"
        return f"The code imports {module}, but that internal module path does not match the repository layout."
    if category == "file_not_found":
        missing = extract_missing_file(record) or "a referenced file"
        return f"The workflow or application references {missing}, but that file path is missing or incorrect."
    if category == "bad_github_actions_yaml":
        return "The GitHub Actions workflow configuration is invalid."
    return record.reason


def infer_expected_fix(category: str, record: FailureRecord, snippets: dict[str, str]) -> str:
    if category == "missing_env_variable":
        variable = extract_env_var(record) or "the required environment variable"
        if "Dockerfile" in record.context:
            return f"Set {variable} in the workflow environment or Docker build arguments before the build step."
        return f"Define {variable} in GitHub Actions env/secrets before running the tests."
    if category == "failed_unit_test":
        if "assert false" in record.context.lower():
            return "Remove the intentional failing assertion or replace it with a real expected condition."
        return "Update the implementation or the test expectation so the assertion matches the intended behavior."
    if category == "docker_build_failure":
        return "Inspect the Dockerfile step shown in the log and fix the failing build command or dependency setup."
    if category == "wrong_python_version":
        version_match = PYTHON_VERSION_RE.search(record.reason)
        if version_match:
            return f"Set actions/setup-python to Python {version_match.group(1)} or newer."
        return "Align the GitHub Actions Python version with the runtime required by the code."
    if category == "missing_dependency":
        module = extract_missing_module(record) or "the missing package"
        return f"Add {module} to requirements.txt or install it in the workflow before running tests."
    if category == "import_path_error":
        return "Fix the import path to match the repository package structure and ensure packages are defined correctly."
    if category == "file_not_found":
        missing = extract_missing_file(record) or "the missing file"
        return f"Add {missing}, correct its path, or fix the workflow working directory."
    if category == "bad_github_actions_yaml":
        return "Fix the syntax or structure of .github/workflows/ci.yml."
    return f"Inspect and fix the underlying CI/CD issue reported as: {record.reason}"


def build_evidence(record: FailureRecord, relevant_files: list[str]) -> list[str]:
    evidence = [record.reason]
    evidence.extend(normalize_dataset_path(path) for path in relevant_files[:2])
    return evidence[:3]


def build_expert_steps(actions: list[str]) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for index, action in enumerate(actions):
        if index == 0:
            summary = "No evidence yet; first inspect the workflow failure log."
        elif action == "inspect_repo":
            summary = "Use the log signal to inspect the relevant repository files at the failing commit."
        elif action == "search_docs":
            summary = "Consult internal docs for a known CI/CD failure pattern."
        elif action == "run_static_check":
            summary = "Use the deterministic static check to confirm the failure pattern."
        else:
            summary = "Enough evidence has been collected to provide the diagnosis and fix."
        steps.append({"state_summary": summary, "action": action})
    return steps


def extract_relevant_snippet(content: str, record: FailureRecord) -> str:
    terms = []
    variable = extract_env_var(record)
    if variable:
        terms.append(variable)
    module = extract_missing_module(record)
    if module:
        terms.append(module)
    missing_file = extract_missing_file(record)
    if missing_file:
        terms.append(missing_file)
    terms.extend(
        token
        for token in [
            "assert False",
            "RuntimeError",
            "python-version",
            "requirements.txt",
            "docker build",
            "Missing required environment variable",
        ]
    )

    lines = content.splitlines()
    for index, line in enumerate(lines):
        if any(term and term.lower() in line.lower() for term in terms):
            start = max(0, index - 2)
            end = min(len(lines), index + 4)
            return "\n".join(lines[start:end]).strip()

    return "\n".join(lines[: min(20, len(lines))]).strip()


def extract_env_var(record: FailureRecord) -> str | None:
    match = ENV_VAR_RE.search(record.reason) or ENV_VAR_RE.search(record.context)
    return match.group(1) if match else None


def extract_missing_module(record: FailureRecord) -> str | None:
    match = MODULE_RE.search(record.reason) or MODULE_RE.search(record.context)
    return match.group(1) if match else None


def extract_missing_file(record: FailureRecord) -> str | None:
    match = FILE_RE.search(record.reason) or FILE_RE.search(record.context)
    return match.group(1) if match else None


def git_show_file(repo_root: Path, commit: str, path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout


def git_list_paths(repo_root: Path, commit: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-tree", "-r", "--name-only", commit],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def git_ls_tree(repo_root: Path, commit: str) -> str:
    paths = git_list_paths(repo_root, commit)
    return "\n".join(paths)


def normalize_dataset_path(path: str) -> str:
    return path.replace("ml_cicd_demo/", "")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def summarize_splits(failure_cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in failure_cases:
        split = str(record.get("split", "unknown"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def write_dataset(output_root: Path, dataset: dict[str, list[dict[str, Any]]], *, docs_source: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_root / "failure_cases.jsonl", dataset["failure_cases"])
    write_jsonl(output_root / "logs.jsonl", dataset["logs"])
    write_jsonl(output_root / "repo_snapshots.jsonl", dataset["repo_snapshots"])
    write_jsonl(output_root / "tool_observations.jsonl", dataset["tool_observations"])
    write_jsonl(output_root / "expert_trajectories.jsonl", dataset["expert_trajectories"])
    write_jsonl(output_root / "dpo_preferences.jsonl", dataset["dpo_preferences"])
    (output_root / "docs_kb.jsonl").write_text(docs_source.read_text(encoding="utf-8"), encoding="utf-8")
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(build_manifest(dataset), indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def build_manifest(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    categories = sorted({record["category"] for record in dataset["failure_cases"]})
    return {
        "created_at": "2026-06-13T00:00:00Z",
        "source_document": "github_failure_summary.md + local git commit snapshots",
        "project": "GitHub failure-derived draft dataset for tool-use diagnosis",
        "num_categories": len(categories),
        "categories": categories,
        "cases_per_category": None,
        "total_cases": len(dataset["failure_cases"]),
        "provided_cases": len(dataset["failure_cases"]),
        "synthetic_extension_cases": 0,
        "datasets": {
            "failure_cases.jsonl": len(dataset["failure_cases"]),
            "logs.jsonl": len(dataset["logs"]),
            "repo_snapshots.jsonl": len(dataset["repo_snapshots"]),
            "tool_observations.jsonl": len(dataset["tool_observations"]),
            "expert_trajectories.jsonl": len(dataset["expert_trajectories"]),
            "dpo_preferences.jsonl": len(dataset["dpo_preferences"]),
        },
        "training_action_space": list(TRAINING_ACTIONS),
        "test_time_only_tool": "web_search",
        "note": (
            "This dataset was generated automatically from GitHub Actions failure summaries and local git history. "
            "Root-cause labels, fixes, and expert trajectories are heuristic drafts and should be reviewed."
        ),
    }


if __name__ == "__main__":
    main()
