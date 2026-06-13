#!/usr/bin/env python3

"""Prepare one controlled GitHub-failure mutation in CI_CD_Workflow.

This is step 1 of the loop:

1. Change the repo in a controlled way.
2. Push that change to GitHub.
3. Collect the resulting failure logs.
4. Convert the failure into dataset rows.

The script focuses on step 1 and writes a manifest so later collection steps
know the intended root cause and expected fix.

Terminology:

- "template" means a reusable failure pattern, for example `import_path_error`
- "case template id" means a concrete template variant, for example
  `import_path_error_1`
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_REPO_ROOT = Path("CI_CD_Workflow")
DEFAULT_MANIFEST_ROOT = Path("agi_tool/generated_failure_manifests")


@dataclass(frozen=True)
class FileMutation:
    path: str
    old: str
    new: str


@dataclass(frozen=True)
class MutationSpec:
    mutation_id: str
    failure_type: str
    description: str
    expected_reason: str
    expected_fix: str
    file_mutations: tuple[FileMutation, ...]


MUTATIONS: tuple[MutationSpec, ...] = (
    MutationSpec(
        mutation_id="import_path_error_1",
        failure_type="import_path_error",
        description="Break the Python import path used by the unit test.",
        expected_reason="ModuleNotFoundError for app.models because the package path is wrong.",
        expected_fix="Restore the import to app.model or fix the package/module layout.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/tests/test_model.py",
                old="from app.model import load_model\n",
                new="from app.models import load_model\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="import_path_error_2",
        failure_type="import_path_error",
        description="Import load_model from a missing top-level module in the unit test.",
        expected_reason="ModuleNotFoundError because model is not importable from the test root.",
        expected_fix="Restore the import to app.model.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/tests/test_model.py",
                old="from app.model import load_model\n",
                new="from model import load_model\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="import_path_error_3",
        failure_type="import_path_error",
        description="Import load_model from the wrong package module in the unit test.",
        expected_reason="ImportError because app.main does not export load_model.",
        expected_fix="Restore the import to app.model.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/tests/test_model.py",
                old="from app.model import load_model\n",
                new="from app.main import load_model\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="failed_unit_test_1",
        failure_type="failed_unit_test",
        description="Change the test assertion so the model output shape check fails.",
        expected_reason="AssertionError because the expected output shape no longer matches the model.",
        expected_fix="Restore the expected shape to the correct value or change the model intentionally.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/tests/test_model.py",
                old="    assert y.shape == (1, 2)\n",
                new="    assert y.shape == (1, 3)\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="failed_unit_test_2",
        failure_type="failed_unit_test",
        description="Change the expected batch shape in the unit test.",
        expected_reason="AssertionError because the asserted output shape is wrong.",
        expected_fix="Restore the expected shape to (1, 2).",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/tests/test_model.py",
                old="    assert y.shape == (1, 2)\n",
                new="    assert y.shape == (2, 2)\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="failed_unit_test_3",
        failure_type="failed_unit_test",
        description="Feed the model an input tensor with the wrong feature dimension.",
        expected_reason="RuntimeError during the forward pass because the input shape no longer matches the model.",
        expected_fix="Restore the input tensor shape to (1, 4).",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/tests/test_model.py",
                old="    x = torch.randn(1, 4)\n",
                new="    x = torch.randn(1, 5)\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="failed_unit_test_4",
        failure_type="failed_unit_test",
        description="Change the model output size so the test expectation no longer matches.",
        expected_reason="AssertionError because the model now returns the wrong number of output actions.",
        expected_fix="Restore action_dim to 2.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/app/model.py",
                old="    def __init__(self, input_dim: int = 4, action_dim: int = 2):\n",
                new="    def __init__(self, input_dim: int = 4, action_dim: int = 3):\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="failed_unit_test_5",
        failure_type="failed_unit_test",
        description="Change the model input size so the test input no longer fits the layer.",
        expected_reason="RuntimeError because the first linear layer expects the wrong input dimension.",
        expected_fix="Restore input_dim to 4.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/app/model.py",
                old="    def __init__(self, input_dim: int = 4, action_dim: int = 2):\n",
                new="    def __init__(self, input_dim: int = 3, action_dim: int = 2):\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="file_not_found_1",
        failure_type="file_not_found",
        description="Point pip install at a requirements file that does not exist.",
        expected_reason="requirements-missing.txt cannot be found during the CI install step.",
        expected_fix="Restore the correct requirements.txt path or add the missing file intentionally.",
        file_mutations=(
            FileMutation(
                path=".github/workflows/ci.yml",
                old="          python -m pip install -r requirements.txt\n",
                new="          python -m pip install -r requirements-missing.txt\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="file_not_found_2",
        failure_type="file_not_found",
        description="Run pytest on a tests directory that does not exist.",
        expected_reason="pytest fails because tests_missing/ is not present.",
        expected_fix="Restore the test path to tests/.",
        file_mutations=(
            FileMutation(
                path=".github/workflows/ci.yml",
                old="          python -m pytest tests/\n",
                new="          python -m pytest tests_missing/\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="file_not_found_3",
        failure_type="file_not_found",
        description="Change the workflow working directory to a folder that does not exist.",
        expected_reason="GitHub Actions cannot enter the missing working directory before running steps.",
        expected_fix="Restore the working directory to ml_cicd_demo.",
        file_mutations=(
            FileMutation(
                path=".github/workflows/ci.yml",
                old="        working-directory: ml_cicd_demo\n",
                new="        working-directory: ml_cicd_demo_missing\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="file_not_found_4",
        failure_type="file_not_found",
        description="Copy a missing directory inside the Docker build.",
        expected_reason="Docker build fails because the requested source path does not exist.",
        expected_fix="Restore the Docker COPY source to app.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old="COPY app ./app\n",
                new="COPY app_missing ./app\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="docker_build_failure_1",
        failure_type="docker_build_failure",
        description="Break the Docker build by referencing a missing requirements file in Dockerfile.",
        expected_reason="Docker build fails because requirements-prod.txt is copied but does not exist.",
        expected_fix="Restore requirements.txt in the Dockerfile or add the intended file.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old="COPY requirements.txt .\n",
                new="COPY requirements-prod.txt .\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="docker_build_failure_2",
        failure_type="docker_build_failure",
        description="Use a Python base image tag that does not exist.",
        expected_reason="Docker build cannot pull the invalid Python base image.",
        expected_fix="Restore the base image to python:3.10-slim.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old="FROM python:3.10-slim\n",
                new="FROM python:3.10-missingtag\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="docker_build_failure_3",
        failure_type="docker_build_failure",
        description="Force the Docker dependency-install layer to fail explicitly.",
        expected_reason="Docker build stops because the install layer now returns a non-zero exit code.",
        expected_fix="Remove the injected false command from the Dockerfile.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old="RUN python -m pip install --upgrade pip && \\\n",
                new="RUN python -m pip install --upgrade pip && false && \\\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="docker_build_failure_4",
        failure_type="docker_build_failure",
        description="Misspell the RUN instruction in the Dockerfile.",
        expected_reason="Dockerfile parsing fails because RNU is not a valid instruction.",
        expected_fix="Restore the RUN instruction spelling.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old="RUN python -m pip install --upgrade pip && \\\n",
                new="RNU python -m pip install --upgrade pip && \\\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="docker_build_failure_5",
        failure_type="docker_build_failure",
        description="Call a missing executable during the Docker dependency-install layer.",
        expected_reason="Docker build fails because pythonx is not installed in the base image.",
        expected_fix="Restore the command to python -m pip.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old="RUN python -m pip install --upgrade pip && \\\n",
                new="RUN pythonx -m pip install --upgrade pip && \\\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="missing_env_variable_1",
        failure_type="missing_env_variable",
        description="Require an environment variable that GitHub Actions does not set.",
        expected_reason="Docker build fails with a missing required environment variable.",
        expected_fix="Set the env var in the workflow or remove the hard requirement from the build step.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old="RUN python -m pip install --upgrade pip && \\\n",
                new=(
                    "RUN test -n \"$MODEL_NAME\" || "
                    "(echo \"Missing required environment variable: MODEL_NAME\" && exit 1)\n"
                    "RUN python -m pip install --upgrade pip && \\\n"
                ),
            ),
        ),
    ),
    MutationSpec(
        mutation_id="missing_env_variable_2",
        failure_type="missing_env_variable",
        description="Require a different environment variable during the Docker build.",
        expected_reason="Docker build fails with a missing required environment variable.",
        expected_fix="Set the env var in the workflow or remove the hard requirement from the build step.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old='RUN test -n "$APP_ENV" || (echo "Missing required environment variable: APP_ENV" && exit 1)\n',
                new='RUN test -n "$TEST_RUN_ID" || (echo "Missing required environment variable: TEST_RUN_ID" && exit 1)\n',
            ),
        ),
    ),
    MutationSpec(
        mutation_id="missing_env_variable_3",
        failure_type="missing_env_variable",
        description="Require a secret-style environment variable during the Docker build.",
        expected_reason="Docker build fails with a missing required environment variable.",
        expected_fix="Set the env var in the workflow or remove the hard requirement from the build step.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/Dockerfile",
                old='RUN test -n "$APP_ENV" || (echo "Missing required environment variable: APP_ENV" && exit 1)\n',
                new='RUN test -n "$SERVICE_TOKEN" || (echo "Missing required environment variable: SERVICE_TOKEN" && exit 1)\n',
            ),
        ),
    ),
    MutationSpec(
        mutation_id="missing_dependency_1",
        failure_type="missing_dependency",
        description="Remove pytest from the dependency list so the test runner cannot be imported.",
        expected_reason="The CI install succeeds incompletely and pytest is missing when tests are invoked.",
        expected_fix="Restore pytest to requirements.txt.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/requirements.txt",
                old="pytest\n",
                new="pytest_missing\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="missing_dependency_2",
        failure_type="missing_dependency",
        description="Replace torch with a missing package name in requirements.",
        expected_reason="Dependency installation fails because the requested package does not exist.",
        expected_fix="Restore torch in requirements.txt.",
        file_mutations=(
            FileMutation(
                path="ml_cicd_demo/requirements.txt",
                old="torch\n",
                new="torch_missing_pkg\n",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="wrong_python_version_1",
        failure_type="wrong_python_version",
        description="Add a version check that expects Python 3.11 even though CI uses 3.10.",
        expected_reason="The CI step fails because the asserted Python version does not match the workflow setup.",
        expected_fix="Restore the version expectation to 3.10 or change the workflow intentionally.",
        file_mutations=(
            FileMutation(
                path=".github/workflows/ci.yml",
                old="          python -m pytest tests/\n",
                new='          python -c "import sys; assert sys.version.startswith(\\"3.11\\"), sys.version"\n          python -m pytest tests/\n',
            ),
        ),
    ),
    MutationSpec(
        mutation_id="wrong_python_version_2",
        failure_type="wrong_python_version",
        description="Configure the workflow to use Python 3.8 while the added assertion expects 3.10.",
        expected_reason="The CI step fails because the workflow is now using the wrong Python version.",
        expected_fix="Restore python-version to 3.10.",
        file_mutations=(
            FileMutation(
                path=".github/workflows/ci.yml",
                old='          python-version: "3.10"\n',
                new='          python-version: "3.8"\n',
            ),
            FileMutation(
                path=".github/workflows/ci.yml",
                old="          python -m pytest tests/\n",
                new='          python -c "import sys; assert sys.version.startswith(\\"3.10\\"), sys.version"\n          python -m pytest tests/\n',
            ),
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one controlled failure mutation.")
    parser.add_argument("--repo-root", type=str, default=str(DEFAULT_REPO_ROOT), help="Target git repo.")
    parser.add_argument(
        "--manifest-root",
        type=str,
        default=str(DEFAULT_MANIFEST_ROOT),
        help="Directory where mutation manifests are written.",
    )
    parser.add_argument(
        "--template",
        "--mutation",
        dest="template_id",
        type=str,
        default=None,
        help="Case template id to apply, for example import_path_error_1.",
    )
    parser.add_argument("--list", action="store_true", help="List available case templates and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root)
    manifest_root = Path(args.manifest_root)

    if args.list:
        print(json.dumps({"mutations": [serialize_mutation(spec) for spec in MUTATIONS]}, indent=2))
        return

    if not args.template_id:
        raise SystemExit("--template is required unless --list is used.")

    spec = get_mutation(args.template_id)
    git_status = git(repo_root, "status", "--short")
    if git_status.strip():
        raise SystemExit(
            "Refusing to mutate a dirty repository. Commit/stash changes in CI_CD_Workflow first."
        )

    base_commit = git(repo_root, "rev-parse", "HEAD").strip()
    branch_name = f"generated-failure/{spec.mutation_id}"
    changed_files = preview_or_apply_mutation(repo_root, spec, dry_run=args.dry_run)
    manifest_path = write_manifest(
        manifest_root=manifest_root,
        repo_root=repo_root,
        spec=spec,
        base_commit=base_commit,
        branch_name=branch_name,
        changed_files=changed_files,
        dry_run=args.dry_run,
    )

    summary = {
        "mutation_id": spec.mutation_id,
        "template_id": spec.mutation_id,
        "failure_type": spec.failure_type,
        "repo_root": str(repo_root),
        "base_commit": base_commit,
        "changed_files": changed_files,
        "dry_run": args.dry_run,
        "manifest_path": str(manifest_path),
        "next_steps": [
            f"git -C {repo_root} checkout -b {branch_name}",
            f"git -C {repo_root} add {' '.join(changed_files)}",
            f"git -C {repo_root} commit -m \"{spec.mutation_id}\"",
            f"git -C {repo_root} push -u origin {branch_name}",
            "python agi_tool/dataops/github_repo_take_failure.py",
            "python agi_tool/dataops/github_failures_to_dataset.py",
        ],
    }
    print(json.dumps(summary, indent=2))


def get_mutation(mutation_id: str) -> MutationSpec:
    for spec in MUTATIONS:
        if spec.mutation_id == mutation_id:
            return spec
    raise SystemExit(f"Unknown mutation_id: {mutation_id}")


def serialize_mutation(spec: MutationSpec) -> dict[str, object]:
    return {
        "mutation_id": spec.mutation_id,
        "failure_type": spec.failure_type,
        "description": spec.description,
        "expected_reason": spec.expected_reason,
        "expected_fix": spec.expected_fix,
        "files": [item.path for item in spec.file_mutations],
    }


def git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def preview_or_apply_mutation(repo_root: Path, spec: MutationSpec, *, dry_run: bool) -> list[str]:
    changed_files: list[str] = []
    for item in spec.file_mutations:
        target = repo_root / item.path
        text = target.read_text(encoding="utf-8")
        if item.old not in text:
            raise SystemExit(
                f"Expected text not found in {target}. Repo contents do not match this mutation template."
            )
        if not dry_run:
            target.write_text(text.replace(item.old, item.new, 1), encoding="utf-8")
        changed_files.append(item.path)
    return changed_files


def write_manifest(
    *,
    manifest_root: Path,
    repo_root: Path,
    spec: MutationSpec,
    base_commit: str,
    branch_name: str,
    changed_files: list[str],
    dry_run: bool,
) -> Path:
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{spec.mutation_id}.json"
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "base_commit": base_commit,
        "suggested_branch": branch_name,
        "mutation_id": spec.mutation_id,
        "failure_type": spec.failure_type,
        "description": spec.description,
        "expected_reason": spec.expected_reason,
        "expected_fix": spec.expected_fix,
        "changed_files": changed_files,
        "dry_run": dry_run,
        "collection_plan": {
            "step_1": "Apply one controlled mutation to the clean repo.",
            "step_2": "Create a branch, commit the mutation, and push it to GitHub.",
            "step_3": "Let GitHub Actions fail and collect the logs with github_repo_take_failure.py.",
            "step_4": "Convert collected failures into JSONL rows with github_failures_to_dataset.py.",
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    main()
