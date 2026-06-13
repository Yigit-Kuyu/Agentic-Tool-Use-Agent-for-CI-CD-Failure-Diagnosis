"""Template-based final answer generation from collected tool evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


MODULE_NOT_FOUND_PATTERN = re.compile(r"No module named ['\"]?([A-Za-z0-9_\.]+)['\"]?")
MISSING_FILE_PATTERN = re.compile(r"No such file or directory: ['\"]([^'\"]+)['\"]")
MISSING_ENV_PATTERN = re.compile(r"Missing required environment variable: ([A-Z0-9_]+)")
DOCKER_IMAGE_PATTERN = re.compile(r"docker\.io/library/([A-Za-z0-9_\-\.]+:[A-Za-z0-9_\-\.]+): not found")


@dataclass(frozen=True)
class GeneratedFinalAnswer:
    """Structured final answer used by the environment and evaluation code."""

    failure_type: str
    diagnosis: str
    fix: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "diagnosis": self.diagnosis,
            "fix": self.fix,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
        }


class TemplateFinalAnswerGenerator:
    """Heuristic generator that turns gathered evidence into a diagnosis/fix pair."""

    INTERNAL_MODULE_NAMES = {"app", "apps", "application", "ml_cicd_demo", "src"}

    STATIC_CHECK_TO_FAILURE_TYPE = {
        "dependency_or_requirements_path_problem_detected": "missing_dependency",
        "python_version_mismatch_detected": "wrong_python_version",
        "internal_import_path_missing": "import_path_error",
        "referenced_file_missing": "file_not_found",
        "github_actions_yaml_invalid": "bad_github_actions_yaml",
        "dockerfile_build_context_or_dependency_problem": "docker_build_failure",
        "unit_test_logic_or_shape_failure_detected": "failed_unit_test",
        "required_environment_variable_missing": "missing_env_variable",
    }

    def generate(self, state: Mapping[str, Any]) -> dict[str, Any]:
        log_text = "\n".join(str(item) for item in state.get("retrieved_logs", []))
        repo_evidence = list(state.get("repo_evidence", []))
        static_checks = [str(item) for item in state.get("static_check_results", [])]
        repo_text = self._stringify(repo_evidence)
        combined_text = "\n".join([log_text, repo_text, "\n".join(static_checks)])

        failure_type = self._infer_failure_type(
            log_text=log_text,
            repo_text=repo_text,
            static_checks=static_checks,
            combined_text=combined_text,
        )
        builder = getattr(self, f"_build_{failure_type}_answer", self._build_generic_answer)
        answer = builder(log_text=log_text, repo_text=repo_text, combined_text=combined_text)
        return answer.to_dict()

    def _infer_failure_type(
        self,
        *,
        log_text: str,
        repo_text: str,
        static_checks: list[str],
        combined_text: str,
    ) -> str:
        for check_name in static_checks:
            mapped = self.STATIC_CHECK_TO_FAILURE_TYPE.get(check_name)
            if mapped is not None:
                return mapped

        lower_text = combined_text.lower()
        if "invalid workflow file" in lower_text:
            return "bad_github_actions_yaml"
        if "missing required environment variable" in lower_text:
            return "missing_env_variable"
        if "docker.io/library/" in lower_text or "docker build" in lower_text:
            return "docker_build_failure"
        if "requires python 3.11 or newer" in lower_text or "requires python 3.10 or newer" in lower_text:
            return "wrong_python_version"
        if "syntaxerror: invalid syntax" in lower_text and ("match " in lower_text or "case " in lower_text):
            return "wrong_python_version"
        module_name = self._extract_missing_module_name(log_text)
        if module_name is not None:
            if module_name.lower() in self.INTERNAL_MODULE_NAMES:
                return "import_path_error"
            return "missing_dependency"
        if "could not open requirements file" in lower_text and "requirements.txt" in repo_text.lower():
            return "missing_dependency"
        if "no such file or directory" in lower_text or "could not open requirements file" in lower_text:
            return "file_not_found"
        if "assertionerror" in lower_text or "assert false" in lower_text:
            return "failed_unit_test"
        return "failed_unit_test"

    def _build_missing_dependency_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        if "could not open requirements file" in combined_text.lower() and "requirement.txt" in combined_text:
            diagnosis = "The workflow references requirement.txt, but the repository file is requirements.txt."
            fix = "Change the workflow command to pip install -r requirements.txt."
            evidence = ["Could not open requirements file: requirement.txt", ".github/workflows/ci.yml", "requirements.txt"]
            return GeneratedFinalAnswer(
                failure_type="missing_dependency",
                diagnosis=diagnosis,
                fix=fix,
                evidence=evidence,
                confidence=0.9,
            )

        package_name = self._extract_missing_module_name(log_text) or "the required package"
        diagnosis = (
            f"{package_name} is imported in the project tests or code but is not installed in the CI environment."
        )
        fix = (
            f"Add {package_name} to requirements.txt or install it in the workflow before running pytest."
        )
        evidence = [
            f"ModuleNotFoundError: No module named {package_name}",
            *self._extract_file_mentions(repo_text),
        ]
        return GeneratedFinalAnswer(
            failure_type="missing_dependency",
            diagnosis=diagnosis,
            fix=fix,
            evidence=evidence[:3],
            confidence=0.9,
        )

    def _build_wrong_python_version_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        if "requires python 3.11 or newer" in combined_text.lower():
            diagnosis = "The code explicitly requires Python 3.11 or newer, but the workflow uses an older Python version."
            fix = "Set python-version to 3.11 or newer in .github/workflows/ci.yml."
            evidence = ["RuntimeError: This project requires Python 3.11 or newer", *self._extract_file_mentions(repo_text)]
        else:
            diagnosis = (
                "The workflow uses an older Python version, but the code uses syntax that requires Python 3.10 or newer."
            )
            fix = "Set actions/setup-python to Python 3.10 or 3.11, or remove the newer syntax from the code."
            evidence = ["SyntaxError: invalid syntax", *self._extract_file_mentions(repo_text)]
        return GeneratedFinalAnswer(
            failure_type="wrong_python_version",
            diagnosis=diagnosis,
            fix=fix,
            evidence=evidence[:3],
            confidence=0.9,
        )

    def _build_import_path_error_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        module_name = self._extract_missing_module_name(log_text) or "the referenced module"
        diagnosis = f"An import references {module_name}, but that internal module/package does not exist in the repository layout."
        fix = "Correct the import to match the repository package layout and ensure __init__.py exists where needed."
        evidence = [
            f"ModuleNotFoundError: No module named {module_name}",
            *self._extract_file_mentions(repo_text),
        ]
        return GeneratedFinalAnswer(
            failure_type="import_path_error",
            diagnosis=diagnosis,
            fix=fix,
            evidence=evidence[:3],
            confidence=0.9,
        )

    def _build_file_not_found_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        missing_file = self._extract_missing_file_name(combined_text) or "the referenced file"
        diagnosis = f"The workflow or code references {missing_file}, but that file is missing or the working directory is wrong."
        fix = f"Add {missing_file}, correct the file path, or fix the workflow/Docker working directory."
        evidence = [missing_file, *self._extract_file_mentions(repo_text)]
        return GeneratedFinalAnswer(
            failure_type="file_not_found",
            diagnosis=diagnosis,
            fix=fix,
            evidence=evidence[:3],
            confidence=0.85,
        )

    def _build_bad_github_actions_yaml_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        diagnosis = "The GitHub Actions workflow YAML has invalid keys, indentation, or misplaced fields."
        fix = "Edit .github/workflows/ci.yml so the jobs and steps structure uses valid keys and indentation."
        return GeneratedFinalAnswer(
            failure_type="bad_github_actions_yaml",
            diagnosis=diagnosis,
            fix=fix,
            evidence=["Invalid workflow file", ".github/workflows/ci.yml"],
            confidence=0.95,
        )

    def _build_docker_build_failure_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        if "no matching distribution found" in combined_text.lower():
            package_match = re.search(r"requirement ([A-Za-z0-9_\-\.]+==[A-Za-z0-9_\-\.]+)", combined_text)
            package_pin = package_match.group(1) if package_match else "an unavailable pinned package version"
            diagnosis = f"Docker build fails because requirements.txt pins an unavailable package version ({package_pin})."
            fix = "Use a valid package version in requirements.txt before building the image."
            evidence = ["No matching distribution found during docker build", "Dockerfile", "requirements.txt"]
            return GeneratedFinalAnswer(
                failure_type="docker_build_failure",
                diagnosis=diagnosis,
                fix=fix,
                evidence=evidence,
                confidence=0.9,
            )

        image_tag = self._extract_docker_image_tag(combined_text)
        if image_tag is not None:
            diagnosis = f"Dockerfile uses a non-existent base image tag {image_tag}."
            fix = "Use a valid Python base image tag such as python:3.10-slim or python:3.11-slim."
            evidence = [f"{image_tag}: not found", "Dockerfile"]
        else:
            diagnosis = "The Docker build is failing because the Dockerfile uses an invalid image, path, or dependency step."
            fix = "Inspect the Dockerfile base image, COPY paths, and install commands, then correct the failing build step."
            evidence = ["docker build failure", "Dockerfile"]
        return GeneratedFinalAnswer(
            failure_type="docker_build_failure",
            diagnosis=diagnosis,
            fix=fix,
            evidence=evidence,
            confidence=0.9,
        )

    def _build_failed_unit_test_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        if "assert false" in combined_text.lower():
            diagnosis = "The failing test intentionally asserts False, so the unit test is guaranteed to fail."
            fix = "Remove the intentional failing assertion or replace it with a real expected condition."
            evidence = ["AssertionError: intentional failing assertion", *self._extract_file_mentions(repo_text)]
        elif "torch.size" in combined_text.lower():
            diagnosis = "The model output shape does not match the shape expected by the test."
            fix = "Align the model output dimension with the test expectation, or update the test to the intended output shape."
            evidence = ["AssertionError: shape mismatch", *self._extract_file_mentions(repo_text)]
        elif "assert 'negative' == 'positive'" in combined_text.lower() or "predicted label mismatch" in combined_text.lower():
            diagnosis = "The model predicted a label different from the expected label in the unit test."
            fix = "Fix deterministic test data or model logic, or update the expected label if the test expectation is wrong."
            evidence = ["AssertionError: predicted label mismatch", *self._extract_file_mentions(repo_text)]
        else:
            diagnosis = "A unit test assertion is failing because the implementation behavior does not match the expected result."
            fix = "Inspect the failing test and implementation, then align the code or the expected assertion."
            evidence = ["AssertionError", *self._extract_file_mentions(repo_text)]
        return GeneratedFinalAnswer(
            failure_type="failed_unit_test",
            diagnosis=diagnosis,
            fix=fix,
            evidence=evidence[:3],
            confidence=0.8,
        )

    def _build_missing_env_variable_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        variable_name = self._extract_missing_env_name(combined_text) or "the required environment variable"
        diagnosis = f"The code or workflow requires environment variable {variable_name}, but it is not defined in the CI environment."
        fix = f"Define {variable_name} in GitHub Actions env/secrets or inject it in the relevant test or build step."
        evidence = [f"Missing required environment variable: {variable_name}", *self._extract_file_mentions(repo_text)]
        return GeneratedFinalAnswer(
            failure_type="missing_env_variable",
            diagnosis=diagnosis,
            fix=fix,
            evidence=evidence[:3],
            confidence=0.9,
        )

    def _build_generic_answer(
        self,
        *,
        log_text: str,
        repo_text: str,
        combined_text: str,
    ) -> GeneratedFinalAnswer:
        return GeneratedFinalAnswer(
            failure_type="failed_unit_test",
            diagnosis="The collected evidence is not enough for a category-specific diagnosis, but the failure is reproducible from the retrieved logs.",
            fix="Inspect the retrieved log and relevant files, then align the failing code path with the expected CI behavior.",
            evidence=self._extract_file_mentions(repo_text)[:3],
            confidence=0.3,
        )

    @staticmethod
    def _extract_missing_module_name(text: str) -> str | None:
        match = MODULE_NOT_FOUND_PATTERN.search(text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_missing_file_name(text: str) -> str | None:
        match = MISSING_FILE_PATTERN.search(text)
        if match:
            return match.group(1)
        if "requirement.txt" in text:
            return "requirement.txt"
        return None

    @staticmethod
    def _extract_missing_env_name(text: str) -> str | None:
        match = MISSING_ENV_PATTERN.search(text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_docker_image_tag(text: str) -> str | None:
        match = DOCKER_IMAGE_PATTERN.search(text)
        if match:
            return match.group(1)
        from_match = re.search(r"FROM\s+([A-Za-z0-9_\-\.]+:[A-Za-z0-9_\-\.]+)", text)
        return from_match.group(1) if from_match else None

    @staticmethod
    def _extract_file_mentions(text: str) -> list[str]:
        mentions = []
        for candidate in (
            ".github/workflows/ci.yml",
            "requirements.txt",
            "Dockerfile",
            "tests/test_model.py",
            "tests/test_api.py",
            "app/model.py",
            "app/main.py",
            "app/__init__.py",
        ):
            if candidate in text:
                mentions.append(candidate)
        return mentions

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(TemplateFinalAnswerGenerator._stringify(item) for item in value)
        if isinstance(value, dict):
            pieces: list[str] = []
            for key, item in value.items():
                pieces.append(f"{key}: {TemplateFinalAnswerGenerator._stringify(item)}")
            return "\n".join(pieces)
        return str(value)
