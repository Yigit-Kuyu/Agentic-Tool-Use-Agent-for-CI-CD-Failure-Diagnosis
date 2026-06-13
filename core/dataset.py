"""Dataset loading utilities for the CI/CD diagnosis project.

The project dataset lives in the sibling ``dataset/`` directory and is split
across JSONL files. This module loads those files into a validated, indexed API
that later training and environment code can reuse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


JsonDict = dict[str, Any]


class DatasetValidationError(ValueError):
    """Raised when the on-disk dataset does not match the expected schema."""


def _read_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise DatasetValidationError(f"{path} must contain a JSON object.")
    return data


def _read_jsonl(path: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"{path}:{line_number} contains invalid JSON: {exc.msg}"
                raise DatasetValidationError(message) from exc

            if not isinstance(item, dict):
                raise DatasetValidationError(
                    f"{path}:{line_number} must contain a JSON object per line."
                )

            records.append(item)

    return records


def _index_by_key(
    records: Iterable[JsonDict],
    *,
    key: str,
    dataset_name: str,
) -> dict[str, JsonDict]:
    indexed: dict[str, JsonDict] = {}
    for record in records:
        record_key = record.get(key)
        if not isinstance(record_key, str) or not record_key:
            raise DatasetValidationError(
                f"{dataset_name} contains a record without a valid '{key}'."
            )
        if record_key in indexed:
            raise DatasetValidationError(
                f"{dataset_name} contains duplicate '{key}' value '{record_key}'."
            )
        indexed[record_key] = record
    return indexed


def _require_keys(record: Mapping[str, Any], *, required: set[str], dataset_name: str) -> None:
    missing = sorted(key for key in required if key not in record)
    if missing:
        raise DatasetValidationError(
            f"{dataset_name} record is missing required keys: {', '.join(missing)}"
        )


@dataclass(frozen=True)
class DatasetPaths:
    """Canonical file locations for all dataset assets."""

    root: Path
    failure_cases: Path
    logs: Path
    repo_snapshots: Path
    tool_observations: Path
    expert_trajectories: Path
    dpo_preferences: Path
    docs_kb: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "DatasetPaths":
        dataset_root = Path(root).expanduser().resolve()
        return cls(
            root=dataset_root,
            failure_cases=dataset_root / "failure_cases.jsonl",
            logs=dataset_root / "logs.jsonl",
            repo_snapshots=dataset_root / "repo_snapshots.jsonl",
            tool_observations=dataset_root / "tool_observations.jsonl",
            expert_trajectories=dataset_root / "expert_trajectories.jsonl",
            dpo_preferences=dataset_root / "dpo_preferences.jsonl",
            docs_kb=dataset_root / "docs_kb.jsonl",
            manifest=dataset_root / "dataset_manifest.json",
        )


@dataclass(frozen=True)
class CaseBundle:
    """A joined view of all records related to a single case."""

    case: JsonDict
    log: JsonDict
    repo_snapshot: JsonDict
    tool_observations: JsonDict
    expert_trajectory: JsonDict | None
    dpo_preference: JsonDict | None


class CICDDiagnosisDataset:
    """Loads and validates the static CI/CD diagnosis dataset."""

    FAILURE_CASE_REQUIRED_KEYS = {
        "case_id",
        "category",
        "user_query",
        "failure_type",
        "root_cause",
        "expected_fix",
        "related_files",
        "log_id",
        "web_augmentation_query",
    }
    LOG_REQUIRED_KEYS = {"case_id", "log_id", "raw_log"}
    REPO_REQUIRED_KEYS = {"case_id", "relevant_files", "file_snippets"}
    TOOL_OBSERVATION_REQUIRED_KEYS = {"case_id", "observations", "target_final_answer"}
    EXPERT_TRAJECTORY_REQUIRED_KEYS = {"case_id", "expert_actions", "steps"}
    DPO_REQUIRED_KEYS = {"case_id", "prompt", "chosen", "rejected"}
    DOC_REQUIRED_KEYS = {"doc_id", "category", "content"}

    def __init__(self, dataset_root: str | Path | None = None, *, validate: bool = True) -> None:
        default_root = Path(__file__).with_name("dataset")
        self.paths = DatasetPaths.from_root(dataset_root or default_root)

        self.manifest = _read_json(self.paths.manifest)
        self.failure_cases = _index_by_key(
            _read_jsonl(self.paths.failure_cases),
            key="case_id",
            dataset_name="failure_cases.jsonl",
        )
        self.logs = _index_by_key(
            _read_jsonl(self.paths.logs),
            key="case_id",
            dataset_name="logs.jsonl",
        )
        self.repo_snapshots = _index_by_key(
            _read_jsonl(self.paths.repo_snapshots),
            key="case_id",
            dataset_name="repo_snapshots.jsonl",
        )
        self.tool_observations = _index_by_key(
            _read_jsonl(self.paths.tool_observations),
            key="case_id",
            dataset_name="tool_observations.jsonl",
        )
        self.expert_trajectories = _index_by_key(
            _read_jsonl(self.paths.expert_trajectories),
            key="case_id",
            dataset_name="expert_trajectories.jsonl",
        )
        self.dpo_preferences = _index_by_key(
            _read_jsonl(self.paths.dpo_preferences),
            key="case_id",
            dataset_name="dpo_preferences.jsonl",
        )
        self.docs_kb = _index_by_key(
            _read_jsonl(self.paths.docs_kb),
            key="doc_id",
            dataset_name="docs_kb.jsonl",
        )

        if validate:
            self.validate()

    def validate(self) -> None:
        """Run cross-file schema and consistency checks."""

        for record in self.failure_cases.values():
            _require_keys(
                record,
                required=self.FAILURE_CASE_REQUIRED_KEYS,
                dataset_name="failure_cases.jsonl",
            )

        for record in self.logs.values():
            _require_keys(record, required=self.LOG_REQUIRED_KEYS, dataset_name="logs.jsonl")

        for record in self.repo_snapshots.values():
            _require_keys(
                record,
                required=self.REPO_REQUIRED_KEYS,
                dataset_name="repo_snapshots.jsonl",
            )

        for record in self.tool_observations.values():
            _require_keys(
                record,
                required=self.TOOL_OBSERVATION_REQUIRED_KEYS,
                dataset_name="tool_observations.jsonl",
            )

        for record in self.expert_trajectories.values():
            _require_keys(
                record,
                required=self.EXPERT_TRAJECTORY_REQUIRED_KEYS,
                dataset_name="expert_trajectories.jsonl",
            )

        for record in self.dpo_preferences.values():
            _require_keys(
                record,
                required=self.DPO_REQUIRED_KEYS,
                dataset_name="dpo_preferences.jsonl",
            )

        for record in self.docs_kb.values():
            _require_keys(record, required=self.DOC_REQUIRED_KEYS, dataset_name="docs_kb.jsonl")

        case_ids = set(self.failure_cases)
        for dataset_name, records in (
            ("logs.jsonl", self.logs),
            ("repo_snapshots.jsonl", self.repo_snapshots),
            ("tool_observations.jsonl", self.tool_observations),
            ("expert_trajectories.jsonl", self.expert_trajectories),
            ("dpo_preferences.jsonl", self.dpo_preferences),
        ):
            record_ids = set(records)
            if record_ids != case_ids:
                missing = sorted(case_ids - record_ids)
                extras = sorted(record_ids - case_ids)
                parts: list[str] = []
                if missing:
                    parts.append(f"missing case_ids: {missing}")
                if extras:
                    parts.append(f"unexpected case_ids: {extras}")
                raise DatasetValidationError(f"{dataset_name} case_id mismatch ({'; '.join(parts)}).")

        total_cases = self.manifest.get("total_cases")
        if isinstance(total_cases, int) and total_cases != len(self.failure_cases):
            raise DatasetValidationError(
                "dataset_manifest.json total_cases does not match loaded failure cases."
            )

        manifest_datasets = self.manifest.get("datasets", {})
        if isinstance(manifest_datasets, dict):
            expected_counts = {
                "failure_cases.jsonl": len(self.failure_cases),
                "logs.jsonl": len(self.logs),
                "repo_snapshots.jsonl": len(self.repo_snapshots),
                "tool_observations.jsonl": len(self.tool_observations),
                "expert_trajectories.jsonl": len(self.expert_trajectories),
                "dpo_preferences.jsonl": len(self.dpo_preferences),
            }
            for filename, actual_count in expected_counts.items():
                manifest_count = manifest_datasets.get(filename)
                if manifest_count is not None and manifest_count != actual_count:
                    raise DatasetValidationError(
                        f"dataset_manifest.json count mismatch for {filename}: "
                        f"expected {manifest_count}, loaded {actual_count}."
                    )

    def __len__(self) -> int:
        return len(self.failure_cases)

    @property
    def categories(self) -> list[str]:
        categories = {record["category"] for record in self.failure_cases.values()}
        return sorted(categories)

    def list_case_ids(
        self,
        *,
        split: str | None = None,
        category: str | None = None,
        source: str | None = None,
    ) -> list[str]:
        case_ids: list[str] = []
        for case_id, record in self.failure_cases.items():
            if split is not None and record.get("split") != split:
                continue
            if category is not None and record.get("category") != category:
                continue
            if source is not None and record.get("source") != source:
                continue
            case_ids.append(case_id)
        return sorted(case_ids)

    def get_case(self, case_id: str) -> JsonDict:
        return self._get_from_index(self.failure_cases, case_id, "failure case")

    def get_training_case(self, case_id: str) -> JsonDict:
        """Return the case metadata with reward-only labels removed."""

        record = dict(self.get_case(case_id))
        record.pop("root_cause", None)
        record.pop("expected_fix", None)
        return record

    def get_log(self, case_id: str) -> JsonDict:
        return self._get_from_index(self.logs, case_id, "log")

    def get_repo_snapshot(self, case_id: str) -> JsonDict:
        return self._get_from_index(self.repo_snapshots, case_id, "repo snapshot")

    def get_tool_observations(self, case_id: str) -> JsonDict:
        return self._get_from_index(self.tool_observations, case_id, "tool observations")

    def get_expert_trajectory(self, case_id: str) -> JsonDict:
        return self._get_from_index(self.expert_trajectories, case_id, "expert trajectory")

    def get_dpo_preference(self, case_id: str) -> JsonDict:
        return self._get_from_index(self.dpo_preferences, case_id, "DPO preference")

    def get_doc(self, doc_id: str) -> JsonDict:
        return self._get_from_index(self.docs_kb, doc_id, "documentation record")

    def get_docs_for_category(self, category: str) -> list[JsonDict]:
        docs = [record for record in self.docs_kb.values() if record.get("category") == category]
        return sorted(docs, key=lambda record: record["doc_id"])

    def get_case_bundle(self, case_id: str) -> CaseBundle:
        return CaseBundle(
            case=self.get_case(case_id),
            log=self.get_log(case_id),
            repo_snapshot=self.get_repo_snapshot(case_id),
            tool_observations=self.get_tool_observations(case_id),
            expert_trajectory=self.get_expert_trajectory(case_id),
            dpo_preference=self.get_dpo_preference(case_id),
        )

    def summary(self) -> JsonDict:
        """Return a compact snapshot useful for debugging or setup checks."""

        manifest_categories = self.manifest.get("categories")
        return {
            "dataset_root": str(self.paths.root),
            "num_cases": len(self),
            "categories": list(manifest_categories) if isinstance(manifest_categories, list) else self.categories,
            "num_docs": len(self.docs_kb),
        }

    @staticmethod
    def _get_from_index(index: Mapping[str, JsonDict], item_id: str, label: str) -> JsonDict:
        try:
            return index[item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown {label} id '{item_id}'.") from exc


def load_dataset(dataset_root: str | Path | None = None, *, validate: bool = True) -> CICDDiagnosisDataset:
    """Convenience constructor used by scripts and notebooks."""

    return CICDDiagnosisDataset(dataset_root=dataset_root, validate=validate)

