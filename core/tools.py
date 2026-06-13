"""Deterministic training-time tool wrappers for CI/CD diagnosis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import EVIDENCE_ACTIONS, FINAL_ACTION, is_valid_evidence_action
from .dataset import CICDDiagnosisDataset, load_dataset


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class ToolExecutionError(ValueError):
    """Raised when a tool invocation is invalid."""


class UnknownToolError(ToolExecutionError):
    """Raised when the caller requests a tool that is not implemented."""


@dataclass(frozen=True)
class ToolResult:
    """Normalized output shape for deterministic tool calls."""

    tool_name: str
    case_id: str
    observation: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class DeterministicToolExecutor:
    """Expose stable training-time tools backed by saved dataset observations."""

    def __init__(
        self,
        dataset: CICDDiagnosisDataset | None = None,
        *,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.dataset = dataset or load_dataset(dataset_root=dataset_root)

    @property
    def available_tools(self) -> tuple[str, ...]:
        return EVIDENCE_ACTIONS

    def run_tool(self, tool_name: str, case_id: str, **kwargs: Any) -> ToolResult:
        """Dispatch a deterministic evidence tool by name."""

        if tool_name == FINAL_ACTION:
            raise UnknownToolError(
                "final_answer is not an evidence-retrieval tool. Let the environment "
                "handle answer generation separately."
            )
        if not is_valid_evidence_action(tool_name):
            raise UnknownToolError(f"Unsupported tool '{tool_name}'.")

        handler = getattr(self, tool_name)
        return handler(case_id, **kwargs)

    def retrieve_logs(self, case_id: str) -> ToolResult:
        observation = self._get_cached_observation(case_id, "retrieve_logs")
        log_record = self.dataset.get_log(case_id)
        return ToolResult(
            tool_name="retrieve_logs",
            case_id=case_id,
            observation=observation,
            metadata={"log_id": log_record["log_id"], "source": "tool_observations.jsonl"},
        )

    def inspect_repo(self, case_id: str) -> ToolResult:
        observation = self._get_cached_observation(case_id, "inspect_repo")
        snapshot = self.dataset.get_repo_snapshot(case_id)
        return ToolResult(
            tool_name="inspect_repo",
            case_id=case_id,
            observation=observation,
            metadata={
                "relevant_files": list(snapshot.get("relevant_files", [])),
                "source": "tool_observations.jsonl",
            },
        )

    def search_docs(self, case_id: str, query: str | None = None, *, top_k: int = 1) -> ToolResult:
        """Return the cached docs observation or run a deterministic keyword lookup."""

        if top_k < 1:
            raise ToolExecutionError("top_k must be at least 1.")

        if query is None:
            observation = self._get_cached_observation(case_id, "search_docs")
            return ToolResult(
                tool_name="search_docs",
                case_id=case_id,
                observation=observation,
                metadata={"mode": "cached", "source": "tool_observations.jsonl"},
            )

        case = self.dataset.get_case(case_id)
        matches = self.keyword_search_docs(query, top_k=top_k, category=case["category"])
        if matches:
            observation = "\n\n".join(doc["content"] for doc in matches)
        else:
            observation = ""
        return ToolResult(
            tool_name="search_docs",
            case_id=case_id,
            observation=observation,
            metadata={
                "mode": "keyword_search",
                "query": query,
                "matched_doc_ids": [doc["doc_id"] for doc in matches],
                "source": "docs_kb.jsonl",
            },
        )

    def run_static_check(self, case_id: str) -> ToolResult:
        observation = self._get_cached_observation(case_id, "run_static_check")
        return ToolResult(
            tool_name="run_static_check",
            case_id=case_id,
            observation=observation,
            metadata={"source": "tool_observations.jsonl"},
        )

    def get_reference_final_answer(self, case_id: str) -> dict[str, Any]:
        """Return the saved target answer for evaluation code, not agent observation."""

        record = self.dataset.get_tool_observations(case_id)
        return dict(record["target_final_answer"])

    def keyword_search_docs(
        self,
        query: str,
        *,
        top_k: int = 1,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a tiny deterministic keyword search over the local docs KB."""

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        scored_docs: list[tuple[int, int, dict[str, Any]]] = []
        for doc in self.dataset.docs_kb.values():
            if category is not None and doc.get("category") != category:
                continue
            content_tokens = set(self._tokenize(doc.get("content", "")))
            overlap = len(query_tokens & content_tokens)
            if overlap == 0:
                continue
            scored_docs.append((overlap, -len(doc.get("content", "")), doc))

        scored_docs.sort(key=lambda item: (-item[0], item[1], item[2]["doc_id"]))
        return [doc for _, _, doc in scored_docs[:top_k]]

    def _get_cached_observation(self, case_id: str, tool_name: str) -> Any:
        record = self.dataset.get_tool_observations(case_id)
        observations = record.get("observations", {})
        if tool_name not in observations:
            raise ToolExecutionError(
                f"No cached observation available for tool '{tool_name}' and case '{case_id}'."
            )
        return observations[tool_name]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())
