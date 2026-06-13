"""Shared action definitions for the CI/CD diagnosis agent."""

from __future__ import annotations


TRAINING_ACTIONS: tuple[str, ...] = (
    "retrieve_logs",
    "inspect_repo",
    "search_docs",
    "run_static_check",
    "final_answer",
)

EVIDENCE_ACTIONS: tuple[str, ...] = TRAINING_ACTIONS[:-1]
FINAL_ACTION: str = TRAINING_ACTIONS[-1]
TEST_TIME_ONLY_TOOLS: tuple[str, ...] = ("web_search",)

ACTION_TO_INDEX: dict[str, int] = {action: index for index, action in enumerate(TRAINING_ACTIONS)}
INDEX_TO_ACTION: dict[int, str] = {index: action for action, index in ACTION_TO_INDEX.items()}


def is_valid_training_action(action: str) -> bool:
    return action in ACTION_TO_INDEX


def is_valid_evidence_action(action: str) -> bool:
    return action in EVIDENCE_ACTIONS
