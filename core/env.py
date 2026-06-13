"""Environment for sequential CI/CD diagnosis tool selection."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .actions import ACTION_TO_INDEX, FINAL_ACTION, INDEX_TO_ACTION, TRAINING_ACTIONS
from .dataset import CICDDiagnosisDataset, load_dataset
from .final_answer import TemplateFinalAnswerGenerator
from .tools import DeterministicToolExecutor, ToolResult


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights for the CI/CD diagnosis environment."""

    correct_failure_type: float = 1.0
    correct_root_cause: float = 1.0
    correct_fix: float = 0.5
    evidence_bonus: float = 0.3
    efficient_sequence_bonus: float = 0.2
    per_tool_cost: float = -0.05
    premature_final_answer_penalty: float = -0.5
    wrong_diagnosis_penalty: float = -1.0
    wrong_tool_penalty: float = -0.3
    duplicate_tool_penalty: float = -0.1
    truncation_penalty: float = -0.5


class CICDDiagnosisEnv:
    """A small deterministic environment for RL-style tool selection."""

    def __init__(
        self,
        dataset: CICDDiagnosisDataset | None = None,
        tool_executor: DeterministicToolExecutor | None = None,
        answer_generator: TemplateFinalAnswerGenerator | None = None,
        *,
        dataset_root: str | Path | None = None,
        split: str | None = None,
        max_steps: int = 5,
        seed: int | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.dataset = dataset or load_dataset(dataset_root=dataset_root)
        self.tool_executor = tool_executor or DeterministicToolExecutor(self.dataset)
        self.answer_generator = answer_generator or TemplateFinalAnswerGenerator()
        self.default_split = split
        self.max_steps = max_steps
        self.reward_config = reward_config or RewardConfig()
        self.rng = random.Random(seed)

        self._active_case_ids = self.dataset.list_case_ids(split=split)
        if not self._active_case_ids:
            raise ValueError("No cases matched the requested environment split.")

        self._clear_episode()

    @property
    def action_space(self) -> tuple[str, ...]:
        return TRAINING_ACTIONS

    def reset(
        self,
        *,
        case_id: str | None = None,
        split: str | None = None,
        category: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Start a new episode and return the initial observable state."""

        chosen_case_id = case_id or self._sample_case_id(split=split, category=category, source=source)
        case_record = self.dataset.get_case(chosen_case_id)
        expert_record = self.dataset.get_expert_trajectory(chosen_case_id)
        expert_actions = list(expert_record.get("expert_actions", []))
        required_tools = [action for action in expert_actions if action != FINAL_ACTION]

        self.current_case_id = chosen_case_id
        self.current_case = case_record
        self.reference_final_answer = self.tool_executor.get_reference_final_answer(chosen_case_id)
        self.expert_actions = expert_actions
        self.required_tools_before_final = required_tools
        self.tool_history: list[str] = []
        self.tool_results: list[ToolResult] = []
        self.last_observation: Any = None
        self.retrieved_logs: list[str] = []
        self.repo_evidence: list[Any] = []
        self.doc_evidence: list[str] = []
        self.static_check_results: list[str] = []
        self.step_count = 0
        self.done = False
        self.truncated = False

        return self.get_state()

    def step(
        self,
        action: str | int,
        *,
        answer: Mapping[str, Any] | None = None,
        tool_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the episode by one tool-selection action."""

        self._require_active_episode()
        if self.done or self.truncated:
            raise RuntimeError("Episode already finished. Call reset() before step().")

        action_name = self._normalize_action(action)
        reward_breakdown: dict[str, float] = {}
        info: dict[str, Any] = {"case_id": self.current_case_id, "action": action_name}

        self._add_reward(reward_breakdown, "per_tool_cost", self.reward_config.per_tool_cost)
        self.step_count += 1
        self.tool_history.append(action_name)

        if action_name == FINAL_ACTION:
            resolved_answer = dict(answer) if answer is not None else self.answer_generator.generate(self.get_state())
            final_reward_breakdown, answer_scores = self._score_final_answer(resolved_answer)
            for key, value in final_reward_breakdown.items():
                self._add_reward(reward_breakdown, key, value)
            self.done = True
            info["final_answer_reference"] = self.reference_final_answer
            info["final_answer"] = resolved_answer
            info["final_answer_scores"] = answer_scores
            info["episode_summary"] = self.get_episode_summary()
            info["reward_breakdown"] = reward_breakdown
            return self.get_state(), sum(reward_breakdown.values()), True, False, info

        if self._was_duplicate_tool(action_name):
            self._add_reward(
                reward_breakdown,
                "duplicate_tool_penalty",
                self.reward_config.duplicate_tool_penalty,
            )

        tool_result = self.tool_executor.run_tool(
            action_name,
            self.current_case_id,
            **dict(tool_kwargs or {}),
        )
        self._record_tool_result(tool_result)
        info["tool_result"] = tool_result

        if not self._is_tool_relevant(action_name):
            self._add_reward(
                reward_breakdown,
                "wrong_tool_penalty",
                self.reward_config.wrong_tool_penalty,
            )

        if self.step_count >= self.max_steps:
            self.truncated = True
            self._add_reward(
                reward_breakdown,
                "truncation_penalty",
                self.reward_config.truncation_penalty,
            )

        info["reward_breakdown"] = reward_breakdown
        info["episode_summary"] = self.get_episode_summary()
        return self.get_state(), sum(reward_breakdown.values()), False, self.truncated, info

    def get_state(self) -> dict[str, Any]:
        """Return the current observable state for the policy."""

        self._require_active_episode()
        return {
            "case_id": self.current_case_id,
            "user_query": self.current_case["user_query"],
            "tool_history": list(self.tool_history),
            "last_observation": self.last_observation,
            "retrieved_logs": list(self.retrieved_logs),
            "repo_evidence": list(self.repo_evidence),
            "doc_evidence": list(self.doc_evidence),
            "static_check_results": list(self.static_check_results),
            "step_count": self.step_count,
            "remaining_steps": max(self.max_steps - self.step_count, 0),
            "done": self.done,
            "truncated": self.truncated,
            "available_actions": list(TRAINING_ACTIONS),
            "feature_state": self.get_feature_state(),
            "state_text": self.get_state_text(),
        }

    def get_feature_state(self) -> dict[str, int]:
        """Return a simple feature-based state encoding."""

        log_text = "\n".join(self.retrieved_logs).lower()
        repo_text = self._stringify(self.repo_evidence).lower()
        return {
            "has_seen_log": int(bool(self.retrieved_logs)),
            "has_inspected_repo": int(bool(self.repo_evidence)),
            "has_searched_docs": int(bool(self.doc_evidence)),
            "has_static_check": int(bool(self.static_check_results)),
            "step_count": self.step_count,
            "remaining_steps": max(self.max_steps - self.step_count, 0),
            "log_mentions_module_not_found": int("modulenotfounderror" in log_text),
            "log_mentions_docker": int("docker" in log_text),
            "log_mentions_yaml": int("yaml" in log_text or "workflow" in log_text),
            "repo_mentions_requirements": int("requirements.txt" in repo_text),
            "repo_mentions_tests": int("tests/" in repo_text or "test_" in repo_text),
        }

    def get_state_text(self) -> str:
        """Return a text-form state useful for simple encoders."""

        last_observation = self._stringify(self.last_observation) if self.last_observation is not None else "None"
        tool_history = ", ".join(self.tool_history) if self.tool_history else "None"
        return (
            f"User query: {self.current_case['user_query']}\n"
            f"Tool history: {tool_history}\n"
            f"Last observation: {last_observation}\n"
            f"Remaining steps: {max(self.max_steps - self.step_count, 0)}"
        )

    def get_episode_summary(self) -> dict[str, Any]:
        self._require_active_episode()
        return {
            "case_id": self.current_case_id,
            "category": self.current_case["category"],
            "failure_type": self.current_case["failure_type"],
            "tool_history": list(self.tool_history),
            "required_tools_before_final": list(self.required_tools_before_final),
            "done": self.done,
            "truncated": self.truncated,
        }

    def _score_final_answer(self, answer: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, bool]]:
        reward_breakdown: dict[str, float] = {}
        collected_required_tools = all(
            tool_name in self.tool_history for tool_name in self.required_tools_before_final
        )
        has_primary_evidence = bool(self.retrieved_logs or self.repo_evidence)

        if not has_primary_evidence:
            self._add_reward(
                reward_breakdown,
                "premature_final_answer_penalty",
                self.reward_config.premature_final_answer_penalty,
            )

        if has_primary_evidence:
            self._add_reward(
                reward_breakdown,
                "evidence_bonus",
                self.reward_config.evidence_bonus,
            )

        if collected_required_tools:
            self._add_reward(
                reward_breakdown,
                "efficient_sequence_bonus",
                self.reward_config.efficient_sequence_bonus,
            )

        answer_dict = dict(answer)
        failure_type_correct = self._matches_label(answer_dict.get("failure_type"), self.current_case["failure_type"])
        diagnosis_correct = self._text_matches_reference(
            answer_dict.get("diagnosis") or answer_dict.get("root_cause"),
            self.current_case["root_cause"],
        )
        fix_correct = self._text_matches_reference(
            answer_dict.get("fix") or answer_dict.get("expected_fix"),
            self.current_case["expected_fix"],
        )

        if failure_type_correct:
            self._add_reward(
                reward_breakdown,
                "correct_failure_type",
                self.reward_config.correct_failure_type,
            )

        if diagnosis_correct:
            self._add_reward(
                reward_breakdown,
                "correct_root_cause",
                self.reward_config.correct_root_cause,
            )

        if fix_correct:
            self._add_reward(
                reward_breakdown,
                "correct_fix",
                self.reward_config.correct_fix,
            )

        positive_reward = sum(value for value in reward_breakdown.values() if value > 0)
        if positive_reward < self.reward_config.correct_root_cause:
            self._add_reward(
                reward_breakdown,
                "wrong_diagnosis_penalty",
                self.reward_config.wrong_diagnosis_penalty,
            )

        return reward_breakdown, {
            "failure_type_correct": failure_type_correct,
            "diagnosis_correct": diagnosis_correct,
            "fix_correct": fix_correct,
        }

    def _record_tool_result(self, tool_result: ToolResult) -> None:
        self.tool_results.append(tool_result)
        self.last_observation = tool_result.observation
        if tool_result.tool_name == "retrieve_logs":
            self.retrieved_logs.append(str(tool_result.observation))
        elif tool_result.tool_name == "inspect_repo":
            self.repo_evidence.append(tool_result.observation)
        elif tool_result.tool_name == "search_docs":
            self.doc_evidence.append(str(tool_result.observation))
        elif tool_result.tool_name == "run_static_check":
            self.static_check_results.append(str(tool_result.observation))

    def _sample_case_id(
        self,
        *,
        split: str | None = None,
        category: str | None = None,
        source: str | None = None,
    ) -> str:
        filtered_case_ids = self.dataset.list_case_ids(
            split=split if split is not None else self.default_split,
            category=category,
            source=source,
        )
        if not filtered_case_ids:
            raise ValueError("No cases matched the requested reset filters.")
        return self.rng.choice(filtered_case_ids)

    def _normalize_action(self, action: str | int) -> str:
        if isinstance(action, int):
            try:
                return INDEX_TO_ACTION[action]
            except KeyError as exc:
                raise ValueError(f"Unknown action index '{action}'.") from exc
        if action not in ACTION_TO_INDEX:
            raise ValueError(f"Unknown action '{action}'.")
        return action

    def _was_duplicate_tool(self, action_name: str) -> bool:
        return action_name in self.tool_history[:-1]

    def _is_tool_relevant(self, action_name: str) -> bool:
        if action_name in self.required_tools_before_final:
            return True
        if action_name == "search_docs" and self.current_case["category"] in {
            "missing_dependency",
            "wrong_python_version",
            "bad_github_actions_yaml",
            "docker_build_failure",
        }:
            return True
        if action_name == "run_static_check" and self.current_case["category"] in {
            "failed_unit_test",
            "bad_github_actions_yaml",
            "docker_build_failure",
            "file_not_found",
        }:
            return True
        return False

    def _clear_episode(self) -> None:
        self.current_case_id: str | None = None
        self.current_case: dict[str, Any] | None = None
        self.reference_final_answer: dict[str, Any] | None = None
        self.expert_actions: list[str] = []
        self.required_tools_before_final: list[str] = []
        self.tool_history: list[str] = []
        self.tool_results: list[ToolResult] = []
        self.last_observation: Any = None
        self.retrieved_logs: list[str] = []
        self.repo_evidence: list[Any] = []
        self.doc_evidence: list[str] = []
        self.static_check_results: list[str] = []
        self.step_count = 0
        self.done = False
        self.truncated = False

    def _require_active_episode(self) -> None:
        if self.current_case_id is None or self.current_case is None:
            raise RuntimeError("No active episode. Call reset() first.")

    @staticmethod
    def _add_reward(breakdown: dict[str, float], key: str, value: float) -> None:
        breakdown[key] = breakdown.get(key, 0.0) + value

    @staticmethod
    def _matches_label(predicted: Any, target: str) -> bool:
        return isinstance(predicted, str) and predicted.strip().lower() == target.strip().lower()

    @classmethod
    def _text_matches_reference(cls, candidate: Any, reference: str) -> bool:
        if not isinstance(candidate, str) or not candidate.strip():
            return False
        candidate_tokens = set(cls._tokenize(candidate))
        reference_tokens = set(cls._tokenize(reference))
        if not candidate_tokens or not reference_tokens:
            return False
        overlap = len(candidate_tokens & reference_tokens) / len(reference_tokens)
        return overlap >= 0.6

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True)
