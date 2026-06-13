"""Evaluation helpers for BC and RL policy rollouts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..core.dataset import CICDDiagnosisDataset, load_dataset
from ..core.env import CICDDiagnosisEnv
from .bc import BehaviorCloningPolicy
from .rl import RLFineTunedPolicy


class ActionPolicy(Protocol):
    """Minimal policy interface shared by BC and RL evaluators."""

    def predict_action(self, state: dict[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class CaseEvaluationResult:
    """Per-case rollout summary for one policy."""

    case_id: str
    category: str
    split: str
    total_reward: float
    success: bool
    action_accuracy: float
    failure_type_correct: bool
    diagnosis_correct: bool
    fix_correct: bool
    exact_trajectory_match: bool
    terminated: bool
    truncated: bool
    steps: int
    tool_history: list[str]
    expert_actions: list[str]
    required_tools_before_final: list[str]


@dataclass(frozen=True)
class AggregateEvaluationResult:
    """Aggregate metrics for a set of rollout results."""

    policy_name: str
    split: str
    num_cases: int
    average_reward: float
    success_rate: float
    action_accuracy: float
    diagnosis_accuracy: float
    fix_accuracy: float
    exact_trajectory_match_rate: float
    average_steps: float
    category_breakdown: dict[str, dict[str, float]]


@dataclass(frozen=True)
class PolicyEvaluationReport:
    """Full report for one policy, including per-case details."""

    aggregate: AggregateEvaluationResult
    cases: list[CaseEvaluationResult]


@dataclass(frozen=True)
class ComparisonReport:
    """Side-by-side report for BC and RL policies."""

    split: str
    bc_report: PolicyEvaluationReport
    rl_report: PolicyEvaluationReport
    reward_gap_rl_minus_bc: float
    success_gap_rl_minus_bc: float
    trajectory_gap_rl_minus_bc: float


def load_policy(policy_path: str | Path) -> ActionPolicy:
    path = Path(policy_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "q_model" in payload:
        return RLFineTunedPolicy.load(path)
    if "model" in payload:
        return BehaviorCloningPolicy.load(path)
    raise ValueError(f"Unrecognized policy format at {path}.")


def rollout_policy_on_case(
    policy: ActionPolicy,
    *,
    env: CICDDiagnosisEnv,
    case_id: str,
) -> CaseEvaluationResult:
    state = env.reset(case_id=case_id)
    total_reward = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] | None = None

    while not terminated and not truncated:
        action = policy.predict_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    if info is None:
        raise RuntimeError("Policy rollout produced no terminal info.")

    summary = info["episode_summary"]
    expert_actions = list(env.dataset.get_expert_trajectory(case_id)["expert_actions"])
    tool_history = list(summary["tool_history"])
    required_tools = set(summary["required_tools_before_final"])

    # Success is defined in terms of evidence gathering rather than answer text because the current
    # policies only learn tool selection. That keeps BC/RL evaluation aligned with what the agent controls.
    answer_scores = dict(info.get("final_answer_scores", {}))
    failure_type_correct = bool(answer_scores.get("failure_type_correct", False))
    diagnosis_correct = bool(answer_scores.get("diagnosis_correct", False))
    fix_correct = bool(answer_scores.get("fix_correct", False))
    action_accuracy = _compute_action_accuracy(tool_history, expert_actions)
    success = bool(
        terminated
        and required_tools.issubset(tool_history)
        and failure_type_correct
        and diagnosis_correct
        and fix_correct
    )
    exact_trajectory_match = tool_history == expert_actions
    case_record = env.dataset.get_case(case_id)
    return CaseEvaluationResult(
        case_id=case_id,
        category=case_record["category"],
        split=case_record.get("split", "unknown"),
        total_reward=total_reward,
        success=success,
        action_accuracy=action_accuracy,
        failure_type_correct=failure_type_correct,
        diagnosis_correct=diagnosis_correct,
        fix_correct=fix_correct,
        exact_trajectory_match=exact_trajectory_match,
        terminated=terminated,
        truncated=truncated,
        steps=len(tool_history),
        tool_history=tool_history,
        expert_actions=expert_actions,
        required_tools_before_final=list(summary["required_tools_before_final"]),
    )


def evaluate_policy(
    policy: ActionPolicy,
    *,
    env: CICDDiagnosisEnv,
    case_ids: Sequence[str],
    policy_name: str,
    split: str,
) -> PolicyEvaluationReport:
    case_results = [rollout_policy_on_case(policy, env=env, case_id=case_id) for case_id in case_ids]
    aggregate = summarize_case_results(case_results, policy_name=policy_name, split=split)
    return PolicyEvaluationReport(aggregate=aggregate, cases=case_results)


def summarize_case_results(
    case_results: Sequence[CaseEvaluationResult],
    *,
    policy_name: str,
    split: str,
) -> AggregateEvaluationResult:
    if not case_results:
        nan = math.nan
        return AggregateEvaluationResult(
            policy_name=policy_name,
            split=split,
            num_cases=0,
            average_reward=nan,
            success_rate=nan,
            action_accuracy=nan,
            diagnosis_accuracy=nan,
            fix_accuracy=nan,
            exact_trajectory_match_rate=nan,
            average_steps=nan,
            category_breakdown={},
        )

    num_cases = len(case_results)
    average_reward = sum(result.total_reward for result in case_results) / num_cases
    success_rate = sum(result.success for result in case_results) / num_cases
    action_accuracy = sum(result.action_accuracy for result in case_results) / num_cases
    diagnosis_accuracy = sum(result.diagnosis_correct for result in case_results) / num_cases
    fix_accuracy = sum(result.fix_correct for result in case_results) / num_cases
    exact_trajectory_match_rate = sum(result.exact_trajectory_match for result in case_results) / num_cases
    average_steps = sum(result.steps for result in case_results) / num_cases

    category_breakdown: dict[str, list[CaseEvaluationResult]] = {}
    for result in case_results:
        category_breakdown.setdefault(result.category, []).append(result)

    category_metrics: dict[str, dict[str, float]] = {}
    for category, results in sorted(category_breakdown.items()):
        category_count = len(results)
        category_metrics[category] = {
            "num_cases": float(category_count),
            "average_reward": sum(item.total_reward for item in results) / category_count,
            "success_rate": sum(item.success for item in results) / category_count,
            "action_accuracy": sum(item.action_accuracy for item in results) / category_count,
            "diagnosis_accuracy": sum(item.diagnosis_correct for item in results) / category_count,
            "fix_accuracy": sum(item.fix_correct for item in results) / category_count,
            "exact_trajectory_match_rate": sum(item.exact_trajectory_match for item in results)
            / category_count,
            "average_steps": sum(item.steps for item in results) / category_count,
        }

    return AggregateEvaluationResult(
        policy_name=policy_name,
        split=split,
        num_cases=num_cases,
        average_reward=average_reward,
        success_rate=success_rate,
        action_accuracy=action_accuracy,
        diagnosis_accuracy=diagnosis_accuracy,
        fix_accuracy=fix_accuracy,
        exact_trajectory_match_rate=exact_trajectory_match_rate,
        average_steps=average_steps,
        category_breakdown=category_metrics,
    )


def compare_policies(
    *,
    bc_policy_path: str | Path,
    rl_policy_path: str | Path,
    dataset: CICDDiagnosisDataset | None = None,
    dataset_root: str | Path | None = None,
    split: str = "test",
    case_ids: Sequence[str] | None = None,
    max_steps: int = 5,
) -> ComparisonReport:
    dataset = dataset or load_dataset(dataset_root=dataset_root)
    selected_case_ids = list(case_ids) if case_ids is not None else dataset.list_case_ids(split=split)
    env = CICDDiagnosisEnv(dataset=dataset, max_steps=max_steps)

    bc_policy = load_policy(bc_policy_path)
    rl_policy = load_policy(rl_policy_path)
    bc_report = evaluate_policy(
        bc_policy,
        env=env,
        case_ids=selected_case_ids,
        policy_name="bc",
        split=split,
    )
    rl_report = evaluate_policy(
        rl_policy,
        env=env,
        case_ids=selected_case_ids,
        policy_name="rl",
        split=split,
    )

    return ComparisonReport(
        split=split,
        bc_report=bc_report,
        rl_report=rl_report,
        reward_gap_rl_minus_bc=rl_report.aggregate.average_reward - bc_report.aggregate.average_reward,
        success_gap_rl_minus_bc=rl_report.aggregate.success_rate - bc_report.aggregate.success_rate,
        trajectory_gap_rl_minus_bc=(
            rl_report.aggregate.exact_trajectory_match_rate - bc_report.aggregate.exact_trajectory_match_rate
        ),
    )


def _compute_action_accuracy(predicted_actions: Sequence[str], expert_actions: Sequence[str]) -> float:
    if not expert_actions:
        return math.nan
    max_length = max(len(predicted_actions), len(expert_actions))
    matches = 0
    for index in range(max_length):
        predicted = predicted_actions[index] if index < len(predicted_actions) else None
        expert = expert_actions[index] if index < len(expert_actions) else None
        if predicted == expert:
            matches += 1
    return matches / max_length


def write_comparison_report(report: ComparisonReport, output_path: str | Path) -> Path:
    output = Path(output_path)
    payload = asdict(report)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output
