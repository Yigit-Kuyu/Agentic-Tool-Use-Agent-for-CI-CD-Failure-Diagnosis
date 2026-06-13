"""Inference helpers for running a saved policy on a diagnosis case."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..core.dataset import CICDDiagnosisDataset, load_dataset
from ..core.env import CICDDiagnosisEnv
from ..training.eval import load_policy


class InferencePolicy(Protocol):
    """Minimal interface required for diagnosis-time action selection."""

    def predict_action(self, state: dict[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class DiagnosisRunResult:
    """Single policy rollout plus the generated final diagnosis."""

    policy_name: str
    case_id: str
    split: str
    category: str
    user_query: str
    tool_history: list[str]
    total_reward: float
    terminated: bool
    truncated: bool
    final_answer: dict[str, Any]
    final_answer_scores: dict[str, bool]
    final_answer_reference: dict[str, Any]
    reward_breakdown: dict[str, float]
    web_augmentation_query: str


def run_diagnosis(
    *,
    policy: InferencePolicy,
    dataset: CICDDiagnosisDataset | None = None,
    dataset_root: str | Path | None = None,
    case_id: str | None = None,
    split: str = "test",
    max_steps: int = 5,
    seed: int | None = None,
    policy_name: str = "policy",
) -> DiagnosisRunResult:
    dataset = dataset or load_dataset(dataset_root=dataset_root)
    env = CICDDiagnosisEnv(dataset=dataset, max_steps=max_steps, seed=seed)
    state = env.reset(case_id=case_id, split=split)
    total_reward = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] | None = None

    while not terminated and not truncated:
        action = policy.predict_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    if info is None:
        raise RuntimeError("Inference rollout did not produce terminal info.")

    case_record = dataset.get_case(info["case_id"])
    summary = info["episode_summary"]
    return DiagnosisRunResult(
        policy_name=policy_name,
        case_id=info["case_id"],
        split=case_record.get("split", split),
        category=case_record["category"],
        user_query=case_record["user_query"],
        tool_history=list(summary["tool_history"]),
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        final_answer=dict(info["final_answer"]),
        final_answer_scores=dict(info["final_answer_scores"]),
        final_answer_reference=dict(info["final_answer_reference"]),
        reward_breakdown=dict(info["reward_breakdown"]),
        web_augmentation_query=case_record["web_augmentation_query"],
    )


def run_diagnosis_from_path(
    *,
    policy_path: str | Path,
    dataset_root: str | Path | None = None,
    case_id: str | None = None,
    split: str = "test",
    max_steps: int = 5,
    seed: int | None = None,
) -> DiagnosisRunResult:
    policy = load_policy(policy_path)
    return run_diagnosis(
        policy=policy,
        dataset_root=dataset_root,
        case_id=case_id,
        split=split,
        max_steps=max_steps,
        seed=seed,
        policy_name=Path(policy_path).stem,
    )


def save_diagnosis_result(result: DiagnosisRunResult, output_path: str | Path) -> Path:
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)
    return output


def load_diagnosis_result(input_path: str | Path) -> DiagnosisRunResult:
    input_file = Path(input_path)
    with input_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DiagnosisRunResult(**payload)


def render_final_answer(result: DiagnosisRunResult) -> str:
    answer = result.final_answer
    evidence = answer.get("evidence", [])
    evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "- None"
    # Keep the text renderer compact because this is meant for quick CLI diagnosis checks.
    return (
        f"Case: {result.case_id}\n"
        f"Category: {result.category}\n"
        f"Policy: {result.policy_name}\n"
        f"Tool trajectory: {' -> '.join(result.tool_history)}\n"
        f"Diagnosis: {answer.get('diagnosis', '')}\n"
        f"Fix: {answer.get('fix', '')}\n"
        f"Confidence: {answer.get('confidence', '')}\n"
        f"Evidence:\n{evidence_text}"
    )
