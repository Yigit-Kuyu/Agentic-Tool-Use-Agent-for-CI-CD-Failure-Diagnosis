"""Direct preference optimization utilities for the CI/CD diagnosis action policy."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.actions import ACTION_TO_INDEX
from ..core.dataset import load_dataset
from ..core.env import CICDDiagnosisEnv
from .bc import BehaviorCloningPolicy, StateVectorizer


DEFAULT_DPO_PREFERENCES_PATH = Path("agi_tool/dataset/dpo_preferences_v2.jsonl")
DEFAULT_DPO_INIT_POLICY_PATH = Path("agi_tool/bc_policy.json")


@dataclass(frozen=True)
class DPOPreferenceExample:
    """One usable preference example at a single shared env state."""

    preference_id: str
    case_id: str
    pair_type: str
    split: str
    state_text: str
    feature_state: dict[str, int]
    chosen_action: str
    rejected_action: str
    reference_margin: float


@dataclass(frozen=True)
class DPODatasetBuildSummary:
    """Bookkeeping for usable vs skipped preference records."""

    total_records: int
    usable_examples: int
    skipped_examples: int
    skipped_by_reason: dict[str, int]


@dataclass(frozen=True)
class DPOTrainConfig:
    """Hyperparameters for lightweight DPO fine-tuning."""

    epochs: int = 200
    learning_rate: float = 0.05
    beta: float = 0.5
    l2_weight: float = 1e-5
    seed: int = 13
    max_steps: int = 6


@dataclass(frozen=True)
class DPOTrainResult:
    """Summary returned by DPO training."""

    train_examples: int
    val_examples: int
    train_preference_accuracy: float
    val_preference_accuracy: float
    epochs_completed: int
    loss_history: list[float]
    build_summary: DPODatasetBuildSummary


class DPOPolicy(BehaviorCloningPolicy):
    """BC-compatible policy wrapper produced by DPO fine-tuning."""

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        payload = {
            "vectorizer": self.vectorizer.to_dict(),
            "model": self.model.to_dict(),
            "training_actions": list(ACTION_TO_INDEX.keys()),
            "policy_type": "dpo",
        }
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output

    @classmethod
    def from_behavior_cloning(cls, bc_policy: BehaviorCloningPolicy) -> "DPOPolicy":
        return cls(
            vectorizer=StateVectorizer.from_dict(bc_policy.vectorizer.to_dict()),
            model=bc_policy.model.from_dict(bc_policy.model.to_dict()),
        )


class DPOPreferenceDatasetBuilder:
    """Builds action-level preference examples from trajectory preference records."""

    def __init__(
        self,
        *,
        dataset_root: str | Path | None = None,
        preference_path: str | Path | None = None,
        reference_policy: BehaviorCloningPolicy,
        max_steps: int = 6,
    ) -> None:
        self.dataset = load_dataset(dataset_root=dataset_root)
        self.preference_path = Path(preference_path or DEFAULT_DPO_PREFERENCES_PATH)
        self.reference_policy = reference_policy
        self.env = CICDDiagnosisEnv(dataset=self.dataset, max_steps=max_steps)

    def build_examples(
        self,
        *,
        train_split: str = "train",
        validation_split: str = "test",
    ) -> tuple[list[DPOPreferenceExample], list[DPOPreferenceExample], DPODatasetBuildSummary]:
        records = self._read_preference_records()
        skipped_by_reason: dict[str, int] = {}
        train_examples: list[DPOPreferenceExample] = []
        val_examples: list[DPOPreferenceExample] = []

        for record in records:
            example, skip_reason = self._build_example(record)
            if example is None:
                skipped_by_reason[skip_reason] = skipped_by_reason.get(skip_reason, 0) + 1
                continue
            if example.split == train_split:
                train_examples.append(example)
            elif example.split == validation_split:
                val_examples.append(example)

        summary = DPODatasetBuildSummary(
            total_records=len(records),
            usable_examples=len(train_examples) + len(val_examples),
            skipped_examples=sum(skipped_by_reason.values()),
            skipped_by_reason=skipped_by_reason,
        )
        return train_examples, val_examples, summary

    def _build_example(
        self,
        record: Mapping[str, Any],
    ) -> tuple[DPOPreferenceExample | None, str]:
        chosen_trajectory = _parse_trajectory(record["chosen"])
        rejected_trajectory = _parse_trajectory(record["rejected"])
        divergence_index = _find_divergence_index(chosen_trajectory, rejected_trajectory)
        if divergence_index is None:
            return None, "same_trajectory"

        chosen_action = chosen_trajectory[divergence_index]
        rejected_action = rejected_trajectory[divergence_index]
        if chosen_action == rejected_action:
            return None, "same_divergence_action"
        if chosen_action not in ACTION_TO_INDEX or rejected_action not in ACTION_TO_INDEX:
            return None, "unknown_action"

        state = self._build_shared_state(record["case_id"], chosen_trajectory[:divergence_index], rejected_trajectory[:divergence_index])
        reference_probs = self.reference_policy.predict_action_probabilities(state)
        reference_margin = math.log(reference_probs[chosen_action] + 1e-12) - math.log(
            reference_probs[rejected_action] + 1e-12
        )
        return (
            DPOPreferenceExample(
                preference_id=str(record["preference_id"]),
                case_id=str(record["case_id"]),
                pair_type=str(record["pair_type"]),
                split=str(record.get("split", "train")),
                state_text=str(state["state_text"]),
                feature_state=dict(state["feature_state"]),
                chosen_action=chosen_action,
                rejected_action=rejected_action,
                reference_margin=float(reference_margin),
            ),
            "",
        )

    def _build_shared_state(
        self,
        case_id: str,
        chosen_prefix: Sequence[str],
        rejected_prefix: Sequence[str],
    ) -> dict[str, Any]:
        if list(chosen_prefix) != list(rejected_prefix):
            raise ValueError("Chosen and rejected prefixes must match before divergence.")

        self.env.reset(case_id=case_id)
        for action in chosen_prefix:
            self.env.step(action)
        return self.env.get_state()

    def _read_preference_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.preference_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records


def train_dpo_policy(
    *,
    init_policy_path: str | Path | None = None,
    preference_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    train_split: str = "train",
    validation_split: str = "test",
    config: DPOTrainConfig | None = None,
) -> tuple[DPOPolicy, DPOTrainResult]:
    config = config or DPOTrainConfig()
    reference_policy = _load_reference_policy(init_policy_path)
    if not reference_policy.model.is_linear_only():
        raise ValueError(
            "The current DPO trainer assumes a linear BC policy checkpoint. "
            "Train DPO from an older linear checkpoint or extend dpo.py for nonlinear BC."
        )
    policy = DPOPolicy.from_behavior_cloning(reference_policy)

    builder = DPOPreferenceDatasetBuilder(
        dataset_root=dataset_root,
        preference_path=preference_path,
        reference_policy=reference_policy,
        max_steps=config.max_steps,
    )
    train_examples, val_examples, build_summary = builder.build_examples(
        train_split=train_split,
        validation_split=validation_split,
    )
    if not train_examples:
        raise ValueError("No usable DPO training examples were found.")

    loss_history = _fit_dpo_policy(policy, train_examples, config=config)
    train_accuracy = evaluate_dpo_preferences(policy, train_examples)
    val_accuracy = evaluate_dpo_preferences(policy, val_examples)
    return policy, DPOTrainResult(
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        train_preference_accuracy=train_accuracy,
        val_preference_accuracy=val_accuracy,
        epochs_completed=config.epochs,
        loss_history=loss_history,
        build_summary=build_summary,
    )


def evaluate_dpo_preferences(policy: DPOPolicy, examples: Sequence[DPOPreferenceExample]) -> float:
    if not examples:
        return math.nan
    correct = 0
    for example in examples:
        margin = _policy_margin(policy, example)
        if margin > example.reference_margin:
            correct += 1
    return correct / len(examples)


def _fit_dpo_policy(
    policy: DPOPolicy,
    examples: Sequence[DPOPreferenceExample],
    *,
    config: DPOTrainConfig,
) -> list[float]:
    rng = random.Random(config.seed)
    loss_history: list[float] = []
    weights = policy.model.weights
    bias = policy.model.bias
    if weights is None or bias is None:
        raise ValueError("DPO policy must start from a trained BC policy.")

    for _ in range(config.epochs):
        batch = list(examples)
        rng.shuffle(batch)
        epoch_loss = 0.0

        for example in batch:
            feature_row = policy.vectorizer.transform([_example_to_bc_like(example)])[0]
            chosen_index = ACTION_TO_INDEX[example.chosen_action]
            rejected_index = ACTION_TO_INDEX[example.rejected_action]

            current_margin = float(feature_row @ weights[:, chosen_index] + bias[chosen_index])
            current_margin -= float(feature_row @ weights[:, rejected_index] + bias[rejected_index])
            z = config.beta * (current_margin - example.reference_margin)
            sigmoid = 1.0 / (1.0 + math.exp(-z))
            epoch_loss += -math.log(sigmoid + 1e-12)

            # This is the key DPO update: push the chosen action logit above the rejected one
            # relative to the frozen reference margin, using the shared state at trajectory divergence.
            scale = config.learning_rate * config.beta * (1.0 - sigmoid)
            weights *= 1.0 - config.learning_rate * config.l2_weight
            weights[:, chosen_index] += scale * feature_row
            weights[:, rejected_index] -= scale * feature_row
            bias[chosen_index] += scale
            bias[rejected_index] -= scale

        loss_history.append(epoch_loss / len(batch))

    return loss_history


def _policy_margin(policy: DPOPolicy, example: DPOPreferenceExample) -> float:
    feature_row = policy.vectorizer.transform([_example_to_bc_like(example)])[0]
    weights = policy.model.weights
    bias = policy.model.bias
    if weights is None or bias is None:
        raise ValueError("Policy model is not initialized.")
    chosen_index = ACTION_TO_INDEX[example.chosen_action]
    rejected_index = ACTION_TO_INDEX[example.rejected_action]
    return float(feature_row @ weights[:, chosen_index] + bias[chosen_index]) - float(
        feature_row @ weights[:, rejected_index] + bias[rejected_index]
    )


def _example_to_bc_like(example: DPOPreferenceExample) -> Any:
    class _StateLike:
        state_text = example.state_text
        feature_state = example.feature_state

    return _StateLike


def _parse_trajectory(candidate_text: str) -> list[str]:
    first_line = candidate_text.splitlines()[0].strip()
    return [action.strip() for action in first_line.split("->")]


def _find_divergence_index(chosen: Sequence[str], rejected: Sequence[str]) -> int | None:
    max_prefix = min(len(chosen), len(rejected))
    for index in range(max_prefix):
        if chosen[index] != rejected[index]:
            return index
    if len(chosen) != len(rejected):
        return max_prefix
    return None


def _load_reference_policy(init_policy_path: str | Path | None) -> BehaviorCloningPolicy:
    resolved = Path(init_policy_path) if init_policy_path is not None else DEFAULT_DPO_INIT_POLICY_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Initial BC policy not found at {resolved}. Train behavior cloning first or pass --init-policy."
        )
    return BehaviorCloningPolicy.load(resolved)
