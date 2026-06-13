"""Behavior cloning utilities for the CI/CD diagnosis agent."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.actions import ACTION_TO_INDEX, INDEX_TO_ACTION, TRAINING_ACTIONS
from ..core.dataset import CICDDiagnosisDataset, load_dataset
from ..core.env import CICDDiagnosisEnv


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class BCExample:
    """One supervised state-action pair for behavior cloning."""

    case_id: str
    state_text: str
    feature_state: dict[str, int]
    target_action: str
    step_index: int


@dataclass(frozen=True)
class DatasetSplit:
    """Case-level train/validation split used by the BC trainer."""

    train_case_ids: list[str]
    val_case_ids: list[str]


@dataclass(frozen=True)
class TrainResult:
    """Summary returned after behavior-cloning training."""

    train_examples: int
    val_examples: int
    train_accuracy: float
    val_accuracy: float
    epochs_completed: int
    loss_history: list[float]


class BehaviorCloningDatasetBuilder:
    """Roll out expert trajectories through the env to build training examples."""

    def __init__(
        self,
        dataset: CICDDiagnosisDataset | None = None,
        env: CICDDiagnosisEnv | None = None,
        *,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.dataset = dataset or load_dataset(dataset_root=dataset_root)
        self.env = env or CICDDiagnosisEnv(dataset=self.dataset)

    def build_examples(self, case_ids: Sequence[str]) -> list[BCExample]:
        examples: list[BCExample] = []
        for case_id in case_ids:
            self.env.reset(case_id=case_id)
            trajectory = self.dataset.get_expert_trajectory(case_id)
            for step_index, action in enumerate(trajectory["expert_actions"]):
                state = self.env.get_state()
                examples.append(
                    BCExample(
                        case_id=case_id,
                        state_text=state["state_text"],
                        feature_state=dict(state["feature_state"]),
                        target_action=action,
                        step_index=step_index,
                    )
                )
                self.env.step(action)
        return examples

    def split_case_ids(
        self,
        *,
        split: str = "train",
        validation_fraction: float = 0.2,
        seed: int = 7,
    ) -> DatasetSplit:
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0).")

        case_ids = self.dataset.list_case_ids(split=split)
        if len(case_ids) < 2 or validation_fraction == 0.0:
            return DatasetSplit(train_case_ids=case_ids, val_case_ids=[])

        rng = random.Random(seed)
        shuffled = list(case_ids)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * validation_fraction)))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1
        return DatasetSplit(
            train_case_ids=sorted(shuffled[val_count:]),
            val_case_ids=sorted(shuffled[:val_count]),
        )


class StateVectorizer:
    """Hybrid vectorizer over numeric env features and state text tokens."""

    def __init__(self, *, max_tokens: int = 512, min_token_count: int = 1) -> None:
        self.max_tokens = max_tokens
        self.min_token_count = min_token_count
        self.numeric_feature_names: list[str] = []
        self.token_to_index: dict[str, int] = {}

    def fit(self, examples: Sequence[BCExample]) -> "StateVectorizer":
        if not examples:
            raise ValueError("Cannot fit vectorizer on an empty example list.")

        self.numeric_feature_names = sorted(examples[0].feature_state.keys())
        token_counts: dict[str, int] = {}
        for example in examples:
            for token in self._tokenize(example.state_text):
                token_counts[token] = token_counts.get(token, 0) + 1

        kept_tokens = [
            token for token, count in sorted(token_counts.items()) if count >= self.min_token_count
        ][: self.max_tokens]
        self.token_to_index = {token: index for index, token in enumerate(kept_tokens)}
        return self

    def fit_transform(self, examples: Sequence[BCExample]) -> np.ndarray:
        self.fit(examples)
        return self.transform(examples)

    def transform(self, examples: Sequence[BCExample]) -> np.ndarray:
        if not self.numeric_feature_names:
            raise ValueError("Vectorizer has not been fit yet.")

        num_rows = len(examples)
        num_numeric = len(self.numeric_feature_names)
        num_text = len(self.token_to_index)
        matrix = np.zeros((num_rows, num_numeric + num_text), dtype=np.float64)

        for row_index, example in enumerate(examples):
            for feature_index, feature_name in enumerate(self.numeric_feature_names):
                matrix[row_index, feature_index] = float(example.feature_state.get(feature_name, 0))

            # Binary token indicators work well here because the state texts are short and repetitive.
            seen_tokens = set(self._tokenize(example.state_text))
            for token in seen_tokens:
                token_index = self.token_to_index.get(token)
                if token_index is None:
                    continue
                matrix[row_index, num_numeric + token_index] = 1.0

        return matrix

    def transform_state(self, state: Mapping[str, Any]) -> np.ndarray:
        example = BCExample(
            case_id=str(state.get("case_id", "")),
            state_text=str(state["state_text"]),
            feature_state=dict(state["feature_state"]),
            target_action=TRAINING_ACTIONS[0],
            step_index=int(state["feature_state"].get("step_count", 0)),
        )
        return self.transform([example])

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "min_token_count": self.min_token_count,
            "numeric_feature_names": self.numeric_feature_names,
            "token_to_index": self.token_to_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateVectorizer":
        vectorizer = cls(
            max_tokens=int(payload["max_tokens"]),
            min_token_count=int(payload["min_token_count"]),
        )
        vectorizer.numeric_feature_names = list(payload["numeric_feature_names"])
        vectorizer.token_to_index = {
            str(token): int(index) for token, index in dict(payload["token_to_index"]).items()
        }
        return vectorizer

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())


class SoftmaxPolicyModel:
    """Small residual-MLP multiclass policy trained with gradient descent."""

    def __init__(self, *, hidden_dim: int = 64) -> None:
        self.hidden_dim = hidden_dim
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.hidden_weights: np.ndarray | None = None
        self.hidden_bias: np.ndarray | None = None
        self.output_weights: np.ndarray | None = None
        self.num_classes = len(TRAINING_ACTIONS)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 300,
        learning_rate: float = 0.1,
        l2_weight: float = 1e-4,
        seed: int = 7,
    ) -> list[float]:
        if features.ndim != 2:
            raise ValueError("features must be a 2D array.")
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array.")

        num_samples, num_features = features.shape
        rng = np.random.default_rng(seed)
        self.weights = np.zeros((num_features, self.num_classes), dtype=np.float64)
        self.bias = np.zeros(self.num_classes, dtype=np.float64)
        self.hidden_weights = rng.normal(0.0, 0.05, size=(num_features, self.hidden_dim))
        self.hidden_bias = np.zeros(self.hidden_dim, dtype=np.float64)
        self.output_weights = rng.normal(0.0, 0.05, size=(self.hidden_dim, self.num_classes))
        loss_history: list[float] = []

        one_hot = np.eye(self.num_classes, dtype=np.float64)[labels]

        for _ in range(epochs):
            hidden_linear = features @ self.hidden_weights + self.hidden_bias
            hidden_activation = np.maximum(hidden_linear, 0.0)
            logits = features @ self.weights + self.bias + hidden_activation @ self.output_weights
            probabilities = self._softmax(logits)

            # This is the only non-trivial math block: compute cross-entropy and its gradients.
            loss = -np.mean(np.sum(one_hot * np.log(probabilities + 1e-12), axis=1))
            loss += 0.5 * l2_weight * float(np.sum(self.weights * self.weights))
            loss += 0.5 * l2_weight * float(np.sum(self.hidden_weights * self.hidden_weights))
            loss += 0.5 * l2_weight * float(np.sum(self.output_weights * self.output_weights))
            loss_history.append(float(loss))

            error = (probabilities - one_hot) / num_samples
            grad_weights = features.T @ error + l2_weight * self.weights
            grad_bias = np.sum(error, axis=0)
            grad_output_weights = hidden_activation.T @ error + l2_weight * self.output_weights
            hidden_error = (error @ self.output_weights.T) * (hidden_linear > 0.0)
            grad_hidden_weights = features.T @ hidden_error + l2_weight * self.hidden_weights
            grad_hidden_bias = np.sum(hidden_error, axis=0)

            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias
            self.output_weights -= learning_rate * grad_output_weights
            self.hidden_weights -= learning_rate * grad_hidden_weights
            self.hidden_bias -= learning_rate * grad_hidden_bias

        return loss_history

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._require_fit()
        hidden_linear = features @ self.hidden_weights + self.hidden_bias
        hidden_activation = np.maximum(hidden_linear, 0.0)
        logits = features @ self.weights + self.bias + hidden_activation @ self.output_weights
        return self._softmax(logits)

    def to_dict(self) -> dict[str, Any]:
        self._require_fit()
        return {
            "model_type": "residual_mlp",
            "hidden_dim": self.hidden_dim,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "hidden_weights": self.hidden_weights.tolist(),
            "hidden_bias": self.hidden_bias.tolist(),
            "output_weights": self.output_weights.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SoftmaxPolicyModel":
        hidden_dim = int(payload.get("hidden_dim", 0) or 0)
        model = cls(hidden_dim=max(hidden_dim, 1))
        model.weights = np.asarray(payload["weights"], dtype=np.float64)
        model.bias = np.asarray(payload["bias"], dtype=np.float64)
        if "hidden_weights" in payload and "hidden_bias" in payload and "output_weights" in payload:
            model.hidden_weights = np.asarray(payload["hidden_weights"], dtype=np.float64)
            model.hidden_bias = np.asarray(payload["hidden_bias"], dtype=np.float64)
            model.output_weights = np.asarray(payload["output_weights"], dtype=np.float64)
            model.hidden_dim = model.hidden_bias.shape[0]
        else:
            # Backward compatibility for older linear-only checkpoints.
            model.hidden_dim = 1
            model.hidden_weights = np.zeros((model.weights.shape[0], 1), dtype=np.float64)
            model.hidden_bias = np.zeros(1, dtype=np.float64)
            model.output_weights = np.zeros((1, model.num_classes), dtype=np.float64)
        return model

    def is_linear_only(self) -> bool:
        self._require_fit()
        return bool(
            np.allclose(self.hidden_weights, 0.0)
            and np.allclose(self.hidden_bias, 0.0)
            and np.allclose(self.output_weights, 0.0)
        )

    def _require_fit(self) -> None:
        if (
            self.weights is None
            or self.bias is None
            or self.hidden_weights is None
            or self.hidden_bias is None
            or self.output_weights is None
        ):
            raise ValueError("Model has not been fit yet.")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class BehaviorCloningPolicy:
    """End-to-end BC policy with vectorization, training, and inference helpers."""

    def __init__(
        self,
        *,
        vectorizer: StateVectorizer | None = None,
        model: SoftmaxPolicyModel | None = None,
    ) -> None:
        self.vectorizer = vectorizer or StateVectorizer()
        self.model = model or SoftmaxPolicyModel()

    def fit(
        self,
        train_examples: Sequence[BCExample],
        *,
        epochs: int = 300,
        learning_rate: float = 0.1,
        l2_weight: float = 1e-4,
        seed: int = 7,
    ) -> list[float]:
        features = self.vectorizer.fit_transform(train_examples)
        labels = np.asarray([ACTION_TO_INDEX[example.target_action] for example in train_examples], dtype=np.int64)
        return self.model.fit(
            features,
            labels,
            epochs=epochs,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            seed=seed,
        )

    def predict_action(self, state: Mapping[str, Any]) -> str:
        features = self.vectorizer.transform_state(state)
        action_index = int(self.model.predict(features)[0])
        return INDEX_TO_ACTION[action_index]

    def predict_action_probabilities(self, state: Mapping[str, Any]) -> dict[str, float]:
        features = self.vectorizer.transform_state(state)
        probabilities = self.model.predict_proba(features)[0]
        return {INDEX_TO_ACTION[index]: float(prob) for index, prob in enumerate(probabilities)}

    def evaluate(self, examples: Sequence[BCExample]) -> float:
        if not examples:
            return math.nan
        features = self.vectorizer.transform(examples)
        labels = np.asarray([ACTION_TO_INDEX[example.target_action] for example in examples], dtype=np.int64)
        predictions = self.model.predict(features)
        return float(np.mean(predictions == labels))

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        payload = {
            "vectorizer": self.vectorizer.to_dict(),
            "model": self.model.to_dict(),
            "training_actions": list(TRAINING_ACTIONS),
        }
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output

    @classmethod
    def load(cls, model_path: str | Path) -> "BehaviorCloningPolicy":
        with Path(model_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            vectorizer=StateVectorizer.from_dict(payload["vectorizer"]),
            model=SoftmaxPolicyModel.from_dict(payload["model"]),
        )


def train_behavior_cloning_policy(
    *,
    dataset: CICDDiagnosisDataset | None = None,
    env: CICDDiagnosisEnv | None = None,
    dataset_root: str | Path | None = None,
    split: str = "train",
    validation_fraction: float = 0.2,
    seed: int = 7,
    epochs: int = 300,
    learning_rate: float = 0.1,
    l2_weight: float = 1e-4,
    hidden_dim: int = 64,
) -> tuple[BehaviorCloningPolicy, TrainResult]:
    dataset = dataset or load_dataset(dataset_root=dataset_root)
    env = env or CICDDiagnosisEnv(dataset=dataset)
    builder = BehaviorCloningDatasetBuilder(dataset=dataset, env=env)
    case_split = builder.split_case_ids(
        split=split,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    train_examples = builder.build_examples(case_split.train_case_ids)
    val_examples = builder.build_examples(case_split.val_case_ids)

    policy = BehaviorCloningPolicy(model=SoftmaxPolicyModel(hidden_dim=hidden_dim))
    loss_history = policy.fit(
        train_examples,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_weight=l2_weight,
        seed=seed,
    )
    train_accuracy = policy.evaluate(train_examples)
    val_accuracy = policy.evaluate(val_examples)

    return policy, TrainResult(
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        epochs_completed=epochs,
        loss_history=loss_history,
    )
