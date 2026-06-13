"""RL fine-tuning utilities for the CI/CD diagnosis agent."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.actions import ACTION_TO_INDEX, INDEX_TO_ACTION, TRAINING_ACTIONS
from ..core.dataset import CICDDiagnosisDataset, load_dataset
from ..core.env import CICDDiagnosisEnv
from .bc import BehaviorCloningDatasetBuilder, BehaviorCloningPolicy, DatasetSplit, StateVectorizer


DEFAULT_BC_POLICY_PATH = Path("agi_tool/bc_policy.json")


@dataclass(frozen=True)
class RLTrainConfig:
    """Hyperparameters for small DQN-style fine-tuning."""

    episodes: int = 300
    gamma: float = 0.95
    learning_rate: float = 0.01
    epsilon_start: float = 0.25
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.99
    weight_decay: float = 1e-5
    td_clip: float = 5.0
    evaluation_interval: int = 25
    validation_fraction: float = 0.2
    seed: int = 11
    max_steps: int = 5
    hidden_dim: int = 64
    replay_capacity: int = 4096
    batch_size: int = 32
    warmup_steps: int = 32
    target_sync_steps: int = 50
    gradient_steps_per_env_step: int = 1
    huber_delta: float = 1.0


@dataclass(frozen=True)
class EvaluationResult:
    """Greedy rollout summary for a set of evaluation cases."""

    average_reward: float
    success_rate: float
    average_steps: float
    num_cases: int


@dataclass(frozen=True)
class RLTrainResult:
    """Summary produced by RL fine-tuning."""

    episodes_completed: int
    final_epsilon: float
    train_case_count: int
    val_case_count: int
    total_env_steps: int
    replay_size: int
    reward_history: list[float]
    evaluation_history: list[dict[str, Any]]
    final_train_eval: EvaluationResult
    final_val_eval: EvaluationResult


@dataclass(frozen=True)
class ReplayTransition:
    """One transition stored in the replay buffer."""

    state_features: np.ndarray
    action_index: int
    reward: float
    next_state_features: np.ndarray
    done: bool


@dataclass(frozen=True)
class _QSnapshot:
    linear_weights: np.ndarray
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray


class ReplayBuffer:
    """Ring-buffer replay storage used by the DQN trainer."""

    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._storage: list[ReplayTransition | None] = [None] * capacity
        self._next_index = 0
        self._size = 0

    def add(self, transition: ReplayTransition) -> None:
        self._storage[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, *, rng: random.Random) -> list[ReplayTransition]:
        if batch_size > self._size:
            raise ValueError("Not enough transitions in replay buffer to sample the requested batch size.")
        indices = rng.sample(range(self._size), batch_size)
        return [self._storage[index] for index in indices if self._storage[index] is not None]

    def __len__(self) -> int:
        return self._size


class DQNQNetwork:
    """Small MLP Q-network with a BC-initialized linear skip branch."""

    def __init__(self, *, num_actions: int = len(TRAINING_ACTIONS), hidden_dim: int = 64) -> None:
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.linear_weights: np.ndarray | None = None
        self.hidden_weights: np.ndarray | None = None
        self.hidden_bias: np.ndarray | None = None
        self.output_weights: np.ndarray | None = None
        self.output_bias: np.ndarray | None = None

    def initialize(self, feature_dim: int, *, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        self.linear_weights = np.zeros((feature_dim, self.num_actions), dtype=np.float64)
        self.hidden_weights = rng.normal(0.0, 0.05, size=(feature_dim, self.hidden_dim)).astype(np.float64)
        self.hidden_bias = np.zeros(self.hidden_dim, dtype=np.float64)
        self.output_weights = np.zeros((self.hidden_dim, self.num_actions), dtype=np.float64)
        self.output_bias = np.zeros(self.num_actions, dtype=np.float64)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        self._require_initialized()
        if features.ndim != 2:
            raise ValueError("features must be a 2D array.")
        hidden_pre = features @ self.hidden_weights + self.hidden_bias
        hidden = np.maximum(hidden_pre, 0.0)
        return features @ self.linear_weights + hidden @ self.output_weights + self.output_bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.q_values(features), axis=1)

    def train_batch(
        self,
        *,
        state_batch: np.ndarray,
        action_batch: np.ndarray,
        target_batch: np.ndarray,
        learning_rate: float,
        weight_decay: float,
        td_clip: float,
        huber_delta: float,
    ) -> float:
        self._require_initialized()
        hidden_pre = state_batch @ self.hidden_weights + self.hidden_bias
        hidden = np.maximum(hidden_pre, 0.0)
        q_values = state_batch @ self.linear_weights + hidden @ self.output_weights + self.output_bias
        chosen_q = q_values[np.arange(len(action_batch)), action_batch]

        td_error = np.clip(target_batch - chosen_q, -td_clip, td_clip)
        abs_error = np.abs(td_error)
        quadratic = np.minimum(abs_error, huber_delta)
        linear = abs_error - quadratic
        loss = np.mean(0.5 * quadratic * quadratic + huber_delta * linear)

        # This is the main DQN gradient step: compute a minibatch TD loss from replayed
        # transitions and push the online Q-network toward the frozen target-network values.
        grad_pred = np.where(abs_error <= huber_delta, chosen_q - target_batch, huber_delta * np.sign(chosen_q - target_batch))
        grad_pred /= len(action_batch)

        action_grads = np.zeros_like(q_values)
        action_grads[np.arange(len(action_batch)), action_batch] = grad_pred

        grad_output_bias = np.sum(action_grads, axis=0)
        grad_linear_weights = state_batch.T @ action_grads + weight_decay * self.linear_weights
        grad_output_weights = hidden.T @ action_grads + weight_decay * self.output_weights

        hidden_grad = action_grads @ self.output_weights.T
        hidden_grad[hidden_pre <= 0.0] = 0.0
        grad_hidden_weights = state_batch.T @ hidden_grad + weight_decay * self.hidden_weights
        grad_hidden_bias = np.sum(hidden_grad, axis=0)

        self.linear_weights -= learning_rate * grad_linear_weights
        self.output_weights -= learning_rate * grad_output_weights
        self.output_bias -= learning_rate * grad_output_bias
        self.hidden_weights -= learning_rate * grad_hidden_weights
        self.hidden_bias -= learning_rate * grad_hidden_bias
        return float(loss)

    def to_dict(self) -> dict[str, Any]:
        self._require_initialized()
        return {
            "network_type": "small_dqn",
            "hidden_dim": self.hidden_dim,
            "linear_weights": self.linear_weights.tolist(),
            "hidden_weights": self.hidden_weights.tolist(),
            "hidden_bias": self.hidden_bias.tolist(),
            "output_weights": self.output_weights.tolist(),
            "output_bias": self.output_bias.tolist(),
        }

    def snapshot(self) -> _QSnapshot:
        self._require_initialized()
        return _QSnapshot(
            linear_weights=self.linear_weights.copy(),
            hidden_weights=self.hidden_weights.copy(),
            hidden_bias=self.hidden_bias.copy(),
            output_weights=self.output_weights.copy(),
            output_bias=self.output_bias.copy(),
        )

    def restore(self, snapshot: _QSnapshot) -> None:
        self.linear_weights = snapshot.linear_weights.copy()
        self.hidden_weights = snapshot.hidden_weights.copy()
        self.hidden_bias = snapshot.hidden_bias.copy()
        self.output_weights = snapshot.output_weights.copy()
        self.output_bias = snapshot.output_bias.copy()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DQNQNetwork":
        if "weights" in payload and "bias" in payload:
            # Backward compatibility for the earlier linear-Q checkpoint format.
            linear_weights = np.asarray(payload["weights"], dtype=np.float64)
            output_bias = np.asarray(payload["bias"], dtype=np.float64)
            model = cls(hidden_dim=0)
            feature_dim, num_actions = linear_weights.shape
            model.num_actions = num_actions
            model.linear_weights = linear_weights
            model.hidden_weights = np.zeros((feature_dim, 0), dtype=np.float64)
            model.hidden_bias = np.zeros(0, dtype=np.float64)
            model.output_weights = np.zeros((0, num_actions), dtype=np.float64)
            model.output_bias = output_bias
            return model

        model = cls(
            hidden_dim=int(payload["hidden_dim"]),
        )
        model.linear_weights = np.asarray(payload["linear_weights"], dtype=np.float64)
        model.hidden_weights = np.asarray(payload["hidden_weights"], dtype=np.float64)
        model.hidden_bias = np.asarray(payload["hidden_bias"], dtype=np.float64)
        model.output_weights = np.asarray(payload["output_weights"], dtype=np.float64)
        model.output_bias = np.asarray(payload["output_bias"], dtype=np.float64)
        model.num_actions = model.output_bias.shape[0]
        return model

    def _require_initialized(self) -> None:
        if (
            self.linear_weights is None
            or self.hidden_weights is None
            or self.hidden_bias is None
            or self.output_weights is None
            or self.output_bias is None
        ):
            raise ValueError("Q network is not initialized.")


class RLFineTunedPolicy:
    """Greedy action policy backed by a small DQN-style Q-network."""

    def __init__(
        self,
        *,
        vectorizer: StateVectorizer,
        q_model: DQNQNetwork | None = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.q_model = q_model or DQNQNetwork()

    @classmethod
    def from_behavior_cloning(
        cls,
        bc_policy: BehaviorCloningPolicy,
        *,
        hidden_dim: int = 64,
        seed: int = 11,
    ) -> "RLFineTunedPolicy":
        vectorizer = StateVectorizer.from_dict(bc_policy.vectorizer.to_dict())
        q_model = DQNQNetwork(hidden_dim=hidden_dim)
        feature_dim = len(vectorizer.numeric_feature_names) + len(vectorizer.token_to_index)
        q_model.initialize(feature_dim, rng=np.random.default_rng(seed))

        bc_weights = bc_policy.model.weights
        bc_bias = bc_policy.model.bias
        if bc_weights is None or bc_bias is None:
            raise ValueError("Behavior-cloning policy must be trained before RL warm start.")

        # The linear skip branch preserves the BC policy's action ordering on day one, while the
        # hidden branch gives DQN room to learn nonlinear refinements from replayed transitions.
        q_model.linear_weights = np.asarray(bc_weights, dtype=np.float64).copy()
        q_model.output_bias = np.asarray(bc_bias, dtype=np.float64).copy()
        return cls(vectorizer=vectorizer, q_model=q_model)

    def predict_action(self, state: Mapping[str, Any]) -> str:
        features = self.vectorizer.transform_state(state)
        action_index = int(self.q_model.predict(features)[0])
        return INDEX_TO_ACTION[action_index]

    def predict_q_values(self, state: Mapping[str, Any]) -> dict[str, float]:
        features = self.vectorizer.transform_state(state)
        q_values = self.q_model.q_values(features)[0]
        return {INDEX_TO_ACTION[index]: float(value) for index, value in enumerate(q_values)}

    def select_action(self, state: Mapping[str, Any], *, epsilon: float, rng: random.Random) -> str:
        if rng.random() < epsilon:
            return rng.choice(list(TRAINING_ACTIONS))
        return self.predict_action(state)

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        payload = {
            "vectorizer": self.vectorizer.to_dict(),
            "q_model": self.q_model.to_dict(),
            "training_actions": list(TRAINING_ACTIONS),
        }
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output

    @classmethod
    def load(cls, model_path: str | Path) -> "RLFineTunedPolicy":
        with Path(model_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            vectorizer=StateVectorizer.from_dict(payload["vectorizer"]),
            q_model=DQNQNetwork.from_dict(payload["q_model"]),
        )


def load_bc_policy_or_raise(policy_path: str | Path | None = None) -> BehaviorCloningPolicy:
    resolved = Path(policy_path) if policy_path is not None else DEFAULT_BC_POLICY_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Behavior-cloning policy not found at {resolved}. Train behavior cloning first or pass --bc-policy."
        )
    return BehaviorCloningPolicy.load(resolved)


def split_rl_cases(
    dataset: CICDDiagnosisDataset,
    *,
    split: str = "train",
    validation_fraction: float = 0.2,
    seed: int = 11,
) -> DatasetSplit:
    builder = BehaviorCloningDatasetBuilder(dataset=dataset)
    return builder.split_case_ids(split=split, validation_fraction=validation_fraction, seed=seed)


def evaluate_rl_policy(
    policy: RLFineTunedPolicy,
    *,
    env: CICDDiagnosisEnv,
    case_ids: Sequence[str],
) -> EvaluationResult:
    if not case_ids:
        return EvaluationResult(
            average_reward=math.nan,
            success_rate=math.nan,
            average_steps=math.nan,
            num_cases=0,
        )

    total_reward = 0.0
    successes = 0
    total_steps = 0

    for case_id in case_ids:
        state = env.reset(case_id=case_id)
        episode_reward = 0.0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = policy.predict_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        total_reward += episode_reward
        total_steps += len(info["episode_summary"]["tool_history"])
        summary = info["episode_summary"]
        required_tools = set(summary["required_tools_before_final"])
        used_tools = set(summary["tool_history"])
        if terminated and required_tools.issubset(used_tools):
            successes += 1

    num_cases = len(case_ids)
    return EvaluationResult(
        average_reward=total_reward / num_cases,
        success_rate=successes / num_cases,
        average_steps=total_steps / num_cases,
        num_cases=num_cases,
    )


def train_rl_fine_tuning(
    *,
    dataset: CICDDiagnosisDataset | None = None,
    bc_policy: BehaviorCloningPolicy | None = None,
    bc_policy_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    split: str = "train",
    config: RLTrainConfig | None = None,
) -> tuple[RLFineTunedPolicy, RLTrainResult]:
    config = config or RLTrainConfig()
    dataset = dataset or load_dataset(dataset_root=dataset_root)
    bc_policy = bc_policy or load_bc_policy_or_raise(bc_policy_path)

    case_split = split_rl_cases(
        dataset,
        split=split,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_env = CICDDiagnosisEnv(dataset=dataset, max_steps=config.max_steps, seed=config.seed)
    eval_env = CICDDiagnosisEnv(dataset=dataset, max_steps=config.max_steps, seed=config.seed)

    policy = RLFineTunedPolicy.from_behavior_cloning(
        bc_policy,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )
    target_network = DQNQNetwork.from_dict(policy.q_model.to_dict())
    replay_buffer = ReplayBuffer(capacity=config.replay_capacity)
    rng = random.Random(config.seed)
    epsilon = config.epsilon_start
    total_env_steps = 0
    reward_history: list[float] = []
    evaluation_history: list[dict[str, Any]] = []
    best_snapshot = policy.q_model.snapshot()

    initial_train_eval = evaluate_rl_policy(policy, env=eval_env, case_ids=case_split.train_case_ids)
    initial_val_eval = evaluate_rl_policy(policy, env=eval_env, case_ids=case_split.val_case_ids)
    best_score = _evaluation_score(initial_train_eval, initial_val_eval)
    evaluation_history.append(
        {
            "episode": 0,
            "epsilon": epsilon,
            "replay_size": len(replay_buffer),
            "train_average_reward": initial_train_eval.average_reward,
            "train_success_rate": initial_train_eval.success_rate,
            "val_average_reward": initial_val_eval.average_reward,
            "val_success_rate": initial_val_eval.success_rate,
        }
    )

    for episode_index in range(config.episodes):
        case_id = rng.choice(case_split.train_case_ids)
        state = train_env.reset(case_id=case_id)
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not terminated and not truncated:
            action = policy.select_action(state, epsilon=epsilon, rng=rng)
            action_index = ACTION_TO_INDEX[action]
            state_features = policy.vectorizer.transform_state(state)[0]

            next_state, reward, terminated, truncated, _info = train_env.step(action)
            next_state_features = policy.vectorizer.transform_state(next_state)[0]
            replay_buffer.add(
                ReplayTransition(
                    state_features=state_features.copy(),
                    action_index=action_index,
                    reward=float(reward),
                    next_state_features=next_state_features.copy(),
                    done=bool(terminated or truncated),
                )
            )

            state = next_state
            episode_reward += reward
            total_env_steps += 1

            if len(replay_buffer) >= config.batch_size and total_env_steps >= config.warmup_steps:
                for _ in range(config.gradient_steps_per_env_step):
                    batch = replay_buffer.sample(config.batch_size, rng=rng)
                    _train_dqn_batch(
                        policy=policy,
                        target_network=target_network,
                        batch=batch,
                        gamma=config.gamma,
                        learning_rate=config.learning_rate,
                        weight_decay=config.weight_decay,
                        td_clip=config.td_clip,
                        huber_delta=config.huber_delta,
                    )

            if total_env_steps % config.target_sync_steps == 0:
                target_network.restore(policy.q_model.snapshot())

        reward_history.append(episode_reward)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        if (episode_index + 1) % config.evaluation_interval == 0 or episode_index == config.episodes - 1:
            train_eval = evaluate_rl_policy(policy, env=eval_env, case_ids=case_split.train_case_ids)
            val_eval = evaluate_rl_policy(policy, env=eval_env, case_ids=case_split.val_case_ids)
            current_score = _evaluation_score(train_eval, val_eval)
            if current_score >= best_score:
                best_score = current_score
                best_snapshot = policy.q_model.snapshot()
            evaluation_history.append(
                {
                    "episode": episode_index + 1,
                    "epsilon": epsilon,
                    "replay_size": len(replay_buffer),
                    "train_average_reward": train_eval.average_reward,
                    "train_success_rate": train_eval.success_rate,
                    "val_average_reward": val_eval.average_reward,
                    "val_success_rate": val_eval.success_rate,
                }
            )

    # DQN on a tiny deterministic dataset can oscillate, so we return the best checkpoint seen
    # under evaluation instead of assuming the final online-network weights are the strongest.
    policy.q_model.restore(best_snapshot)
    final_train_eval = evaluate_rl_policy(policy, env=eval_env, case_ids=case_split.train_case_ids)
    final_val_eval = evaluate_rl_policy(policy, env=eval_env, case_ids=case_split.val_case_ids)
    return policy, RLTrainResult(
        episodes_completed=config.episodes,
        final_epsilon=epsilon,
        train_case_count=len(case_split.train_case_ids),
        val_case_count=len(case_split.val_case_ids),
        total_env_steps=total_env_steps,
        replay_size=len(replay_buffer),
        reward_history=reward_history,
        evaluation_history=evaluation_history,
        final_train_eval=final_train_eval,
        final_val_eval=final_val_eval,
    )


def _train_dqn_batch(
    *,
    policy: RLFineTunedPolicy,
    target_network: DQNQNetwork,
    batch: Sequence[ReplayTransition],
    gamma: float,
    learning_rate: float,
    weight_decay: float,
    td_clip: float,
    huber_delta: float,
) -> float:
    state_batch = np.asarray([transition.state_features for transition in batch], dtype=np.float64)
    action_batch = np.asarray([transition.action_index for transition in batch], dtype=np.int64)
    reward_batch = np.asarray([transition.reward for transition in batch], dtype=np.float64)
    next_state_batch = np.asarray([transition.next_state_features for transition in batch], dtype=np.float64)
    done_mask = np.asarray([transition.done for transition in batch], dtype=np.float64)

    next_q_values = target_network.q_values(next_state_batch)
    target_batch = reward_batch + (1.0 - done_mask) * gamma * np.max(next_q_values, axis=1)
    return policy.q_model.train_batch(
        state_batch=state_batch,
        action_batch=action_batch,
        target_batch=target_batch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        td_clip=td_clip,
        huber_delta=huber_delta,
    )


def _evaluation_score(train_eval: EvaluationResult, val_eval: EvaluationResult) -> tuple[float, float, float, float]:
    if math.isnan(val_eval.success_rate):
        return (
            train_eval.success_rate,
            train_eval.average_reward,
            -train_eval.average_steps,
            train_eval.num_cases,
        )
    return (
        val_eval.success_rate,
        val_eval.average_reward,
        train_eval.success_rate,
        train_eval.average_reward,
    )
