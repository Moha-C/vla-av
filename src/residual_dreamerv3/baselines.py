"""Frozen sequence baselines and world-model validation gates.

The residual policy is not allowed to train until its action-conditioned RSSM
beats both persistence and a ridge dynamics model on seed-disjoint episodes.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import GateConfig
from .data import Episode
from .model import CategoricalRSSM


EPSILON = 1.0e-6


def _safe_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).std(axis=axis)
    return np.where(result < 1.0e-3, 1.0, result)


def _transition_arrays(episodes: Sequence[Episode]) -> Dict[str, np.ndarray]:
    if not episodes:
        raise ValueError("at least one episode is required")
    return {
        "observation": np.concatenate([item.observations[:-1] for item in episodes], axis=0),
        "action": np.concatenate([item.actions for item in episodes], axis=0),
        "next_observation": np.concatenate([item.observations[1:] for item in episodes], axis=0),
        "reward": np.concatenate([item.rewards for item in episodes], axis=0),
        "continuation": np.concatenate([item.continuation for item in episodes], axis=0),
        "risk": np.concatenate([item.risk for item in episodes], axis=0),
        "collision": np.concatenate([item.collision for item in episodes], axis=0),
        "offroad": np.concatenate([item.offroad for item in episodes], axis=0),
    }


@dataclass
class Normalization:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    reward_mean: float
    reward_std: float

    @classmethod
    def fit(cls, episodes: Sequence[Episode]) -> "Normalization":
        arrays = _transition_arrays(episodes)
        observations = np.concatenate((arrays["observation"], arrays["next_observation"]), axis=0)
        reward_std = float(np.std(arrays["reward"]))
        return cls(
            observation_mean=observations.mean(axis=0).astype(np.float32),
            observation_std=_safe_std(observations).astype(np.float32),
            reward_mean=float(np.mean(arrays["reward"])),
            reward_std=max(1.0e-3, reward_std),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_mean": self.observation_mean.tolist(),
            "observation_std": self.observation_std.tolist(),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
        }


class FrozenDynamicsBaseline:
    name = "baseline"

    def predict(self, observation: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class PersistenceBaseline(FrozenDynamicsBaseline):
    name = "persistence"

    def __init__(self, train_episodes: Sequence[Episode]):
        arrays = _transition_arrays(train_episodes)
        self.reward = float(arrays["reward"].mean())
        self.continuation = float(arrays["continuation"].mean())
        self.risk = float(arrays["risk"].mean())
        self.collision = float(arrays["collision"].mean())
        self.offroad = float(arrays["offroad"].mean())

    def predict(self, observation: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        batch = observation.shape[0]
        return {
            "observation": observation.copy(),
            "reward": np.full(batch, self.reward, dtype=np.float32),
            "continuation": np.full(batch, self.continuation, dtype=np.float32),
            "risk": np.full(batch, self.risk, dtype=np.float32),
            "collision": np.full(batch, self.collision, dtype=np.float32),
            "offroad": np.full(batch, self.offroad, dtype=np.float32),
        }


class RidgeDynamicsBaseline(FrozenDynamicsBaseline):
    """Small action-conditioned linear dynamics baseline.

    It predicts normalized observation deltas and scalar heads. This is kept
    deliberately strong enough that a useless RSSM cannot pass by copying its
    input, while remaining deterministic and auditable.
    """

    name = "action_conditioned_ridge"

    def __init__(self, ridge: float = 1.0e-2):
        self.ridge = float(ridge)
        self.input_mean: Optional[np.ndarray] = None
        self.input_std: Optional[np.ndarray] = None
        self.observation_std: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    def fit(self, episodes: Sequence[Episode]) -> "RidgeDynamicsBaseline":
        arrays = _transition_arrays(episodes)
        observation = arrays["observation"].astype(np.float64)
        action = arrays["action"].astype(np.float64)
        next_observation = arrays["next_observation"].astype(np.float64)
        inputs = np.concatenate((observation, action), axis=1)
        self.input_mean = inputs.mean(axis=0)
        self.input_std = _safe_std(inputs)
        self.observation_std = _safe_std(np.concatenate((observation, next_observation), axis=0))
        normalized = (inputs - self.input_mean) / self.input_std
        design = np.concatenate((normalized, np.ones((len(normalized), 1))), axis=1)
        targets = np.concatenate(
            (
                (next_observation - observation) / self.observation_std,
                arrays["reward"][:, None],
                arrays["continuation"][:, None],
                arrays["risk"][:, None],
                arrays["collision"][:, None],
                arrays["offroad"][:, None],
            ),
            axis=1,
        )
        identity = np.eye(design.shape[1], dtype=np.float64)
        identity[-1, -1] = 0.0
        self.weights = np.linalg.solve(design.T @ design + self.ridge * identity, design.T @ targets)
        return self

    def _check_fitted(self) -> None:
        if any(value is None for value in (self.input_mean, self.input_std, self.observation_std, self.weights)):
            raise RuntimeError("ridge baseline has not been fitted")

    def predict(self, observation: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        self._check_fitted()
        inputs = np.concatenate((observation, action), axis=1).astype(np.float64)
        normalized = (inputs - self.input_mean) / self.input_std
        design = np.concatenate((normalized, np.ones((len(normalized), 1))), axis=1)
        output = design @ self.weights
        observation_dim = observation.shape[1]
        return {
            "observation": (observation + output[:, :observation_dim] * self.observation_std).astype(np.float32),
            "reward": output[:, observation_dim].astype(np.float32),
            "continuation": np.clip(output[:, observation_dim + 1], 0.0, 1.0).astype(np.float32),
            "risk": np.clip(output[:, observation_dim + 2], 0.0, 1.0).astype(np.float32),
            "collision": np.clip(output[:, observation_dim + 3], 0.0, 1.0).astype(np.float32),
            "offroad": np.clip(output[:, observation_dim + 4], 0.0, 1.0).astype(np.float32),
        }

    def save(self, path: Path) -> None:
        self._check_fitted()
        np.savez_compressed(
            str(path),
            ridge=np.asarray([self.ridge]),
            input_mean=self.input_mean,
            input_std=self.input_std,
            observation_std=self.observation_std,
            weights=self.weights,
        )

    @classmethod
    def load(cls, path: Path) -> "RidgeDynamicsBaseline":
        payload = np.load(str(path))
        result = cls(float(payload["ridge"][0]))
        result.input_mean = payload["input_mean"]
        result.input_std = payload["input_std"]
        result.observation_std = payload["observation_std"]
        result.weights = payload["weights"]
        return result


def _metric_bucket() -> Dict[str, float]:
    return {
        "count": 0.0,
        "observation_squared_error": 0.0,
        "reward_absolute_error": 0.0,
        "continuation_brier": 0.0,
        "risk_brier": 0.0,
        "collision_brier": 0.0,
        "offroad_brier": 0.0,
    }


def _record(
    bucket: Dict[str, float],
    prediction: Mapping[str, np.ndarray],
    episode: Episode,
    target_index: int,
    normalization: Normalization,
) -> None:
    observation_error = (
        (prediction["observation"][0] - episode.observations[target_index + 1])
        / normalization.observation_std
    )
    bucket["count"] += 1.0
    bucket["observation_squared_error"] += float(np.mean(observation_error ** 2))
    bucket["reward_absolute_error"] += abs(
        float(prediction["reward"][0]) - float(episode.rewards[target_index])
    ) / normalization.reward_std
    for key in ("continuation", "risk", "collision", "offroad"):
        target = float(getattr(episode, key)[target_index])
        bucket[key + "_brier"] += (float(prediction[key][0]) - target) ** 2


def _finalize(buckets: Mapping[int, Dict[str, float]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"horizons": {}}
    aggregate: Dict[str, List[float]] = {}
    for horizon, raw in sorted(buckets.items()):
        count = max(1.0, raw["count"])
        row = {key: value / count for key, value in raw.items() if key != "count"}
        row["count"] = int(raw["count"])
        result["horizons"][str(horizon)] = row
        for key, value in row.items():
            if key != "count":
                aggregate.setdefault(key, []).append(float(value))
    result["aggregate"] = {key: float(np.mean(values)) for key, values in aggregate.items()}
    return result


def evaluate_baseline(
    baseline: FrozenDynamicsBaseline,
    episodes: Sequence[Episode],
    normalization: Normalization,
    horizons: Sequence[int],
    maximum_starts_per_episode: int = 256,
) -> Dict[str, Any]:
    horizons = tuple(sorted(set(int(value) for value in horizons)))
    maximum_horizon = max(horizons)
    buckets = {value: _metric_bucket() for value in horizons}
    for episode in episodes:
        maximum_start = episode.transitions - maximum_horizon
        if maximum_start < 0:
            continue
        starts = np.linspace(0, maximum_start, min(maximum_start + 1, maximum_starts_per_episode), dtype=int)
        for start in np.unique(starts):
            observation = episode.observations[start : start + 1].copy()
            for step in range(maximum_horizon):
                prediction = baseline.predict(observation, episode.actions[start + step : start + step + 1])
                observation = prediction["observation"]
                horizon = step + 1
                if horizon in buckets:
                    _record(buckets[horizon], prediction, episode, start + step, normalization)
    result = _finalize(buckets)
    result["name"] = baseline.name
    return result


@torch.no_grad()
def evaluate_world_model(
    model: CategoricalRSSM,
    episodes: Sequence[Episode],
    normalization: Normalization,
    horizons: Sequence[int],
    device: torch.device,
    maximum_starts_per_episode: int = 128,
) -> Dict[str, Any]:
    model.eval()
    horizons = tuple(sorted(set(int(value) for value in horizons)))
    maximum_horizon = max(horizons)
    buckets = {value: _metric_bucket() for value in horizons}
    for episode in episodes:
        maximum_start = episode.transitions - maximum_horizon
        if maximum_start < 0:
            continue
        starts = np.linspace(0, maximum_start, min(maximum_start + 1, maximum_starts_per_episode), dtype=int)
        for start in np.unique(starts):
            observation = torch.as_tensor(episode.observations[start : start + 1], device=device)
            state = model.observe_initial(observation, deterministic=True)
            for step in range(maximum_horizon):
                action = torch.as_tensor(episode.actions[start + step : start + step + 1], device=device)
                state = model.imagine_step(state, action, deterministic=True)
                predicted = model.prediction(state, observation)
                observation = predicted.observation
                payload = {
                    "observation": predicted.observation.cpu().numpy(),
                    "reward": predicted.reward.cpu().numpy(),
                    "continuation": predicted.continuation.cpu().numpy(),
                    "risk": predicted.risk.cpu().numpy(),
                    "collision": predicted.collision.cpu().numpy(),
                    "offroad": predicted.offroad.cpu().numpy(),
                }
                horizon = step + 1
                if horizon in buckets:
                    _record(buckets[horizon], payload, episode, start + step, normalization)
    result = _finalize(buckets)
    result["name"] = "categorical_rssm"
    return result


@torch.no_grad()
def action_sensitivity(
    model: CategoricalRSSM,
    episodes: Sequence[Episode],
    normalization: Normalization,
    device: torch.device,
    maximum_samples: int = 512,
) -> Dict[str, float]:
    model.eval()
    spreads: List[float] = []
    collapsed = 0
    seen = 0
    for episode in episodes:
        for index in range(0, episode.transitions, max(1, episode.transitions // 64)):
            observation = torch.as_tensor(episode.observations[index : index + 1], device=device)
            state = model.observe_initial(observation, deterministic=True)
            base = torch.as_tensor(episode.actions[index : index + 1], device=device)
            alternatives = torch.stack(
                (
                    base[0],
                    torch.tensor([-0.65, 0.75, 0.0], device=device),
                    torch.tensor([0.65, 0.75, 0.0], device=device),
                    torch.tensor([0.0, 0.0, 1.0], device=device),
                )
            )
            repeated = type(state)(
                state.deterministic.repeat(len(alternatives), 1),
                state.stochastic.repeat(len(alternatives), 1, 1),
                state.logits.repeat(len(alternatives), 1, 1),
            )
            next_state = model.imagine_step(repeated, alternatives, deterministic=True)
            reference = observation.repeat(len(alternatives), 1)
            predictions = model.prediction(next_state, reference).observation.cpu().numpy()
            normalized = predictions / normalization.observation_std[None, :]
            spread = float(np.mean(np.std(normalized, axis=0)))
            spreads.append(spread)
            collapsed += int(spread < 1.0e-4)
            seen += 1
            if seen >= maximum_samples:
                break
        if seen >= maximum_samples:
            break
    return {
        "samples": float(seen),
        "mean_transition_spread": float(np.mean(spreads)) if spreads else 0.0,
        "median_transition_spread": float(np.median(spreads)) if spreads else 0.0,
        "collapse_fraction": float(collapsed / max(1, seen)),
    }


def build_gate_report(
    model_metrics: Mapping[str, Any],
    baseline_metrics: Sequence[Mapping[str, Any]],
    sensitivity: Mapping[str, float],
    config: GateConfig,
    split_name: str,
) -> Dict[str, Any]:
    best_observation = min(
        float(item["aggregate"]["observation_squared_error"]) for item in baseline_metrics
    )
    best_reward = min(
        float(item["aggregate"]["reward_absolute_error"]) for item in baseline_metrics
    )
    best_risk = min(float(item["aggregate"]["risk_brier"]) for item in baseline_metrics)
    model_observation = float(model_metrics["aggregate"]["observation_squared_error"])
    model_reward = float(model_metrics["aggregate"]["reward_absolute_error"])
    model_risk = float(model_metrics["aggregate"]["risk_brier"])
    observation_improvement = (best_observation - model_observation) / max(EPSILON, best_observation)
    reward_improvement = (best_reward - model_reward) / max(EPSILON, best_reward)
    risk_improvement = (best_risk - model_risk) / max(EPSILON, best_risk)
    finite_values = [
        best_observation, best_reward, best_risk, model_observation, model_reward, model_risk,
        float(sensitivity.get("mean_transition_spread", 0.0)),
        float(sensitivity.get("collapse_fraction", 1.0)),
    ]
    checks = {
        "all_finite": all(math.isfinite(value) for value in finite_values),
        "observation_beats_best_baseline": observation_improvement >= config.minimum_observation_improvement,
        "reward_beats_best_baseline": reward_improvement >= config.minimum_reward_improvement,
        "risk_beats_best_baseline": risk_improvement >= config.minimum_risk_improvement,
        "action_not_collapsed": float(sensitivity.get("collapse_fraction", 1.0)) <= config.maximum_action_collapse_fraction,
        "action_spread_sufficient": float(sensitivity.get("mean_transition_spread", 0.0)) >= config.minimum_action_transition_spread,
    }
    if not config.require_all_finite:
        checks["all_finite"] = True
    return {
        "schema_version": "residual_dreamerv3_world_model_gate_v1",
        "split": split_name,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": asdict(config),
        "model": dict(model_metrics),
        "baselines": [dict(item) for item in baseline_metrics],
        "action_sensitivity": dict(sensitivity),
        "improvement": {
            "observation": observation_improvement,
            "reward": reward_improvement,
            "risk": risk_improvement,
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
