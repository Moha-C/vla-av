"""Versioned configuration for the isolated residual DreamerV3 branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class DataConfig:
    observation_dim: int = 32
    action_dim: int = 3
    sequence_length: int = 32
    split_seed: int = 20260830
    train_ratio: float = 0.50
    validation_ratio: float = 0.25
    test_ratio: float = 0.25
    minimum_train_seeds: int = 3
    minimum_validation_seeds: int = 2
    minimum_test_seeds: int = 2
    require_native_simlingo: bool = True
    require_bench2drive_ground_truth: bool = True
    maximum_step_progress_m: float = 5.0
    event_window_weight: float = 32.0
    danger_window_weight: float = 4.0
    danger_risk_threshold: float = 0.65


@dataclass
class ModelConfig:
    observation_dim: int = 32
    action_dim: int = 3
    encoder_dim: int = 192
    hidden_dim: int = 384
    deterministic_size: int = 256
    stochastic_size: int = 16
    categorical_classes: int = 16
    unimix: float = 0.01
    free_nats: float = 1.0
    dynamics_kl_scale: float = 0.5
    representation_kl_scale: float = 0.1
    reward_bins: int = 255
    reward_low: float = -20.0
    reward_high: float = 20.0
    value_bins: int = 255
    value_low: float = -20.0
    value_high: float = 20.0


@dataclass
class LossConfig:
    observation: float = 1.0
    reward: float = 1.0
    continuation: float = 1.0
    risk: float = 2.0
    collision: float = 4.0
    offroad: float = 3.0
    prior_prediction: float = 1.0
    collision_positive_weight: float = 8.0
    offroad_positive_weight: float = 6.0


@dataclass
class RewardConfig:
    progress_scale: float = 1.0
    safe_scale: float = 0.25
    collision_penalty: float = 25.0
    offroad_penalty: float = 12.0
    control_change_penalty: float = 0.08
    completion_bonus: float = 5.0
    intervention_penalty: float = 0.03
    residual_change_penalty: float = 0.04


@dataclass
class ActorConfig:
    hidden_dim: int = 384
    imagination_horizon: int = 15
    discount: float = 0.997
    lambda_return: float = 0.95
    entropy_scale: float = 3.0e-4
    slow_critic_fraction: float = 0.02
    slow_critic_regularization: float = 1.0
    maximum_steer_residual: float = 0.30
    maximum_longitudinal_residual: float = 0.60
    initial_authority: float = 0.02
    minimum_std: float = 0.05
    maximum_std: float = 1.0


@dataclass
class TrainingConfig:
    batch_size: int = 32
    world_model_epochs: int = 40
    actor_epochs: int = 40
    world_model_learning_rate: float = 1.0e-4
    actor_learning_rate: float = 3.0e-5
    critic_learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-6
    gradient_clip: float = 100.0
    maximum_windows: int = 0
    device: str = "cuda"


@dataclass
class GateConfig:
    horizons: tuple = (1, 5, 10, 20)
    minimum_observation_improvement: float = 0.02
    minimum_reward_improvement: float = 0.0
    minimum_risk_improvement: float = 0.0
    maximum_action_collapse_fraction: float = 0.10
    minimum_action_transition_spread: float = 1.0e-3
    require_all_finite: bool = True


@dataclass
class ResidualDreamerConfig:
    schema_version: str = "residual_dreamerv3_v2"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    def validate(self) -> None:
        if self.data.observation_dim != self.model.observation_dim:
            raise ValueError("data/model observation dimensions differ")
        if self.data.action_dim != self.model.action_dim:
            raise ValueError("data/model action dimensions differ")
        if self.data.observation_dim != 32 or self.data.action_dim != 3:
            raise ValueError("the SimLingo contract is observation=32 and action=3")
        ratios = self.data.train_ratio + self.data.validation_ratio + self.data.test_ratio
        if abs(ratios - 1.0) > 1.0e-6:
            raise ValueError("train/validation/test ratios must sum to one")
        if self.data.sequence_length < max(self.gate.horizons):
            raise ValueError("sequence_length must cover every evaluation horizon")
        if not 0.0 < self.actor.initial_authority < 1.0:
            raise ValueError("initial_authority must be in (0, 1)")
        if self.actor.imagination_horizon < 2:
            raise ValueError("imagination_horizon must be at least two")
        for name in ("minimum_train_seeds", "minimum_validation_seeds", "minimum_test_seeds"):
            if int(getattr(self.data, name)) < 1:
                raise ValueError("data.%s must be positive" % name)
        if self.data.event_window_weight < 1.0 or self.data.danger_window_weight < 1.0:
            raise ValueError("event/danger window weights must be at least one")
        if not 0.0 <= self.data.danger_risk_threshold <= 1.0:
            raise ValueError("danger_risk_threshold must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(instance: T, values: Mapping[str, Any]) -> T:
    known = {item.name for item in fields(instance)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError("unknown configuration keys: %s" % ", ".join(unknown))
    for key, value in values.items():
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise TypeError("configuration section %s must be a mapping" % key)
            _merge(current, value)
        else:
            if key in ("horizons",) and isinstance(value, list):
                value = tuple(value)
            setattr(instance, key, value)
    return instance


def load_config(path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None) -> ResidualDreamerConfig:
    config = ResidualDreamerConfig()
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("configuration root must be a mapping")
        _merge(config, raw)
    if overrides:
        _merge(config, overrides)
    config.validate()
    return config
