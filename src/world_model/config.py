"""Versioned configuration for the report-aligned Dreamer branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class ObservationConfig:
    max_speed_mps: float = 30.0
    max_accel_mps2: float = 10.0
    max_progress_delta_m: float = 5.0
    max_lane_distance_m: float = 5.0
    max_clearance_m: float = 80.0
    max_relative_speed_mps: float = 40.0
    max_ttc_s: float = 20.0
    max_blocked_ticks: float = 200.0
    max_return_distance_m: float = 50.0
    max_curvature: float = 0.25


@dataclass
class RSSMConfig:
    observation_dim: int = 32
    action_dim: int = 3
    encoder_dim: int = 128
    hidden_dim: int = 256
    deterministic_size: int = 128
    stochastic_size: int = 16
    categorical_classes: int = 16
    unimix: float = 0.01
    free_nats: float = 1.0
    dynamics_kl_scale: float = 0.5
    representation_kl_scale: float = 0.1
    imagination_horizon: int = 8


@dataclass
class PredictionLossConfig:
    observation: float = 1.0
    progress: float = 1.0
    risk: float = 2.0
    continuation: float = 1.0
    value: float = 0.5
    collision: float = 2.0
    offroad: float = 1.5
    # Posterior heads are useful for representation learning, but closed-loop
    # planning queries the action-conditioned prior.  Keep these additions at
    # zero by default so historical checkpoints retain their exact semantics.
    prior_prediction: float = 0.0
    action_contrastive: float = 0.0
    action_contrastive_margin: float = 0.05
    action_safety_monotonic: float = 0.0
    action_safety_margin: float = 0.02
    hazard_front_clearance: float = 0.20
    hazard_oncoming_ttc: float = 0.40
    hazard_vru_distance: float = 0.15
    collision_positive_weight: float = 1.0
    offroad_positive_weight: float = 1.0


@dataclass
class CandidateConfig:
    slow_factors: tuple = (0.70, 0.40)
    emergency_brake_levels: tuple = ()
    steer_offsets: tuple = (-0.18, -0.08, 0.08, 0.18)
    overtake_steer: float = 0.28
    return_steer: float = 0.20
    assumed_alpha: float = 0.60
    max_steer_delta: float = 0.35
    include_actor_candidate: bool = True


@dataclass
class EvaluatorConfig:
    lambda_progress: float = 1.0
    lambda_risk: float = 3.0
    lambda_change: float = 0.20
    continuation_discount: float = 0.99
    pairwise_scale: float = 0.25


@dataclass
class RewardConfig:
    w_progress: float = 1.0
    w_safe: float = 1.0
    w_collision: float = 20.0
    w_offroad: float = 10.0
    w_jerk: float = 0.20
    w_alpha: float = 0.05
    collision_terminal: bool = True


@dataclass
class AuthorityConfig:
    enabled: bool = True
    learned: bool = True
    fixed_alpha: float = 0.15
    max_alpha: float = 0.70
    smoothing: float = 0.80
    max_delta_per_step: float = 0.08
    exact_native_epsilon: float = 1.0e-8


@dataclass
class PairwiseConfig:
    enabled: bool = False
    checkpoint: str = ""
    hidden_dim: int = 128


@dataclass
class PolicyConfig:
    hidden_dim: int = 256
    min_std: float = 0.05
    max_std: float = 0.70
    max_steer_residual: float = 0.35
    max_longitudinal_residual: float = 0.60
    initial_alpha: float = 0.05
    entropy_scale: float = 0.001
    lambda_return: float = 0.95


@dataclass
class TrainingConfig:
    batch_size: int = 32
    sequence_length: int = 32
    learning_rate: float = 3.0e-4
    actor_lr: float = 8.0e-5
    critic_lr: float = 8.0e-5
    pairwise_lr: float = 1.0e-4
    world_model_epochs: int = 30
    policy_epochs: int = 30
    pairwise_epochs: int = 20
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    minimum_train_seeds: int = 2
    minimum_validation_seeds: int = 2
    minimum_test_seeds: int = 2
    split_seed: int = 20260818
    gradient_clip: float = 100.0
    checkpoint_dir: str = "checkpoints/report_aligned_dreamer"
    # Production training follows the report's Phase 1: trajectories produced
    # by native SimLingo only. Historical guard/RL traces remain available for
    # explicitly requested diagnostics, never as an implicit fallback.
    source_policy: str = "simlingo_native"
    require_event_ground_truth: bool = True


@dataclass
class RuntimeConfig:
    ablation: str = "D"
    shadow: bool = False
    deterministic_latent: bool = True
    deterministic_policy: bool = True
    trace_path: str = ""


@dataclass
class DreamerConfig:
    version: str = "report_aligned_v2"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    prediction_loss: PredictionLossConfig = field(default_factory=PredictionLossConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    authority: AuthorityConfig = field(default_factory=AuthorityConfig)
    pairwise: PairwiseConfig = field(default_factory=PairwiseConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.rssm.observation_dim != 32:
            raise ValueError("The compact DreamerObservation schema has 32 features")
        if self.rssm.action_dim != 3:
            raise ValueError("RSSM action must be [steer, throttle, brake]")
        if self.runtime.ablation not in ("A", "B", "C", "D", "E"):
            raise ValueError("runtime.ablation must be one of A, B, C, D, E")
        ratios = (
            self.training.train_ratio
            + self.training.validation_ratio
            + self.training.test_ratio
        )
        if abs(ratios - 1.0) > 1.0e-6:
            raise ValueError("train/validation/test ratios must sum to 1")
        for name in ("fixed_alpha", "max_alpha", "smoothing"):
            value = float(getattr(self.authority, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("authority.%s must be in [0, 1]" % name)
        if self.training.source_policy not in ("simlingo_native", "any"):
            raise ValueError(
                "training.source_policy must be simlingo_native or any"
            )
        for name in (
            "minimum_train_seeds",
            "minimum_validation_seeds",
            "minimum_test_seeds",
        ):
            if int(getattr(self.training, name)) < 1:
                raise ValueError("training.%s must be at least 1" % name)
        for name in (
            "observation",
            "progress",
            "risk",
            "continuation",
            "value",
            "collision",
            "offroad",
            "prior_prediction",
            "action_contrastive",
            "action_contrastive_margin",
            "action_safety_monotonic",
            "action_safety_margin",
            "hazard_front_clearance",
            "hazard_oncoming_ttc",
            "hazard_vru_distance",
            "collision_positive_weight",
            "offroad_positive_weight",
        ):
            if float(getattr(self.prediction_loss, name)) < 0.0:
                raise ValueError("prediction_loss.%s must be non-negative" % name)
        for level in self.candidates.emergency_brake_levels:
            if not 0.0 < float(level) <= 1.0:
                raise ValueError(
                    "candidates.emergency_brake_levels must be in (0, 1]"
                )
        if not 0.0 < float(self.policy.initial_alpha) < 1.0:
            raise ValueError("policy.initial_alpha must be in (0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: T, values: Mapping[str, Any]) -> T:
    known = {item.name: item for item in fields(instance)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError("Unknown configuration keys: %s" % ", ".join(unknown))
    for key, value in values.items():
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise TypeError("Configuration section %s must be a mapping" % key)
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None) -> DreamerConfig:
    config = DreamerConfig()
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("Dreamer configuration root must be a mapping")
        _merge_dataclass(config, raw)
    if overrides:
        _merge_dataclass(config, overrides)
    config.validate()
    return config
