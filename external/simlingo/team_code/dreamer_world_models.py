"""Temporal world-model building blocks for the SimLingo Dreamer complement.

The legacy runtime uses a deterministic one-step MLP.  This module adds a
compact DreamerV3-inspired recurrent state-space model (RSSM) without changing
that legacy checkpoint format.  The RSSM consumes the complete policy
observation, keeps a recurrent latent state, and predicts driving outcomes used
for multi-step imagination.

This is intentionally a small PyTorch implementation tailored to the existing
Bench2Drive traces.  It is not presented as a drop-in copy of DreamerV3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


WORLD_MODEL_TYPE = "categorical_rssm_v2"
POLICY_MODEL_TYPE = "rssm_conditioned_actor_critic_v2"
UTILITY_MODEL_TYPE = "pairwise_latent_utility_v2"
EVENT_NAMES = ("collision", "offroad", "vru_danger", "red_light", "blocked")

# The 49-D policy observation contains map-invariant placeholders and several
# permanently constant slots inherited from the original Youma state vector.
# Asking a learned model to reconstruct those slots only teaches tiny rollout
# noise. The RSSM predicts the dimensions that can actually change or affect a
# driving decision; all other deltas are exactly zero (persistence).
PREDICTED_OBSERVATION_INDICES = (
    2, 4, 6, 8, 10,             # ego dynamics, route shape, traffic light
    13, 14, 16, 18, 21, 23, 26, # front vehicle and VRU geometry
    28, 29, 30,                  # current SimLingo command
    31, 32, 33, 34, 35,          # blockage, side clearance and TTC
    36, 37, 38, 39, 40, 41,      # adjacent oncoming traffic and lane validity
    42, 43, 44,                  # oncoming traffic in the current lane
)

# Map-invariant context made available to the learned pairwise utility. These
# are normalized observations, not hand-written decisions: the calibrator must
# still learn how they affect preference from data. Global pose and route-ID
# proxies are deliberately excluded to reduce seed/map memorization.
UTILITY_CONTEXT_OBSERVATION_INDICES = (
    2,                           # ego speed
    28, 29, 30,                 # current SimLingo command
    31, 32, 33, 34, 35,         # blockage, side clearance and TTC
    36, 37, 38, 39, 40, 41,     # adjacent oncoming traffic and lane validity
    42, 43, 44,                 # oncoming traffic in the current lane
)


def symlog(value: torch.Tensor) -> torch.Tensor:
    """Signed logarithm used to keep large rewards/progress numerically tame."""

    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`."""

    return torch.sign(value) * torch.expm1(torch.abs(value))


@dataclass
class RSSMConfig:
    observation_dim: int = 49
    action_dim: int = 4
    encoder_dim: int = 128
    hidden_dim: int = 256
    deter_dim: int = 128
    stoch_dim: int = 16
    classes: int = 16
    event_dim: int = len(EVENT_NAMES)
    unimix: float = 0.01
    free_nats: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    deterministic_state_mode: str = "argmax"

    @property
    def stochastic_size(self) -> int:
        return int(self.stoch_dim * self.classes)

    @property
    def feature_dim(self) -> int:
        return int(self.deter_dim + self.stochastic_size)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "RSSMConfig":
        if not raw:
            return cls()
        fields = cls.__dataclass_fields__
        values = {key: raw[key] for key in fields if key in raw}
        return cls(**values)


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            logits=self.logits.detach(),
        )

    def index(self, indices: torch.Tensor) -> "RSSMState":
        return RSSMState(
            deter=self.deter[indices],
            stoch=self.stoch[indices],
            logits=self.logits[indices],
        )


class _DenseBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class TemporalRSSMWorldModel(nn.Module):
    """Categorical recurrent state-space model for structured driving traces.

    At time ``t`` the posterior summarizes all observations through ``t``.  An
    action advances the prior to ``t+1``.  Prediction heads are attached to the
    prior feature so the same heads can be used during imagined rollouts where
    future observations are unavailable.
    """

    def __init__(self, config: Optional[RSSMConfig] = None):
        super().__init__()
        self.config = config or RSSMConfig()
        cfg = self.config

        self.encoder = nn.Sequential(
            nn.Linear(cfg.observation_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.encoder_dim),
            nn.LayerNorm(cfg.encoder_dim),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(cfg.stochastic_size + cfg.action_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.deter_dim)
        self.prior = _DenseBlock(
            cfg.deter_dim,
            cfg.stochastic_size,
            cfg.hidden_dim,
        )
        self.posterior = _DenseBlock(
            cfg.deter_dim + cfg.encoder_dim,
            cfg.stochastic_size,
            cfg.hidden_dim,
        )

        feature_dim = cfg.feature_dim
        self.head_observation_delta = _DenseBlock(
            feature_dim, cfg.observation_dim, cfg.hidden_dim
        )
        self.head_reward = _DenseBlock(feature_dim, 1, cfg.hidden_dim)
        self.head_continuation = _DenseBlock(feature_dim, 1, cfg.hidden_dim)
        self.head_risk = _DenseBlock(feature_dim, 1, cfg.hidden_dim)
        self.head_progress = _DenseBlock(feature_dim, 1, cfg.hidden_dim)
        self.head_events = _DenseBlock(feature_dim, cfg.event_dim, cfg.hidden_dim)

        observation_mask = torch.zeros(cfg.observation_dim, dtype=torch.float32)
        valid_indices = [
            index for index in PREDICTED_OBSERVATION_INDICES
            if index < cfg.observation_dim
        ]
        if valid_indices:
            observation_mask[valid_indices] = 1.0
        else:
            # Small synthetic test configurations do not share the production
            # 49-D schema, so keep every dimension trainable there.
            observation_mask.fill_(1.0)
        self.register_buffer("observation_delta_mask", observation_mask)

        # A fresh model starts as the strong persistence baseline. Training has
        # to earn every non-zero correction instead of injecting random drift.
        observation_output = self.head_observation_delta.net[-1]
        nn.init.zeros_(observation_output.weight)
        nn.init.zeros_(observation_output.bias)

    @property
    def observation_dim(self) -> int:
        return self.config.observation_dim

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def _reshape_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.reshape(*logits.shape[:-1], self.config.stoch_dim, self.config.classes)

    def _probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        if self.config.unimix > 0.0:
            uniform = torch.full_like(probabilities, 1.0 / self.config.classes)
            probabilities = (
                (1.0 - self.config.unimix) * probabilities
                + self.config.unimix * uniform
            )
        return probabilities

    def _sample(self, logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
        probabilities = self._probabilities(logits)
        if deterministic:
            if self.config.deterministic_state_mode == "probabilities":
                # Small structured datasets do not support a stable hard
                # categorical assignment for every continuous traffic state.
                # The expected categorical state preserves that information
                # while remaining deterministic at validation/runtime.
                return probabilities.reshape(*probabilities.shape[:-2], -1)
            indices = probabilities.argmax(dim=-1)
            one_hot = F.one_hot(indices, self.config.classes).to(probabilities.dtype)
        else:
            flat = probabilities.reshape(-1, self.config.classes)
            indices = torch.multinomial(flat, 1).reshape(probabilities.shape[:-1])
            one_hot = F.one_hot(indices, self.config.classes).to(probabilities.dtype)
        # Straight-through categorical sample: discrete forward pass, useful
        # posterior gradients during sequence training.
        straight_through = one_hot + probabilities - probabilities.detach()
        return straight_through.reshape(*straight_through.shape[:-2], -1)

    def initial(self, batch_size: int, device: Optional[torch.device] = None) -> RSSMState:
        device = device or next(self.parameters()).device
        deter = torch.zeros(batch_size, self.config.deter_dim, device=device)
        logits = torch.zeros(
            batch_size,
            self.config.stoch_dim,
            self.config.classes,
            device=device,
        )
        stoch = self._sample(logits, deterministic=True)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def feature(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def observe_initial(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> RSSMState:
        initial = self.initial(observation.shape[0], observation.device)
        encoded = self.encoder(observation)
        logits = self._reshape_logits(
            self.posterior(torch.cat([initial.deter, encoded], dim=-1))
        )
        stoch = self._sample(logits, deterministic=deterministic)
        return RSSMState(deter=initial.deter, stoch=stoch, logits=logits)

    def img_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> RSSMState:
        encoded = self.action_encoder(torch.cat([previous.stoch, action], dim=-1))
        deter = self.gru(encoded, previous.deter)
        logits = self._reshape_logits(self.prior(deter))
        stoch = self._sample(logits, deterministic=deterministic)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def obs_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[RSSMState, RSSMState]:
        prior = self.img_step(previous, action, deterministic=deterministic)
        encoded = self.encoder(observation)
        logits = self._reshape_logits(
            self.posterior(torch.cat([prior.deter, encoded], dim=-1))
        )
        posterior = RSSMState(
            deter=prior.deter,
            stoch=self._sample(logits, deterministic=deterministic),
            logits=logits,
        )
        return posterior, prior

    def heads(self, state: RSSMState) -> Dict[str, torch.Tensor]:
        feature = self.feature(state)
        return {
            "observation_delta": (
                self.head_observation_delta(feature) * self.observation_delta_mask
            ),
            "reward_symlog": self.head_reward(feature).squeeze(-1),
            "continuation_logit": self.head_continuation(feature).squeeze(-1),
            "risk_logit": self.head_risk(feature).squeeze(-1),
            "progress_symlog": self.head_progress(feature).squeeze(-1),
            "event_logits": self.head_events(feature),
        }

    def imagine_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[RSSMState, Dict[str, torch.Tensor]]:
        prior = self.img_step(previous, action, deterministic=deterministic)
        return prior, self.heads(prior)

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Observe ``T+1`` observations and predict ``T`` transitions."""

        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("observations/actions must have shape [batch, time, features]")
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("a sequence needs one more observation than action")
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected observation dim {self.observation_dim}, got {observations.shape[-1]}"
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"expected action dim {self.action_dim}, got {actions.shape[-1]}")

        posterior = self.observe_initial(observations[:, 0], deterministic=deterministic)
        prior_logits = []
        posterior_logits = []
        posterior_features = [self.feature(posterior)]
        posterior_deter = [posterior.deter]
        posterior_stoch = [posterior.stoch]
        posterior_state_logits = [posterior.logits]
        predictions: Dict[str, list] = {
            "observation_delta": [],
            "reward_symlog": [],
            "continuation_logit": [],
            "risk_logit": [],
            "progress_symlog": [],
            "event_logits": [],
        }
        for step in range(actions.shape[1]):
            posterior, prior = self.obs_step(
                posterior,
                actions[:, step],
                observations[:, step + 1],
                deterministic=deterministic,
            )
            head = self.heads(prior)
            prior_logits.append(prior.logits)
            posterior_logits.append(posterior.logits)
            posterior_features.append(self.feature(posterior))
            posterior_deter.append(posterior.deter)
            posterior_stoch.append(posterior.stoch)
            posterior_state_logits.append(posterior.logits)
            for key in predictions:
                predictions[key].append(head[key])

        result = {
            key: torch.stack(values, dim=1)
            for key, values in predictions.items()
        }
        result.update({
            "prior_logits": torch.stack(prior_logits, dim=1),
            "posterior_logits": torch.stack(posterior_logits, dim=1),
            "posterior_features": torch.stack(posterior_features, dim=1),
            "posterior_deter": torch.stack(posterior_deter, dim=1),
            "posterior_stoch": torch.stack(posterior_stoch, dim=1),
            "posterior_state_logits": torch.stack(
                posterior_state_logits, dim=1
            ),
        })
        return result

    def kl_loss(
        self,
        posterior_logits: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Balanced categorical KL with free nats, following Dreamer practice."""

        def categorical_kl(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
            left = self._probabilities(left_logits)
            right = self._probabilities(right_logits)
            value = left * (
                torch.log(left.clamp_min(1e-8)) - torch.log(right.clamp_min(1e-8))
            )
            return value.sum(dim=-1).sum(dim=-1)

        dyn_kl = categorical_kl(posterior_logits.detach(), prior_logits)
        rep_kl = categorical_kl(posterior_logits, prior_logits.detach())
        free = torch.as_tensor(
            self.config.free_nats,
            dtype=dyn_kl.dtype,
            device=dyn_kl.device,
        )
        dyn_loss = torch.maximum(dyn_kl, free).mean()
        rep_loss = torch.maximum(rep_kl, free).mean()
        return {
            "loss": self.config.dyn_scale * dyn_loss + self.config.rep_scale * rep_loss,
            "dynamic": dyn_kl.mean(),
            "representation": rep_kl.mean(),
        }


class PairwiseUtilityCalibrator(nn.Module):
    """Learn a continuous utility correction between two imagined futures.

    The physical RSSM heads remain responsible for risk and progress. This
    small low-rank network only learns the residual preference between the
    SimLingo future and a Dreamer proposal. Its inputs are latent imagined
    futures plus their learned risk/progress deltas; no CARLA distance, TTC or
    hand-written scene threshold is consulted at runtime.

    Every branch is expressed relative to the SimLingo branch. A zero branch
    delta therefore produces exactly zero correction, which keeps the native
    SimLingo score unchanged by construction.
    """

    def __init__(
        self,
        feature_dim: int,
        observation_dim: int = 0,
        hidden_dim: int = 32,
        output_scale: float = 1.5,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_scale = float(output_scale)
        if (
            self.feature_dim <= 0
            or self.observation_dim < 0
            or self.hidden_dim <= 0
        ):
            raise ValueError(
                "feature_dim/hidden_dim must be positive and observation_dim non-negative"
            )
        if self.output_scale <= 0.0:
            raise ValueError("output_scale must be positive")

        self.context_projection = nn.Linear(
            self.feature_dim + self.observation_dim,
            self.hidden_dim,
            bias=True,
        )
        self.delta_projection = nn.Linear(
            self.feature_dim, self.hidden_dim, bias=False
        )
        self.metric_projection = nn.Linear(5, self.hidden_dim, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 3, 1, bias=False),
        )

        # Installing an untrained calibrator is behavior preserving. Training
        # must earn every non-zero residual utility correction.
        nn.init.zeros_(self.head[-1].weight)

    def forward(
        self,
        base_feature: torch.Tensor,
        candidate_feature: torch.Tensor,
        progress_delta: torch.Tensor,
        risk_delta: torch.Tensor,
        control_delta: torch.Tensor,
        current_observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_feature.shape != candidate_feature.shape:
            raise ValueError("base and candidate latent features must match")
        if base_feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected latent feature dim {self.feature_dim}, "
                f"got {base_feature.shape[-1]}"
            )
        if control_delta.shape[:-1] != base_feature.shape[:-1] or control_delta.shape[-1] != 3:
            raise ValueError("control_delta must have shape [..., 3]")
        if self.observation_dim:
            if current_observation is None:
                raise ValueError(
                    "current_observation is required by this utility calibrator"
                )
            if (
                current_observation.shape[:-1] != base_feature.shape[:-1]
                or current_observation.shape[-1] != self.observation_dim
            ):
                raise ValueError(
                    "current_observation shape does not match calibrator metadata"
                )
            context_input = torch.cat(
                [base_feature, current_observation], dim=-1
            )
        else:
            context_input = base_feature
        delta_feature = candidate_feature - base_feature
        context = torch.tanh(self.context_projection(context_input))
        delta = self.delta_projection(delta_feature)
        metrics = torch.stack(
            [progress_delta, risk_delta], dim=-1
        )
        metrics = torch.cat([metrics, control_delta], dim=-1)
        metric_feature = self.metric_projection(metrics)
        joint = torch.cat([delta, context * delta, metric_feature], dim=-1)
        raw = self.head(joint).squeeze(-1)
        return self.output_scale * torch.tanh(raw)


def discounted_feature_pool(
    features: torch.Tensor,
    continuation: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Pool imagined latent features with continuation and time discount."""

    if features.ndim != 3:
        raise ValueError("features must have shape [batch, horizon, feature]")
    if continuation.shape != features.shape[:2]:
        raise ValueError(
            "continuation must have shape [batch, horizon] matching features"
        )
    horizon = features.shape[1]
    discounts = torch.pow(
        torch.as_tensor(discount, dtype=features.dtype, device=features.device),
        torch.arange(horizon, dtype=features.dtype, device=features.device),
    ).reshape(1, horizon)
    weights = continuation * discounts
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (features * weights.unsqueeze(-1)).sum(dim=1) / denominator


def expand_actor_input_state_dict(
    state_dict: Dict[str, torch.Tensor],
    new_input_dim: int,
) -> Dict[str, torch.Tensor]:
    """Expand an ActorCritic input while preserving its exact old outputs.

    New latent columns are initialized to zero.  Therefore a migrated policy is
    behavior-identical before any V2 policy training.
    """

    weight = state_dict.get("trunk.0.weight")
    if weight is None:
        raise KeyError("actor checkpoint is missing trunk.0.weight")
    old_input_dim = int(weight.shape[1])
    if new_input_dim < old_input_dim:
        raise ValueError(f"cannot shrink actor input {old_input_dim} -> {new_input_dim}")
    if new_input_dim == old_input_dim:
        return dict(state_dict)
    expanded = weight.new_zeros((weight.shape[0], new_input_dim))
    expanded[:, :old_input_dim] = weight
    result = dict(state_dict)
    result["trunk.0.weight"] = expanded
    return result
