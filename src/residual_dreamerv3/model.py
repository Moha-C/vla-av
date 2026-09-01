"""PyTorch DreamerV3 mechanisms for a residual SimLingo complement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ActorConfig, LossConfig, ModelConfig, ResidualDreamerConfig
from .transforms import symlog, symexp, two_hot_loss, two_hot_mean


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, layers: int = 2):
        super().__init__()
        modules = []
        current = input_dim
        for _ in range(layers):
            modules.extend((nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
            current = hidden_dim
        modules.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@dataclass
class RSSMState:
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    logits: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(
            self.deterministic.detach(),
            self.stochastic.detach(),
            self.logits.detach(),
        )


@dataclass
class Prediction:
    observation: torch.Tensor
    reward: torch.Tensor
    continuation: torch.Tensor
    risk: torch.Tensor
    collision: torch.Tensor
    offroad: torch.Tensor


class CategoricalRSSM(nn.Module):
    """Categorical RSSM with balanced KL, free nats and unimix."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = MLP(config.observation_dim, config.encoder_dim, config.hidden_dim)
        core_input = self.stochastic_flat_dim + config.action_dim
        self.core_input = MLP(core_input, config.hidden_dim, config.hidden_dim, layers=1)
        self.recurrent = nn.GRUCell(config.hidden_dim, config.deterministic_size)
        self.prior = MLP(config.deterministic_size, self.stochastic_flat_dim, config.hidden_dim)
        self.posterior = MLP(
            config.deterministic_size + config.encoder_dim,
            self.stochastic_flat_dim,
            config.hidden_dim,
        )
        feature_dim = self.feature_dim
        self.decoder = MLP(feature_dim, config.observation_dim, config.hidden_dim)
        # The untrained transition model is exact persistence. Learning starts
        # from zero delta instead of injecting arbitrary drift into rollouts.
        nn.init.zeros_(self.decoder.network[-1].weight)
        nn.init.zeros_(self.decoder.network[-1].bias)
        self.reward_head = MLP(feature_dim, config.reward_bins, config.hidden_dim)
        self.continue_head = MLP(feature_dim, 1, config.hidden_dim)
        self.risk_head = MLP(feature_dim, 1, config.hidden_dim)
        self.collision_head = MLP(feature_dim, 1, config.hidden_dim)
        self.offroad_head = MLP(feature_dim, 1, config.hidden_dim)
        self.register_buffer(
            "reward_bins",
            torch.linspace(config.reward_low, config.reward_high, config.reward_bins),
        )

    @property
    def stochastic_flat_dim(self) -> int:
        return self.config.stochastic_size * self.config.categorical_classes

    @property
    def feature_dim(self) -> int:
        return self.config.deterministic_size + self.stochastic_flat_dim

    def initial(self, batch_size: int, device: Optional[torch.device] = None) -> RSSMState:
        device = device or next(self.parameters()).device
        deterministic = torch.zeros(batch_size, self.config.deterministic_size, device=device)
        logits = torch.zeros(
            batch_size,
            self.config.stochastic_size,
            self.config.categorical_classes,
            device=device,
        )
        stochastic = self.sample(logits, deterministic=True)
        return RSSMState(deterministic, stochastic, logits)

    def feature(self, state: RSSMState) -> torch.Tensor:
        return torch.cat((state.deterministic, state.stochastic.reshape(*state.stochastic.shape[:-2], -1)), dim=-1)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        if self.config.unimix:
            probabilities = (
                (1.0 - self.config.unimix) * probabilities
                + self.config.unimix / float(self.config.categorical_classes)
            )
        return probabilities

    def sample(self, logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        probabilities = self.probabilities(logits)
        if deterministic:
            indices = probabilities.argmax(dim=-1)
            return F.one_hot(indices, self.config.categorical_classes).to(probabilities.dtype)
        sampled = torch.distributions.OneHotCategorical(probs=probabilities).sample()
        return sampled + probabilities - probabilities.detach()

    def _logits(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.reshape(
            *flat.shape[:-1],
            self.config.stochastic_size,
            self.config.categorical_classes,
        )

    def observe_initial(self, observation: torch.Tensor, deterministic: bool = False) -> RSSMState:
        state = self.initial(observation.shape[0], observation.device)
        embedding = self.encoder(symlog(observation))
        logits = self._logits(self.posterior(torch.cat((state.deterministic, embedding), dim=-1)))
        return RSSMState(state.deterministic, self.sample(logits, deterministic), logits)

    def imagine_step(self, previous: RSSMState, action: torch.Tensor, deterministic: bool = False) -> RSSMState:
        action = action / torch.maximum(torch.ones_like(action), action.abs()).detach()
        recurrent_input = self.core_input(
            torch.cat((previous.stochastic.reshape(action.shape[0], -1), action), dim=-1)
        )
        deterministic_state = self.recurrent(recurrent_input, previous.deterministic)
        logits = self._logits(self.prior(deterministic_state))
        return RSSMState(deterministic_state, self.sample(logits, deterministic), logits)

    def observe_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[RSSMState, RSSMState]:
        prior = self.imagine_step(previous, action, deterministic)
        embedding = self.encoder(symlog(observation))
        logits = self._logits(
            self.posterior(torch.cat((prior.deterministic, embedding), dim=-1))
        )
        posterior = RSSMState(prior.deterministic, self.sample(logits, deterministic), logits)
        return posterior, prior

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tuple[RSSMState, ...], Tuple[RSSMState, ...]]:
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("observations/actions must be [batch, time, feature]")
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("a sequence requires T+1 observations and T actions")
        state = self.observe_initial(observations[:, 0], deterministic)
        posteriors = []
        priors = []
        for index in range(actions.shape[1]):
            state, prior = self.observe_step(
                state, actions[:, index], observations[:, index + 1], deterministic
            )
            posteriors.append(state)
            priors.append(prior)
        return tuple(posteriors), tuple(priors)

    def prediction(
        self,
        state: RSSMState,
        reference_observation: torch.Tensor,
    ) -> Prediction:
        """Decode one transition relative to its current observation.

        Predicting an observation delta gives the prior a persistence anchor. A
        zero decoded delta is therefore the trivial persistence model that the
        learned dynamics must improve upon at the frozen validation gate.
        """
        feature = self.feature(state)
        reward_logits = self.reward_head(feature)
        return Prediction(
            observation=reference_observation + symexp(self.decoder(feature)),
            reward=two_hot_mean(reward_logits, self.reward_bins),
            continuation=torch.sigmoid(self.continue_head(feature).squeeze(-1)),
            risk=torch.sigmoid(self.risk_head(feature).squeeze(-1)),
            collision=torch.sigmoid(self.collision_head(feature).squeeze(-1)),
            offroad=torch.sigmoid(self.offroad_head(feature).squeeze(-1)),
        )

    def _prediction_logits(self, state: RSSMState) -> Dict[str, torch.Tensor]:
        feature = self.feature(state)
        return {
            "observation": self.decoder(feature),
            "reward": self.reward_head(feature),
            "continuation": self.continue_head(feature).squeeze(-1),
            "risk": self.risk_head(feature).squeeze(-1),
            "collision": self.collision_head(feature).squeeze(-1),
            "offroad": self.offroad_head(feature).squeeze(-1),
        }

    def balanced_kl(self, posterior: RSSMState, prior: RSSMState) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior_probability = self.probabilities(posterior.logits)
        prior_probability = self.probabilities(prior.logits)
        posterior_log = torch.log(posterior_probability.clamp_min(1.0e-8))
        prior_log = torch.log(prior_probability.clamp_min(1.0e-8))
        dynamics = (
            posterior_probability.detach() * (posterior_log.detach() - prior_log)
        ).sum(dim=(-1, -2))
        representation = (
            posterior_probability * (posterior_log - prior_log.detach())
        ).sum(dim=(-1, -2))
        return (
            dynamics.clamp_min(self.config.free_nats),
            representation.clamp_min(self.config.free_nats),
        )

    @staticmethod
    def _weighted_bce(logits: torch.Tensor, target: torch.Tensor, positive_weight: float) -> torch.Tensor:
        weights = torch.where(target > 0.5, torch.full_like(target, positive_weight), torch.ones_like(target))
        return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()

    def loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        weights: LossConfig,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        posteriors, priors = self.observe_sequence(observations, actions)
        posterior_outputs = [self._prediction_logits(state) for state in posteriors]
        prior_outputs = [self._prediction_logits(state) for state in priors]
        observation_delta = observations[:, 1:] - observations[:, :-1]

        def prediction_losses(
            outputs: Sequence[Dict[str, torch.Tensor]],
            prefix: str,
        ) -> Dict[str, torch.Tensor]:
            observation_logits = torch.stack(
                [item["observation"] for item in outputs], dim=1
            )
            reward_logits = torch.stack([item["reward"] for item in outputs], dim=1)
            continuation_logits = torch.stack(
                [item["continuation"] for item in outputs], dim=1
            )
            risk_logits = torch.stack([item["risk"] for item in outputs], dim=1)
            collision_logits = torch.stack(
                [item["collision"] for item in outputs], dim=1
            )
            offroad_logits = torch.stack([item["offroad"] for item in outputs], dim=1)
            return {
                prefix + "observation": F.mse_loss(
                    observation_logits, symlog(observation_delta)
                ),
                prefix + "reward": two_hot_loss(
                    reward_logits, targets["rewards"], self.reward_bins
                ).mean(),
                prefix + "continuation": F.binary_cross_entropy_with_logits(
                    continuation_logits, targets["continuation"]
                ),
                prefix + "risk": F.binary_cross_entropy_with_logits(
                    risk_logits, targets["risk"]
                ),
                prefix + "collision": self._weighted_bce(
                    collision_logits,
                    targets["collision"],
                    weights.collision_positive_weight,
                ),
                prefix + "offroad": self._weighted_bce(
                    offroad_logits,
                    targets["offroad"],
                    weights.offroad_positive_weight,
                ),
            }

        losses = prediction_losses(posterior_outputs, "")
        losses.update(prediction_losses(prior_outputs, "prior_"))
        dynamics = []
        representation = []
        for posterior, prior in zip(posteriors, priors):
            first, second = self.balanced_kl(posterior, prior)
            dynamics.append(first.mean())
            representation.append(second.mean())
        losses["dynamics_kl"] = torch.stack(dynamics).mean()
        losses["representation_kl"] = torch.stack(representation).mean()
        posterior_prediction = (
            weights.observation * losses["observation"]
            + weights.reward * losses["reward"]
            + weights.continuation * losses["continuation"]
            + weights.risk * losses["risk"]
            + weights.collision * losses["collision"]
            + weights.offroad * losses["offroad"]
        )
        prior_prediction = (
            weights.observation * losses["prior_observation"]
            + weights.reward * losses["prior_reward"]
            + weights.continuation * losses["prior_continuation"]
            + weights.risk * losses["prior_risk"]
            + weights.collision * losses["prior_collision"]
            + weights.offroad * losses["prior_offroad"]
        )
        losses["posterior_prediction"] = posterior_prediction
        losses["prior_prediction"] = prior_prediction
        total = (
            posterior_prediction
            + weights.prior_prediction * prior_prediction
            + self.config.dynamics_kl_scale * losses["dynamics_kl"]
            + self.config.representation_kl_scale * losses["representation_kl"]
        )
        losses["total"] = total
        return total, losses


@dataclass
class ActorOutput:
    final_action: torch.Tensor
    proposal: torch.Tensor
    residual: torch.Tensor
    authority: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor


def signed_longitudinal(action: torch.Tensor) -> torch.Tensor:
    return action[..., 1] - action[..., 2]


def physical_control(steer: torch.Tensor, longitudinal: torch.Tensor) -> torch.Tensor:
    longitudinal = longitudinal.clamp(-1.0, 1.0)
    return torch.stack(
        (steer.clamp(-1.0, 1.0), longitudinal.clamp_min(0.0), (-longitudinal).clamp_min(0.0)),
        dim=-1,
    )


class ResidualActor(nn.Module):
    """Stochastic residual and continuous authority around SimLingo."""

    def __init__(self, feature_dim: int, observation_dim: int, config: ActorConfig):
        super().__init__()
        self.config = config
        self.trunk = MLP(feature_dim + observation_dim, config.hidden_dim, config.hidden_dim)
        self.mean = nn.Linear(config.hidden_dim, 3)
        self.log_std = nn.Parameter(torch.full((3,), -1.5))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        with torch.no_grad():
            probability = float(config.initial_authority)
            self.mean.bias[2] = math.log(probability / (1.0 - probability))

    def forward(
        self,
        feature: torch.Tensor,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> ActorOutput:
        hidden = self.trunk(torch.cat((feature, symlog(observation)), dim=-1))
        mean = self.mean(hidden)
        std = self.log_std.exp().clamp(self.config.minimum_std, self.config.maximum_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        raw = mean if deterministic else distribution.rsample()
        bounded = torch.tanh(raw[..., :2])
        authority = torch.sigmoid(raw[..., 2])
        residual = torch.stack(
            (
                bounded[..., 0] * self.config.maximum_steer_residual,
                bounded[..., 1] * self.config.maximum_longitudinal_residual,
            ),
            dim=-1,
        )
        native = observation[..., 2:5]
        proposal = physical_control(
            native[..., 0] + residual[..., 0],
            signed_longitudinal(native) + residual[..., 1],
        )
        final_action = physical_control(
            native[..., 0] + authority * residual[..., 0],
            signed_longitudinal(native) + authority * residual[..., 1],
        )
        log_probability = distribution.log_prob(raw).sum(dim=-1)
        log_probability -= torch.log(1.0 - bounded.square() + 1.0e-6).sum(dim=-1)
        log_probability -= torch.log(authority * (1.0 - authority) + 1.0e-6)
        return ActorOutput(
            final_action=final_action,
            proposal=proposal,
            residual=residual,
            authority=authority,
            log_probability=log_probability,
            entropy=distribution.entropy().sum(dim=-1),
        )


class TwoHotCritic(nn.Module):
    def __init__(self, feature_dim: int, config: ModelConfig, actor_config: ActorConfig):
        super().__init__()
        self.network = MLP(feature_dim, config.value_bins, actor_config.hidden_dim)
        self.register_buffer("bins", torch.linspace(config.value_low, config.value_high, config.value_bins))

    def logits(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return two_hot_mean(self.logits(feature), self.bins)

    def loss(self, feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return two_hot_loss(self.logits(feature), target, self.bins)


class ResidualDreamerV3(nn.Module):
    def __init__(self, config: Optional[ResidualDreamerConfig] = None):
        super().__init__()
        self.config = config or ResidualDreamerConfig()
        self.config.validate()
        self.world_model = CategoricalRSSM(self.config.model)
        self.actor = ResidualActor(
            self.world_model.feature_dim,
            self.config.model.observation_dim,
            self.config.actor,
        )
        self.critic = TwoHotCritic(self.world_model.feature_dim, self.config.model, self.config.actor)
        self.slow_critic = TwoHotCritic(self.world_model.feature_dim, self.config.model, self.config.actor)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        for parameter in self.slow_critic.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_slow_critic(self) -> None:
        fraction = float(self.config.actor.slow_critic_fraction)
        for slow, current in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow.data.lerp_(current.data, fraction)
