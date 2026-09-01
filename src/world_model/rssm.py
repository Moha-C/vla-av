"""Compact categorical RSSM adapted from DreamerV3 mechanisms.

The implementation is PyTorch-native because SimLingo runs in a Python 3.8
PyTorch environment. It adapts the latent-state, balanced-KL and imagination
mechanisms of the MIT-licensed DreamerV3 code bundled by CarDreamer; it does
not import CarDreamer's environment or autonomous policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PredictionLossConfig, RSSMConfig


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

    def repeat_interleave(self, repeats: int) -> "RSSMState":
        return RSSMState(
            self.deterministic.repeat_interleave(repeats, dim=0),
            self.stochastic.repeat_interleave(repeats, dim=0),
            self.logits.repeat_interleave(repeats, dim=0),
        )


@dataclass
class WorldModelOutput:
    observation: torch.Tensor
    progress: torch.Tensor
    risk: torch.Tensor
    continuation: torch.Tensor
    value: torch.Tensor
    collision: torch.Tensor
    offroad: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "observation": self.observation,
            "progress": self.progress,
            "risk": self.risk,
            "continuation": self.continuation,
            "value": self.value,
            "collision": self.collision,
            "offroad": self.offroad,
        }


class DenseBlock(nn.Module):
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


class PredictionHeads(nn.Module):
    """Separate physical heads so every prediction can be evaluated alone."""

    def __init__(self, feature_dim: int, observation_dim: int, hidden_dim: int):
        super().__init__()
        self.observation = DenseBlock(feature_dim, observation_dim, hidden_dim)
        self.progress = DenseBlock(feature_dim, 1, hidden_dim)
        self.risk = DenseBlock(feature_dim, 1, hidden_dim)
        self.continuation = DenseBlock(feature_dim, 1, hidden_dim)
        self.value = DenseBlock(feature_dim, 1, hidden_dim)
        self.collision = DenseBlock(feature_dim, 1, hidden_dim)
        self.offroad = DenseBlock(feature_dim, 1, hidden_dim)

    def forward(self, feature: torch.Tensor) -> WorldModelOutput:
        return WorldModelOutput(
            observation=torch.tanh(self.observation(feature)),
            progress=self.progress(feature),
            risk=torch.sigmoid(self.risk(feature)),
            continuation=torch.sigmoid(self.continuation(feature)),
            value=self.value(feature),
            collision=torch.sigmoid(self.collision(feature)),
            offroad=torch.sigmoid(self.offroad(feature)),
        )


class CompactRSSM(nn.Module):
    """Recurrent deterministic state plus categorical stochastic state."""

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
            nn.Linear(self.stochastic_flat_dim + cfg.action_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(cfg.hidden_dim, cfg.deterministic_size)
        self.prior_network = DenseBlock(
            cfg.deterministic_size,
            self.stochastic_flat_dim,
            cfg.hidden_dim,
        )
        self.posterior_network = DenseBlock(
            cfg.deterministic_size + cfg.encoder_dim,
            self.stochastic_flat_dim,
            cfg.hidden_dim,
        )
        self.prediction_heads = PredictionHeads(
            self.feature_dim,
            cfg.observation_dim,
            cfg.hidden_dim,
        )

    @property
    def stochastic_flat_dim(self) -> int:
        return int(self.config.stochastic_size * self.config.categorical_classes)

    @property
    def feature_dim(self) -> int:
        return int(self.config.deterministic_size + self.stochastic_flat_dim)

    def initial(self, batch_size: int, device: Optional[torch.device] = None) -> RSSMState:
        parameter = next(self.parameters())
        device = device or parameter.device
        deterministic = torch.zeros(batch_size, self.config.deterministic_size, device=device)
        logits = torch.zeros(
            batch_size,
            self.config.stochastic_size,
            self.config.categorical_classes,
            device=device,
        )
        stochastic = self._sample(logits, deterministic=True)
        return RSSMState(deterministic, stochastic, logits)

    def feature(self, state: RSSMState) -> torch.Tensor:
        return torch.cat((state.deterministic, state.stochastic), dim=-1)

    def _reshape_logits(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.reshape(
            *flat.shape[:-1],
            self.config.stochastic_size,
            self.config.categorical_classes,
        )

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        if self.config.unimix > 0.0:
            uniform = torch.full_like(
                probabilities, 1.0 / self.config.categorical_classes
            )
            probabilities = (
                (1.0 - self.config.unimix) * probabilities
                + self.config.unimix * uniform
            )
        return probabilities

    def _sample(self, logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
        probabilities = self.probabilities(logits)
        if deterministic:
            # Runtime evaluation must be reproducible.  A categorical RSSM uses
            # the modal category here; returning the probability vector would
            # silently change the latent representation between training and
            # deterministic closed-loop inference.
            indices = probabilities.argmax(dim=-1)
            one_hot = F.one_hot(
                indices, num_classes=self.config.categorical_classes
            ).to(probabilities.dtype)
        else:
            distribution = torch.distributions.OneHotCategorical(probs=probabilities)
            sampled = distribution.sample()
            one_hot = sampled + probabilities - probabilities.detach()
        return one_hot.reshape(*one_hot.shape[:-2], -1)

    def posterior_from_embedding(
        self,
        deterministic_state: torch.Tensor,
        embedding: torch.Tensor,
        deterministic: bool,
    ) -> RSSMState:
        logits = self._reshape_logits(
            self.posterior_network(torch.cat((deterministic_state, embedding), dim=-1))
        )
        stochastic = self._sample(logits, deterministic)
        return RSSMState(deterministic_state, stochastic, logits)

    def observe_initial(self, observation: torch.Tensor, deterministic: bool = False) -> RSSMState:
        initial = self.initial(observation.shape[0], observation.device)
        embedding = self.encoder(observation)
        return self.posterior_from_embedding(
            initial.deterministic, embedding, deterministic
        )

    def imagine_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> RSSMState:
        recurrent_input = self.action_encoder(
            torch.cat((previous.stochastic, action), dim=-1)
        )
        deter = self.recurrent(recurrent_input, previous.deterministic)
        logits = self._reshape_logits(self.prior_network(deter))
        stochastic = self._sample(logits, deterministic)
        return RSSMState(deter, stochastic, logits)

    def observe_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[RSSMState, RSSMState]:
        prior = self.imagine_step(previous, action, deterministic)
        embedding = self.encoder(observation)
        posterior = self.posterior_from_embedding(
            prior.deterministic, embedding, deterministic
        )
        return posterior, prior

    def heads(self, state: RSSMState) -> WorldModelOutput:
        return self.prediction_heads(self.feature(state))

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[RSSMState, Tuple[RSSMState, ...], Tuple[RSSMState, ...]]:
        state, _, posteriors, priors = self._unroll_sequence(
            observations, actions, deterministic
        )
        return state, posteriors, priors

    def _unroll_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[
        RSSMState,
        RSSMState,
        Tuple[RSSMState, ...],
        Tuple[RSSMState, ...],
    ]:
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("observations/actions must be [batch, time, feature]")
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("an RSSM sequence needs T+1 observations and T actions")
        initial = self.observe_initial(observations[:, 0], deterministic)
        state = initial
        posteriors = []
        priors = []
        for index in range(actions.shape[1]):
            state, prior = self.observe_step(
                state,
                actions[:, index],
                observations[:, index + 1],
                deterministic,
            )
            posteriors.append(state)
            priors.append(prior)
        return state, initial, tuple(posteriors), tuple(priors)

    @staticmethod
    def _weighted_binary_cross_entropy(
        prediction: torch.Tensor,
        target: torch.Tensor,
        positive_weight: float,
    ) -> torch.Tensor:
        prediction = prediction.clamp(1.0e-6, 1.0 - 1.0e-6)
        positive = -float(positive_weight) * target * torch.log(prediction)
        negative = -(1.0 - target) * torch.log1p(-prediction)
        return (positive + negative).mean()

    def _action_contrastive_loss(
        self,
        initial: RSSMState,
        posteriors: Tuple[RSSMState, ...],
        priors: Tuple[RSSMState, ...],
        actions: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """Make the imagined transition identify the action that produced it.

        Each observed transition supplies a positive pair: previous latent plus
        the executed control must be closer to the next posterior than clearly
        different physical controls.  This does not label an alternative as
        safe or unsafe; it only prevents an action-agnostic dynamics model.
        """

        previous_states = (initial,) + posteriors[:-1]
        if not previous_states:
            return actions.sum() * 0.0
        previous = self._flatten_time_states(previous_states)
        target = torch.stack(
            [state.deterministic for state in posteriors], dim=1
        ).reshape(-1, self.config.deterministic_size).detach()
        true_prior = torch.stack(
            [state.deterministic for state in priors], dim=1
        ).reshape(-1, self.config.deterministic_size)
        action = actions.reshape(-1, self.config.action_dim)
        true_distance = F.mse_loss(
            true_prior, target, reduction="none"
        ).mean(dim=-1)

        hard_brake = action.clone()
        hard_brake[:, 1] = 0.0
        hard_brake[:, 2] = 1.0
        full_throttle = action.clone()
        full_throttle[:, 1] = 1.0
        full_throttle[:, 2] = 0.0
        steer_left = action.clone()
        steer_left[:, 0] = (steer_left[:, 0] - 0.35).clamp(-1.0, 1.0)
        steer_right = action.clone()
        steer_right[:, 0] = (steer_right[:, 0] + 0.35).clamp(-1.0, 1.0)
        alternatives = torch.stack(
            (hard_brake, full_throttle, steer_left, steer_right), dim=1
        )
        count = int(alternatives.shape[1])
        alternative_prior = self.imagine_step(
            previous.repeat_interleave(count),
            alternatives.reshape(-1, self.config.action_dim),
            deterministic=False,
        )
        alternative_distance = F.mse_loss(
            alternative_prior.deterministic,
            target.repeat_interleave(count, dim=0),
            reduction="none",
        ).mean(dim=-1).reshape(-1, count)
        different = (
            torch.linalg.vector_norm(
                alternatives - action.unsqueeze(1), dim=-1
            )
            > 5.0e-2
        ).to(action.dtype)
        terms = (
            F.relu(
                float(margin)
                + true_distance.unsqueeze(1)
                - alternative_distance
            )
            * different
        )
        numerator = terms.sum()
        denominator = different.sum().clamp_min(1.0)
        return numerator / denominator

    @staticmethod
    def _flatten_time_states(states: Tuple[RSSMState, ...]) -> RSSMState:
        def flatten(values: Tuple[torch.Tensor, ...]) -> torch.Tensor:
            stacked = torch.stack(values, dim=1)
            return stacked.reshape(-1, *stacked.shape[2:])

        return RSSMState(
            flatten(tuple(state.deterministic for state in states)),
            flatten(tuple(state.stochastic for state in states)),
            flatten(tuple(state.logits for state in states)),
        )

    def _action_safety_monotonic_loss(
        self,
        initial: RSSMState,
        posteriors: Tuple[RSSMState, ...],
        observations: torch.Tensor,
        actions: torch.Tensor,
        weights: PredictionLossConfig,
    ) -> Dict[str, torch.Tensor]:
        """Apply a disclosed physical prior to hazardous counterfactuals.

        Offline demonstrations are confounded: braking is observed mostly when
        the scene is already dangerous.  Without intervention data, a model can
        therefore learn that braking *causes* risk.  For hazard states only, we
        require hard braking to predict no more risk/collision/speed/progress
        than full throttle.  This shapes the learned world model; it never
        forces a control action in closed loop.
        """

        previous_states = (initial,) + posteriors[:-1]
        margin = float(weights.action_safety_margin)
        previous = self._flatten_time_states(previous_states)
        observation = observations[:, :-1].reshape(-1, observations.shape[-1])
        hazard = (
            (observation[:, 15] <= float(weights.hazard_front_clearance))
            | (observation[:, 19] <= float(weights.hazard_oncoming_ttc))
            | (observation[:, 21] <= float(weights.hazard_oncoming_ttc))
            | (observation[:, 23] <= float(weights.hazard_oncoming_ttc))
            | (observation[:, 26] <= float(weights.hazard_vru_distance))
        ).to(observation.dtype)
        native = actions.reshape(-1, self.config.action_dim)
        hard_brake = native.clone()
        hard_brake[:, 1] = 0.0
        hard_brake[:, 2] = 1.0
        full_throttle = native.clone()
        full_throttle[:, 1] = 1.0
        full_throttle[:, 2] = 0.0
        brake_output = self.heads(
            self.imagine_step(previous, hard_brake, deterministic=False)
        )
        throttle_output = self.heads(
            self.imagine_step(previous, full_throttle, deterministic=False)
        )
        denominator = hazard.sum().clamp_min(1.0)

        def average(value: torch.Tensor) -> torch.Tensor:
            return (F.relu(value) * hazard).sum() / denominator

        risk = average(
            margin
            + brake_output.risk.squeeze(-1)
            - throttle_output.risk.squeeze(-1)
        )
        collision = average(
            margin
            + brake_output.collision.squeeze(-1)
            - throttle_output.collision.squeeze(-1)
        )
        speed = average(
            margin
            + brake_output.observation[:, 0]
            - throttle_output.observation[:, 0]
        )
        progress = average(
            margin
            + brake_output.progress.squeeze(-1)
            - throttle_output.progress.squeeze(-1)
        )
        return {
            "action_safety_risk": risk,
            "action_safety_collision": collision,
            "action_safety_speed": speed,
            "action_safety_progress": progress,
            "action_safety_hazard_fraction": hazard.mean(),
            "action_safety_monotonic": (risk + collision + speed + progress)
            / 4.0,
        }

    def balanced_kl(self, posterior: RSSMState, prior: RSSMState) -> Tuple[torch.Tensor, torch.Tensor]:
        post_prob = self.probabilities(posterior.logits)
        prior_prob = self.probabilities(prior.logits)
        post_log = torch.log(post_prob.clamp_min(1.0e-8))
        prior_log = torch.log(prior_prob.clamp_min(1.0e-8))
        dynamics = (
            post_prob.detach() * (post_log.detach() - prior_log)
        ).sum(dim=(-1, -2))
        representation = (
            post_prob * (post_log - prior_log.detach())
        ).sum(dim=(-1, -2))
        free = float(self.config.free_nats)
        return dynamics.clamp_min(free), representation.clamp_min(free)

    def loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        loss_config: Optional[PredictionLossConfig] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        weights = loss_config or PredictionLossConfig()
        _, initial, posteriors, priors = self._unroll_sequence(
            observations, actions
        )
        # Dreamer trains decoders/reward heads from posterior features, while
        # the balanced KL aligns the prior used by imagination with that
        # posterior.  Training every head directly from the prior makes a
        # difficult dynamics problem masquerade as a decoder error and is not
        # the DreamerV3 mechanism adapted here.
        predicted = [self.heads(state) for state in posteriors]
        observation_prediction = torch.stack([item.observation for item in predicted], dim=1)
        progress_prediction = torch.stack([item.progress for item in predicted], dim=1).squeeze(-1)
        risk_prediction = torch.stack([item.risk for item in predicted], dim=1).squeeze(-1)
        continuation_prediction = torch.stack([item.continuation for item in predicted], dim=1).squeeze(-1)
        value_prediction = torch.stack([item.value for item in predicted], dim=1).squeeze(-1)
        collision_prediction = torch.stack([item.collision for item in predicted], dim=1).squeeze(-1)
        offroad_prediction = torch.stack([item.offroad for item in predicted], dim=1).squeeze(-1)

        losses = {
            "observation": F.mse_loss(observation_prediction, observations[:, 1:]),
            "progress": F.mse_loss(progress_prediction, targets["progress"]),
            "risk": F.binary_cross_entropy(risk_prediction, targets["risk"]),
            "continuation": F.binary_cross_entropy(
                continuation_prediction, targets["continuation"]
            ),
            "value": F.smooth_l1_loss(value_prediction, targets["value"]),
            "collision": self._weighted_binary_cross_entropy(
                collision_prediction,
                targets["collision"],
                weights.collision_positive_weight,
            ),
            "offroad": self._weighted_binary_cross_entropy(
                offroad_prediction,
                targets["offroad"],
                weights.offroad_positive_weight,
            ),
        }
        prior_predicted = [self.heads(state) for state in priors]
        prior_observation = torch.stack(
            [item.observation for item in prior_predicted], dim=1
        )
        prior_progress = torch.stack(
            [item.progress for item in prior_predicted], dim=1
        ).squeeze(-1)
        prior_risk = torch.stack(
            [item.risk for item in prior_predicted], dim=1
        ).squeeze(-1)
        prior_continuation = torch.stack(
            [item.continuation for item in prior_predicted], dim=1
        ).squeeze(-1)
        prior_value = torch.stack(
            [item.value for item in prior_predicted], dim=1
        ).squeeze(-1)
        prior_collision = torch.stack(
            [item.collision for item in prior_predicted], dim=1
        ).squeeze(-1)
        prior_offroad = torch.stack(
            [item.offroad for item in prior_predicted], dim=1
        ).squeeze(-1)
        losses.update(
            {
                "prior_observation": F.mse_loss(
                    prior_observation, observations[:, 1:]
                ),
                "prior_progress": F.mse_loss(
                    prior_progress, targets["progress"]
                ),
                "prior_risk": F.binary_cross_entropy(
                    prior_risk, targets["risk"]
                ),
                "prior_continuation": F.binary_cross_entropy(
                    prior_continuation, targets["continuation"]
                ),
                "prior_value": F.smooth_l1_loss(
                    prior_value, targets["value"]
                ),
                "prior_collision": self._weighted_binary_cross_entropy(
                    prior_collision,
                    targets["collision"],
                    weights.collision_positive_weight,
                ),
                "prior_offroad": self._weighted_binary_cross_entropy(
                    prior_offroad,
                    targets["offroad"],
                    weights.offroad_positive_weight,
                ),
            }
        )
        dynamic_terms = []
        representation_terms = []
        for posterior, prior in zip(posteriors, priors):
            dynamic, representation = self.balanced_kl(posterior, prior)
            dynamic_terms.append(dynamic.mean())
            representation_terms.append(representation.mean())
        losses["dynamics_kl"] = torch.stack(dynamic_terms).mean()
        losses["representation_kl"] = torch.stack(representation_terms).mean()
        posterior_total = (
            weights.observation * losses["observation"]
            + weights.progress * losses["progress"]
            + weights.risk * losses["risk"]
            + weights.continuation * losses["continuation"]
            + weights.value * losses["value"]
            + weights.collision * losses["collision"]
            + weights.offroad * losses["offroad"]
        )
        prior_total = (
            weights.observation * losses["prior_observation"]
            + weights.progress * losses["prior_progress"]
            + weights.risk * losses["prior_risk"]
            + weights.continuation * losses["prior_continuation"]
            + weights.value * losses["prior_value"]
            + weights.collision * losses["prior_collision"]
            + weights.offroad * losses["prior_offroad"]
        )
        losses["prior_prediction_total"] = prior_total
        losses["action_contrastive"] = self._action_contrastive_loss(
            initial,
            posteriors,
            priors,
            actions,
            weights.action_contrastive_margin,
        )
        losses.update(
            self._action_safety_monotonic_loss(
                initial,
                posteriors,
                observations,
                actions,
                weights,
            )
        )
        total = (
            posterior_total
            + weights.prior_prediction * prior_total
            + weights.action_contrastive * losses["action_contrastive"]
            + weights.action_safety_monotonic
            * losses["action_safety_monotonic"]
            + self.config.dynamics_kl_scale * losses["dynamics_kl"]
            + self.config.representation_kl_scale * losses["representation_kl"]
        )
        losses["total"] = total
        return total, losses
