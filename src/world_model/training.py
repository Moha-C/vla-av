"""Imagined actor/critic optimization for the report-aligned Dreamer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import DreamerConfig
from .policy import LatentCritic, ResidualActor, blend_control
from .rssm import CompactRSSM, RSSMState


@dataclass
class ImaginedTrainingBatch:
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    objective: torch.Tensor
    mean_risk: torch.Tensor
    mean_collision: torch.Tensor
    mean_offroad: torch.Tensor
    mean_alpha: torch.Tensor


def lambda_returns(
    rewards: torch.Tensor,
    continuations: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    discount: float,
    lambda_: float,
) -> torch.Tensor:
    """Compute continuation-aware lambda returns for [batch, horizon]."""

    next_values = torch.cat((values[:, 1:], bootstrap.unsqueeze(1)), dim=1)
    result: List[torch.Tensor] = []
    accumulator = bootstrap
    for index in range(rewards.shape[1] - 1, -1, -1):
        mixed = (1.0 - lambda_) * next_values[:, index] + lambda_ * accumulator
        accumulator = rewards[:, index] + discount * continuations[:, index] * mixed
        result.append(accumulator)
    return torch.stack(list(reversed(result)), dim=1)


def imagine_actor_critic(
    world_model: CompactRSSM,
    actor: ResidualActor,
    critic: LatentCritic,
    start: RSSMState,
    native: torch.Tensor,
    config: DreamerConfig,
    deterministic: bool = False,
) -> ImaginedTrainingBatch:
    state = start
    previous_control = native
    rewards: List[torch.Tensor] = []
    continuations: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    features: List[torch.Tensor] = []
    risks: List[torch.Tensor] = []
    collisions: List[torch.Tensor] = []
    offroads: List[torch.Tensor] = []
    alphas: List[torch.Tensor] = []
    cfg = config.reward
    for _ in range(config.rssm.imagination_horizon):
        feature = world_model.feature(state)
        actor_output = actor(feature, native, deterministic)
        alpha = actor_output.alpha.clamp(0.0, config.authority.max_alpha)
        final_control = blend_control(native, actor_output.proposal, alpha)
        state = world_model.imagine_step(state, final_control, deterministic=False)
        prediction = world_model.heads(state)
        next_feature = world_model.feature(state)
        jerk = torch.linalg.vector_norm(final_control - previous_control, dim=-1)
        safety = 1.0 - 2.0 * prediction.risk.squeeze(-1)
        reward = (
            cfg.w_progress * prediction.progress.squeeze(-1)
            + cfg.w_safe * safety
            - cfg.w_collision * prediction.collision.squeeze(-1)
            - cfg.w_offroad * prediction.offroad.squeeze(-1)
            - cfg.w_jerk * jerk
            - cfg.w_alpha * alpha
        )
        rewards.append(reward)
        continuations.append(prediction.continuation.squeeze(-1))
        entropies.append(actor_output.entropy)
        features.append(next_feature)
        risks.append(prediction.risk.squeeze(-1))
        collisions.append(prediction.collision.squeeze(-1))
        offroads.append(prediction.offroad.squeeze(-1))
        alphas.append(alpha)
        previous_control = final_control
    reward_tensor = torch.stack(rewards, dim=1)
    continuation_tensor = torch.stack(continuations, dim=1)
    feature_tensor = torch.stack(features, dim=1)
    value_tensor = critic(feature_tensor.detach().reshape(-1, feature_tensor.shape[-1])).reshape(
        feature_tensor.shape[:2]
    )
    bootstrap = critic(world_model.feature(state)).detach()
    returns = lambda_returns(
        reward_tensor,
        continuation_tensor,
        value_tensor.detach(),
        bootstrap,
        config.evaluator.continuation_discount,
        config.policy.lambda_return,
    )
    weights = torch.ones_like(continuation_tensor)
    if weights.shape[1] > 1:
        weights[:, 1:] = torch.cumprod(
            config.evaluator.continuation_discount
            * continuation_tensor[:, :-1].detach(),
            dim=1,
        )
    entropy = torch.stack(entropies, dim=1)
    objective = (weights * returns).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0e-6)
    actor_loss = -objective.mean() - config.policy.entropy_scale * entropy.mean()
    critic_loss = F.smooth_l1_loss(value_tensor, returns.detach())
    return ImaginedTrainingBatch(
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        objective=objective.mean(),
        mean_risk=torch.stack(risks, dim=1).mean(),
        mean_collision=torch.stack(collisions, dim=1).mean(),
        mean_offroad=torch.stack(offroads, dim=1).mean(),
        mean_alpha=torch.stack(alphas, dim=1).mean(),
    )
