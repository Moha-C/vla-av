"""Residual actor and latent critic trained in RSSM imagination."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import PolicyConfig


def signed_longitudinal(control: torch.Tensor) -> torch.Tensor:
    return control[..., 1] - control[..., 2]


def split_longitudinal(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    value = value.clamp(-1.0, 1.0)
    return value.clamp_min(0.0), (-value).clamp_min(0.0)


@dataclass
class ActorOutput:
    proposal: torch.Tensor
    alpha: torch.Tensor
    raw_action: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor


class ResidualActor(nn.Module):
    """Propose a bounded correction around the native SimLingo control."""

    def __init__(self, feature_dim: int, config: Optional[PolicyConfig] = None):
        super().__init__()
        self.config = config or PolicyConfig()
        hidden = self.config.hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, 3)
        self.log_std = nn.Parameter(torch.zeros(3))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        with torch.no_grad():
            probability = float(self.config.initial_alpha)
            self.mean.bias[2] = math.log(probability / (1.0 - probability))

    def distribution(self, feature: torch.Tensor, native: torch.Tensor) -> torch.distributions.Normal:
        hidden = self.trunk(torch.cat((feature, native), dim=-1))
        mean = self.mean(hidden)
        minimum = float(self.config.min_std)
        maximum = float(self.config.max_std)
        std = self.log_std.exp().clamp(minimum, maximum).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def forward(
        self,
        feature: torch.Tensor,
        native: torch.Tensor,
        deterministic: bool = False,
    ) -> ActorOutput:
        distribution = self.distribution(feature, native)
        raw = distribution.mean if deterministic else distribution.rsample()
        steer_delta = torch.tanh(raw[..., 0]) * self.config.max_steer_residual
        long_delta = torch.tanh(raw[..., 1]) * self.config.max_longitudinal_residual
        alpha = torch.sigmoid(raw[..., 2])
        steer = (native[..., 0] + steer_delta).clamp(-1.0, 1.0)
        longitudinal = (signed_longitudinal(native) + long_delta).clamp(-1.0, 1.0)
        throttle, brake = split_longitudinal(longitudinal)
        proposal = torch.stack((steer, throttle, brake), dim=-1)
        return ActorOutput(
            proposal=proposal,
            alpha=alpha,
            raw_action=raw,
            log_probability=distribution.log_prob(raw).sum(dim=-1),
            entropy=distribution.entropy().sum(dim=-1),
        )


class LatentCritic(nn.Module):
    def __init__(self, feature_dim: int, config: Optional[PolicyConfig] = None):
        super().__init__()
        cfg = config or PolicyConfig()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature).squeeze(-1)


def blend_control(native: torch.Tensor, proposal: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.unsqueeze(-1) if alpha.ndim == native.ndim - 1 else alpha
    blended = (1.0 - alpha) * native + alpha * proposal
    steer = blended[..., 0].clamp(-1.0, 1.0)
    longitudinal = signed_longitudinal(blended).clamp(-1.0, 1.0)
    throttle, brake = split_longitudinal(longitudinal)
    return torch.stack((steer, throttle, brake), dim=-1)
