"""Numerically stable DreamerV3 scalar transforms and two-hot targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    # Dreamer heads live in symlog space. Clamping prevents an untrained head
    # from overflowing before the validation gate has had a chance to reject it.
    value = value.clamp(-20.0, 20.0)
    return torch.sign(value) * torch.expm1(torch.abs(value))


def two_hot(value: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate a scalar onto two adjacent categorical bins."""

    value = symlog(value).clamp(float(bins[0]), float(bins[-1]))
    position = (value - bins[0]) / (bins[-1] - bins[0]) * (len(bins) - 1)
    lower = position.floor().long().clamp(0, len(bins) - 1)
    upper = (lower + 1).clamp(0, len(bins) - 1)
    upper_weight = position - lower.to(position.dtype)
    lower_weight = 1.0 - upper_weight
    target = torch.zeros(*value.shape, len(bins), device=value.device, dtype=value.dtype)
    target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target


def two_hot_loss(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    labels = two_hot(target, bins)
    return -(labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def two_hot_mean(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    expected_symlog = (torch.softmax(logits, dim=-1) * bins).sum(dim=-1)
    return symexp(expected_symlog)
