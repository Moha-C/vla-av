"""Loss functions for supervised VLA action-head training."""

from __future__ import annotations

import torch


def action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted MSE over [steering, throttle, brake]."""

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must share shape, got {pred.shape} and {target.shape}.")
    if pred.shape[-1] != 3:
        raise ValueError(f"Expected action tensors with last dim 3, got {pred.shape}.")

    if weights is None:
        weights = torch.tensor([2.0, 1.0, 1.0], dtype=pred.dtype, device=pred.device)
    else:
        weights = weights.to(device=pred.device, dtype=pred.dtype)

    squared_error = (pred - target) ** 2
    return (squared_error * weights).mean()
