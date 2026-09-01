"""Optional pairwise candidate utility calibrator."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PairwiseCalibrator(nn.Module):
    """Predict whether candidate A is preferable to candidate B.

    Candidate features are expected to contain predicted progression, risk,
    continuation, value, and action-change cost. The antisymmetric difference
    prevents a hidden candidate-index preference.
    """

    def __init__(self, feature_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def logits(self, candidate_a: torch.Tensor, candidate_b: torch.Tensor) -> torch.Tensor:
        return self.network(candidate_a - candidate_b).squeeze(-1)

    def forward(self, candidate_a: torch.Tensor, candidate_b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(candidate_a, candidate_b))

    def tournament_bonus(self, features: torch.Tensor) -> torch.Tensor:
        count = features.shape[0]
        if count <= 1:
            return torch.zeros(count, device=features.device)
        bonuses = []
        for index in range(count):
            repeated = features[index].unsqueeze(0).expand(count, -1)
            probability = self.forward(repeated, features)
            mask = torch.arange(count, device=features.device) != index
            bonuses.append((probability[mask] - 0.5).mean())
        return torch.stack(bonuses)
