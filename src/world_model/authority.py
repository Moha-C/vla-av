"""Continuous learned/fixed authority with temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import AuthorityConfig


@dataclass(frozen=True)
class AuthorityDecision:
    alpha_raw: float
    alpha: float
    native_weight: float
    dreamer_weight: float


class LearnedAuthorityController:
    def __init__(self, config: Optional[AuthorityConfig] = None):
        self.config = config or AuthorityConfig()
        self.reset()

    def reset(self) -> None:
        self._previous_alpha = 0.0

    def decide(
        self,
        learned_alpha: float,
        candidate_is_native: bool = False,
        force_alpha: Optional[float] = None,
    ) -> AuthorityDecision:
        cfg = self.config
        if not cfg.enabled or candidate_is_native:
            raw = 0.0
        elif force_alpha is not None:
            raw = float(force_alpha)
        elif cfg.learned:
            raw = float(learned_alpha)
        else:
            raw = float(cfg.fixed_alpha)
        raw = float(np.clip(raw, 0.0, cfg.max_alpha))
        smoothed = cfg.smoothing * self._previous_alpha + (1.0 - cfg.smoothing) * raw
        low = self._previous_alpha - cfg.max_delta_per_step
        high = self._previous_alpha + cfg.max_delta_per_step
        alpha = float(np.clip(smoothed, max(0.0, low), min(cfg.max_alpha, high)))
        if raw <= cfg.exact_native_epsilon:
            alpha = 0.0
        self._previous_alpha = alpha
        return AuthorityDecision(raw, alpha, 1.0 - alpha, alpha)

    def blend(
        self,
        native: Sequence[float],
        dreamer: Sequence[float],
        alpha: float,
    ) -> np.ndarray:
        native_array = np.asarray(native, dtype=np.float32)[:3]
        dreamer_array = np.asarray(dreamer, dtype=np.float32)[:3]
        if alpha <= self.config.exact_native_epsilon:
            return native_array.copy()
        mixed = (1.0 - alpha) * native_array + alpha * dreamer_array
        steer = float(np.clip(mixed[0], -1.0, 1.0))
        longitudinal = float(np.clip(mixed[1] - mixed[2], -1.0, 1.0))
        return np.asarray(
            [steer, max(0.0, longitudinal), max(0.0, -longitudinal)],
            dtype=np.float32,
        )
