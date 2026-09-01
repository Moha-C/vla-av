"""Configurable reward specified by the internship report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .config import RewardConfig


@dataclass(frozen=True)
class RewardSignals:
    progress_delta: float
    safety: float
    collision: float = 0.0
    offroad: float = 0.0
    jerk: float = 0.0
    alpha: float = 0.0


@dataclass(frozen=True)
class RewardResult:
    total: float
    progress: float
    safety: float
    collision: float
    offroad: float
    jerk: float
    authority: float
    terminal: bool

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class DreamerReward:
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def __call__(self, signals: RewardSignals) -> RewardResult:
        cfg = self.config
        parts = {
            "progress": cfg.w_progress * float(signals.progress_delta),
            "safety": cfg.w_safe * float(np.clip(signals.safety, -1.0, 1.0)),
            "collision": -cfg.w_collision * float(bool(signals.collision)),
            "offroad": -cfg.w_offroad * float(bool(signals.offroad)),
            "jerk": -cfg.w_jerk * abs(float(signals.jerk)),
            "authority": -cfg.w_alpha * float(np.clip(signals.alpha, 0.0, 1.0)),
        }
        return RewardResult(
            total=float(sum(parts.values())),
            terminal=bool(cfg.collision_terminal and signals.collision),
            **parts,
        )

    @staticmethod
    def action_jerk(current: Sequence[float], previous: Sequence[float], dt: float = 0.05) -> float:
        now = np.asarray(current, dtype=np.float32)[:3]
        before = np.asarray(previous, dtype=np.float32)[:3]
        return float(np.linalg.norm(now - before) / max(dt, 1.0e-3))

    @staticmethod
    def safety_from_risk(risk: float) -> float:
        return float(1.0 - 2.0 * np.clip(risk, 0.0, 1.0))
