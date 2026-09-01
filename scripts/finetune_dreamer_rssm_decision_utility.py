#!/usr/bin/env python3
"""Fit the RSSM risk/progress utility without adding a runtime guard.

The recurrent dynamics, observation model, actor and SimLingo are frozen. Only
the learned progress head changes while the independently calibrated risk head
stays frozen. Clean completed overtakes teach that a useful actor proposal
should outrank a blocked SimLingo command, while opposing-traffic traces teach
that an unsafe proposal should rank below it.

Distance and TTC are used exclusively to label offline training/validation
pairs. They are never read by the runtime arbiter as a threshold or veto.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from external.simlingo.team_code.dreamer_guard import rssm_authority_confidence
from external.simlingo.team_code.dreamer_world_models import (
    RSSMConfig,
    RSSMState,
    TemporalRSSMWorldModel,
    symexp,
)
from scripts.finetune_dreamer_rssm_stationary_oncoming import (
    base_action,
    decision_replay_metrics,
    normalized_episode,
    pair_metrics as stationary_pair_metrics,
    posterior_states,
    stationary_pairs,
)
from scripts.train_dreamer_rssm_v2 import (
    ROOT,
    Episode,
    actor_from_checkpoint,
    atomic_json,
    atomic_torch_save,
    decoded_actor_output,
    discover_traces,
    evaluate_horizons,
    load_episodes,
    rssm_quality_gate,
    sha256,
    split_routes,
)


DEFAULT_CHECKPOINT = (
    ROOT / "external/simlingo/checkpoints/dreamer_ppo_rssm_v2/candidate_model.pt"
)
DEFAULT_POSITIVE_TRAIN = (
    "logs/action_dreaming_collect/action_dreaming_20260716_101125.jsonl",
)
DEFAULT_POSITIVE_VALIDATION = (
    "logs/action_dreaming_collect/action_dreaming_20260715_154252.jsonl",
)
DEFAULT_NEGATIVE_TRAIN = (
    "logs/dreamer_online_rl/"
    "webapp_20260810_161558_ppo_route_70_seed_5349/trace.jsonl",
)
DEFAULT_NEGATIVE_VALIDATION = (
    "logs/dreamer_online_rl/"
    "webapp_20260811_135402_ppo_route_70_seed_451052/trace.jsonl",
)
DEFAULT_REPLAY_PATTERNS = (
    "logs/dreamer_online_rl/webapp_*/trace.jsonl",
    "logs/dreamer_online_rl/*/traces/*.jsonl",
    "logs/dreamer_rl_campaign/*/traces/*.jsonl",
    "logs/action_dreaming_collect/*.jsonl",
)


@dataclass
class UtilityPairs:
    current_observation: torch.Tensor
    base_features: torch.Tensor
    candidate_features: torch.Tensor
    base_continuation: torch.Tensor
    candidate_continuation: torch.Tensor
    action_delta: torch.Tensor
    control_delta: torch.Tensor
    target_margin: torch.Tensor
    severity: torch.Tensor
    source: Tuple[str, ...]

    @property
    def samples(self) -> int:
        return int(self.target_margin.shape[0])

    def index(self, indices: torch.Tensor) -> "UtilityPairs":
        rows = indices.detach().cpu().numpy().tolist()
        return UtilityPairs(
            current_observation=self.current_observation[indices],
            base_features=self.base_features[indices],
            candidate_features=self.candidate_features[indices],
            base_continuation=self.base_continuation[indices],
            candidate_continuation=self.candidate_continuation[indices],
            action_delta=self.action_delta[indices],
            control_delta=self.control_delta[indices],
            target_margin=self.target_margin[indices],
            severity=self.severity[indices],
            source=tuple(self.source[int(index)] for index in rows),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--promote-to", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-replay", type=int, default=5000)
    parser.add_argument("--positive-stride", type=int, default=1)
    parser.add_argument("--negative-stride", type=int, default=3)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--positive-train-trace", action="append", default=[])
    parser.add_argument(
        "--positive-validation-trace", action="append", default=[]
    )
    parser.add_argument("--negative-train-trace", action="append", default=[])
    parser.add_argument(
        "--negative-validation-trace", action="append", default=[]
    )
    parser.add_argument("--replay-trace-pattern", action="append", default=[])
    return parser.parse_args()


def resolve_paths(patterns: Iterable[str]) -> List[Path]:
    return discover_traces(list(patterns), 0)


def actor_proposal(
    actor: torch.nn.Module,
    model: TemporalRSSMWorldModel,
    state: RSSMState,
    observation: np.ndarray,
    checkpoint: Dict[str, Any],
) -> np.ndarray:
    """Reproduce the actor-gated proposal scored by the CARLA runtime."""

    policy_mean = np.asarray(checkpoint["policy_state_mean"], dtype=np.float32)
    policy_std = np.maximum(
        np.asarray(checkpoint["policy_state_std"], dtype=np.float32), 1e-6
    )
    normalized = torch.from_numpy(
        ((observation - policy_mean) / policy_std)[None, :].astype(np.float32)
    )
    actor_input = torch.cat([normalized, model.feature(state)], dim=-1)
    target = decoded_actor_output(actor, actor_input)[0].detach().cpu().numpy()
    base = base_action(observation)
    gate = float(np.clip(target[3], 0.0, 1.0))
    steering = float(base[0]) + gate * (float(target[0]) - float(base[0]))
    base_longitudinal = float(base[1]) - float(base[2])
    target_longitudinal = float(target[1]) - float(target[2])
    longitudinal = base_longitudinal + gate * (
        target_longitudinal - base_longitudinal
    )
    return np.asarray(
        [
            np.clip(steering, -1.0, 1.0),
            max(0.0, longitudinal),
            max(0.0, -longitudinal),
            gate,
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def imagined_branch(
    model: TemporalRSSMWorldModel,
    state: RSSMState,
    immediate_action: np.ndarray,
    base: np.ndarray,
    checkpoint: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    metadata = checkpoint.get("rssm_v2") or {}
    arbitration = metadata.get("arbitration") or {}
    horizon = max(1, int(metadata.get("planning_horizon", 5)))
    commit = max(
        1,
        min(horizon, int(arbitration.get("candidate_commit_horizon", 1))),
    )
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.maximum(
        np.asarray(checkpoint["action_std"], dtype=np.float32), 1e-6
    )

    def normalized(action: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            ((action - action_mean) / action_std)[None, :].astype(np.float32)
        )

    candidate_t = normalized(immediate_action)
    base_t = normalized(base)
    imagined = state.detach()
    features: List[torch.Tensor] = []
    continuation_prefix: List[torch.Tensor] = []
    continuation = torch.ones(1, dtype=torch.float32)
    for step in range(horizon):
        action_t = candidate_t if step < commit else base_t
        imagined = model.img_step(imagined, action_t, deterministic=True)
        feature = model.feature(imagined).squeeze(0).detach()
        features.append(feature)
        continuation_prefix.append(continuation.squeeze(0).detach())
        continuation = continuation * torch.sigmoid(
            model.head_continuation(feature[None, :]).squeeze(-1)
        )
    return torch.stack(features), torch.stack(continuation_prefix)


def empty_pairs(model: TemporalRSSMWorldModel) -> UtilityPairs:
    horizon = 5
    return UtilityPairs(
        current_observation=torch.empty(0, model.observation_dim),
        base_features=torch.empty(0, horizon, model.feature_dim),
        candidate_features=torch.empty(0, horizon, model.feature_dim),
        base_continuation=torch.empty(0, horizon),
        candidate_continuation=torch.empty(0, horizon),
        action_delta=torch.empty(0),
        control_delta=torch.empty(0, 3),
        target_margin=torch.empty(0),
        severity=torch.empty(0),
        source=(),
    )


def make_pairs(
    model: TemporalRSSMWorldModel,
    actor: torch.nn.Module,
    episodes: Sequence[Episode],
    checkpoint: Dict[str, Any],
    kind: str,
    stride: int,
) -> UtilityPairs:
    metadata = checkpoint.get("rssm_v2") or {}
    arbitration = metadata.get("arbitration") or {}
    temperature = max(
        1e-6, float(arbitration.get("authority_temperature", 0.35))
    )
    base_features: List[torch.Tensor] = []
    current_observations: List[torch.Tensor] = []
    candidate_features: List[torch.Tensor] = []
    base_continuation: List[torch.Tensor] = []
    candidate_continuation: List[torch.Tensor] = []
    action_deltas: List[float] = []
    control_deltas: List[torch.Tensor] = []
    margins: List[float] = []
    severities: List[float] = []
    sources: List[str] = []
    accepted_index = 0
    for episode in episodes:
        observations, actions = normalized_episode(episode, checkpoint)
        states = posterior_states(model, observations, actions)
        for step in range(episode.transitions):
            observation = episode.observations[step]
            base = base_action(observation)
            proposal = actor_proposal(
                actor, model, states[step], observation, checkpoint
            )
            delta = proposal[:3] - base[:3]
            action_delta = float(np.abs(delta).mean())
            if action_delta <= 0.005:
                continue
            target_margin = 0.0
            severity = 0.0
            if kind == "positive":
                if (
                    episode.teacher_mask[step] <= 0.5
                    or episode.teacher_targets[step, 3] < 0.5
                ):
                    continue
                teacher = episode.teacher_targets[step]
                base_longitudinal = float(base[1] - base[2])
                teacher_longitudinal = float(teacher[1] - teacher[2])
                proposal_longitudinal = float(proposal[1] - proposal[2])
                teacher_delta = teacher[:3] - base[:3]
                denominator = float(np.dot(delta, delta))
                if denominator <= 1e-6:
                    continue
                alignment_denominator = float(
                    np.linalg.norm(delta) * np.linalg.norm(teacher_delta)
                )
                alignment = (
                    float(np.dot(delta, teacher_delta))
                    / max(1e-6, alignment_denominator)
                )
                same_side = (
                    abs(float(teacher_delta[0])) >= 0.04
                    and abs(float(delta[0])) >= 0.04
                    and np.sign(teacher_delta[0]) == np.sign(delta[0])
                )
                # A clean pass may release SimLingo's brake *or* preserve an
                # already-high throttle while adding lateral motion. Requiring
                # a throttle increase would wrongly discard the second half of
                # every successful manoeuvre.
                useful_longitudinal = (
                    float(teacher[1]) >= 0.30
                    and float(teacher[2]) <= 0.10
                    and float(proposal[2]) <= 0.35
                    and (
                        proposal_longitudinal >= base_longitudinal + 0.20
                        or (
                            base_longitudinal >= 0.35
                            and proposal_longitudinal >= base_longitudinal - 0.35
                        )
                    )
                )
                if not same_side or not useful_longitudinal or alignment < 0.45:
                    continue
                projected_authority = float(
                    np.dot(teacher_delta, delta) / denominator
                )
                desired_authority = float(
                    np.clip(projected_authority, 0.55, 0.93)
                )
                # This is the inverse of the runtime's smooth confidence map.
                # It calibrates a learned score margin, not a scene threshold.
                target_margin = float(
                    np.clip(
                        -temperature * math.log1p(-desired_authority),
                        0.25,
                        1.05,
                    )
                )
                severity = desired_authority
            elif kind == "negative":
                steer_delta = float(proposal[0] - base[0])
                side = -1 if steer_delta < -0.04 else 1 if steer_delta > 0.04 else 0
                if side == 0:
                    continue
                distance_index = 36 if side < 0 else 37
                ttc_index = 38 if side < 0 else 39
                lane_index = 40 if side < 0 else 41
                distance = float(observation[distance_index])
                ttc = float(observation[ttc_index])
                lane_available = bool(observation[lane_index] >= 0.5)
                stationary_opposing = distance < 35.0 and ttc >= 98.0
                closing_opposing = distance < 45.0 and ttc < 8.0
                if (
                    not lane_available
                    or not (stationary_opposing or closing_opposing)
                ):
                    continue
                # Evaluate a genuine counterfactual engagement even when the
                # historical actor had already learned to brake. This teaches
                # the RSSM what the unsafe alternative would produce; it does
                # not force this action or add a runtime geometric veto.
                counterfactual_longitudinal = max(
                    0.45,
                    float(base[1] - base[2]) + 0.35,
                    float(proposal[1] - proposal[2]),
                )
                proposal = np.asarray(
                    [
                        proposal[0],
                        min(1.0, counterfactual_longitudinal),
                        0.0,
                        max(0.70, float(proposal[3])),
                    ],
                    dtype=np.float32,
                )
                delta = proposal[:3] - base[:3]
                action_delta = float(np.abs(delta).mean())
                distance_severity = float(np.clip((45.0 - distance) / 45.0, 0.0, 1.0))
                ttc_severity = (
                    0.0 if ttc >= 98.0
                    else float(np.clip((8.0 - ttc) / 8.0, 0.0, 1.0))
                )
                severity = max(distance_severity, ttc_severity)
                target_margin = -(0.18 + 0.45 * severity)
            else:
                raise ValueError(f"unknown pair kind: {kind}")

            if accepted_index % max(1, int(stride)) != 0:
                accepted_index += 1
                continue
            accepted_index += 1
            base_feature, base_prefix = imagined_branch(
                model, states[step], base, base, checkpoint
            )
            candidate_feature, candidate_prefix = imagined_branch(
                model, states[step], proposal, base, checkpoint
            )
            base_features.append(base_feature)
            current_observations.append(
                observations[step].detach().clone()
            )
            candidate_features.append(candidate_feature)
            base_continuation.append(base_prefix)
            candidate_continuation.append(candidate_prefix)
            action_deltas.append(action_delta)
            control_deltas.append(
                torch.from_numpy((proposal[:3] - base[:3]).copy())
            )
            margins.append(target_margin)
            severities.append(severity)
            sources.append(f"{episode.route_id}:{episode.seed}:{step}")
    if not margins:
        return empty_pairs(model)
    return UtilityPairs(
        current_observation=torch.stack(current_observations),
        base_features=torch.stack(base_features),
        candidate_features=torch.stack(candidate_features),
        base_continuation=torch.stack(base_continuation),
        candidate_continuation=torch.stack(candidate_continuation),
        action_delta=torch.tensor(action_deltas, dtype=torch.float32),
        control_delta=torch.stack(control_deltas).float(),
        target_margin=torch.tensor(margins, dtype=torch.float32),
        severity=torch.tensor(severities, dtype=torch.float32),
        source=tuple(sources),
    )


def concatenate_pairs(parts: Sequence[UtilityPairs]) -> UtilityPairs:
    present = [part for part in parts if part.samples]
    if not present:
        raise RuntimeError("no utility pairs were produced")
    return UtilityPairs(
        current_observation=torch.cat(
            [part.current_observation for part in present]
        ),
        base_features=torch.cat([part.base_features for part in present]),
        candidate_features=torch.cat([part.candidate_features for part in present]),
        base_continuation=torch.cat(
            [part.base_continuation for part in present]
        ),
        candidate_continuation=torch.cat(
            [part.candidate_continuation for part in present]
        ),
        action_delta=torch.cat([part.action_delta for part in present]),
        control_delta=torch.cat([part.control_delta for part in present]),
        target_margin=torch.cat([part.target_margin for part in present]),
        severity=torch.cat([part.severity for part in present]),
        source=tuple(source for part in present for source in part.source),
    )


def utility_scores(
    model: TemporalRSSMWorldModel,
    pairs: UtilityPairs,
    checkpoint: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    metadata = checkpoint.get("rssm_v2") or {}
    arbitration = metadata.get("arbitration") or {}
    discount_factor = float(metadata.get("planning_discount", 0.95))
    progress_weight = float(arbitration.get("progress_weight", 1.0))
    risk_weight = float(arbitration.get("risk_weight", 2.0))
    risk_curvature = float(arbitration.get("risk_curvature", 0.0))
    action_penalty = float(arbitration.get("action_penalty", 0.2))
    horizon = pairs.base_features.shape[1]
    discounts = torch.pow(
        torch.tensor(discount_factor, dtype=torch.float32),
        torch.arange(horizon, dtype=torch.float32),
    ).reshape(1, -1)

    def branch(
        features: torch.Tensor,
        continuation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        risk = torch.sigmoid(model.head_risk(flat)).reshape(shape[0], shape[1])
        progress_symlog = model.head_progress(flat).reshape(shape[0], shape[1])
        progress = symexp(torch.clamp(progress_symlog, -5.0, 5.0))
        future_risk = (risk * continuation).amax(dim=1)
        future_progress = (progress * continuation * discounts).sum(dim=1)
        return future_risk, future_progress

    base_risk, base_progress = branch(
        pairs.base_features, pairs.base_continuation
    )
    candidate_risk, candidate_progress = branch(
        pairs.candidate_features, pairs.candidate_continuation
    )
    base_risk_cost = base_risk + risk_curvature * base_risk.square()
    candidate_risk_cost = (
        candidate_risk + risk_curvature * candidate_risk.square()
    )
    base_score = progress_weight * base_progress - risk_weight * base_risk_cost
    candidate_score = (
        progress_weight * candidate_progress
        - risk_weight * candidate_risk_cost
        - action_penalty * pairs.action_delta
    )
    return {
        "base_risk": base_risk,
        "candidate_risk": candidate_risk,
        "base_progress": base_progress,
        "candidate_progress": candidate_progress,
        "base_score": base_score,
        "candidate_score": candidate_score,
        "margin": candidate_score - base_score,
    }


@torch.no_grad()
def utility_metrics(
    model: TemporalRSSMWorldModel,
    pairs: UtilityPairs,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    if not pairs.samples:
        return {"samples": 0, "accuracy": 0.0}
    values = utility_scores(model, pairs, checkpoint)
    margin = values["margin"]
    positive = pairs.target_margin > 0.0
    correct = torch.where(positive, margin > 0.0, margin < 0.0)
    temperature = float(
        ((checkpoint.get("rssm_v2") or {}).get("arbitration") or {}).get(
            "authority_temperature", 0.35
        )
    )
    confidences = [
        rssm_authority_confidence(float(value), temperature)
        for value in margin[positive].tolist()
    ]
    margin_error = (margin - pairs.target_margin).abs()
    return {
        "samples": pairs.samples,
        "positive_samples": int(positive.sum()),
        "negative_samples": int((~positive).sum()),
        "accuracy": float(correct.float().mean()),
        "target_margin_mae": float(margin_error.mean()),
        "mean_margin": float(margin.mean()),
        "mean_positive_margin": (
            float(margin[positive].mean()) if bool(positive.any()) else 0.0
        ),
        "mean_negative_margin": (
            float(margin[~positive].mean()) if bool((~positive).any()) else 0.0
        ),
        "positive_accuracy": (
            float((margin[positive] > 0.0).float().mean())
            if bool(positive.any()) else 0.0
        ),
        "negative_accuracy": (
            float((margin[~positive] < 0.0).float().mean())
            if bool((~positive).any()) else 0.0
        ),
        "mean_positive_authority_confidence": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        "p10_positive_authority_confidence": (
            float(np.percentile(confidences, 10)) if confidences else 0.0
        ),
        "mean_risk_delta": float(
            (values["candidate_risk"] - values["base_risk"]).mean()
        ),
        "mean_progress_delta": float(
            (values["candidate_progress"] - values["base_progress"]).mean()
        ),
    }


@torch.no_grad()
def replay_targets(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[Episode],
    checkpoint: Dict[str, Any],
    maximum: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: List[torch.Tensor] = []
    for episode in episodes:
        observations, actions = normalized_episode(episode, checkpoint)
        states = posterior_states(model, observations, actions)
        for step in range(episode.transitions):
            action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
            action_std = np.maximum(
                np.asarray(checkpoint["action_std"], dtype=np.float32), 1e-6
            )
            action = episode.actions[step]
            action_t = torch.from_numpy(
                ((action - action_mean) / action_std)[None, :].astype(np.float32)
            )
            prior = model.img_step(states[step], action_t, deterministic=True)
            features.append(model.feature(prior).squeeze(0).detach())
    if len(features) > maximum > 0:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(features), maximum, replace=False))
        features = [features[int(index)] for index in selected]
    stacked = torch.stack(features)
    risk = torch.sigmoid(model.head_risk(stacked).squeeze(-1)).detach()
    progress = model.head_progress(stacked).squeeze(-1).detach()
    return stacked, risk, progress


def blended_head_state(
    parent: Dict[str, torch.Tensor],
    trained: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return {
        key: parent[key] + alpha * (trained[key] - parent[key])
        for key in parent
    }


def head_state(model: TemporalRSSMWorldModel) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        "risk": copy.deepcopy(model.head_risk.state_dict()),
        "progress": copy.deepcopy(model.head_progress.state_dict()),
    }


def load_head_state(
    model: TemporalRSSMWorldModel,
    state: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    model.head_risk.load_state_dict(state["risk"])
    model.head_progress.load_state_dict(state["progress"])


def blended_heads(
    parent: Dict[str, Dict[str, torch.Tensor]],
    trained: Dict[str, Dict[str, torch.Tensor]],
    alpha: float,
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        name: blended_head_state(parent[name], trained[name], alpha)
        for name in ("risk", "progress")
    }


def pair_gate(
    parent_positive: Dict[str, Any],
    candidate_positive: Dict[str, Any],
    parent_negative: Dict[str, Any],
    candidate_negative: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    positive_accuracy_floor = max(
        0.70, float(parent_positive["positive_accuracy"]) + 0.05
    )
    negative_accuracy_floor = max(
        0.90, float(parent_negative["negative_accuracy"]) - 0.02
    )
    positive_margin_gain = (
        float(candidate_positive["mean_positive_margin"])
        - float(parent_positive["mean_positive_margin"])
    )
    passed = bool(
        candidate_positive["positive_accuracy"] >= positive_accuracy_floor
        and positive_margin_gain >= 0.10
        and candidate_positive["mean_positive_authority_confidence"] >= 0.30
        and candidate_negative["negative_accuracy"] >= negative_accuracy_floor
        and candidate_negative["mean_negative_margin"] <= -0.05
    )
    return passed, {
        "positive_accuracy_floor": positive_accuracy_floor,
        "negative_accuracy_floor": negative_accuracy_floor,
        "positive_margin_gain_min": 0.10,
        "positive_authority_confidence_min": 0.30,
        "negative_mean_margin_max": -0.05,
        "positive_margin_gain": positive_margin_gain,
    }


def targeted_head_quality_gate(
    parent: Dict[str, Any],
    candidate: Dict[str, Any],
    parent_forced: Dict[str, Any],
    candidate_forced: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Compare only outputs that the targeted fit is allowed to change."""

    checks: Dict[str, Dict[str, float | bool]] = {}
    passed = True
    for horizon in (1, 5, 15):
        key = str(horizon)
        parent_row = parent.get(key) or {}
        candidate_row = candidate.get(key) or {}
        risk_limit = max(
            0.18 if horizon <= 5 else 0.22,
            float(parent_row.get("risk_mae", math.inf)) + 0.01,
        )
        progress_limit = float(parent_row.get("progress_mae_m", math.inf)) + 0.035
        risk_value = float(candidate_row.get("risk_mae", math.inf))
        progress_value = float(candidate_row.get("progress_mae_m", math.inf))
        row_passed = bool(
            math.isfinite(risk_value)
            and math.isfinite(progress_value)
            and risk_value <= risk_limit
            and progress_value <= progress_limit
        )
        passed = passed and row_passed
        checks[key] = {
            "risk_mae": risk_value,
            "risk_limit": risk_limit,
            "progress_mae_m": progress_value,
            "progress_limit": progress_limit,
            "passed": row_passed,
        }
    forced_parent = parent_forced.get("5") or {}
    forced_candidate = candidate_forced.get("5") or {}
    forced_limit = max(
        0.22,
        float(forced_parent.get("risk_mae", math.inf)) + 0.01,
    )
    forced_value = float(forced_candidate.get("risk_mae", math.inf))
    forced_passed = bool(
        math.isfinite(forced_value) and forced_value <= forced_limit
    )
    passed = passed and forced_passed
    return bool(passed), {
        "checks": checks,
        "forced_horizon_5_risk_mae": forced_value,
        "forced_horizon_5_risk_limit": forced_limit,
        "forced_passed": forced_passed,
        "note": (
            "The encoder, RSSM dynamics, observation and event heads are "
            "frozen. Their absolute gate is reported separately; promotion "
            "compares only risk/progress outputs changed by this fit."
        ),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else checkpoint_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RSSMConfig.from_dict(checkpoint.get("world_model_config"))
    model = TemporalRSSMWorldModel(config)
    model.load_state_dict(checkpoint["world_model"])
    model.eval()
    actor = actor_from_checkpoint(checkpoint, torch.device("cpu"))
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    path_groups = {
        "positive_train": resolve_paths(
            args.positive_train_trace or DEFAULT_POSITIVE_TRAIN
        ),
        "positive_validation": resolve_paths(
            args.positive_validation_trace or DEFAULT_POSITIVE_VALIDATION
        ),
        "negative_train": resolve_paths(
            args.negative_train_trace or DEFAULT_NEGATIVE_TRAIN
        ),
        "negative_validation": resolve_paths(
            args.negative_validation_trace or DEFAULT_NEGATIVE_VALIDATION
        ),
    }
    if any(not paths for paths in path_groups.values()):
        missing = [name for name, paths in path_groups.items() if not paths]
        raise RuntimeError("missing trace group(s): " + ", ".join(missing))
    episode_groups: Dict[str, List[Episode]] = {}
    audits: Dict[str, Any] = {}
    for name, paths in path_groups.items():
        episodes, audit = load_episodes(paths, args.sequence_length)
        if not episodes:
            raise RuntimeError(f"trace group {name} produced no episodes")
        episode_groups[name] = episodes
        audits[name] = audit

    positive_train = make_pairs(
        model,
        actor,
        episode_groups["positive_train"],
        checkpoint,
        "positive",
        args.positive_stride,
    )
    positive_validation = make_pairs(
        model,
        actor,
        episode_groups["positive_validation"],
        checkpoint,
        "positive",
        args.positive_stride,
    )
    negative_train = make_pairs(
        model,
        actor,
        episode_groups["negative_train"],
        checkpoint,
        "negative",
        args.negative_stride,
    )
    negative_validation = make_pairs(
        model,
        actor,
        episode_groups["negative_validation"],
        checkpoint,
        "negative",
        args.negative_stride,
    )
    if min(
        positive_train.samples,
        positive_validation.samples,
        negative_train.samples,
        negative_validation.samples,
    ) < 20:
        raise RuntimeError("each independent utility pair split needs >=20 samples")
    training_pairs = concatenate_pairs([positive_train, negative_train])
    stationary_train = stationary_pairs(
        model, episode_groups["negative_train"], checkpoint
    )
    stationary_validation = stationary_pairs(
        model, episode_groups["negative_validation"], checkpoint
    )
    if stationary_train[0].shape[0] < 100 or stationary_validation[0].shape[0] < 100:
        raise RuntimeError("stationary opposing-traffic calibration needs >=100 pairs")

    validation_paths = {
        path.resolve()
        for name in ("positive_validation", "negative_validation")
        for path in path_groups[name]
    }
    replay_paths = [
        path for path in resolve_paths(
            args.replay_trace_pattern or DEFAULT_REPLAY_PATTERNS
        )
        if path.resolve() not in validation_paths
    ]
    replay_episodes, replay_audit = load_episodes(
        replay_paths, args.sequence_length
    )
    if not replay_episodes:
        raise RuntimeError("generic replay pool is empty")
    replay_x, _, replay_parent_progress = replay_targets(
        model, replay_episodes, checkpoint, args.max_replay, args.seed
    )

    parent_heads = head_state(model)
    before_positive = utility_metrics(
        model, positive_validation, checkpoint
    )
    before_negative = utility_metrics(
        model, negative_validation, checkpoint
    )
    before_train = utility_metrics(model, training_pairs, checkpoint)
    risk_curvature = float(
        ((checkpoint.get("rssm_v2") or {}).get("arbitration") or {}).get(
            "risk_curvature", 0.0
        )
    )
    parent_stationary = stationary_pair_metrics(
        model, stationary_validation, risk_curvature
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # The stationary-opposing calibration already passed its independent gate.
    # Keep that learned risk representation immutable and adjust only how much
    # useful progress an imagined action is expected to produce.
    for parameter in model.head_progress.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.head_progress.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history: List[Dict[str, Any]] = []
    positive_count = int((training_pairs.target_margin > 0.0).sum())
    negative_count = training_pairs.samples - positive_count
    for epoch in range(max(1, args.epochs)):
        permutation = torch.randperm(training_pairs.samples, generator=generator)
        losses: List[float] = []
        for start in range(0, training_pairs.samples, max(1, args.batch_size)):
            indices = permutation[start:start + max(1, args.batch_size)]
            batch = training_pairs.index(indices)
            values = utility_scores(model, batch, checkpoint)
            margin = values["margin"]
            positive = batch.target_margin > 0.0
            class_weights = torch.where(
                positive,
                torch.full_like(margin, training_pairs.samples / max(1, 2 * positive_count)),
                torch.full_like(margin, training_pairs.samples / max(1, 2 * negative_count)),
            )
            regression = F.smooth_l1_loss(
                margin,
                batch.target_margin,
                reduction="none",
                beta=0.10,
            )
            signed = torch.where(positive, margin, -margin)
            ranking = F.softplus(-signed / 0.10)
            progress_delta = (
                values["candidate_progress"] - values["base_progress"]
            )
            useful_progress = torch.where(
                positive,
                F.relu(0.08 - progress_delta),
                torch.zeros_like(progress_delta),
            )

            replay_size = min(max(32, batch.target_margin.shape[0]), replay_x.shape[0])
            replay_indices = torch.randint(
                replay_x.shape[0], (replay_size,), generator=generator
            )
            replay_batch = replay_x[replay_indices]
            replay_progress = model.head_progress(replay_batch).squeeze(-1)
            progress_distillation = F.smooth_l1_loss(
                replay_progress,
                replay_parent_progress[replay_indices],
                beta=0.05,
            )
            pair_loss = (
                class_weights
                * (
                    2.5 * regression
                    + 0.35 * ranking
                    + 0.55 * useful_progress
                )
            ).mean()
            total = (
                pair_loss
                + 4.0 * progress_distillation
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head_progress.parameters(),
                5.0,
            )
            optimizer.step()
            losses.append(float(total.detach()))
        model.eval()
        positive_metrics = utility_metrics(
            model, positive_validation, checkpoint
        )
        negative_metrics = utility_metrics(
            model, negative_validation, checkpoint
        )
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "positive": positive_metrics,
            "negative": negative_metrics,
        }
        history.append(row)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[rssm-utility] epoch={epoch + 1}/{args.epochs} "
                f"loss={row['loss']:.4f} "
                f"positive_acc={positive_metrics['positive_accuracy']:.3f} "
                f"positive_margin={positive_metrics['mean_positive_margin']:+.3f} "
                f"negative_acc={negative_metrics['negative_accuracy']:.3f} "
                f"negative_margin={negative_metrics['mean_negative_margin']:+.3f}",
                flush=True,
            )
    trained_heads = head_state(model)

    blend_step = float(np.clip(args.blend_step, 0.01, 1.0))
    blend_audit: List[Dict[str, Any]] = []
    selected_alpha: Optional[float] = None
    selected_pair_gate: Dict[str, Any] = {}
    for raw_alpha in np.arange(blend_step, 1.0 + 0.5 * blend_step, blend_step):
        alpha = float(min(1.0, raw_alpha))
        load_head_state(model, blended_heads(parent_heads, trained_heads, alpha))
        positive_metrics = utility_metrics(
            model, positive_validation, checkpoint
        )
        negative_metrics = utility_metrics(
            model, negative_validation, checkpoint
        )
        passed, gate = pair_gate(
            before_positive,
            positive_metrics,
            before_negative,
            negative_metrics,
        )
        stationary_metrics = stationary_pair_metrics(
            model, stationary_validation, risk_curvature
        )
        stationary_passed = bool(
            stationary_metrics["mean_risk_cost_margin"]
            >= parent_stationary["mean_risk_cost_margin"] - 0.002
            and stationary_metrics["positive_fraction"]
            >= parent_stationary["positive_fraction"] - 0.005
        )
        passed = bool(passed and stationary_passed)
        blend_audit.append(
            {
                "alpha": alpha,
                "passed": passed,
                "positive": positive_metrics,
                "negative": negative_metrics,
                "stationary": stationary_metrics,
                "stationary_passed": stationary_passed,
                "gate": gate,
            }
        )
        # Use the smallest sufficient update. Larger interpolation factors may
        # improve the training objective while needlessly moving calibration.
        if passed and selected_alpha is None:
            selected_alpha = alpha
            selected_pair_gate = gate
    if selected_alpha is None:
        load_head_state(model, parent_heads)
    else:
        load_head_state(
            model, blended_heads(parent_heads, trained_heads, selected_alpha)
        )
    model.eval()
    after_positive = utility_metrics(model, positive_validation, checkpoint)
    after_negative = utility_metrics(model, negative_validation, checkpoint)

    candidate_stationary = stationary_pair_metrics(
        model, stationary_validation, risk_curvature
    )
    stationary_passed = bool(
        candidate_stationary["mean_risk_cost_margin"]
        >= parent_stationary["mean_risk_cost_margin"] - 0.002
        and candidate_stationary["positive_fraction"]
        >= parent_stationary["positive_fraction"] - 0.005
    )

    all_paths = resolve_paths(args.replay_trace_pattern or DEFAULT_REPLAY_PATTERNS)
    forced_path_set = {
        path.resolve()
        for paths in path_groups.values()
        for path in paths
    }
    pool_paths = [path for path in all_paths if path.resolve() not in forced_path_set]
    pool_episodes, _ = load_episodes(pool_paths, args.sequence_length)
    regular_train, regular_validation, regular_routes = split_routes(
        pool_episodes, args.seed
    )
    del regular_train
    forced_episodes = (
        episode_groups["positive_validation"]
        + episode_groups["negative_validation"]
    )
    observation_mean = np.asarray(
        checkpoint["world_observation_mean"], dtype=np.float32
    )
    observation_std = np.asarray(
        checkpoint["world_observation_std"], dtype=np.float32
    )
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    # Evaluate parent and candidate on exactly the same episodes. The generic
    # absolute quality gate is retained for visibility, but a head-only update
    # cannot repair or degrade frozen observation dynamics.
    selected_heads = head_state(model)
    load_head_state(model, parent_heads)
    parent_forced_validation = evaluate_horizons(
        model,
        forced_episodes,
        observation_mean,
        observation_std,
        action_mean,
        action_std,
        torch.device("cpu"),
    )
    parent_combined_validation = evaluate_horizons(
        model,
        list(regular_validation) + forced_episodes,
        observation_mean,
        observation_std,
        action_mean,
        action_std,
        torch.device("cpu"),
    )
    load_head_state(model, selected_heads)
    forced_validation = evaluate_horizons(
        model,
        forced_episodes,
        observation_mean,
        observation_std,
        action_mean,
        action_std,
        torch.device("cpu"),
    )
    combined_validation = evaluate_horizons(
        model,
        list(regular_validation) + forced_episodes,
        observation_mean,
        observation_std,
        action_mean,
        action_std,
        torch.device("cpu"),
    )
    absolute_quality_passed, absolute_quality_gate = rssm_quality_gate(
        combined_validation, forced_validation
    )
    quality_passed, quality_gate = targeted_head_quality_gate(
        parent_combined_validation,
        combined_validation,
        parent_forced_validation,
        forced_validation,
    )
    positive_replay = decision_replay_metrics(
        model, episode_groups["positive_validation"], checkpoint
    )
    negative_replay = decision_replay_metrics(
        model, episode_groups["negative_validation"], checkpoint
    )
    pair_passed, pair_gate_details = pair_gate(
        before_positive,
        after_positive,
        before_negative,
        after_negative,
    )
    passed = bool(
        selected_alpha is not None
        and pair_passed
        and stationary_passed
        and quality_passed
        and int(negative_replay["risk_increase_selected"]) == 0
    )

    candidate = copy.deepcopy(checkpoint)
    candidate_world_model = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    changed_keys = sorted(
        key
        for key, value in candidate_world_model.items()
        if key not in checkpoint["world_model"]
        or not torch.equal(value, checkpoint["world_model"][key])
    )
    unexpected = [
        key for key in changed_keys
        if not key.startswith("head_progress.")
    ]
    if unexpected:
        raise RuntimeError(
            "decision utility training changed frozen parameters: "
            + ", ".join(unexpected)
        )
    candidate["world_model"] = candidate_world_model
    metadata = copy.deepcopy(candidate.get("rssm_v2") or {})
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata.update(
        {
            "decision_utility_finetuned_at": created_at,
            "decision_utility_parent_sha256": sha256(checkpoint_path),
            "decision_utility_heads_only": True,
            "decision_utility_progress_head_only": True,
            "decision_utility_risk_head_unchanged": True,
            "decision_utility_changed_world_model_keys": changed_keys,
            "decision_utility_selected_head_blend": selected_alpha,
            "decision_utility_pair_gate_passed": pair_passed,
            "decision_utility_stationary_gate_passed": stationary_passed,
            "decision_utility_quality_gate_passed": quality_passed,
            "runtime_guard": False,
            "hard_safety_thresholds": False,
            "complementary_to_simlingo": True,
        }
    )
    candidate["rssm_v2"] = metadata
    attempt_path = output_dir / "decision_utility_last_attempt.pt"
    atomic_torch_save(attempt_path, candidate)
    report: Dict[str, Any] = {
        "status": "validated" if passed else "quality_gate_rejected",
        "created_at": created_at,
        "parent_checkpoint": str(checkpoint_path),
        "parent_sha256": sha256(checkpoint_path),
        "paths": {
            name: [str(path) for path in paths]
            for name, paths in path_groups.items()
        },
        "audits": audits,
        "pair_counts": {
            "positive_train": positive_train.samples,
            "positive_validation": positive_validation.samples,
            "negative_train": negative_train.samples,
            "negative_validation": negative_validation.samples,
        },
        "replay_samples": int(replay_x.shape[0]),
        "before_train": before_train,
        "before_positive_validation": before_positive,
        "after_positive_validation": after_positive,
        "before_negative_validation": before_negative,
        "after_negative_validation": after_negative,
        "selected_head_blend": selected_alpha,
        "pair_gate_passed": pair_passed,
        "pair_gate": pair_gate_details or selected_pair_gate,
        "stationary_gate_passed": stationary_passed,
        "parent_stationary": parent_stationary,
        "candidate_stationary": candidate_stationary,
        "quality_gate_passed": quality_passed,
        "quality_gate": quality_gate,
        "absolute_quality_gate_passed": absolute_quality_passed,
        "absolute_quality_gate": absolute_quality_gate,
        "parent_combined_validation": parent_combined_validation,
        "parent_forced_validation": parent_forced_validation,
        "combined_validation": combined_validation,
        "forced_validation": forced_validation,
        "positive_replay": positive_replay,
        "negative_replay": negative_replay,
        "changed_world_model_keys": changed_keys,
        "actor_checkpoint_unchanged": True,
        "risk_head_unchanged": True,
        "recurrent_world_model_unchanged": True,
        "simlingo_unchanged": True,
        "runtime_guard": False,
        "hard_safety_thresholds": False,
        "regular_validation_routes": regular_routes,
        "blend_audit": blend_audit,
        "history": history,
        "attempt": str(attempt_path),
        "promoted": False,
    }
    if args.promote and passed:
        promote_to = args.promote_to.expanduser().resolve()
        backup = output_dir / (
            "candidate_model_before_decision_utility_"
            + time.strftime("%Y%m%d_%H%M%S")
            + ".pt"
        )
        if promote_to.exists():
            shutil.copy2(promote_to, backup)
        atomic_torch_save(promote_to, candidate)
        report.update(
            {
                "promoted": True,
                "promoted_to": str(promote_to),
                "promoted_sha256": sha256(promote_to),
                "backup": str(backup),
                "backup_sha256": sha256(backup),
            }
        )
    report_path = output_dir / "decision_utility_finetune_report.json"
    atomic_json(report_path, report)
    print(
        f"[rssm-utility] gate={'PASS' if passed else 'REJECT'} "
        f"pairs={int(pair_passed)} stationary={int(stationary_passed)} "
        f"multi_horizon={int(quality_passed)} alpha={selected_alpha} "
        f"promoted={int(report['promoted'])}",
        flush=True,
    )
    print(f"[rssm-utility] report={report_path}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
