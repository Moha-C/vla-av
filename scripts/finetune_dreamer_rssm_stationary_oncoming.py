#!/usr/bin/env python3
"""Calibrate RSSM risk for stationary opposing traffic without a hard guard.

Only the learned risk head is updated. The recurrent dynamics, observation
model, PPO actor and SimLingo command remain frozen. Promotion requires the
existing multi-horizon quality gate plus an independent directional test: an
overtake action toward a nearby opposite-facing stopped vehicle must predict
more risk than keeping SimLingo's command. The parent risk distribution is
distilled during the update and the final head is selected by weight-space
interpolation, so a local correction cannot silently erase useful overtakes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from external.simlingo.team_code.dreamer_world_models import (
    RSSMConfig,
    RSSMState,
    TemporalRSSMWorldModel,
    symexp,
)
from external.simlingo.team_code.dreamer_guard import rssm_authority_confidence
from scripts.train_dreamer_rssm_v2 import (
    ROOT,
    Episode,
    atomic_json,
    atomic_torch_save,
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
DEFAULT_PARENT_CHECKPOINT = (
    ROOT
    / "external/simlingo/checkpoints/dreamer_ppo_rssm_v2/"
    "candidate_model_before_stationary_risk_head_20260812.pt"
)
DEFAULT_TRAIN_TRACE = (
    "logs/dreamer_online_rl/"
    "webapp_20260810_161558_ppo_route_70_seed_5349/trace.jsonl"
)
DEFAULT_VALIDATION_TRACES = (
    "logs/dreamer_online_rl/"
    "webapp_20260811_135402_ppo_route_70_seed_451052/trace.jsonl",
    "logs/dreamer_online_rl/"
    "webapp_20260807_153143_ppo_route_148_seed_208080/trace.jsonl",
)
DEFAULT_TRAIN_PRESERVATION_TRACE = (
    "logs/dreamer_online_rl/"
    "webapp_20260731_172937_ppo_route_148_seed_265911/trace.jsonl"
)
DEFAULT_PATTERNS = (
    "logs/dreamer_online_rl/webapp_*/trace.jsonl",
    "logs/dreamer_online_rl/*/traces/*.jsonl",
    "logs/dreamer_rl_campaign/*/traces/*.jsonl",
    "logs/action_dreaming_collect/*.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument(
        "--promote-to",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Active checkpoint path; kept separate from the immutable parent.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    # Keep exactly the route split used to validate the parent checkpoint so a
    # head-only calibration is compared against the same held-out population.
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-replay", type=int, default=8000)
    parser.add_argument("--blocked-route-id", default="70")
    parser.add_argument("--preservation-route-id", default="148")
    parser.add_argument("--minimum-preservation", type=float, default=0.70)
    parser.add_argument(
        "--maximum-blocked-selection-ratio", type=float, default=0.50
    )
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--trace-pattern", action="append", default=[])
    parser.add_argument(
        "--train-stationary-trace",
        action="append",
        default=[],
        help="Trace containing stationary opposing traffic; may be repeated.",
    )
    parser.add_argument(
        "--validation-trace",
        action="append",
        default=[],
        help="Independent forced-validation trace; may be repeated.",
    )
    parser.add_argument(
        "--train-preservation-trace",
        action="append",
        default=[],
        help="Clean successful trace used to preserve useful interventions.",
    )
    return parser.parse_args()


def normalized_episode(
    episode: Episode,
    checkpoint: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    observation_mean = np.asarray(
        checkpoint["world_observation_mean"], dtype=np.float32
    )
    observation_std = np.asarray(
        checkpoint["world_observation_std"], dtype=np.float32
    )
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    observations = (
        episode.observations - observation_mean
    ) / np.maximum(observation_std, 1e-6)
    actions = (
        episode.actions - action_mean
    ) / np.maximum(action_std, 1e-6)
    return (
        torch.from_numpy(observations.astype(np.float32)),
        torch.from_numpy(actions.astype(np.float32)),
    )


@torch.no_grad()
def posterior_states(
    model: TemporalRSSMWorldModel,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> List[RSSMState]:
    state = model.observe_initial(observations[0:1], deterministic=True)
    states = [state.detach()]
    for step in range(actions.shape[0]):
        state, _ = model.obs_step(
            state,
            actions[step:step + 1],
            observations[step + 1:step + 2],
            deterministic=True,
        )
        states.append(state.detach())
    return states


def side_from_stationary_observation(observation: np.ndarray) -> Tuple[int, float]:
    candidates = []
    for side, distance_index, ttc_index, lane_index in (
        (-1, 36, 38, 40),
        (1, 37, 39, 41),
    ):
        distance = float(observation[distance_index])
        ttc = float(observation[ttc_index])
        lane_available = float(observation[lane_index]) >= 0.5
        if lane_available and distance < 35.0 and ttc >= 98.0:
            candidates.append((distance, side))
    if not candidates:
        return 0, 80.0
    distance, side = min(candidates)
    return int(side), float(distance)


def base_action(observation: np.ndarray) -> np.ndarray:
    return np.asarray(
        [observation[28], observation[29], observation[30], 0.0],
        dtype=np.float32,
    )


def overtake_action(base: np.ndarray, side: int, steer_delta: float) -> np.ndarray:
    return np.asarray([
        float(np.clip(base[0] + side * steer_delta, -1.0, 1.0)),
        float(max(0.32, base[1])),
        0.0,
        1.0,
    ], dtype=np.float32)


def stationary_risk_margin(distance: float) -> float:
    """Small learned ranking margin, increasing continuously with proximity."""

    severity = float(np.clip((35.0 - float(distance)) / 35.0, 0.0, 1.0))
    return float(0.006 + 0.010 * severity)


@torch.no_grad()
def imagined_feature(
    model: TemporalRSSMWorldModel,
    state: RSSMState,
    action: np.ndarray,
    checkpoint: Dict[str, Any],
) -> torch.Tensor:
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.maximum(
        np.asarray(checkpoint["action_std"], dtype=np.float32), 1e-6
    )
    action_t = torch.from_numpy(((action - action_mean) / action_std)[None, :])
    prior = model.img_step(state, action_t, deterministic=True)
    return model.feature(prior).squeeze(0).detach()


def stationary_pairs(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[Episode],
    checkpoint: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_features: List[torch.Tensor] = []
    candidate_features: List[torch.Tensor] = []
    parent_base_probabilities: List[float] = []
    parent_candidate_probabilities: List[float] = []
    margins: List[float] = []
    for episode in episodes:
        observations, actions = normalized_episode(episode, checkpoint)
        states = posterior_states(model, observations, actions)
        for step in range(episode.transitions):
            observation = episode.observations[step]
            side, distance = side_from_stationary_observation(observation)
            if side == 0:
                continue
            base = base_action(observation)
            base_feature = imagined_feature(model, states[step], base, checkpoint)
            for delta in (0.20, 0.35, 0.50):
                candidate_feature = imagined_feature(
                    model,
                    states[step],
                    overtake_action(base, side, delta),
                    checkpoint,
                )
                base_features.append(base_feature)
                candidate_features.append(candidate_feature)
                parent_base_probabilities.append(float(torch.sigmoid(
                    model.head_risk(base_feature.unsqueeze(0)).squeeze()
                )))
                parent_candidate_probabilities.append(float(torch.sigmoid(
                    model.head_risk(candidate_feature.unsqueeze(0)).squeeze()
                )))
                margins.append(stationary_risk_margin(distance))
    if not base_features:
        empty = torch.empty(0, model.feature_dim)
        scalar = torch.empty(0)
        return empty, empty, scalar, scalar, scalar
    return (
        torch.stack(base_features),
        torch.stack(candidate_features),
        torch.tensor(parent_base_probabilities, dtype=torch.float32),
        torch.tensor(parent_candidate_probabilities, dtype=torch.float32),
        torch.tensor(margins, dtype=torch.float32),
    )


@torch.no_grad()
def replay_features(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[Episode],
    checkpoint: Dict[str, Any],
    maximum: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features: List[torch.Tensor] = []
    parent_probabilities: List[float] = []
    for episode in episodes:
        observations, actions = normalized_episode(episode, checkpoint)
        states = posterior_states(model, observations, actions)
        for step in range(episode.transitions):
            feature = imagined_feature(
                model, states[step], episode.actions[step], checkpoint
            )
            features.append(feature)
            parent_probabilities.append(float(torch.sigmoid(
                model.head_risk(feature.unsqueeze(0)).squeeze()
            )))
    if len(features) > maximum > 0:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(features), maximum, replace=False))
        features = [features[int(index)] for index in indices]
        parent_probabilities = [
            parent_probabilities[int(index)] for index in indices
        ]
    return (
        torch.stack(features),
        torch.tensor(parent_probabilities, dtype=torch.float32),
    )


@torch.no_grad()
def pair_metrics(
    model: TemporalRSSMWorldModel,
    pairs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    risk_curvature: float,
) -> Dict[str, float]:
    base_features, candidate_features, _, _, requested_margin = pairs
    if base_features.shape[0] == 0:
        return {
            "samples": 0,
            "mean_margin": -math.inf,
            "mean_risk_cost_margin": -math.inf,
            "positive_fraction": 0.0,
            "requested_margin_fraction": 0.0,
            "base_risk": math.inf,
            "candidate_risk": math.inf,
        }
    base = torch.sigmoid(model.head_risk(base_features).squeeze(-1))
    candidate = torch.sigmoid(model.head_risk(candidate_features).squeeze(-1))
    margin = candidate - base
    base_cost = base + float(risk_curvature) * base.square()
    candidate_cost = candidate + float(risk_curvature) * candidate.square()
    return {
        "samples": int(base.shape[0]),
        "mean_margin": float(margin.mean()),
        "mean_risk_cost_margin": float((candidate_cost - base_cost).mean()),
        "positive_fraction": float((margin > 0.0).float().mean()),
        "requested_margin_fraction": float(
            (margin >= requested_margin).float().mean()
        ),
        "base_risk": float(base.mean()),
        "candidate_risk": float(candidate.mean()),
    }


def blended_risk_head_state(
    parent: Dict[str, torch.Tensor],
    calibrated: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return {
        key: parent[key] + alpha * (calibrated[key] - parent[key])
        for key in parent
    }


@torch.no_grad()
def decision_replay_metrics(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[Episode],
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """Replay base/candidate choices with the exact continuous RSSM utility."""

    metadata = checkpoint.get("rssm_v2") or {}
    arbitration = metadata.get("arbitration") or {}
    planning_horizon = max(1, int(metadata.get("planning_horizon", 5)))
    commit_horizon = max(
        1,
        min(
            planning_horizon,
            int(arbitration.get("candidate_commit_horizon", 1)),
        ),
    )
    discount_factor = float(metadata.get("planning_discount", 0.95))
    progress_weight = float(arbitration.get("progress_weight", 1.0))
    risk_weight = float(arbitration.get("risk_weight", 2.0))
    risk_curvature = float(arbitration.get("risk_curvature", 0.0))
    action_penalty = float(arbitration.get("action_penalty", 0.2))
    calibration_basis = arbitration.get("calibration_basis") or {}
    authority_temperature = float(arbitration.get(
        "authority_temperature",
        float(calibration_basis.get("progress_mae_m", 0.15))
        + risk_weight * float(calibration_basis.get("risk_mae", 0.18)),
    ))
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.maximum(
        np.asarray(checkpoint["action_std"], dtype=np.float32), 1e-6
    )
    material = 0
    selected = 0
    risk_increase_selected = 0
    score_margins: List[float] = []
    authority_confidences: List[float] = []
    effective_authorities: List[float] = []
    effective_control_deltas: List[float] = []
    for episode in episodes:
        observations, actions = normalized_episode(episode, checkpoint)
        states = posterior_states(model, observations, actions)
        state_indices: List[int] = []
        action_rows: List[np.ndarray] = []
        action_deltas: List[np.ndarray] = []
        for step in range(episode.transitions):
            rows = np.stack((
                base_action(episode.observations[step]),
                np.asarray(episode.actions[step], dtype=np.float32),
            )).astype(np.float32)
            action_delta = np.abs(
                rows[:, :3] - rows[0:1, :3]
            ).mean(axis=1)
            if float(action_delta[1]) <= 0.005:
                continue
            state_indices.append(step)
            action_rows.append(rows)
            action_deltas.append(action_delta)
        if not state_indices:
            continue
        batch_size = len(state_indices)
        material += batch_size
        rows_np = np.stack(action_rows)
        delta_np = np.stack(action_deltas)
        normalized_actions = torch.from_numpy(
            ((rows_np - action_mean) / action_std).astype(np.float32)
        ).reshape(batch_size * 2, -1)
        deter = torch.cat([states[index].deter for index in state_indices])
        stoch = torch.cat([states[index].stoch for index in state_indices])
        logits = torch.cat([states[index].logits for index in state_indices])
        imagined = RSSMState(
            deter=deter.repeat_interleave(2, dim=0),
            stoch=stoch.repeat_interleave(2, dim=0),
            logits=logits.repeat_interleave(2, dim=0),
        )
        base_actions = normalized_actions.reshape(batch_size, 2, -1)[:, 0:1]
        base_actions = base_actions.repeat(1, 2, 1).reshape(batch_size * 2, -1)
        continuation = torch.ones(batch_size * 2, dtype=torch.float32)
        discount = 1.0
        risks: List[torch.Tensor] = []
        progress: List[torch.Tensor] = []
        for horizon_step in range(planning_horizon):
            rollout_actions = (
                normalized_actions
                if horizon_step < commit_horizon
                else base_actions
            )
            imagined, prediction = model.imagine_step(
                imagined, rollout_actions, deterministic=True
            )
            risks.append(torch.sigmoid(prediction["risk_logit"]) * continuation)
            progress.append(
                symexp(prediction["progress_symlog"])
                * continuation
                * discount
            )
            continuation = continuation * torch.sigmoid(
                prediction["continuation_logit"]
            )
            discount *= discount_factor
        risk = torch.stack(risks).amax(dim=0).reshape(batch_size, 2).cpu().numpy()
        predicted_progress = (
            torch.stack(progress).sum(dim=0).reshape(batch_size, 2).cpu().numpy()
        )
        risk_cost = risk + risk_curvature * risk * risk
        score = (
            progress_weight * predicted_progress
            - risk_weight * risk_cost
            - action_penalty * delta_np
        )
        selected_mask = score[:, 1] > score[:, 0]
        selected += int(selected_mask.sum())
        if selected_mask.any():
            selected_margins = (
                score[selected_mask, 1] - score[selected_mask, 0]
            )
            selected_confidences = np.asarray([
                rssm_authority_confidence(margin, authority_temperature)
                for margin in selected_margins
            ], dtype=np.float32)
            actor_authorities = np.clip(
                rows_np[selected_mask, 1, 3], 0.0, 1.0
            )
            score_margins.extend(selected_margins.tolist())
            authority_confidences.extend(selected_confidences.tolist())
            effective_authorities.extend(
                (actor_authorities * selected_confidences).tolist()
            )
            effective_control_deltas.extend(
                (delta_np[selected_mask, 1] * selected_confidences).tolist()
            )
            risk_increase_selected += int(
                (risk[selected_mask, 1] > risk[selected_mask, 0] + 0.02).sum()
            )
    return {
        "episodes": len(episodes),
        "transitions": int(sum(row.transitions for row in episodes)),
        "material_candidates": material,
        "selected_candidates": selected,
        "selection_rate": selected / max(1, material),
        "risk_increase_selected": risk_increase_selected,
        "mean_positive_score_margin": (
            float(np.mean(score_margins)) if score_margins else 0.0
        ),
        "authority_temperature": authority_temperature,
        "mean_authority_confidence": (
            float(np.mean(authority_confidences))
            if authority_confidences else 0.0
        ),
        "p95_authority_confidence": (
            float(np.percentile(authority_confidences, 95))
            if authority_confidences else 0.0
        ),
        "mean_effective_authority": (
            float(np.mean(effective_authorities))
            if effective_authorities else 0.0
        ),
        "mean_effective_control_delta": (
            float(np.mean(effective_control_deltas))
            if effective_control_deltas else 0.0
        ),
    }


def behavior_preservation_gate(
    parent_blocked: Dict[str, Any],
    candidate_blocked: Dict[str, Any],
    parent_preservation: Dict[str, Any],
    candidate_preservation: Dict[str, Any],
    minimum_preservation: float,
    maximum_blocked_selection_ratio: float,
) -> Tuple[bool, Dict[str, Any]]:
    parent_blocked_count = int(parent_blocked["selected_candidates"])
    parent_preservation_count = int(parent_preservation["selected_candidates"])
    maximum_blocked = int(math.floor(
        parent_blocked_count * float(maximum_blocked_selection_ratio)
    ))
    minimum_preserved = int(math.ceil(
        parent_preservation_count * float(minimum_preservation)
    ))
    candidate_blocked_count = int(candidate_blocked["selected_candidates"])
    candidate_preservation_count = int(
        candidate_preservation["selected_candidates"]
    )
    passed = bool(
        candidate_blocked_count <= maximum_blocked
        and candidate_preservation_count >= minimum_preserved
        and int(candidate_blocked["risk_increase_selected"]) == 0
        and int(candidate_preservation["risk_increase_selected"]) == 0
    )
    return passed, {
        "parent_blocked_selected": parent_blocked_count,
        "candidate_blocked_selected": candidate_blocked_count,
        "maximum_blocked_selected": maximum_blocked,
        "parent_preservation_selected": parent_preservation_count,
        "candidate_preservation_selected": candidate_preservation_count,
        "minimum_preserved_selected": minimum_preserved,
        "minimum_preservation_fraction": float(minimum_preservation),
        "maximum_blocked_selection_ratio": float(
            maximum_blocked_selection_ratio
        ),
        "candidate_blocked_risk_increase_selected": int(
            candidate_blocked["risk_increase_selected"]
        ),
        "candidate_preservation_risk_increase_selected": int(
            candidate_preservation["risk_increase_selected"]
        ),
    }


def resolve_paths(patterns: Iterable[str]) -> List[Path]:
    return discover_traces(list(patterns), 0)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else checkpoint_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    parent_arbitration = (checkpoint.get("rssm_v2") or {}).get("arbitration") or {}
    risk_curvature = float(parent_arbitration.get("risk_curvature", 0.0))
    config = RSSMConfig.from_dict(checkpoint.get("world_model_config"))
    model = TemporalRSSMWorldModel(config)
    model.load_state_dict(checkpoint["world_model"])
    model.eval()

    train_stationary_paths = resolve_paths(
        args.train_stationary_trace or [DEFAULT_TRAIN_TRACE]
    )
    preservation_train_paths = resolve_paths(
        args.train_preservation_trace or [DEFAULT_TRAIN_PRESERVATION_TRACE]
    )
    forced_paths = resolve_paths(
        args.validation_trace or list(DEFAULT_VALIDATION_TRACES)
    )
    if not train_stationary_paths or not preservation_train_paths or not forced_paths:
        raise RuntimeError(
            "stationary, preservation-training and validation traces are required"
        )
    excluded = {path.resolve() for path in forced_paths}
    all_paths = resolve_paths(args.trace_pattern or list(DEFAULT_PATTERNS))
    pool_paths = [path for path in all_paths if path.resolve() not in excluded]
    pool_episodes, _ = load_episodes(pool_paths, args.sequence_length)
    forced_episodes, _ = load_episodes(forced_paths, args.sequence_length)
    train_stationary_episodes, _ = load_episodes(
        train_stationary_paths, args.sequence_length
    )
    preservation_train_episodes, _ = load_episodes(
        preservation_train_paths, args.sequence_length
    )
    training_episodes, regular_validation, regular_routes = split_routes(
        pool_episodes, args.seed
    )
    # The stationary trace is intentionally reserved for the targeted fit even
    # if its route lands in the generic route split. Independent seed 451052
    # remains completely held out for the directional gate.
    stationary_keys = {episode.key for episode in train_stationary_episodes}
    training_by_key = {episode.key: episode for episode in training_episodes}
    for episode in train_stationary_episodes:
        training_by_key[episode.key] = episode
    for episode in preservation_train_episodes:
        training_by_key[episode.key] = episode
    training_episodes = list(training_by_key.values())
    regular_validation = [
        episode for episode in regular_validation
        if episode.key not in stationary_keys
    ]

    train_pairs = stationary_pairs(
        model, train_stationary_episodes, checkpoint
    )
    validation_pairs = stationary_pairs(model, forced_episodes, checkpoint)
    if train_pairs[0].shape[0] < 100 or validation_pairs[0].shape[0] < 100:
        raise RuntimeError("insufficient independent stationary-oncoming pairs")
    replay_x, replay_parent_probability = replay_features(
        model, training_episodes, checkpoint, args.max_replay, args.seed
    )
    before = pair_metrics(model, validation_pairs, risk_curvature)
    parent_state = copy.deepcopy(model.head_risk.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head_risk.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.head_risk.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    history = []
    base_x, candidate_x, parent_base, parent_candidate, requested_margin = train_pairs
    for epoch in range(max(1, args.epochs)):
        model.head_risk.train()
        replay_logits = model.head_risk(replay_x).squeeze(-1)
        base_logits = model.head_risk(base_x).squeeze(-1)
        candidate_logits = model.head_risk(candidate_x).squeeze(-1)
        replay_distillation_loss = F.binary_cross_entropy_with_logits(
            replay_logits, replay_parent_probability
        )
        pair_distillation_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(base_logits, parent_base)
            + F.binary_cross_entropy_with_logits(
                candidate_logits, parent_candidate
            )
        )
        base_probability = torch.sigmoid(base_logits)
        candidate_probability = torch.sigmoid(candidate_logits)
        ranking_loss = F.relu(
            requested_margin - (candidate_probability - base_probability)
        ).mean()
        total = (
            4.0 * replay_distillation_loss
            + 3.0 * pair_distillation_loss
            + 1.0 * ranking_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.head_risk.parameters(), 5.0)
        optimizer.step()
        model.head_risk.eval()
        metrics = pair_metrics(model, validation_pairs, risk_curvature)
        history.append({
            "epoch": epoch + 1,
            "loss": float(total.detach()),
            "replay_distillation_loss": float(
                replay_distillation_loss.detach()
            ),
            "pair_distillation_loss": float(pair_distillation_loss.detach()),
            "ranking_loss": float(ranking_loss.detach()),
            **metrics,
        })
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[rssm-stationary] epoch={epoch + 1}/{args.epochs} "
                f"loss={float(total.detach()):.4f} "
                f"val_margin={metrics['mean_margin']:+.4f} "
                f"val_positive={metrics['positive_fraction']:.3f}",
                flush=True,
            )
    model.eval()
    fully_calibrated_state = copy.deepcopy(model.head_risk.state_dict())

    blocked_validation_episodes = [
        episode for episode in forced_episodes
        if episode.route_id == str(args.blocked_route_id)
    ]
    preservation_validation_episodes = [
        episode for episode in forced_episodes
        if episode.route_id == str(args.preservation_route_id)
    ]
    if not blocked_validation_episodes or not preservation_validation_episodes:
        raise RuntimeError(
            "forced validation must contain both blocked and preservation routes"
        )
    model.head_risk.load_state_dict(parent_state)
    parent_blocked_replay = decision_replay_metrics(
        model, blocked_validation_episodes, checkpoint
    )
    parent_preservation_replay = decision_replay_metrics(
        model, preservation_validation_episodes, checkpoint
    )
    blend_step = float(np.clip(args.blend_step, 0.01, 1.0))
    blend_values = np.arange(blend_step, 1.0 + blend_step * 0.5, blend_step)
    blend_audit: List[Dict[str, Any]] = []
    selected_alpha: Optional[float] = None
    selected_behavior_gate: Dict[str, Any] = {}
    selected_blocked_replay: Dict[str, Any] = {}
    selected_preservation_replay: Dict[str, Any] = {}
    # Pick the strongest correction that still preserves the independent
    # successful route. This is weight-space regularization and checkpoint
    # selection, not a runtime distance/TTC rule.
    for raw_alpha in blend_values:
        alpha = float(min(1.0, raw_alpha))
        model.head_risk.load_state_dict(blended_risk_head_state(
            parent_state, fully_calibrated_state, alpha
        ))
        blocked_replay = decision_replay_metrics(
            model, blocked_validation_episodes, checkpoint
        )
        preservation_replay = decision_replay_metrics(
            model, preservation_validation_episodes, checkpoint
        )
        behavior_passed, behavior_gate = behavior_preservation_gate(
            parent_blocked_replay,
            blocked_replay,
            parent_preservation_replay,
            preservation_replay,
            args.minimum_preservation,
            args.maximum_blocked_selection_ratio,
        )
        blend_audit.append({
            "alpha": alpha,
            "passed": behavior_passed,
            "blocked": blocked_replay,
            "preservation": preservation_replay,
            "gate": behavior_gate,
        })
        if behavior_passed:
            selected_alpha = alpha
            selected_behavior_gate = behavior_gate
            selected_blocked_replay = blocked_replay
            selected_preservation_replay = preservation_replay
    if selected_alpha is None:
        model.head_risk.load_state_dict(parent_state)
    else:
        model.head_risk.load_state_dict(blended_risk_head_state(
            parent_state, fully_calibrated_state, selected_alpha
        ))
    model.eval()
    after = pair_metrics(model, validation_pairs, risk_curvature)

    observation_mean = np.asarray(checkpoint["world_observation_mean"], np.float32)
    observation_std = np.asarray(checkpoint["world_observation_std"], np.float32)
    action_mean = np.asarray(checkpoint["action_mean"], np.float32)
    action_std = np.asarray(checkpoint["action_std"], np.float32)
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
        list(regular_validation) + list(forced_episodes),
        observation_mean,
        observation_std,
        action_mean,
        action_std,
        torch.device("cpu"),
    )
    quality_passed, quality_gate = rssm_quality_gate(
        combined_validation, forced_validation
    )
    directional_passed = bool(
        selected_alpha is not None
        and
        after["samples"] >= 100
        and after["mean_margin"] >= before["mean_margin"] + 0.004
        and after["mean_risk_cost_margin"] >= (
            before["mean_risk_cost_margin"] + 0.025
        )
        and after["base_risk"] <= before["base_risk"] + 0.08
    )
    behavior_passed = selected_alpha is not None
    passed = bool(quality_passed and directional_passed and behavior_passed)
    # The parent arbitration was already calibrated after repairing the input
    # representation. Keep it fixed so the behavioral replay gate evaluates
    # exactly the utility that will run in CARLA; only the learned risk head is
    # changed by this targeted calibration.
    arbitration = copy.deepcopy(parent_arbitration)
    candidate = copy.deepcopy(checkpoint)
    candidate_world_model = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    changed_world_model_keys = sorted(
        key
        for key, value in candidate_world_model.items()
        if key not in checkpoint["world_model"]
        or not torch.equal(value, checkpoint["world_model"][key])
    )
    unexpected_changed_keys = [
        key for key in changed_world_model_keys
        if not key.startswith("head_risk.")
    ]
    if unexpected_changed_keys:
        raise RuntimeError(
            "stationary calibration changed frozen RSSM parameters: "
            + ", ".join(unexpected_changed_keys)
        )
    candidate["world_model"] = candidate_world_model
    metadata = copy.deepcopy(candidate.get("rssm_v2") or {})
    metadata.update({
        "stationary_oncoming_risk_finetuned_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        ),
        "stationary_oncoming_parent_sha256": sha256(checkpoint_path),
        "stationary_oncoming_risk_head_only": True,
        "stationary_oncoming_changed_world_model_keys": changed_world_model_keys,
        "stationary_oncoming_directional_validation": after,
        "stationary_oncoming_directional_gate_passed": directional_passed,
        "stationary_oncoming_distilled_parent": True,
        "stationary_oncoming_selected_head_blend": selected_alpha,
        "stationary_oncoming_behavior_gate_passed": behavior_passed,
        "stationary_oncoming_behavior_gate": selected_behavior_gate,
        "stationary_oncoming_blocked_replay": selected_blocked_replay,
        "stationary_oncoming_preservation_replay": (
            selected_preservation_replay
        ),
        "recalibration_validation": combined_validation,
        "recalibration_forced_validation": forced_validation,
        "recalibration_quality_gate_passed": quality_passed,
        "arbitration": arbitration,
        "runtime_guard": False,
        "hard_safety_thresholds": False,
        "complementary_to_simlingo": True,
    })
    candidate["rssm_v2"] = metadata
    attempt_path = output_dir / "stationary_oncoming_last_attempt.pt"
    atomic_torch_save(attempt_path, candidate)
    report = {
        "status": "validated" if passed else "quality_gate_rejected",
        "created_at": metadata["stationary_oncoming_risk_finetuned_at"],
        "parent_checkpoint": str(checkpoint_path),
        "parent_sha256": sha256(checkpoint_path),
        "train_stationary_traces": [str(path) for path in train_stationary_paths],
        "train_preservation_traces": [
            str(path) for path in preservation_train_paths
        ],
        "forced_validation_traces": [str(path) for path in forced_paths],
        "regular_validation_routes": regular_routes,
        "train_pairs": int(train_pairs[0].shape[0]),
        "validation_pairs": int(validation_pairs[0].shape[0]),
        "replay_samples": int(replay_x.shape[0]),
        "changed_world_model_keys": changed_world_model_keys,
        "actor_checkpoint_unchanged": True,
        "before": before,
        "after": after,
        "directional_gate_passed": directional_passed,
        "behavior_gate_passed": behavior_passed,
        "behavior_gate": selected_behavior_gate,
        "selected_head_blend": selected_alpha,
        "parent_blocked_replay": parent_blocked_replay,
        "selected_blocked_replay": selected_blocked_replay,
        "parent_preservation_replay": parent_preservation_replay,
        "selected_preservation_replay": selected_preservation_replay,
        "blend_audit": blend_audit,
        "quality_gate_passed": quality_passed,
        "quality_gate": quality_gate,
        "combined_validation": combined_validation,
        "forced_validation": forced_validation,
        "arbitration": arbitration,
        "history": history,
        "attempt": str(attempt_path),
        "promoted": False,
    }
    if args.promote and passed:
        promote_to = args.promote_to.expanduser().resolve()
        promote_to.parent.mkdir(parents=True, exist_ok=True)
        backup = (
            output_dir
            / "candidate_model_before_conservative_stationary_risk_20260812.pt"
        )
        if promote_to.exists() and not backup.exists():
            shutil.copy2(promote_to, backup)
        atomic_torch_save(promote_to, candidate)
        report["promoted"] = True
        report["promoted_to"] = str(promote_to)
        report["backup"] = str(backup)
        report["backup_sha256"] = sha256(backup) if backup.exists() else None
        report["promoted_sha256"] = sha256(promote_to)
    elif not passed:
        model.head_risk.load_state_dict(parent_state)
    report_path = output_dir / "stationary_oncoming_finetune_report.json"
    atomic_json(report_path, report)
    print(
        f"[rssm-stationary] gate={'PASS' if passed else 'REJECT'} "
        f"directional={int(directional_passed)} behavior={int(behavior_passed)} "
        f"multi_horizon={int(quality_passed)} alpha={selected_alpha} "
        f"promoted={int(report['promoted'])}",
        flush=True,
    )
    print(f"[rssm-stationary] report={report_path}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
