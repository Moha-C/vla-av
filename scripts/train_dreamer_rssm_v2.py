#!/usr/bin/env python3
"""Train and validate a temporal RSSM complement for SimLingo.

This script never overwrites the production PPO/SDBS checkpoints. It migrates
one PPO complement into an isolated V2 checkpoint, fits a recurrent world model
on ordered Bench2Drive traces, validates full held-out routes at several
horizons, and only writes ``candidate_model.pt`` when the quality gate passes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from external.simlingo.team_code.dreamer_guard import ActorCritic
from external.simlingo.team_code.dreamer_world_models import (
    EVENT_NAMES,
    PREDICTED_OBSERVATION_INDICES,
    POLICY_MODEL_TYPE,
    WORLD_MODEL_TYPE,
    RSSMConfig,
    RSSMState,
    TemporalRSSMWorldModel,
    expand_actor_input_state_dict,
    symlog,
    symexp,
)
from scripts import dreamer_online_rl_update as core


ROOT = Path(__file__).resolve().parents[1]
WORLD_STATE_DIM = 28
OBSERVATION_DIM = 49
ACTION_DIM = 4
POLICY_INPUT_SEMANTICS = core.MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS
POLICY_ACTION_SEMANTICS = "simlingo_signed_longitudinal_target_with_learned_gate_v3"
DEFAULT_SOURCE = (
    ROOT
    / "external/simlingo/checkpoints/dreamer_ppo_rl_noguard/production_model.pt"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "external/simlingo/checkpoints/dreamer_ppo_rssm_v2"
)

OBSERVATION_FAMILIES = {
    "decision": tuple(PREDICTED_OBSERVATION_INDICES),
    "ego": (2, 4, 6, 8),
    "hazards": (
        10, 13, 14, 16, 18, 21, 23, 26, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44,
    ),
    "simlingo_command": (28, 29, 30),
}


@dataclass
class Episode:
    key: str
    route_id: str
    seed: str
    source: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    continuation: np.ndarray
    risks: np.ndarray
    progress: np.ndarray
    events: np.ndarray
    teacher_targets: np.ndarray
    teacher_mask: np.ndarray

    @property
    def transitions(self) -> int:
        return int(self.actions.shape[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("status"), dict):
                rows.append(row)
    return rows


def validated_teacher_paths() -> set:
    paths = set()
    for summary_path in ROOT.glob("logs/dreamer_rl_distillation/*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") != "saved":
            continue
        for run in summary.get("runs") or []:
            if run.get("accepted"):
                try:
                    paths.add(str(Path(run["trace"]).expanduser().resolve()))
                except (KeyError, OSError):
                    continue
    return paths


def discover_traces(patterns: Sequence[str], maximum: int = 0) -> List[Path]:
    discovered = set()
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_file() and path.stat().st_size > 0:
                discovered.add(path.resolve())
    paths = sorted(discovered, key=lambda path: (path.stat().st_mtime, str(path)))
    if maximum > 0:
        paths = paths[-maximum:]
    return paths


def result_metrics(route_id: str, seed: str) -> Dict[str, float]:
    path = ROOT / "logs/simlingo_eval" / f"results_bench2drive_{route_id}_seed_{seed}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = ((payload.get("_checkpoint") or {}).get("records") or [])
        record = records[0] if records else {}
        scores = record.get("scores") or {}
        infractions = record.get("infractions") or {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}

    def count(name: str) -> float:
        value = infractions.get(name) or []
        return float(len(value)) if isinstance(value, list) else float(bool(value))

    return {
        "route_score": float(scores.get("score_route", 0.0)),
        "collisions": count("collisions_layout")
        + count("collisions_pedestrian")
        + count("collisions_vehicle"),
        "offroad": count("outside_route_lanes") + count("route_dev"),
        "red_lights": count("red_light"),
        "stop_infractions": count("stop_infraction"),
        "blocked": count("vehicle_blocked"),
        "incomplete": 0.0 if record.get("status") == "Completed" else 1.0,
    }


def split_contiguous(rows: List[Dict[str, Any]], minimum_rows: int) -> List[List[Dict[str, Any]]]:
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    previous_timestamp: Optional[float] = None
    seen = set()
    for row in rows:
        status = row.get("status") or {}
        timestamp = core.as_float(status.get("timestamp"), core.as_float(row.get("collector_time")))
        collector = core.as_float(row.get("collector_time"), timestamp)
        # Guard collectors can retain a stale status while CARLA is starting.
        if abs(collector - timestamp) > 5.0:
            continue
        if timestamp in seen:
            continue
        seen.add(timestamp)
        if previous_timestamp is not None and (timestamp <= previous_timestamp or timestamp - previous_timestamp > 2.0):
            if len(current) >= minimum_rows:
                segments.append(current)
            current = []
        current.append(row)
        previous_timestamp = timestamp
    if len(current) >= minimum_rows:
        segments.append(current)
    return segments


def actual_action(status: Dict[str, Any]) -> np.ndarray:
    base = core.action_dict(status, "base_action")
    chosen = core.action_dict(status, "chosen_action")
    applied = bool(status.get("applied"))
    if str(status.get("mode", "")) == "rl_noguard":
        gate = float(np.clip(core.as_float(status.get("rl_intervention_strength"), chosen[3]), 0.0, 1.0))
        return np.asarray([chosen[0], chosen[1], chosen[2], gate], dtype=np.float32)
    control = chosen if applied else base
    return np.asarray([control[0], control[1], control[2], 1.0 if applied else 0.0], dtype=np.float32)


def observation_from_status(status: Dict[str, Any], previous_action: np.ndarray) -> Optional[np.ndarray]:
    observation = core.policy_state_from_status(
        status,
        OBSERVATION_DIM,
        WORLD_STATE_DIM,
        policy_input_semantics=POLICY_INPUT_SEMANTICS,
    )
    if observation is None:
        return None
    observation = np.asarray(observation, dtype=np.float32).copy()
    observation[-4:] = np.asarray(previous_action, dtype=np.float32)[:4]
    return observation if np.all(np.isfinite(observation)) else None


def geometric_targets(
    status: Dict[str, Any],
    next_status: Dict[str, Any],
    action: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    state = np.asarray(status.get("state_vector") or [], dtype=np.float32)
    nxt = np.asarray(next_status.get("state_vector") or [], dtype=np.float32)
    distance = 0.0
    speed = 0.0
    if state.shape[0] >= 3 and nxt.shape[0] >= 3:
        distance = min(20.0, math.hypot(float(nxt[0] - state[0]), float(nxt[1] - state[1])))
        speed = max(0.0, float(nxt[2]))

    front = core.as_float(status.get("front_vehicle_m"), 80.0)
    walker = core.as_float(status.get("nearest_walker_m"), 80.0)
    bike = core.as_float(status.get("nearest_bike_m"), 80.0)
    current_oncoming = core.current_oncoming_values(status)
    base = core.action_dict(status, "base_action")
    side = core.side_from_action(base, action)
    side_values = core.status_side_values(status, side) if side else {
        "clear": 80.0,
        "ttc": 99.0,
        "oncoming": 80.0,
        "oncoming_ttc": 99.0,
    }
    risks = [
        max(0.0, (12.0 - front) / 12.0),
        max(0.0, (10.0 - min(walker, bike)) / 10.0),
        max(0.0, (8.0 - side_values["clear"]) / 8.0),
        max(0.0, (4.0 - side_values["ttc"]) / 4.0),
        # Opposing traffic is defined by heading, not by its instantaneous
        # velocity. A stopped vehicle facing the ego remains an occupied,
        # dangerous part of the lane selected by the candidate action even
        # though its closing-speed TTC is formally infinite.
        max(0.0, (35.0 - side_values["oncoming"]) / 35.0),
        max(0.0, (6.0 - side_values["oncoming_ttc"]) / 6.0),
        max(0.0, (35.0 - current_oncoming["distance"]) / 35.0),
        max(0.0, (6.0 - current_oncoming["ttc"]) / 6.0),
    ]
    light = str(status.get("traffic_light", "none")).lower()
    if light in ("red", "yellow") and speed > 1.0:
        risks.append(0.75)
    blocked = core.as_float(status.get("blocked_ticks"), 0.0)
    if blocked >= 80.0 and speed < 0.3:
        risks.append(0.55)
    risk = float(np.clip(max(risks), 0.0, 1.0))
    events = np.asarray([
        0.0,
        0.0,
        1.0 if min(walker, bike) < 8.0 else 0.0,
        1.0 if light in ("red", "yellow") and speed > 1.0 else 0.0,
        1.0 if blocked >= 80.0 and speed < 0.3 else 0.0,
    ], dtype=np.float32)
    return risk, distance, events


def build_episode(
    path: Path,
    rows: List[Dict[str, Any]],
    segment_index: int,
    teacher_paths: set,
) -> Optional[Episode]:
    route_id = str(rows[0].get("route_id", "unknown"))
    seed = str(rows[0].get("seed", "unknown"))
    metrics = result_metrics(route_id, seed)
    is_teacher = str(path.resolve()) in teacher_paths
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    continuations: List[float] = []
    risks: List[float] = []
    progress: List[float] = []
    events: List[np.ndarray] = []
    teacher_targets: List[np.ndarray] = []
    teacher_mask: List[float] = []
    first_base = core.action_dict(rows[0].get("status") or {}, "base_action")
    previous_action = np.asarray([first_base[0], first_base[1], first_base[2], 0.0], dtype=np.float32)
    previous_reward_action = previous_action.copy()
    tracker = core.OvertakeRewardTracker()
    stagnant = 0

    for index, (row, next_row) in enumerate(zip(rows, rows[1:])):
        status = row.get("status") or {}
        next_status = next_row.get("status") or {}
        observation = observation_from_status(status, previous_action)
        action = actual_action(status)
        next_observation = observation_from_status(next_status, action)
        if observation is None or next_observation is None:
            continue
        state = np.asarray(status.get("state_vector") or [], dtype=np.float32)
        nxt = np.asarray(next_status.get("state_vector") or [], dtype=np.float32)
        if state.shape[0] < WORLD_STATE_DIM or nxt.shape[0] < WORLD_STATE_DIM:
            continue
        if np.linalg.norm(nxt[:2] - state[:2]) < 0.03 and max(0.0, float(nxt[2])) < 0.2:
            stagnant += 1
        else:
            stagnant = 0
        try:
            reward, _ = core.step_reward(
                status,
                next_status,
                previous_reward_action,
                stagnant_steps=stagnant,
                overtake_tracker=tracker,
            )
        except Exception:
            reward = 0.0
        risk, step_progress, event = geometric_targets(status, next_status, action)
        applied = bool(status.get("applied"))
        base = core.action_dict(status, "base_action")
        chosen = core.action_dict(status, "chosen_action")
        target_control = chosen[:3] if applied else base[:3]
        target = np.asarray([*target_control, 0.995 if applied else 0.005], dtype=np.float32)

        if not observations:
            observations.append(observation)
        observations.append(next_observation)
        actions.append(action)
        rewards.append(float(np.clip(reward, -20.0, 20.0)))
        continuations.append(1.0)
        risks.append(risk)
        progress.append(step_progress)
        events.append(event)
        teacher_targets.append(target)
        teacher_mask.append(1.0 if is_teacher else 0.0)
        previous_action = action
        previous_reward_action = action

    if len(actions) < 8 or len(observations) != len(actions) + 1:
        return None
    continuations[-1] = 0.0
    if metrics.get("collisions", 0.0) > 0.0:
        events[-1][0] = 1.0
        risks[-1] = 1.0
        rewards[-1] = min(rewards[-1], -10.0)
    if metrics.get("offroad", 0.0) > 0.0:
        events[-1][1] = 1.0
        risks[-1] = max(risks[-1], 0.9)
        rewards[-1] = min(rewards[-1], -6.0)

    source = "validated_guard_teacher" if is_teacher else (
        "online_rl" if "dreamer_online_rl" in str(path) else "guard_trace"
    )
    return Episode(
        key=f"{route_id}:{seed}:{segment_index}:{path.name}",
        route_id=route_id,
        seed=seed,
        source=source,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        continuation=np.asarray(continuations, dtype=np.float32),
        risks=np.asarray(risks, dtype=np.float32),
        progress=np.asarray(progress, dtype=np.float32),
        events=np.asarray(events, dtype=np.float32),
        teacher_targets=np.asarray(teacher_targets, dtype=np.float32),
        teacher_mask=np.asarray(teacher_mask, dtype=np.float32),
    )


def load_episodes(paths: Sequence[Path], sequence_length: int) -> Tuple[List[Episode], List[Dict[str, Any]]]:
    teacher_paths = validated_teacher_paths()
    episodes: List[Episode] = []
    audit: List[Dict[str, Any]] = []
    for path in paths:
        rows = read_jsonl(path)
        core.enrich_current_oncoming(rows)
        segments = split_contiguous(rows, sequence_length + 1)
        accepted = 0
        transitions = 0
        for segment_index, segment in enumerate(segments):
            episode = build_episode(path, segment, segment_index, teacher_paths)
            if episode is not None:
                episodes.append(episode)
                accepted += 1
                transitions += episode.transitions
        audit.append({
            "path": str(path),
            "rows": len(rows),
            "segments": len(segments),
            "accepted_segments": accepted,
            "transitions": transitions,
            "validated_teacher": str(path.resolve()) in teacher_paths,
        })
    return episodes, audit


def split_routes(episodes: Sequence[Episode], seed: int) -> Tuple[List[Episode], List[Episode], List[str]]:
    routes = sorted({episode.route_id for episode in episodes})
    if len(routes) < 2:
        raise RuntimeError("RSSM validation requires at least two distinct routes")
    rng = random.Random(seed)
    rng.shuffle(routes)
    validation_count = max(1, int(round(len(routes) * 0.25)))
    validation_routes = set(routes[:validation_count])
    training = [episode for episode in episodes if episode.route_id not in validation_routes]
    validation = [episode for episode in episodes if episode.route_id in validation_routes]
    if not training or not validation:
        raise RuntimeError("route-held-out split is empty")
    return training, validation, sorted(validation_routes)


def calibrated_arbitration(validation: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a continuous, uncertainty-aware RSSM utility from validation.

    Higher held-out risk/progress error increases risk curvature and the cost
    of a large deviation from SimLingo. This is deliberately a soft utility:
    no distance, TTC, risk threshold, or hard veto is introduced.
    """
    horizon = validation.get("5") or validation.get("1") or {}
    risk_error = float(horizon.get("risk_mae", 0.18))
    progress_error = float(horizon.get("progress_mae_m", 0.15))
    if not math.isfinite(risk_error):
        risk_error = 0.18
    if not math.isfinite(progress_error):
        progress_error = 0.15
    risk_weight = 2.0
    authority_temperature = float(np.clip(
        progress_error + risk_weight * risk_error,
        0.10,
        1.50,
    ))
    return {
        "objective": "risk_sensitive_simlingo_complement_v2",
        "progress_weight": 1.0,
        "risk_weight": risk_weight,
        "risk_curvature": float(np.clip(1.5 + 12.0 * risk_error, 2.0, 4.0)),
        "action_penalty": float(np.clip(0.12 + 1.5 * progress_error, 0.18, 0.45)),
        "candidate_commit_horizon": 1,
        "authority_mapping": "one_minus_exp_negative_positive_margin_over_temperature_v1",
        "authority_temperature": authority_temperature,
        "actor_gate_role": "upper_bound_scaled_by_model_confidence",
        "hard_thresholds": False,
        "calibration_basis": {
            "validation_horizon": 5 if "5" in validation else 1,
            "risk_mae": risk_error,
            "progress_mae_m": progress_error,
        },
    }


def rssm_quality_gate(
    validation: Dict[str, Any],
    forced_validation: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    forced_validation = forced_validation or {}
    h1 = validation.get("1", {})
    h5 = validation.get("5", {})
    h15 = validation.get("15", {})
    h1_ego = (h1.get("families") or {}).get("ego") or {}
    h5_decision = (h5.get("families") or {}).get("decision") or {}
    forced_h5 = forced_validation.get("5", {})
    forced_present = bool(forced_validation)
    forced_passed = bool(
        not forced_present
        or (
            float(forced_h5.get("risk_mae", math.inf)) <= 0.22
            and float(forced_h5.get("event_brier", math.inf)) <= 0.10
        )
    )
    passed = bool(
        math.isfinite(float(h1_ego.get("persistence_ratio", math.inf)))
        and float(h1_ego.get("persistence_ratio", math.inf)) <= 1.35
        and float(h5_decision.get("observation_mae_normalized", math.inf)) <= 0.18
        and float(h1.get("idle_decision_noise_mae_normalized", math.inf)) <= 0.025
        and float(h5.get("risk_mae", math.inf)) <= 0.18
        and float(h15.get("risk_mae", math.inf)) <= 0.22
        and float(h5.get("event_brier", math.inf)) <= 0.08
        and forced_passed
    )
    return passed, {
        "horizon_1_ego_persistence_ratio_max": 1.35,
        "horizon_5_decision_mae_normalized_max": 0.18,
        "horizon_1_idle_decision_noise_mae_max": 0.025,
        "horizon_5_risk_mae_max": 0.18,
        "horizon_15_risk_mae_max": 0.22,
        "horizon_5_event_brier_max": 0.08,
        "forced_horizon_5_risk_mae_max": 0.22,
        "forced_horizon_5_event_brier_max": 0.10,
        "forced_validation_present": forced_present,
        "forced_validation_passed": forced_passed,
        "note": (
            "Structured exogenous traffic variables are reported against "
            "persistence but are not the primary safety-planning gate."
        ),
    }


class SequenceWindows(Dataset):
    def __init__(
        self,
        episodes: Sequence[Episode],
        sequence_length: int,
        stride: int,
        observation_mean: np.ndarray,
        observation_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.episodes = list(episodes)
        self.sequence_length = sequence_length
        self.observation_mean = observation_mean
        self.observation_std = observation_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.windows: List[Tuple[int, int]] = []
        self.weights: List[float] = []
        for episode_index, episode in enumerate(self.episodes):
            starts = list(range(0, episode.transitions - sequence_length + 1, max(1, stride)))
            final_start = episode.transitions - sequence_length
            if final_start >= 0 and final_start not in starts:
                starts.append(final_start)
            for start in starts:
                stop = start + sequence_length
                hazard = float(
                    max(
                        episode.risks[start:stop].max(initial=0.0),
                        episode.events[start:stop].max(initial=0.0),
                    )
                )
                self.windows.append((episode_index, start))
                self.weights.append(1.0 + 3.0 * hazard)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_index, start = self.windows[index]
        episode = self.episodes[episode_index]
        stop = start + self.sequence_length
        observations = (
            episode.observations[start:stop + 1] - self.observation_mean
        ) / self.observation_std
        actions = (
            episode.actions[start:stop] - self.action_mean
        ) / self.action_std
        return {
            "observations": torch.from_numpy(observations.astype(np.float32)),
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "rewards": torch.from_numpy(episode.rewards[start:stop]),
            "continuation": torch.from_numpy(episode.continuation[start:stop]),
            "risks": torch.from_numpy(episode.risks[start:stop]),
            "progress": torch.from_numpy(episode.progress[start:stop]),
            "events": torch.from_numpy(episode.events[start:stop]),
        }


def compute_world_observation_normalizer(
    episodes: Sequence[Episode],
    policy_scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit an RSSM-only normalizer without changing the migrated policy.

    The production actor uses the historic fixed-range normalizer.  The world
    model benefits from centered data and empirical scales, but those two
    normalizers must remain separate or migration would silently change the
    already validated SimLingo-complement behavior.
    """

    values = np.concatenate([episode.observations for episode in episodes], axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    empirical_std = values.std(axis=0).astype(np.float32)
    scale_floor = np.maximum(np.asarray(policy_scale, dtype=np.float32) * 0.02, 1e-3)
    std = np.maximum(empirical_std, scale_floor).astype(np.float32)
    return mean, std


def _weighted_observation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Most CARLA ticks barely move. Upweight genuine changes so a trivial
    # persistence predictor cannot dominate the learning signal.
    change_weight = 1.0 + 3.0 * (target.abs() >= 0.025).to(target.dtype)
    weights = change_weight * mask
    element_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return (element_loss * weights).sum() / weights.sum().clamp_min(1.0)


def overshooting_loss(
    model: TemporalRSSMWorldModel,
    predictions: Dict[str, torch.Tensor],
    observations: torch.Tensor,
    actions: torch.Tensor,
    maximum_horizon: int,
    deterministic: bool = False,
) -> torch.Tensor:
    """Train open-loop latent rollouts from every posterior state in a window."""

    time_steps = int(actions.shape[1])
    horizon = min(max(0, int(maximum_horizon)), time_steps)
    if horizon <= 1:
        return observations.new_zeros(())
    starts = time_steps - horizon + 1
    batch_size = int(observations.shape[0])
    state = RSSMState(
        deter=predictions["posterior_deter"][:, :starts].reshape(
            batch_size * starts, -1
        ),
        stoch=predictions["posterior_stoch"][:, :starts].reshape(
            batch_size * starts, -1
        ),
        logits=predictions["posterior_state_logits"][:, :starts].reshape(
            batch_size * starts,
            model.config.stoch_dim,
            model.config.classes,
        ),
    )
    initial_observation = observations[:, :starts]
    imagined_observation = initial_observation.reshape(batch_size * starts, -1)
    mask = model.observation_delta_mask.reshape(1, 1, -1)
    losses = []
    for offset in range(horizon):
        action = actions[:, offset:offset + starts].reshape(batch_size * starts, -1)
        state, heads = model.imagine_step(
            state, action, deterministic=deterministic
        )
        imagined_observation = imagined_observation + heads["observation_delta"]
        target = observations[:, offset + 1:offset + starts + 1]
        predicted = imagined_observation.reshape(batch_size, starts, -1)
        cumulative_delta = target - initial_observation
        losses.append(
            _weighted_observation_loss(
                predicted - initial_observation,
                cumulative_delta,
                mask,
            )
        )
    # One-step supervision already exists below; emphasize truly imagined
    # consequences while retaining a smooth curriculum across horizons.
    return torch.stack(losses[1:]).mean()


def world_model_loss(
    model: TemporalRSSMWorldModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    overshoot_horizon: int = 5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    predictions = model.observe_sequence(observations, actions)
    target_delta = observations[:, 1:] - observations[:, :-1]
    reward_target = symlog(batch["rewards"].to(device))
    progress_target = symlog(batch["progress"].to(device))
    continuation_target = batch["continuation"].to(device)
    risk_target = batch["risks"].to(device)
    event_target = batch["events"].to(device)

    observation_mask = model.observation_delta_mask.reshape(1, 1, -1)
    observation_loss = _weighted_observation_loss(
        predictions["observation_delta"], target_delta, observation_mask
    )
    rollout_loss = overshooting_loss(
        model,
        predictions,
        observations,
        actions,
        overshoot_horizon,
    )
    reward_loss = F.smooth_l1_loss(predictions["reward_symlog"], reward_target)
    continuation_loss = F.binary_cross_entropy_with_logits(
        predictions["continuation_logit"], continuation_target
    )
    risk_loss = F.binary_cross_entropy_with_logits(predictions["risk_logit"], risk_target)
    progress_loss = F.smooth_l1_loss(predictions["progress_symlog"], progress_target)
    event_loss = F.binary_cross_entropy_with_logits(predictions["event_logits"], event_target)
    kl = model.kl_loss(predictions["posterior_logits"], predictions["prior_logits"])
    total = (
        observation_loss
        + 0.45 * rollout_loss
        + 0.50 * reward_loss
        + 0.20 * continuation_loss
        + 0.80 * risk_loss
        + 0.30 * progress_loss
        + 0.40 * event_loss
        + kl["loss"]
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "observation": float(observation_loss.detach().cpu()),
        "overshooting": float(rollout_loss.detach().cpu()),
        "reward": float(reward_loss.detach().cpu()),
        "continuation": float(continuation_loss.detach().cpu()),
        "risk": float(risk_loss.detach().cpu()),
        "progress": float(progress_loss.detach().cpu()),
        "events": float(event_loss.detach().cpu()),
        "kl_dynamic": float(kl["dynamic"].detach().cpu()),
        "kl_representation": float(kl["representation"].detach().cpu()),
    }
    return total, metrics


def normalize_episode(
    episode: Episode,
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    observations = (episode.observations - observation_mean) / observation_std
    actions = (episode.actions - action_mean) / action_std
    return torch.from_numpy(observations.astype(np.float32)), torch.from_numpy(actions.astype(np.float32))


@torch.no_grad()
def evaluate_horizons(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[Episode],
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    horizons: Sequence[int] = (1, 5, 10, 15),
) -> Dict[str, Any]:
    model.eval()
    accumulators = {
        int(horizon): {
            "risk": [],
            "progress": [],
            "event_brier": [],
            "families": {
                name: {"error": [], "persistence": []}
                for name in ("all", *OBSERVATION_FAMILIES)
            },
            "changed_decision_error": [],
            "changed_decision_persistence": [],
            "idle_decision_noise": [],
        }
        for horizon in horizons
    }
    for episode in episodes:
        observations, actions = normalize_episode(
            episode, observation_mean, observation_std, action_mean, action_std
        )
        observations = observations.to(device)
        actions = actions.to(device)
        posterior = model.observe_initial(observations[0:1], deterministic=True)
        posterior_states: List[RSSMState] = [posterior.detach()]
        for step in range(episode.transitions):
            posterior, _ = model.obs_step(
                posterior,
                actions[step:step + 1],
                observations[step + 1:step + 2],
                deterministic=True,
            )
            posterior_states.append(posterior.detach())

        for horizon in horizons:
            horizon = int(horizon)
            available = episode.transitions - horizon + 1
            if available <= 0:
                continue
            stride = max(1, available // 80)
            for start in range(0, available, stride):
                latent = posterior_states[start]
                predicted_observation = observations[start:start + 1].clone()
                risk_predictions = []
                progress_predictions = []
                event_predictions = []
                for offset in range(horizon):
                    latent, heads = model.imagine_step(
                        latent,
                        actions[start + offset:start + offset + 1],
                        deterministic=True,
                    )
                    predicted_observation = predicted_observation + heads["observation_delta"]
                    risk_predictions.append(torch.sigmoid(heads["risk_logit"]))
                    progress_predictions.append(symexp(heads["progress_symlog"]))
                    event_predictions.append(torch.sigmoid(heads["event_logits"]))
                target = observations[start + horizon:start + horizon + 1]
                persistence = observations[start:start + 1]
                absolute_error = (
                    predicted_observation - target
                ).abs().squeeze(0).cpu().numpy()
                persistence_error = (
                    persistence - target
                ).abs().squeeze(0).cpu().numpy()
                family_indices = {
                    "all": tuple(range(OBSERVATION_DIM)),
                    **OBSERVATION_FAMILIES,
                }
                for name, indices in family_indices.items():
                    valid = [index for index in indices if index < absolute_error.shape[0]]
                    accumulators[horizon]["families"][name]["error"].append(
                        float(absolute_error[valid].mean())
                    )
                    accumulators[horizon]["families"][name]["persistence"].append(
                        float(persistence_error[valid].mean())
                    )
                decision_indices = list(OBSERVATION_FAMILIES["decision"])
                decision_change = float(persistence_error[decision_indices].mean())
                decision_error = float(absolute_error[decision_indices].mean())
                if decision_change >= 0.015:
                    accumulators[horizon]["changed_decision_error"].append(decision_error)
                    accumulators[horizon]["changed_decision_persistence"].append(
                        decision_change
                    )
                elif decision_change <= 0.005:
                    accumulators[horizon]["idle_decision_noise"].append(decision_error)
                target_risk = float(episode.risks[start:start + horizon].mean())
                target_progress = float(episode.progress[start:start + horizon].mean())
                accumulators[horizon]["risk"].append(
                    abs(float(torch.stack(risk_predictions).mean().cpu()) - target_risk)
                )
                accumulators[horizon]["progress"].append(
                    abs(float(torch.stack(progress_predictions).mean().cpu()) - target_progress)
                )
                predicted_events = torch.stack(event_predictions).amax(dim=0).squeeze(0)
                target_events = torch.from_numpy(
                    episode.events[start:start + horizon].max(axis=0)
                ).to(predicted_events)
                accumulators[horizon]["event_brier"].append(
                    float(((predicted_events - target_events) ** 2).mean().cpu())
                )

    result: Dict[str, Any] = {}
    for horizon, values in accumulators.items():
        families = {}
        for name, rows in values["families"].items():
            error = float(np.mean(rows["error"])) if rows["error"] else math.inf
            persistence = (
                float(np.mean(rows["persistence"]))
                if rows["persistence"] else math.inf
            )
            families[name] = {
                "observation_mae_normalized": error,
                "persistence_mae_normalized": persistence,
                "persistence_ratio": error / max(persistence, 1e-8),
            }
        changed_error = (
            float(np.mean(values["changed_decision_error"]))
            if values["changed_decision_error"] else math.inf
        )
        changed_persistence = (
            float(np.mean(values["changed_decision_persistence"]))
            if values["changed_decision_persistence"] else math.inf
        )
        result[str(horizon)] = {
            "samples": len(values["families"]["all"]["error"]),
            **families["all"],
            "families": families,
            "changed_decision_samples": len(values["changed_decision_error"]),
            "changed_decision_mae_normalized": changed_error,
            "changed_decision_persistence_mae_normalized": changed_persistence,
            "changed_decision_persistence_ratio": (
                changed_error / max(changed_persistence, 1e-8)
            ),
            "idle_decision_noise_mae_normalized": (
                float(np.mean(values["idle_decision_noise"]))
                if values["idle_decision_noise"] else math.inf
            ),
            "risk_mae": float(np.mean(values["risk"])) if values["risk"] else math.inf,
            "progress_mae_m": float(np.mean(values["progress"])) if values["progress"] else math.inf,
            "event_brier": (
                float(np.mean(values["event_brier"]))
                if values["event_brier"] else math.inf
            ),
        }
    return result


def migrate_source_checkpoint(
    source: Path,
    config: RSSMConfig,
    model: TemporalRSSMWorldModel,
    policy_observation_mean: np.ndarray,
    policy_observation_std: np.ndarray,
    world_observation_mean: np.ndarray,
    world_observation_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch.load(source, map_location="cpu")
    checkpoint, observation_migration = core.upgrade_policy_observation_checkpoint(
        copy.deepcopy(checkpoint)
    )
    policy_state = expand_actor_input_state_dict(
        checkpoint["policy"], OBSERVATION_DIM + config.feature_dim
    )
    checkpoint.pop("model", None)
    checkpoint["world_model"] = model.state_dict()
    checkpoint["world_model_type"] = WORLD_MODEL_TYPE
    checkpoint["world_model_config"] = config.to_dict()
    checkpoint["policy_model_type"] = POLICY_MODEL_TYPE
    checkpoint["base_world_state_dim"] = WORLD_STATE_DIM
    checkpoint["policy_observation_dim"] = OBSERVATION_DIM
    checkpoint["policy"] = policy_state
    checkpoint["policy_input_semantics"] = POLICY_INPUT_SEMANTICS
    checkpoint["policy_action_semantics"] = POLICY_ACTION_SEMANTICS
    checkpoint["policy_state_mean"] = policy_observation_mean.astype(np.float32)
    checkpoint["policy_state_std"] = policy_observation_std.astype(np.float32)
    checkpoint["world_observation_mean"] = world_observation_mean.astype(np.float32)
    checkpoint["world_observation_std"] = world_observation_std.astype(np.float32)
    checkpoint["action_mean"] = action_mean.astype(np.float32)
    checkpoint["action_std"] = action_std.astype(np.float32)
    checkpoint.pop("optimizer_pi", None)
    checkpoint.pop("optimizer_wm", None)
    return checkpoint, observation_migration


def actor_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> ActorCritic:
    state = checkpoint["policy"]
    actor = ActorCritic(
        state_dim=int(state["trunk.0.weight"].shape[1]),
        action_dim=int(state["log_std"].shape[0]),
        hidden=int(state["trunk.0.weight"].shape[0]),
    ).to(device)
    actor.load_state_dict(state)
    return actor


def decoded_actor_output(actor: ActorCritic, inputs: torch.Tensor) -> torch.Tensor:
    mean, _, _ = actor(inputs)
    steering = torch.tanh(mean[:, 0:1])
    longitudinal = torch.tanh(mean[:, 1:2] - mean[:, 2:3])
    throttle = torch.relu(longitudinal)
    brake = torch.relu(-longitudinal)
    gate = torch.sigmoid(mean[:, 3:4])
    return torch.cat([steering, throttle, brake, gate], dim=-1)


@torch.no_grad()
def teacher_features(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[Episode],
    policy_observation_mean: np.ndarray,
    policy_observation_std: np.ndarray,
    world_observation_mean: np.ndarray,
    world_observation_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    routes: List[str] = []
    model.eval()
    for episode in episodes:
        if not bool(episode.teacher_mask.any()):
            continue
        world_observations, actions = normalize_episode(
            episode,
            world_observation_mean,
            world_observation_std,
            action_mean,
            action_std,
        )
        policy_observations = torch.from_numpy((
            (episode.observations - policy_observation_mean)
            / policy_observation_std
        ).astype(np.float32))
        world_observations = world_observations.to(device)
        policy_observations = policy_observations.to(device)
        actions = actions.to(device)
        posterior = model.observe_initial(world_observations[0:1], deterministic=True)
        for step in range(episode.transitions):
            if episode.teacher_mask[step] > 0.5:
                inputs.append(torch.cat([
                    policy_observations[step], model.feature(posterior)[0]
                ], dim=-1))
                targets.append(torch.from_numpy(episode.teacher_targets[step]).to(device))
                routes.append(episode.route_id)
            posterior, _ = model.obs_step(
                posterior,
                actions[step:step + 1],
                world_observations[step + 1:step + 2],
                deterministic=True,
            )
    if not inputs:
        return (
            torch.empty(0, OBSERVATION_DIM + model.feature_dim, device=device),
            torch.empty(0, ACTION_DIM, device=device),
            [],
        )
    return torch.stack(inputs), torch.stack(targets), routes


@torch.no_grad()
def actor_metrics(actor: ActorCritic, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if inputs.shape[0] == 0:
        return {"samples": 0, "control_mae_active": math.inf, "gate_accuracy": 0.0}
    predicted = decoded_actor_output(actor, inputs)
    active = targets[:, 3] >= 0.5
    control_mae = (
        float((predicted[active, :3] - targets[active, :3]).abs().mean().cpu())
        if bool(active.any()) else 0.0
    )
    gate_accuracy = float(((predicted[:, 3] >= 0.5) == active).float().mean().cpu())
    return {
        "samples": int(inputs.shape[0]),
        "control_mae_active": control_mae,
        "gate_accuracy": gate_accuracy,
        "mean_gate": float(predicted[:, 3].mean().cpu()),
    }


def fit_latent_adapter(
    checkpoint: Dict[str, Any],
    model: TemporalRSSMWorldModel,
    training_episodes: Sequence[Episode],
    validation_episodes: Sequence[Episode],
    policy_observation_mean: np.ndarray,
    policy_observation_std: np.ndarray,
    world_observation_mean: np.ndarray,
    world_observation_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    epochs: int,
    seed: int,
) -> Dict[str, Any]:
    actor = actor_from_checkpoint(checkpoint, device)
    reference = copy.deepcopy(actor).eval()
    train_inputs, train_targets, _ = teacher_features(
        model,
        training_episodes,
        policy_observation_mean,
        policy_observation_std,
        world_observation_mean,
        world_observation_std,
        action_mean,
        action_std,
        device,
    )
    validation_inputs, validation_targets, _ = teacher_features(
        model,
        validation_episodes,
        policy_observation_mean,
        policy_observation_std,
        world_observation_mean,
        world_observation_std,
        action_mean,
        action_std,
        device,
    )
    if train_inputs.shape[0] == 0 or validation_inputs.shape[0] == 0:
        return {"accepted": False, "reason": "teacher route split is empty"}

    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    first_weight = actor.trunk[0].weight
    first_weight.requires_grad_(True)
    gradient_mask = torch.zeros_like(first_weight)
    gradient_mask[:, OBSERVATION_DIM:] = 1.0
    first_weight.register_hook(lambda gradient: gradient * gradient_mask)
    optimizer = torch.optim.Adam([first_weight], lr=8e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    before = actor_metrics(actor, validation_inputs, validation_targets)
    history = []
    for epoch in range(max(0, epochs)):
        permutation = torch.randperm(train_inputs.shape[0], generator=generator)
        losses = []
        actor.train()
        for start in range(0, train_inputs.shape[0], 128):
            indices = permutation[start:start + 128].to(device)
            batch_inputs = train_inputs[indices]
            batch_targets = train_targets[indices]
            prediction = decoded_actor_output(actor, batch_inputs)
            active = batch_targets[:, 3] >= 0.5
            weights = torch.where(active, 4.0, 1.0).unsqueeze(-1)
            control_loss = (F.smooth_l1_loss(
                prediction[:, :3], batch_targets[:, :3], reduction="none"
            ) * weights).mean()
            gate_loss = F.binary_cross_entropy(
                prediction[:, 3], batch_targets[:, 3]
            )
            with torch.no_grad():
                reference_prediction = decoded_actor_output(reference, batch_inputs)
            trust_loss = F.smooth_l1_loss(prediction, reference_prediction)
            loss = control_loss + 0.75 * gate_loss + 0.08 * trust_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([first_weight], 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        actor.eval()
        validation = actor_metrics(actor, validation_inputs, validation_targets)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **validation})

    after = actor_metrics(actor, validation_inputs, validation_targets)
    control_improved = bool(
        math.isfinite(after["control_mae_active"])
        and after["control_mae_active"] <= before["control_mae_active"] * 0.99
    )
    gate_not_degraded = bool(
        after["gate_accuracy"] >= max(0.80, before["gate_accuracy"] - 0.005)
    )
    accepted = control_improved and gate_not_degraded
    if accepted:
        checkpoint["policy"] = {
            key: value.detach().cpu() for key, value in actor.state_dict().items()
        }
    return {
        "accepted": accepted,
        "reason": "held-out teacher gate passed" if accepted else "held-out teacher gate rejected adapter",
        "before": before,
        "after": after,
        "training_samples": int(train_inputs.shape[0]),
        "validation_samples": int(validation_inputs.shape[0]),
        "history_tail": history[-5:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--actor-epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--overshoot-horizon", type=int, default=5)
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--trace-pattern",
        action="append",
        default=[],
        help="Repo-relative glob; may be repeated.",
    )
    parser.add_argument(
        "--validation-trace-pattern",
        action="append",
        default=[],
        help=(
            "Repo-relative trace glob reserved for safety validation and excluded "
            "from training; may be repeated."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    patterns = args.trace_pattern or [
        "logs/dreamer_online_rl/webapp_*/trace.jsonl",
        "logs/dreamer_online_rl/*/traces/*.jsonl",
        "logs/dreamer_rl_campaign/*/traces/*.jsonl",
        "logs/action_dreaming_collect/*.jsonl",
    ]
    paths = discover_traces(patterns, args.max_traces)
    forced_validation_paths = discover_traces(
        args.validation_trace_pattern,
        0,
    )
    forced_path_set = {path.resolve() for path in forced_validation_paths}
    training_pool_paths = [
        path for path in paths if path.resolve() not in forced_path_set
    ]
    pool_episodes, pool_audit = load_episodes(
        training_pool_paths,
        args.sequence_length,
    )
    forced_validation_episodes, forced_audit = load_episodes(
        forced_validation_paths,
        args.sequence_length,
    )
    for row in pool_audit:
        row["forced_validation"] = False
    for row in forced_audit:
        row["forced_validation"] = True
    trace_audit = pool_audit + forced_audit
    episodes = pool_episodes + forced_validation_episodes
    if not episodes:
        raise RuntimeError("no usable ordered Dreamer traces were found")
    training_episodes, validation_episodes, validation_routes = split_routes(
        pool_episodes,
        args.seed,
    )
    validation_episodes = list(validation_episodes) + list(forced_validation_episodes)
    validation_routes = sorted({
        *validation_routes,
        *(episode.route_id for episode in forced_validation_episodes),
    })

    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    source = torch.load(source_checkpoint, map_location="cpu")
    source, observation_migration = core.upgrade_policy_observation_checkpoint(copy.deepcopy(source))
    policy_observation_mean = np.asarray(
        source["policy_state_mean"], dtype=np.float32
    )[:OBSERVATION_DIM]
    policy_observation_std = np.maximum(
        np.asarray(source["policy_state_std"], dtype=np.float32)[:OBSERVATION_DIM],
        1e-6,
    )
    world_observation_mean, world_observation_std = (
        compute_world_observation_normalizer(
            training_episodes,
            policy_observation_std,
        )
    )
    all_training_actions = np.concatenate([episode.actions for episode in training_episodes], axis=0)
    action_mean = all_training_actions.mean(axis=0).astype(np.float32)
    action_std = np.maximum(all_training_actions.std(axis=0), 0.05).astype(np.float32)

    dataset = SequenceWindows(
        training_episodes,
        args.sequence_length,
        args.stride,
        world_observation_mean,
        world_observation_std,
        action_mean,
        action_std,
    )
    audit = {
        "status": "inspected" if args.inspect_only else "training",
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": sha256(source_checkpoint),
        "device": str(device),
        "traces_discovered": len(paths),
        "episodes": len(episodes),
        "training_episodes": len(training_episodes),
        "validation_episodes": len(validation_episodes),
        "validation_routes": validation_routes,
        "forced_validation_traces": [
            str(path) for path in forced_validation_paths
        ],
        "forced_validation_episodes": len(forced_validation_episodes),
        "transitions": sum(episode.transitions for episode in episodes),
        "training_windows": len(dataset),
        "teacher_transitions": int(sum(episode.teacher_mask.sum() for episode in episodes)),
        "observation_migration": observation_migration,
        "world_observation_mean": world_observation_mean.tolist(),
        "world_observation_std": world_observation_std.tolist(),
        "trace_audit": trace_audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "dataset_audit.json", audit)
    print(
        f"[rssm-v2] traces={len(paths)} episodes={len(episodes)} "
        f"transitions={audit['transitions']} windows={len(dataset)} "
        f"validation_routes={','.join(validation_routes)} device={device}",
        flush=True,
    )
    if args.inspect_only:
        return 0
    if len(dataset) == 0:
        raise RuntimeError("no sequence windows available")

    config = RSSMConfig(observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)
    model = TemporalRSSMWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )
    history = []
    best_state = None
    best_loss = math.inf
    started = time.time()
    for epoch in range(max(1, args.epochs)):
        model.train()
        rows = []
        for batch in loader:
            loss, metrics = world_model_loss(
                model,
                batch,
                device,
                overshoot_horizon=args.overshoot_horizon,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite RSSM loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            rows.append(metrics)
        epoch_metrics = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
        epoch_metrics["epoch"] = epoch + 1
        history.append(epoch_metrics)
        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[rssm-v2] epoch={epoch + 1}/{args.epochs} "
            f"loss={epoch_metrics['loss']:.4f} obs={epoch_metrics['observation']:.4f} "
            f"risk={epoch_metrics['risk']:.4f} kl={epoch_metrics['kl_dynamic']:.4f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    validation = evaluate_horizons(
        model,
        validation_episodes,
        world_observation_mean,
        world_observation_std,
        action_mean,
        action_std,
        device,
    )
    forced_validation = (
        evaluate_horizons(
            model,
            forced_validation_episodes,
            world_observation_mean,
            world_observation_std,
            action_mean,
            action_std,
            device,
        )
        if forced_validation_episodes else {}
    )
    h1 = validation.get("1", {})
    h5 = validation.get("5", {})
    h1_ego = (h1.get("families") or {}).get("ego") or {}
    h5_decision = (h5.get("families") or {}).get("decision") or {}
    quality_passed, quality_gate = rssm_quality_gate(
        validation,
        forced_validation,
    )
    forced_validation_passed = bool(
        quality_gate["forced_validation_passed"]
    )

    checkpoint, _ = migrate_source_checkpoint(
        source_checkpoint,
        config,
        model.cpu(),
        policy_observation_mean,
        policy_observation_std,
        world_observation_mean,
        world_observation_std,
        action_mean,
        action_std,
    )
    model.to(device)
    actor_adapter = fit_latent_adapter(
        checkpoint,
        model,
        training_episodes,
        validation_episodes,
        policy_observation_mean,
        policy_observation_std,
        world_observation_mean,
        world_observation_std,
        action_mean,
        action_std,
        device,
        args.actor_epochs,
        args.seed,
    )
    arbitration = calibrated_arbitration(validation)
    checkpoint["rssm_v2"] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": sha256(source_checkpoint),
        "runtime_guard": False,
        "complementary_to_simlingo": True,
        "model_based_arbitration": True,
        "planning_horizon": 5,
        "planning_discount": 0.95,
        "arbitration": arbitration,
        "hard_safety_thresholds": False,
        "training_history_tail": history[-5:],
        "validation": validation,
        "forced_validation": forced_validation,
        "quality_gate_passed": quality_passed,
        "forced_validation_passed": forced_validation_passed,
        "actor_latent_adapter": actor_adapter,
        "validation_routes": validation_routes,
        "transitions": audit["transitions"],
        "training_seconds": time.time() - started,
    }
    checkpoint = {
        key: ({inner: value.detach().cpu() for inner, value in item.items()} if isinstance(item, dict) and item and all(torch.is_tensor(value) for value in item.values()) else item)
        for key, item in checkpoint.items()
    }
    attempt_path = args.output_dir / "last_attempt.pt"
    atomic_torch_save(attempt_path, checkpoint)
    report = {
        **audit,
        "status": "candidate_saved" if quality_passed else "quality_gate_rejected",
        "quality_gate_passed": quality_passed,
        "quality_gate": quality_gate,
        "validation": validation,
        "forced_validation": forced_validation,
        "forced_validation_passed": forced_validation_passed,
        "arbitration": arbitration,
        "actor_latent_adapter": actor_adapter,
        "training_history": history,
        "last_attempt": str(attempt_path),
        "candidate": str(args.output_dir / "candidate_model.pt") if quality_passed else "",
        "elapsed_seconds": time.time() - started,
    }
    if quality_passed:
        atomic_torch_save(args.output_dir / "candidate_model.pt", checkpoint)
    atomic_json(args.output_dir / "validation_report.json", report)
    print(
        f"[rssm-v2] quality_gate={'PASS' if quality_passed else 'REJECT'} "
        f"h1_ego_ratio={float(h1_ego.get('persistence_ratio', math.inf)):.3f} "
        f"h5_decision_mae={float(h5_decision.get('observation_mae_normalized', math.inf)):.3f} "
        f"risk_mae_h5={float(h5.get('risk_mae', math.inf)):.3f}",
        flush=True,
    )
    return 0 if quality_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
