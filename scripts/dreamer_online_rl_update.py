#!/usr/bin/env python3
"""Online PPO update from one SimLingo + Dreamer no-guard episode trace.

This script is intentionally tied to the Bench2Drive/SimLingo runtime traces,
not the generic CarlaEnv trainer.  SimLingo remains the base driver; the RL
policy learns a continuous no-guard complement around SimLingo's action.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


MAP_INVARIANT_POLICY_INPUT_SEMANTICS = (
    "world_state_plus_simlingo_map_invariant_temporal_context_v5"
)
CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS = (
    "world_state_plus_simlingo_temporal_current_oncoming_context_v6"
)
MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS = (
    "world_state_plus_simlingo_map_invariant_temporal_current_oncoming_context_v6"
)
MAP_INVARIANT_WORLD_STATE_INDICES = (0, 1, 3)
REWARD_SCHEMA = "simlingo_complement_impact_ttc_contrastive_v4"


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 28, action_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.critic = nn.Linear(hidden, 1)

    @staticmethod
    def _squash(raw):
        steering = torch.tanh(raw[..., 0:1])
        rest = torch.sigmoid(raw[..., 1:4])
        return torch.cat([steering, rest], dim=-1)

    def forward(self, state):
        h = self.trunk(state)
        mean = self.actor_mean(h)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def evaluate(self, state, raw_action):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class WorldModel(nn.Module):
    def __init__(self, state_dim: int = 28, action_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head_state = nn.Linear(hidden, state_dim)
        self.head_risk = nn.Linear(hidden, 1)
        self.head_progress = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.head_state(x), self.sigmoid(self.head_risk(x)), self.head_progress(x)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = row.get("status") or {}
            if status.get("mode") == "rl_noguard":
                rows.append(row)
    rows.sort(key=lambda row: as_float(row.get("collector_time")))
    return rows


def read_collision_events(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "collision":
                events.append(event)
    events.sort(key=lambda event: as_float(event.get("wall_time"), float("inf")))
    return events


def enrich_current_oncoming(rows: List[Dict[str, Any]]) -> None:
    """Backfill the current-lane oncoming signal for legacy and new traces."""
    previous: Dict[int, Tuple[int, float, float]] = {}
    for index, row in enumerate(rows):
        status = row.get("status") or {}
        vehicles = status.get("nearby_vehicles")
        if not isinstance(vehicles, list):
            vehicles = []
        ego_road = int(as_float(status.get("ego_road_id"), 0.0))
        ego_lane = int(as_float(status.get("ego_lane_id"), 0.0))
        lane_width = max(2.4, as_float(status.get("ego_lane_width_m"), 3.5))
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        side_candidates: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {
            "left": [],
            "right": [],
        }
        for vehicle in vehicles:
            if not isinstance(vehicle, dict):
                continue
            actor_id = int(as_float(vehicle.get("id"), -1.0))
            forward = as_float(vehicle.get("forward_m"), 80.0)
            lateral = as_float(vehicle.get("lateral_m"), 80.0)
            heading_dot = as_float(vehicle.get("heading_dot"), 1.0)
            actor_road = int(as_float(vehicle.get("road_id"), 0.0))
            actor_lane = int(as_float(vehicle.get("lane_id"), 0.0))
            same_lane = bool(
                ego_lane != 0
                and actor_lane != 0
                and ego_road == actor_road
                and ego_lane == actor_lane
            )
            geometric_match = abs(lateral) <= max(1.5, 0.48 * lane_width)
            # Older collectors coupled the oncoming label to velocity. That
            # made an opposite-facing vehicle disappear as soon as it stopped.
            # Recover the geometric fact from heading even when the persisted
            # boolean is explicitly false.
            is_oncoming = bool(vehicle.get("is_oncoming", False)) or heading_dot < -0.25
            vehicle["is_oncoming"] = bool(is_oncoming)
            if actor_id < 0 or forward < 0.0 or forward > 80.0 or not is_oncoming:
                continue
            if -6.2 <= lateral <= -0.8:
                side_candidates["left"].append((forward, vehicle))
            elif 0.8 <= lateral <= 6.2:
                side_candidates["right"].append((forward, vehicle))
            if same_lane or geometric_match:
                candidates.append((forward, vehicle))

        if candidates:
            distance, vehicle = min(candidates, key=lambda item: item[0])
            actor_id = int(as_float(vehicle.get("id"), -1.0))
            direct_closing = vehicle.get("closing_speed_mps")
            if direct_closing is None:
                relative_speed = vehicle.get("relative_longitudinal_speed_mps")
                direct_closing = max(0.0, -as_float(relative_speed)) if relative_speed is not None else None
            if direct_closing is None and actor_id in previous:
                prev_index, prev_distance, prev_closing = previous[actor_id]
                elapsed_sim = 0.05 * max(1, index - prev_index)
                estimate = float(np.clip((prev_distance - distance) / elapsed_sim, 0.0, 40.0))
                direct_closing = 0.65 * estimate + 0.35 * prev_closing
            closing = float(np.clip(as_float(direct_closing, 0.0), 0.0, 40.0))
            ttc = distance / max(0.1, closing) if closing > 0.5 else 99.0
            previous[actor_id] = (index, distance, closing)
            status["current_oncoming_distance"] = float(distance)
            status["current_oncoming_distance_m"] = float(distance)
            status["current_oncoming_closing_speed_mps"] = closing
            status["current_oncoming_ttc_s"] = float(min(99.0, ttc))
            status["current_oncoming_actor_id"] = actor_id
        else:
            # Overwrite stale fields from legacy traces. setdefault() retained
            # the old false-negative 80 m / 99 s values during retraining.
            status["current_oncoming_distance"] = 80.0
            status["current_oncoming_distance_m"] = 80.0
            status["current_oncoming_closing_speed_mps"] = 0.0
            status["current_oncoming_ttc_s"] = 99.0
            status["current_oncoming_actor_id"] = -1

        for side, side_rows in side_candidates.items():
            if not side_rows:
                status[f"{side}_oncoming_m"] = 80.0
                status[f"{side}_oncoming_rel_speed_mps"] = 0.0
                status[f"{side}_oncoming_ttc_s"] = 99.0
                continue
            distance, vehicle = min(side_rows, key=lambda item: item[0])
            direct_closing = vehicle.get("closing_speed_mps")
            if direct_closing is None:
                relative_speed = vehicle.get("relative_longitudinal_speed_mps")
                direct_closing = (
                    max(0.0, -as_float(relative_speed))
                    if relative_speed is not None else 0.0
                )
            closing = float(np.clip(as_float(direct_closing, 0.0), 0.0, 40.0))
            ttc = distance / max(0.1, closing) if closing > 0.5 else 99.0
            status[f"{side}_oncoming_m"] = float(distance)
            status[f"{side}_oncoming_rel_speed_mps"] = -closing
            status[f"{side}_oncoming_ttc_s"] = float(min(99.0, ttc))


def infer_legacy_collision_event(
    rows: List[Dict[str, Any]],
    metrics: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    if as_float(metrics.get("collisions"), 0.0) <= 0.0:
        return None
    expected_actor_id = int(as_float(metrics.get("first_collision_actor_id"), -1.0))
    for index, row in enumerate(rows):
        status = row.get("status") or {}
        state = status.get("state_vector") or []
        speed = as_float(state[2], 0.0) if len(state) > 2 else 0.0
        if speed <= 0.1:
            continue
        ego_road = int(as_float(status.get("ego_road_id"), 0.0))
        ego_lane = int(as_float(status.get("ego_lane_id"), 0.0))
        for vehicle in status.get("nearby_vehicles") or []:
            if not isinstance(vehicle, dict):
                continue
            actor_id = int(as_float(vehicle.get("id"), -1.0))
            if expected_actor_id >= 0 and actor_id != expected_actor_id:
                continue
            forward = as_float(vehicle.get("forward_m"), 80.0)
            lateral = abs(as_float(vehicle.get("lateral_m"), 80.0))
            clearance = as_float(vehicle.get("longitudinal_clearance_m"), 80.0)
            heading_dot = as_float(vehicle.get("heading_dot"), 1.0)
            same_lane = bool(
                ego_lane != 0
                and int(as_float(vehicle.get("road_id"), 0.0)) == ego_road
                and int(as_float(vehicle.get("lane_id"), 0.0)) == ego_lane
            )
            actor_matches = expected_actor_id >= 0 and actor_id == expected_actor_id
            strict_geometry = same_lane and heading_dot < -0.25 and lateral <= 1.5
            if (
                -1.0 <= forward <= 8.5
                and clearance <= 0.25
                and (actor_matches or strict_geometry)
            ):
                return {
                    "event": "collision",
                    "source": "legacy_geometry_fallback",
                    "wall_time": as_float(status.get("timestamp"), as_float(row.get("collector_time"))),
                    "collision_kind": "vehicle" if as_float(metrics.get("vehicle_collisions"), 0.0) > 0.0 else "unknown",
                    "other_actor_id": actor_id,
                    "status_index": index,
                    "forward_m": forward,
                    "lateral_m": as_float(vehicle.get("lateral_m"), 0.0),
                    "longitudinal_clearance_m": clearance,
                    "heading_dot": heading_dot,
                    "expected_actor_id": expected_actor_id,
                }
    return None


def truncate_rows_at_first_collision(
    rows: List[Dict[str, Any]],
    collision_events: List[Dict[str, Any]],
    metrics: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    collision = collision_events[0] if collision_events else infer_legacy_collision_event(rows, metrics)
    if collision is None or len(rows) < 2:
        return rows, collision
    if "status_index" in collision:
        impact_status_index = int(collision["status_index"])
    else:
        timestamps = [
            as_float((row.get("status") or {}).get("timestamp"), as_float(row.get("collector_time")))
            for row in rows
        ]
        impact_status_index = bisect_left(timestamps, as_float(collision.get("wall_time")))
    impact_status_index = max(1, min(impact_status_index, len(rows) - 1))
    collision = {**collision, "impact_status_index": impact_status_index, "impact_transition_index": impact_status_index - 1}
    return rows[: impact_status_index + 1], collision


def upgrade_policy_observation_checkpoint(ckpt: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Insert three current-oncoming inputs without changing old outputs."""
    policy_state = ckpt.get("policy")
    world_state = ckpt.get("world_model")
    if not isinstance(policy_state, dict) or not isinstance(world_state, dict):
        return ckpt, {"applied": False, "reason": "checkpoint has no PPO policy/world model"}
    old_weight = policy_state.get("trunk.0.weight")
    log_std = policy_state.get("log_std")
    if old_weight is None or log_std is None:
        return ckpt, {"applied": False, "reason": "policy input layer missing"}
    action_dim = int(log_std.shape[0])
    world_state_dim = int(world_state["fc1.weight"].shape[1]) - action_dim
    old_dim = int(old_weight.shape[1])
    target_dim = world_state_dim + 3 + 11 + 3 + 4
    old_semantics = str(ckpt.get("policy_input_semantics", "world_state_v1"))
    target_semantics = (
        MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS
        if old_semantics == MAP_INVARIANT_POLICY_INPUT_SEMANTICS
        else CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS
    )
    if old_dim == target_dim and old_semantics in (
        CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
        MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
    ):
        return ckpt, {"applied": False, "reason": "already v6", "old_dim": old_dim, "new_dim": target_dim}

    compact_prefix = world_state_dim + 3 + 11
    new_weight = old_weight.new_zeros((old_weight.shape[0], target_dim))
    if old_dim == world_state_dim:
        new_weight[:, :world_state_dim] = old_weight
    elif old_dim == compact_prefix:
        new_weight[:, :compact_prefix] = old_weight
    elif old_dim == compact_prefix + 4:
        new_weight[:, :compact_prefix] = old_weight[:, :compact_prefix]
        new_weight[:, compact_prefix + 3: compact_prefix + 7] = old_weight[:, compact_prefix: compact_prefix + 4]
    elif old_dim == target_dim:
        new_weight.copy_(old_weight)
    else:
        raise ValueError(f"unsupported policy observation migration: {old_dim} -> {target_dim}")

    new_policy_state = dict(policy_state)
    new_policy_state["trunk.0.weight"] = new_weight
    ckpt["policy"] = new_policy_state
    old_mean_raw = ckpt.get("policy_state_mean", ckpt.get("state_mean"))
    old_std_raw = ckpt.get("policy_state_std", ckpt.get("state_std"))
    old_mean = np.asarray(old_mean_raw, dtype=np.float32).reshape(-1) if old_mean_raw is not None else np.zeros(old_dim, dtype=np.float32)
    old_std = np.asarray(old_std_raw, dtype=np.float32).reshape(-1) if old_std_raw is not None else default_policy_state_scale(old_dim)
    new_mean = np.zeros(target_dim, dtype=np.float32)
    new_std = default_policy_state_scale(target_dim)
    if old_dim == compact_prefix + 4:
        new_mean[:compact_prefix] = old_mean[:compact_prefix]
        new_std[:compact_prefix] = old_std[:compact_prefix]
        new_mean[compact_prefix + 3: compact_prefix + 7] = old_mean[compact_prefix: compact_prefix + 4]
        new_std[compact_prefix + 3: compact_prefix + 7] = old_std[compact_prefix: compact_prefix + 4]
    else:
        copied = min(old_dim, target_dim, old_mean.shape[0], old_std.shape[0])
        new_mean[:copied] = old_mean[:copied]
        new_std[:copied] = old_std[:copied]
    ckpt["policy_state_mean"] = new_mean
    ckpt["policy_state_std"] = new_std
    ckpt["policy_input_semantics"] = target_semantics
    ckpt.pop("optimizer_pi", None)
    migration = {
        "applied": True,
        "old_dim": old_dim,
        "new_dim": target_dim,
        "old_semantics": old_semantics,
        "new_semantics": target_semantics,
        "new_columns_initialized_to_zero": 3,
    }
    ckpt["policy_observation_migration"] = migration
    return ckpt, migration


def state_from_status(status: Dict[str, Any], dim: int) -> Optional[np.ndarray]:
    raw = status.get("state_vector")
    if not isinstance(raw, list) or len(raw) < min(28, dim):
        return None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape[0] < dim:
        arr = np.pad(arr, (0, dim - arr.shape[0]), mode="constant")
    elif arr.shape[0] > dim:
        arr = arr[:dim]
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def action_dict(status: Dict[str, Any], key: str) -> np.ndarray:
    raw = status.get(key) or {}
    intervention = as_float(
        raw.get("intervention"),
        as_float(status.get("rl_intervention_strength"), 0.0) if key == "chosen_action" else 0.0,
    )
    return np.asarray([
        np.clip(as_float(raw.get("steer")), -1.0, 1.0),
        np.clip(as_float(raw.get("throttle")), 0.0, 1.0),
        np.clip(as_float(raw.get("brake")), 0.0, 1.0),
        np.clip(intervention, 0.0, 1.0),
    ], dtype=np.float32)


def map_invariant_policy_state(
    observation: np.ndarray,
    world_state_dim: int,
) -> np.ndarray:
    """Remove CARLA's global pose from an otherwise route-relative observation."""
    result = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
    for index in MAP_INVARIANT_WORLD_STATE_INDICES:
        if index < min(world_state_dim, result.shape[0]):
            result[index] = 0.0
    return result


def policy_state_from_status(
    status: Dict[str, Any],
    policy_dim: int,
    world_state_dim: int,
    policy_input_semantics: str = "",
) -> Optional[np.ndarray]:
    recorded = status.get("policy_state_vector")
    if isinstance(recorded, list) and len(recorded) >= policy_dim:
        observation = np.asarray(recorded[:policy_dim], dtype=np.float32).copy()
        # Legacy 49-D traces persisted the old velocity-coupled oncoming
        # representation. enrich_current_oncoming() repairs the structured
        # status first, so patch every temporal traffic slot from that source
        # instead of silently reusing the stale recorded vector.
        if policy_dim in (42, 46, 49):
            left_clear = as_float(status.get("left_clear_m"), 80.0)
            right_clear = as_float(status.get("right_clear_m"), 80.0)
            observation[31:42] = np.asarray([
                as_float(status.get("blocked_ticks"), 0.0),
                min(
                    as_float(status.get("left_front_m"), left_clear),
                    as_float(status.get("left_rear_m"), left_clear),
                ),
                min(
                    as_float(status.get("right_front_m"), right_clear),
                    as_float(status.get("right_rear_m"), right_clear),
                ),
                as_float(status.get("left_ttc_s"), 99.0),
                as_float(status.get("right_ttc_s"), 99.0),
                as_float(status.get("left_oncoming_m"), 80.0),
                as_float(status.get("right_oncoming_m"), 80.0),
                as_float(status.get("left_oncoming_ttc_s"), 99.0),
                as_float(status.get("right_oncoming_ttc_s"), 99.0),
                1.0 if bool(status.get("left_lane_available", True)) else 0.0,
                1.0 if bool(status.get("right_lane_available", True)) else 0.0,
            ], dtype=np.float32)
        if policy_dim == 49:
            observation[42:45] = np.asarray([
                as_float(status.get("current_oncoming_distance_m"), 80.0),
                as_float(status.get("current_oncoming_closing_speed_mps"), 0.0),
                as_float(status.get("current_oncoming_ttc_s"), 99.0),
            ], dtype=np.float32)
    else:
        state = state_from_status(status, world_state_dim)
        if state is None:
            return None
        base = action_dict(status, "base_action")[:3]
        left_clear = as_float(status.get("left_clear_m"), 80.0)
        right_clear = as_float(status.get("right_clear_m"), 80.0)
        if policy_dim in (42, 46, 49):
            context = np.asarray([
                as_float(status.get("blocked_ticks"), 0.0),
                min(as_float(status.get("left_front_m"), left_clear), as_float(status.get("left_rear_m"), left_clear)),
                min(as_float(status.get("right_front_m"), right_clear), as_float(status.get("right_rear_m"), right_clear)),
                as_float(status.get("left_ttc_s"), 99.0),
                as_float(status.get("right_ttc_s"), 99.0),
                as_float(status.get("left_oncoming_m"), 80.0),
                as_float(status.get("right_oncoming_m"), 80.0),
                as_float(status.get("left_oncoming_ttc_s"), 99.0),
                as_float(status.get("right_oncoming_ttc_s"), 99.0),
                1.0 if bool(status.get("left_lane_available", True)) else 0.0,
                1.0 if bool(status.get("right_lane_available", True)) else 0.0,
            ], dtype=np.float32)
        else:
            context = np.asarray([
                as_float(status.get("blocked_ticks"), 0.0),
                as_float(status.get("left_front_m"), left_clear),
                as_float(status.get("left_rear_m"), left_clear),
                as_float(status.get("right_front_m"), right_clear),
                as_float(status.get("right_rear_m"), right_clear),
                as_float(status.get("left_ttc_s"), 99.0),
                as_float(status.get("right_ttc_s"), 99.0),
                as_float(status.get("left_oncoming_m"), 80.0),
                as_float(status.get("right_oncoming_m"), 80.0),
                as_float(status.get("left_oncoming_ttc_s"), 99.0),
                as_float(status.get("right_oncoming_ttc_s"), 99.0),
                1.0 if bool(status.get("left_lane_available", True)) else 0.0,
                1.0 if bool(status.get("right_lane_available", True)) else 0.0,
            ], dtype=np.float32)
        parts = [state, base, context]
        if policy_dim == 49:
            parts.append(np.asarray([
                as_float(status.get("current_oncoming_distance_m"), 80.0),
                as_float(status.get("current_oncoming_closing_speed_mps"), 0.0),
                as_float(status.get("current_oncoming_ttc_s"), 99.0),
            ], dtype=np.float32))
        if policy_dim in (46, 49):
            previous = status.get("rl_previous_policy_action")
            if not isinstance(previous, list) or len(previous) < 4:
                previous = [float(base[0]), float(base[1]), float(base[2]), 0.0]
            parts.append(np.asarray(previous[:4], dtype=np.float32))
        observation = np.concatenate(parts)
        if observation.shape[0] < policy_dim:
            observation = np.pad(observation, (0, policy_dim - observation.shape[0]), mode="constant")
        elif observation.shape[0] > policy_dim:
            observation = observation[:policy_dim]
    if policy_input_semantics in (
        MAP_INVARIANT_POLICY_INPUT_SEMANTICS,
        MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
    ):
        observation = map_invariant_policy_state(observation, world_state_dim)
    if not np.all(np.isfinite(observation)):
        return None
    return observation


def load_metrics(raw: str) -> Dict[str, float]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        path = Path(raw).expanduser()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {str(k): as_float(v) for k, v in data.items() if isinstance(v, (int, float, str, bool))}


def infer_models(ckpt: Dict[str, Any], device: torch.device) -> Tuple[ActorCritic, WorldModel]:
    policy_state = ckpt["policy"]
    world_state = ckpt["world_model"]
    policy_hidden = int(policy_state["trunk.0.weight"].shape[0])
    policy_state_dim = int(policy_state["trunk.0.weight"].shape[1])
    action_dim = int(policy_state["log_std"].shape[0])
    wm_hidden = int(world_state["fc1.weight"].shape[0])
    wm_state_dim = int(world_state["fc1.weight"].shape[1]) - action_dim

    policy = ActorCritic(policy_state_dim, action_dim, policy_hidden).to(device)
    world_model = WorldModel(wm_state_dim, action_dim, wm_hidden).to(device)
    policy.load_state_dict(policy_state)
    world_model.load_state_dict(world_state)
    policy.train()
    world_model.train()
    return policy, world_model


def default_policy_state_scale(dim: int) -> np.ndarray:
    scale = np.ones(max(dim, 49), dtype=np.float32)
    scale[0] = 1000.0
    scale[1] = 1000.0
    scale[2] = 15.0
    scale[3] = math.pi
    scale[4] = 8.0
    scale[6] = 8.0
    scale[8] = math.pi
    scale[10] = 2.0
    scale[11] = 50.0
    for idx in (13, 16, 18, 21, 23, 26):
        scale[idx] = 80.0
    for idx in (14, 19, 20, 24, 25):
        scale[idx] = 20.0
    if dim in (42, 46, 49):
        scale[31] = 200.0
        for idx in (32, 33, 36, 37):
            scale[idx] = 80.0
        for idx in (34, 35, 38, 39):
            scale[idx] = 20.0
    elif dim >= 44:
        scale[31] = 200.0
        for idx in (32, 33, 34, 35, 38, 39):
            scale[idx] = 80.0
        for idx in (36, 37, 40, 41):
            scale[idx] = 20.0
    if dim >= 49:
        scale[42] = 80.0
        scale[43] = 20.0
        scale[44] = 20.0
    return scale[:dim]


def policy_normalizer_from_checkpoint(
    ckpt: Dict[str, Any],
    state_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_raw = ckpt.get("policy_state_mean", ckpt.get("state_mean"))
    std_raw = ckpt.get("policy_state_std", ckpt.get("state_std"))
    if mean_raw is None or std_raw is None:
        mean = np.zeros(state_dim, dtype=np.float32)
        std = default_policy_state_scale(state_dim)
    else:
        mean = np.asarray(mean_raw, dtype=np.float32).reshape(-1)
        std = np.asarray(std_raw, dtype=np.float32).reshape(-1)
        if mean.shape[0] < state_dim:
            mean = np.pad(mean, (0, state_dim - mean.shape[0]), mode="constant")
        if std.shape[0] < state_dim:
            std = np.pad(std, (0, state_dim - std.shape[0]), mode="constant", constant_values=1.0)
        mean = mean[:state_dim]
        std = std[:state_dim]
    return (
        torch.as_tensor(mean, dtype=torch.float32, device=device),
        torch.as_tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6),
    )


def side_from_action(base: np.ndarray, chosen: np.ndarray) -> int:
    delta = float(chosen[0] - base[0])
    if delta < -0.04:
        return -1
    if delta > 0.04:
        return 1
    return 0


def status_side_values(status: Dict[str, Any], side: int) -> Dict[str, float]:
    prefix = "left" if side < 0 else "right"
    return {
        "front": as_float(status.get(f"{prefix}_front_m"), 80.0),
        "rear": as_float(status.get(f"{prefix}_rear_m"), 80.0),
        "clear": as_float(status.get(f"{prefix}_clear_m"), 80.0),
        "ttc": as_float(status.get(f"{prefix}_ttc_s"), 99.0),
        "oncoming": as_float(status.get(f"{prefix}_oncoming_m"), 80.0),
        "oncoming_ttc": as_float(status.get(f"{prefix}_oncoming_ttc_s"), 99.0),
        "lane": 1.0 if bool(status.get(f"{prefix}_lane_available", True)) else 0.0,
    }


def current_oncoming_values(status: Dict[str, Any]) -> Dict[str, float]:
    return {
        "distance": as_float(
            status.get("current_oncoming_distance_m"),
            as_float(status.get("current_oncoming_distance"), 80.0),
        ),
        "closing_speed": as_float(status.get("current_oncoming_closing_speed_mps"), 0.0),
        "ttc": as_float(status.get("current_oncoming_ttc_s"), 99.0),
    }


def oncoming_ttc_severity(distance_m: float, ttc_s: float) -> float:
    """Continuous 0..1 danger signal, active before the hard collision zone."""
    if distance_m >= 79.5 and ttc_s >= 98.0:
        return 0.0
    ttc_component = float(np.clip((6.0 - ttc_s) / 5.0, 0.0, 1.0))
    distance_component = 0.55 * float(np.clip((42.0 - distance_m) / 34.0, 0.0, 1.0))
    return max(ttc_component, distance_component)


def progressive_oncoming_engagement_penalty(
    distance_m: float,
    ttc_s: float,
    commitment: float,
) -> float:
    severity = oncoming_ttc_severity(distance_m, ttc_s)
    commitment = float(np.clip(commitment, 0.0, 1.0))
    penalty = -(0.08 * severity + 1.80 * severity * severity) * commitment
    if ttc_s < 1.25:
        penalty -= 2.00 * commitment
    elif ttc_s < 2.0:
        penalty -= 0.80 * commitment
    return float(penalty)


def lane_signature(status: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    if "ego_lane_id" not in status:
        return None
    return (
        int(as_float(status.get("ego_road_id"), 0.0)),
        int(as_float(status.get("ego_lane_id"), 0.0)),
    )


def tracked_vehicle(status: Dict[str, Any], actor_id: int) -> Optional[Dict[str, float]]:
    if actor_id < 0:
        return None
    rows = status.get("nearby_vehicles")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict) or int(as_float(row.get("id"), -1.0)) != actor_id:
            continue
        return {
            "forward": as_float(row.get("forward_m"), 80.0),
            "lateral": as_float(row.get("lateral_m"), 0.0),
            "distance": as_float(row.get("distance_m"), 80.0),
            "clearance": as_float(row.get("longitudinal_clearance_m"), 0.0),
        }
    return None


def side_is_safe_for_overtake(status: Dict[str, Any], side: int) -> bool:
    vals = status_side_values(status, side)
    return bool(
        vals["lane"] >= 0.5
        and vals["front"] >= 6.0
        and vals["rear"] >= 5.5
        and vals["ttc"] >= 1.8
        and vals["oncoming"] >= 28.0
        and vals["oncoming_ttc"] >= 3.5
    )


def overtake_escape_available(status: Dict[str, Any]) -> bool:
    """Return whether waiting behind the obstacle is currently avoidable.

    This is reward context only. It never filters or changes the policy action.
    A red/yellow light, a nearby VRU, or unsafe adjacent/oncoming traffic makes
    waiting legitimate, so stationary time must not teach the policy to force a
    dangerous overtake.
    """
    traffic_light = str(status.get("traffic_light", "none")).lower()
    nearest_vru = min(
        as_float(status.get("nearest_walker_m"), 80.0),
        as_float(status.get("nearest_bike_m"), 80.0),
    )
    if traffic_light in ("red", "yellow") or nearest_vru <= 10.0:
        return False
    return side_is_safe_for_overtake(status, -1) or side_is_safe_for_overtake(status, 1)


@dataclass
class OvertakeRewardTracker:
    """Episode-local reward state; it never influences runtime control."""

    phase: str = "idle"
    origin_road_id: int = 0
    origin_lane_id: int = 0
    obstacle_id: int = -1
    departure_side: int = 0
    departed: bool = False
    passed: bool = False
    stable_reentry_steps: int = 0
    early_reentry_reported: bool = False
    last_obstacle_clearance_m: float = 0.0
    obstacle_missing_steps: int = 0
    cooldown_steps: int = 0

    def _reset(self, cooldown_steps: int = 0) -> None:
        self.phase = "idle"
        self.origin_road_id = 0
        self.origin_lane_id = 0
        self.obstacle_id = -1
        self.departure_side = 0
        self.departed = False
        self.passed = False
        self.stable_reentry_steps = 0
        self.early_reentry_reported = False
        self.last_obstacle_clearance_m = 0.0
        self.obstacle_missing_steps = 0
        self.cooldown_steps = max(0, int(cooldown_steps))

    def _same_origin_lane(self, status: Dict[str, Any]) -> bool:
        lane = lane_signature(status)
        if lane is None:
            return False
        # Lane IDs encode the driving lane across short road-id changes. Using
        # road_id as a hard requirement would break legitimate re-entry near a
        # CARLA road segment boundary.
        return lane[1] == self.origin_lane_id

    @staticmethod
    def _centered_in_lane(status: Dict[str, Any]) -> bool:
        width = max(2.4, as_float(status.get("ego_lane_width_m"), 3.5))
        offset = abs(as_float(status.get("ego_lane_center_offset_m"), 0.0))
        return offset <= min(0.90, 0.28 * width)

    @staticmethod
    def _departure_detected(status: Dict[str, Any], origin_lane_id: int) -> bool:
        lane = lane_signature(status)
        if lane is None:
            return False
        width = max(2.4, as_float(status.get("ego_lane_width_m"), 3.5))
        offset = abs(as_float(status.get("ego_lane_center_offset_m"), 0.0))
        return lane[1] != origin_lane_id or offset >= max(1.0, 0.34 * width)

    @staticmethod
    def _safe_reentry_gap(speed_mps: float) -> float:
        # Bumper-to-bumper clearance: at least 3 m and roughly 0.8 s at speed.
        return max(3.0, 0.80 * max(0.0, speed_mps))

    def observe(
        self,
        status: Dict[str, Any],
        next_status: Dict[str, Any],
        action_side: int,
        speed_mps: float,
    ) -> Dict[str, float]:
        reward = {
            "clean_pass": 0.0,
            "early_reentry": 0.0,
            "unsafe_reentry": 0.0,
            "overtake_phase": 0.0,
        }
        if self.cooldown_steps > 0:
            self.cooldown_steps -= 1
            return reward

        current_lane = lane_signature(status)
        front = as_float(status.get("front_vehicle_m"), 80.0)
        obstacle_id = int(as_float(status.get("front_vehicle_id"), -1.0))
        base = action_dict(status, "base_action")
        nearest_vru = min(
            as_float(status.get("nearest_walker_m"), 80.0),
            as_float(status.get("nearest_bike_m"), 80.0),
        )
        traffic_light = str(status.get("traffic_light", "none")).lower()

        if self.phase == "idle":
            hazard_cue = bool(
                current_lane is not None
                and current_lane[1] != 0
                and obstacle_id >= 0
                and front <= 16.0
                and nearest_vru > 8.0
                and traffic_light not in ("red", "yellow")
                and (
                    as_float(status.get("blocked_ticks"), 0.0) >= 20.0
                    or base[2] >= 0.55
                    or speed_mps <= 1.0
                )
            )
            if not hazard_cue:
                return reward
            self.phase = "approach"
            self.origin_road_id, self.origin_lane_id = current_lane
            self.obstacle_id = obstacle_id
            self.departure_side = action_side

        if self.departure_side == 0 and action_side != 0:
            self.departure_side = action_side

        obstacle = tracked_vehicle(next_status, self.obstacle_id)
        if obstacle is None:
            self.obstacle_missing_steps += 1
        else:
            self.obstacle_missing_steps = 0
            self.last_obstacle_clearance_m = max(
                self.last_obstacle_clearance_m,
                obstacle["clearance"] if obstacle["forward"] < 0.0 else 0.0,
            )

        if not self.departed and self._departure_detected(next_status, self.origin_lane_id):
            self.departed = True
            self.phase = "passing"
            reward["overtake_phase"] += 0.35

        if self.departed and not self.passed:
            if obstacle is not None and obstacle["forward"] <= -0.5:
                self.passed = True
                self.phase = "passed"
                self.last_obstacle_clearance_m = max(
                    self.last_obstacle_clearance_m,
                    obstacle["clearance"],
                )
                reward["overtake_phase"] += 0.75
            elif obstacle is None and self.obstacle_missing_steps >= 8:
                # Exact actor telemetry can disappear outside the 80 m trace
                # window only after a sustained pass. Never infer this from a
                # one-frame front-distance jump.
                self.passed = True
                self.phase = "passed"

        in_origin_lane = self._same_origin_lane(next_status)
        centered = self._centered_in_lane(next_status)
        if self.departed and in_origin_lane and centered:
            gap_required = self._safe_reentry_gap(speed_mps)
            obstacle = tracked_vehicle(next_status, self.obstacle_id)
            if obstacle is not None and obstacle["forward"] < 0.0:
                rear_clearance = obstacle["clearance"]
                self.last_obstacle_clearance_m = max(self.last_obstacle_clearance_m, rear_clearance)
            elif self.passed and self.obstacle_missing_steps >= 8:
                rear_clearance = 80.0
            else:
                rear_clearance = self.last_obstacle_clearance_m

            if not self.passed:
                self.stable_reentry_steps = 0
                if not self.early_reentry_reported:
                    reward["early_reentry"] -= 4.0
                    self.early_reentry_reported = True
                self.phase = "early_reentry"
            elif rear_clearance < gap_required:
                self.stable_reentry_steps = 0
                severity = float(np.clip((gap_required - rear_clearance) / gap_required, 0.0, 1.0))
                reward["unsafe_reentry"] -= 0.12 + 0.18 * severity
                if not self.early_reentry_reported:
                    reward["early_reentry"] -= 4.0 + 8.0 * severity
                    self.early_reentry_reported = True
                self.phase = "early_reentry"
            else:
                self.phase = "stabilizing"
                self.stable_reentry_steps += 1
                reward["overtake_phase"] += 0.04
                if self.stable_reentry_steps >= 20:
                    reward["clean_pass"] += 8.0
                    self._reset(cooldown_steps=40)
        elif self.departed:
            self.stable_reentry_steps = 0

        if self.phase == "approach" and front > 24.0 and self.obstacle_missing_steps >= 8:
            self._reset(cooldown_steps=20)
        return reward

    def finalize(self) -> Dict[str, float]:
        if self.phase in ("passing", "passed", "early_reentry", "stabilizing"):
            return {"incomplete_overtake": -8.0}
        return {"incomplete_overtake": 0.0}


def step_reward(
    status: Dict[str, Any],
    next_status: Dict[str, Any],
    prev_action: np.ndarray,
    stagnant_steps: int = 0,
    overtake_tracker: Optional[OvertakeRewardTracker] = None,
) -> Tuple[float, Dict[str, float]]:
    state = np.asarray(status.get("state_vector") or [], dtype=np.float32)
    nxt = np.asarray(next_status.get("state_vector") or [], dtype=np.float32)
    if state.shape[0] < 3 or nxt.shape[0] < 3:
        return 0.0, {}

    base = action_dict(status, "base_action")
    chosen = action_dict(status, "chosen_action")
    side = side_from_action(base, chosen)
    speed = float(max(0.0, nxt[2]))
    dx = float(nxt[0] - state[0])
    dy = float(nxt[1] - state[1])
    distance = math.sqrt(dx * dx + dy * dy)
    front = as_float(status.get("front_vehicle_m"), 80.0)
    walker = as_float(status.get("nearest_walker_m"), as_float(status.get("state_vector", [80.0] * 28)[18] if isinstance(status.get("state_vector"), list) and len(status.get("state_vector")) > 18 else 80.0, 80.0))
    next_walker = as_float(next_status.get("nearest_walker_m"), as_float(next_status.get("state_vector", [80.0] * 28)[18] if isinstance(next_status.get("state_vector"), list) and len(next_status.get("state_vector")) > 18 else 80.0, 80.0))
    bike = as_float(status.get("nearest_bike_m"), as_float(status.get("state_vector", [80.0] * 28)[23] if isinstance(status.get("state_vector"), list) and len(status.get("state_vector")) > 23 else 80.0, 80.0))
    next_bike = as_float(next_status.get("nearest_bike_m"), as_float(next_status.get("state_vector", [80.0] * 28)[23] if isinstance(next_status.get("state_vector"), list) and len(next_status.get("state_vector")) > 23 else 80.0, 80.0))
    blocked = as_float(status.get("blocked_ticks"), 0.0)
    chosen_risk = as_float(status.get("chosen_risk"), 0.0)
    intervention = float(np.clip(as_float(status.get("rl_intervention_strength"), chosen[3]), 0.0, 1.0))
    selected_side = status_side_values(status, side) if side != 0 else None
    safe_overtake_context = bool(side != 0 and side_is_safe_for_overtake(status, side))
    raw = status.get("rl_raw_action")
    raw_saturation = 0.0
    if isinstance(raw, list) and raw:
        finite_raw = [abs(as_float(v, 0.0)) for v in raw[:4] if math.isfinite(as_float(v, 0.0))]
        if finite_raw:
            max_raw = max(finite_raw)
            if max_raw > 20.0:
                raw_saturation -= min(12.0, 0.04 * (max_raw - 20.0))
            if max_raw > 100.0:
                raw_saturation -= 8.0

    progress = 0.18 * min(distance, 8.0) + 0.015 * min(speed, 12.0)
    action_delta = np.abs(chosen[:3] - prev_action[:3])
    residual_delta = np.abs(chosen[:3] - base[:3])
    comfort = -0.06 * float(action_delta.sum())
    smoothness = (
        -0.18 * float(action_delta[0] ** 2)
        -0.10 * float((action_delta[1] + action_delta[2]) ** 2)
        -0.04 * float(abs(chosen[0]))
    )
    residual_cost = -0.025 * float(residual_delta.sum())
    intervention_cost = -0.025 * intervention
    unnecessary_intervention = 0.0
    nearest_vru = min(walker, bike)
    if (
        front > 24.0
        and nearest_vru > 24.0
        and str(status.get("traffic_light", "none")).lower() not in ("red", "yellow")
        and chosen_risk < 0.45
    ):
        unnecessary_intervention = -0.045 * intervention
    brake_throttle_conflict = -0.45 if chosen[1] > 0.15 and chosen[2] > 0.15 else 0.0
    # Penalize waiting only when a legal and safe escape is already open.
    # In particular, oncoming traffic must never make the policy impatient.
    escape_available = overtake_escape_available(status)
    stuck = 0.0
    tailgate = -1.5 if front < 4.0 and chosen[1] > 0.20 else 0.0
    risk_penalty = -0.25 * max(0.0, chosen_risk - 0.55)
    high_risk_override = 0.0
    if (
        not safe_overtake_context
        and chosen_risk > 0.85
        and chosen[1] > 0.25
        and chosen[2] < 0.20
    ):
        high_risk_override -= 2.5
    if (
        not safe_overtake_context
        and
        base[2] > 0.65
        and chosen[1] > 0.25
        and chosen[2] < 0.20
        and (front < 18.0 or min(as_float(status.get("left_clear_m"), 80.0), as_float(status.get("right_clear_m"), 80.0)) < 6.0 or chosen_risk > 0.75)
    ):
        high_risk_override -= 4.0

    overtake_attempt = 0.0
    unsafe_side = 0.0
    oncoming_engagement = 0.0
    clean_pass = 0.0
    early_reentry = 0.0
    unsafe_reentry = 0.0
    overtake_phase = 0.0
    if side != 0:
        vals = selected_side
        assert vals is not None
        if vals["lane"] < 0.5:
            unsafe_side -= 2.5
        if vals["clear"] < 2.0:
            unsafe_side -= 5.0
        elif vals["clear"] < 3.0:
            unsafe_side -= 2.0
        if vals["front"] < 3.5:
            unsafe_side -= 2.2
        elif vals["front"] < 6.0:
            unsafe_side -= 0.8
        if vals["rear"] < 3.0:
            unsafe_side -= 2.8
        elif vals["rear"] < 5.5:
            unsafe_side -= 0.8
        if vals["ttc"] < 1.0:
            unsafe_side -= 3.0
        elif vals["ttc"] < 1.8:
            unsafe_side -= 1.0
        steer_commitment = float(np.clip(abs(chosen[0] - base[0]) / 0.25, 0.0, 1.0))
        drive_commitment = float(np.clip(chosen[1] + 0.50 * (1.0 - chosen[2]), 0.0, 1.0))
        side_commitment = max(0.35 * intervention, steer_commitment) * drive_commitment
        oncoming_engagement += progressive_oncoming_engagement_penalty(
            vals["oncoming"],
            vals["oncoming_ttc"],
            side_commitment,
        )

        safe_side = safe_overtake_context
        base_braking = base[2] > 0.70 and base[1] < 0.05
        release_brake = chosen[2] < 0.35
        add_throttle = chosen[1] > 0.10
        if front < 14.0 and safe_side:
            if distance > 0.10 and add_throttle and release_brake:
                overtake_attempt += 0.12
            if base_braking and add_throttle and release_brake:
                overtake_attempt += 0.10
            elif base_braking and chosen[2] >= 0.35 and speed < 0.5:
                stuck -= 0.04
        if front < 18.0 and safe_side and distance > 0.20:
            overtake_attempt += 0.08
    elif front < 10.0 and blocked > 80.0 and escape_available:
        stuck -= 0.12

    current_oncoming = current_oncoming_values(status)
    current_commitment = float(np.clip(
        (0.30 + 0.70 * chosen[1]) * (1.0 - 0.80 * chosen[2]),
        0.08,
        1.0,
    ))
    oncoming_engagement += progressive_oncoming_engagement_penalty(
        current_oncoming["distance"],
        current_oncoming["ttc"],
        current_commitment,
    )

    if (
        stagnant_steps >= 30
        and speed < 0.20
        and front < 14.0
        and nearest_vru > 10.0
        and str(status.get("traffic_light", "none")).lower() not in ("red", "yellow")
    ):
        if escape_available:
            stuck -= min(0.12, 0.08 + 0.002 * float(stagnant_steps - 30))
            if intervention >= 0.20 and chosen[2] > 0.45 and chosen[1] < 0.10:
                stuck -= 0.04

    if overtake_tracker is not None:
        maneuver_reward = overtake_tracker.observe(
            status,
            next_status,
            action_side=side,
            speed_mps=speed,
        )
        clean_pass = maneuver_reward["clean_pass"]
        early_reentry = maneuver_reward["early_reentry"]
        unsafe_reentry = maneuver_reward["unsafe_reentry"]
        overtake_phase = maneuver_reward["overtake_phase"]

    vru_reward = 0.0
    next_nearest_vru = min(next_walker, next_bike)
    vru_closing = nearest_vru - next_nearest_vru
    vru_danger = nearest_vru < 14.0 or (nearest_vru < 24.0 and vru_closing > 1.0)
    if vru_danger:
        if chosen[2] >= 0.28 or chosen[1] <= 0.10:
            vru_reward += 1.2
        if nearest_vru < 10.0 and chosen[2] >= 0.45:
            vru_reward += 1.0
        if speed > 5.5 and chosen[2] < 0.20:
            vru_reward -= 2.5
        if chosen[1] > 0.35 and chosen[2] < 0.20:
            vru_reward -= 2.0
        if nearest_vru < 6.0 and chosen[2] < 0.35:
            vru_reward -= 4.0
        if nearest_vru < 3.0:
            vru_reward -= 8.0
    elif nearest_vru < 24.0 and chosen[1] > 0.65 and speed > 7.0:
        vru_reward -= 0.8

    total = (
        progress
        + comfort
        + smoothness
        + residual_cost
        + intervention_cost
        + unnecessary_intervention
        + brake_throttle_conflict
        + stuck
        + tailgate
        + risk_penalty
        + high_risk_override
        + unsafe_side
        + oncoming_engagement
        + overtake_attempt
        + overtake_phase
        + clean_pass
        + early_reentry
        + unsafe_reentry
        + vru_reward
        + raw_saturation
    )
    parts = {
        "progress": progress,
        "comfort": comfort,
        "smoothness": smoothness,
        "residual_cost": residual_cost,
        "intervention_cost": intervention_cost,
        "unnecessary_intervention": unnecessary_intervention,
        "brake_throttle_conflict": brake_throttle_conflict,
        "stuck": stuck,
        "tailgate": tailgate,
        "risk": risk_penalty,
        "high_risk_override": high_risk_override,
        "unsafe_side": unsafe_side,
        "oncoming_engagement": oncoming_engagement,
        "overtake_attempt": overtake_attempt,
        "overtake_phase": overtake_phase,
        "clean_pass": clean_pass,
        "early_reentry": early_reentry,
        "unsafe_reentry": unsafe_reentry,
        "vru": vru_reward,
        "raw_saturation": raw_saturation,
    }
    return float(total), parts


def terminal_reward(
    metrics: Dict[str, float],
    exclude_collisions: bool = False,
) -> Tuple[float, Dict[str, float]]:
    if not metrics:
        return 0.0, {}
    route = as_float(metrics.get("route_score"), as_float(metrics.get("route_completion"), 0.0))
    driving = as_float(metrics.get("driving_score"), 0.0)
    collisions = as_float(metrics.get("collisions"), 0.0)
    pedestrian_collisions = as_float(metrics.get("pedestrian_collisions"), 0.0)
    vehicle_collisions = as_float(metrics.get("vehicle_collisions"), 0.0)
    layout_collisions = as_float(metrics.get("layout_collisions"), 0.0)
    offroad = as_float(metrics.get("offroad"), 0.0)
    red_lights = as_float(metrics.get("red_lights"), 0.0)
    blocked = as_float(metrics.get("blocked"), 0.0)
    min_speed = as_float(metrics.get("min_speed_infractions"), 0.0)
    incomplete = as_float(metrics.get("incomplete"), 0.0)
    bonus = 0.05 * route + 0.025 * driving
    clean = 8.0 if (
        route >= 95.0
        and collisions == 0
        and offroad == 0
        and red_lights == 0
        and blocked == 0
    ) else 0.0
    # Bench2Drive's min-speed list mixes context-free slow and fast relative
    # traffic observations. Trace-level rewards can distinguish avoidable
    # hesitation from justified waiting, so keep this aggregate diagnostic only.
    min_speed_penalty = 0.0
    collision_penalties = (
        -30.0 * collisions
        -45.0 * pedestrian_collisions
        -28.0 * vehicle_collisions
        -18.0 * layout_collisions
    )
    if exclude_collisions:
        collision_penalties = 0.0
    penalties = (
        -60.0 * incomplete
        + collision_penalties
        -14.0 * offroad
        -8.0 * red_lights
        -10.0 * blocked
        + min_speed_penalty
    )
    total = bonus + clean + penalties
    return float(total), {
        "terminal_route": 0.05 * route,
        "terminal_driving": 0.025 * driving,
        "terminal_clean": clean,
        "terminal_min_speed": min_speed_penalty,
        "terminal_min_speed_observed": min_speed,
        "terminal_penalty": penalties,
        "terminal_incomplete": -60.0 * incomplete,
        "terminal_collision_penalty": collision_penalties,
    }


def impact_collision_penalty(collision_event: Dict[str, Any]) -> float:
    kind = str(collision_event.get("collision_kind", "unknown")).lower()
    if kind == "pedestrian":
        return -75.0
    if kind == "static":
        return -48.0
    return -58.0


def build_rollout(
    rows: List[Dict[str, Any]],
    policy_state_dim: int,
    world_state_dim: int,
    metrics: Dict[str, float],
    policy_input_semantics: str = "",
    collision_event: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    states: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    policy_states: List[np.ndarray] = []
    raw_actions: List[np.ndarray] = []
    chosen_actions: List[np.ndarray] = []
    rewards: List[float] = []
    values: List[float] = []
    log_probs: List[float] = []
    risk_targets: List[float] = []
    progress_targets: List[float] = []
    reward_parts: Dict[str, float] = {}
    prev_action = np.zeros(4, dtype=np.float32)
    intervention_strengths: List[float] = []
    stagnant_steps = 0
    overtake_tracker = OvertakeRewardTracker()

    for a, b in zip(rows, rows[1:]):
        status = a.get("status") or {}
        next_status = b.get("status") or {}
        state = state_from_status(status, world_state_dim)
        next_state = state_from_status(next_status, world_state_dim)
        policy_state = policy_state_from_status(
            status,
            policy_state_dim,
            world_state_dim,
            policy_input_semantics=policy_input_semantics,
        )
        raw = status.get("rl_raw_action")
        if state is None or next_state is None or policy_state is None or not isinstance(raw, list) or len(raw) < 4:
            continue
        raw_action = np.asarray(raw[:4], dtype=np.float32)
        if not np.all(np.isfinite(raw_action)):
            continue
        chosen = action_dict(status, "chosen_action")
        dx = float(next_state[0] - state[0])
        dy = float(next_state[1] - state[1])
        progress = float(max(0.0, min(math.sqrt(dx * dx + dy * dy), 20.0)))
        if progress < 0.03 and float(max(0.0, next_state[2])) < 0.20:
            stagnant_steps += 1
        else:
            stagnant_steps = 0
        reward, parts = step_reward(
            status,
            next_status,
            prev_action,
            stagnant_steps=stagnant_steps,
            overtake_tracker=overtake_tracker,
        )
        for key, value in parts.items():
            reward_parts[key] = reward_parts.get(key, 0.0) + float(value)
        side = side_from_action(action_dict(status, "base_action"), chosen)
        side_vals = status_side_values(status, side) if side else {"clear": 80.0, "ttc": 99.0, "oncoming": 80.0, "oncoming_ttc": 99.0}
        proximity_risk = max(
            0.0,
            (12.0 - min(as_float(status.get("front_vehicle_m"), 80.0), side_vals["clear"])) / 12.0,
            (4.0 - min(side_vals["ttc"], side_vals["oncoming_ttc"])) / 4.0,
            (30.0 - side_vals["oncoming"]) / 30.0,
            (6.0 - current_oncoming_values(status)["ttc"]) / 6.0,
            (35.0 - current_oncoming_values(status)["distance"]) / 35.0,
        )
        states.append(state)
        next_states.append(next_state)
        policy_states.append(policy_state)
        raw_actions.append(raw_action)
        chosen_actions.append(chosen)
        rewards.append(reward)
        values.append(as_float(status.get("rl_value"), 0.0))
        log_probs.append(as_float(status.get("rl_log_prob"), 0.0))
        risk_targets.append(float(np.clip(max(as_float(status.get("chosen_risk"), 0.0), proximity_risk), 0.0, 1.0)))
        progress_targets.append(progress)
        intervention_strengths.append(float(np.clip(as_float(status.get("rl_intervention_strength"), chosen[3]), 0.0, 1.0)))
        prev_action = chosen

    if rewards:
        maneuver_terminal = overtake_tracker.finalize()
        maneuver_terminal_total = float(sum(maneuver_terminal.values()))
        rewards[-1] += maneuver_terminal_total
        for key, value in maneuver_terminal.items():
            reward_parts[key] = reward_parts.get(key, 0.0) + float(value)
        if collision_event is not None:
            collision_penalty = impact_collision_penalty(collision_event)
            rewards[-1] += collision_penalty
            reward_parts["impact_collision"] = collision_penalty
        term, terminal_parts = terminal_reward(
            metrics,
            exclude_collisions=collision_event is not None,
        )
        rewards[-1] += term
        for key, value in terminal_parts.items():
            reward_parts[key] = reward_parts.get(key, 0.0) + float(value)

    return {
        "states": np.asarray(states, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "policy_states": np.asarray(policy_states, dtype=np.float32),
        "raw_actions": np.asarray(raw_actions, dtype=np.float32),
        "chosen_actions": np.asarray(chosen_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "risk_targets": np.asarray(risk_targets, dtype=np.float32),
        "progress_targets": np.asarray(progress_targets, dtype=np.float32),
        "intervention_strengths": np.asarray(intervention_strengths, dtype=np.float32),
        "reward_parts": reward_parts,
        "collision_event": collision_event or {},
    }


def catastrophic_rollout_reason(
    rollout: Dict[str, np.ndarray],
    min_reward_sum: float,
    max_unsafe_side_loss: float,
    max_stuck_loss: float,
) -> str:
    reward_sum = float(np.sum(rollout["rewards"]))
    parts = rollout.get("reward_parts") or {}
    unsafe_side = as_float(parts.get("unsafe_side"), 0.0)
    stuck = as_float(parts.get("stuck"), 0.0)
    if reward_sum < min_reward_sum:
        return f"catastrophic reward_sum {reward_sum:.3f} < {min_reward_sum:.3f}"
    if unsafe_side < max_unsafe_side_loss:
        return f"catastrophic unsafe_side {unsafe_side:.3f} < {max_unsafe_side_loss:.3f}"
    if stuck < max_stuck_loss:
        return f"catastrophic stuck {stuck:.3f} < {max_stuck_loss:.3f}"
    return ""


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    returns = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv.astype(np.float32), returns.astype(np.float32)


def minibatches(n: int, batch_size: int, rng: np.random.Generator):
    indices = np.arange(n)
    rng.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start:start + batch_size]


def discover_positive_replay_samples(
    replay_root: Optional[Path],
    current_trace: Path,
    policy_state_dim: int,
    world_state_dim: int,
    policy_input_semantics: str,
    policy_action_semantics: str,
    max_traces: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Build an auxiliary safe-overtake anchor from successful episodes.

    Stale trajectories are not inserted into PPO ratios. They are only used by
    a small supervised anchor loss, while the current failed episode remains
    the sole source of PPO advantages.
    """
    empty_states = np.empty((0, policy_state_dim), dtype=np.float32)
    empty_actions = np.empty((0, 4), dtype=np.float32)
    if replay_root is None or max_traces <= 0 or not replay_root.exists():
        return empty_states, empty_actions, []
    candidates: List[Tuple[float, Path, Dict[str, Any]]] = []
    for summary_path in replay_root.glob("webapp_*/update_summary.json"):
        trace_path = summary_path.parent / "trace.jsonl"
        if not trace_path.exists() or trace_path.resolve() == current_trace.resolve():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = summary.get("metrics") or {}
        successful = bool(
            as_float(metrics.get("route_score"), 0.0) >= 95.0
            and as_float(metrics.get("collisions"), 0.0) == 0.0
            and as_float(metrics.get("offroad"), 0.0) == 0.0
            and as_float(metrics.get("red_lights"), 0.0) == 0.0
        )
        if successful:
            candidates.append((summary_path.stat().st_mtime, trace_path, summary))
    candidates.sort(key=lambda item: item[0], reverse=True)

    replay_states: List[np.ndarray] = []
    replay_actions: List[np.ndarray] = []
    sources: List[Dict[str, Any]] = []
    for _, trace_path, summary in candidates:
        rows = read_jsonl(trace_path)
        if len(rows) < 2:
            continue
        first_status = rows[0].get("status") or {}
        trace_action_semantics = str(first_status.get("rl_action_semantics", ""))
        if trace_action_semantics and trace_action_semantics != policy_action_semantics:
            continue
        enrich_current_oncoming(rows)
        trace_states: List[np.ndarray] = []
        trace_actions: List[np.ndarray] = []
        for row, next_row in zip(rows, rows[1:]):
            status = row.get("status") or {}
            next_status = next_row.get("status") or {}
            raw = status.get("rl_raw_action")
            if not isinstance(raw, list) or len(raw) < 4:
                continue
            raw_action = np.asarray(raw[:4], dtype=np.float32)
            if not np.all(np.isfinite(raw_action)) or float(np.max(np.abs(raw_action))) > 8.0:
                continue
            base = action_dict(status, "base_action")
            chosen = action_dict(status, "chosen_action")
            side = side_from_action(base, chosen)
            if side == 0 or not side_is_safe_for_overtake(status, side):
                continue
            if as_float(status.get("front_vehicle_m"), 80.0) > 18.0:
                continue
            if current_oncoming_values(status)["ttc"] < 4.0:
                continue
            current_state = state_from_status(status, world_state_dim)
            following_state = state_from_status(next_status, world_state_dim)
            policy_state = policy_state_from_status(
                status,
                policy_state_dim,
                world_state_dim,
                policy_input_semantics=policy_input_semantics,
            )
            if current_state is None or following_state is None or policy_state is None:
                continue
            progress = math.hypot(
                float(following_state[0] - current_state[0]),
                float(following_state[1] - current_state[1]),
            )
            if progress < 0.02:
                continue
            trace_states.append(policy_state)
            trace_actions.append(raw_action)
        if not trace_states:
            continue
        if len(trace_states) > 192:
            sample_indices = np.linspace(0, len(trace_states) - 1, 192, dtype=int)
            trace_states = [trace_states[index] for index in sample_indices]
            trace_actions = [trace_actions[index] for index in sample_indices]
        replay_states.extend(trace_states)
        replay_actions.extend(trace_actions)
        sources.append({
            "trace": str(trace_path),
            "samples": len(trace_states),
            "route_score": as_float((summary.get("metrics") or {}).get("route_score"), 0.0),
        })
        if len(sources) >= max_traces:
            break
    if not replay_states:
        return empty_states, empty_actions, sources
    return (
        np.asarray(replay_states, dtype=np.float32),
        np.asarray(replay_actions, dtype=np.float32),
        sources,
    )


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as tmp:
        torch.save(payload, tmp.name)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", default="")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--collision-events", default="")
    parser.add_argument("--positive-replay-root", default="")
    parser.add_argument("--positive-replay-count", type=int, default=3)
    parser.add_argument("--positive-anchor-coef", type=float, default=0.03)
    parser.add_argument("--collision-epoch-cap", type=int, default=2)
    parser.add_argument("--collision-lr-scale", type=float, default=0.5)
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--lr-world-model", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--min-transitions", type=int, default=64)
    parser.add_argument("--min-save-reward-sum", type=float, default=-250.0)
    parser.add_argument("--max-save-unsafe-side-loss", type=float, default=-100.0)
    parser.add_argument("--max-save-stuck-loss", type=float, default=-60.0)
    parser.add_argument(
        "--learn-from-failures",
        action="store_true",
        help="Apply clipped PPO updates from failed/collision episodes instead of discarding their penalties.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    trace_path = Path(args.trace).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_checkpoint = Path(args.output_checkpoint).expanduser().resolve() if args.output_checkpoint else checkpoint_path
    metrics = load_metrics(args.metrics_json)

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt, observation_migration = upgrade_policy_observation_checkpoint(ckpt)
    policy, world_model = infer_models(ckpt, device)
    policy_state_dim = policy.state_dim
    world_state_dim = world_model.state_dim
    policy_state_mean, policy_state_std = policy_normalizer_from_checkpoint(
        ckpt,
        policy_state_dim,
        device,
    )
    rows = read_jsonl(trace_path)
    original_row_count = len(rows)
    enrich_current_oncoming(rows)
    collision_events_path = Path(args.collision_events).expanduser().resolve() if args.collision_events else None
    collision_events = read_collision_events(collision_events_path)
    rows, collision_event = truncate_rows_at_first_collision(rows, collision_events, metrics)
    negative_example = None
    if collision_event is not None:
        negative_example = {
            "label": "negative_collision_episode",
            "trace": str(trace_path),
            "collision_event": collision_event,
            "original_rows": original_row_count,
            "truncated_rows": len(rows),
            "post_impact_rows_excluded": max(0, original_row_count - len(rows)),
            "metrics": metrics,
            "no_guard": True,
            "complement_to_simlingo": True,
        }
        (trace_path.parent / "negative_example.json").write_text(
            json.dumps(negative_example, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rollout = build_rollout(
        rows,
        policy_state_dim,
        world_state_dim,
        metrics,
        policy_input_semantics=str(ckpt.get("policy_input_semantics", "world_state_v1")),
        collision_event=collision_event,
    )
    n = int(rollout["states"].shape[0])
    if n < args.min_transitions:
        summary = {
            "status": "skipped",
            "reason": f"not enough transitions: {n} < {args.min_transitions}",
            "trace": str(trace_path),
            "checkpoint": str(checkpoint_path),
            "transitions": n,
            "metrics": metrics,
            "collision_event": collision_event,
            "original_rows": original_row_count,
            "truncated_rows": len(rows),
            "policy_observation_migration": observation_migration,
        }
        if args.summary:
            Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return 2

    catastrophic_reason = catastrophic_rollout_reason(
        rollout,
        min_reward_sum=args.min_save_reward_sum,
        max_unsafe_side_loss=args.max_save_unsafe_side_loss,
        max_stuck_loss=args.max_save_stuck_loss,
    )
    if catastrophic_reason and not args.learn_from_failures:
        summary = {
            "status": "skipped_bad_rollout",
            "reason": catastrophic_reason,
            "trace": str(trace_path),
            "checkpoint": str(checkpoint_path),
            "output_checkpoint": str(output_checkpoint),
            "transitions": n,
            "reward_mean": float(np.mean(rollout["rewards"])),
            "reward_sum": float(np.sum(rollout["rewards"])),
            "reward_parts": rollout["reward_parts"],
            "metrics": metrics,
            "checkpoint_saved": False,
            "collision_event": collision_event,
            "original_rows": original_row_count,
            "truncated_rows": len(rows),
            "policy_observation_migration": observation_migration,
        }
        if args.summary:
            Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return 2

    positive_states, positive_actions, positive_sources = discover_positive_replay_samples(
        Path(args.positive_replay_root).expanduser().resolve() if args.positive_replay_root else None,
        trace_path,
        policy_state_dim,
        world_state_dim,
        str(ckpt.get("policy_input_semantics", "world_state_v1")),
        str(ckpt.get("policy_action_semantics", "simlingo_residual_with_learned_gate_v1")),
        max(0, args.positive_replay_count),
    )
    if negative_example is not None:
        negative_example.update({
            "contrastive_positive_sources": positive_sources,
            "contrastive_positive_samples": int(positive_states.shape[0]),
            "reward_schema": REWARD_SCHEMA,
        })
        (trace_path.parent / "negative_example.json").write_text(
            json.dumps(negative_example, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    advantages, returns = compute_gae(rollout["rewards"], rollout["values"], args.gamma, args.lam)
    tensors = {
        "states": torch.as_tensor(rollout["states"], dtype=torch.float32, device=device),
        "next_states": torch.as_tensor(rollout["next_states"], dtype=torch.float32, device=device),
        "raw_actions": torch.as_tensor(rollout["raw_actions"], dtype=torch.float32, device=device),
        "chosen_actions": torch.as_tensor(rollout["chosen_actions"], dtype=torch.float32, device=device),
        "old_log_probs": torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=device),
        "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        "risk_targets": torch.as_tensor(rollout["risk_targets"], dtype=torch.float32, device=device).unsqueeze(-1),
        "progress_targets": torch.as_tensor(rollout["progress_targets"], dtype=torch.float32, device=device).unsqueeze(-1),
    }
    tensors["policy_states"] = (
        torch.as_tensor(rollout["policy_states"], dtype=torch.float32, device=device)
        - policy_state_mean.reshape(1, -1)
    ) / policy_state_std.reshape(1, -1)
    positive_policy_states = None
    positive_raw_actions = None
    if positive_states.shape[0] > 0:
        positive_policy_states = (
            torch.as_tensor(positive_states, dtype=torch.float32, device=device)
            - policy_state_mean.reshape(1, -1)
        ) / policy_state_std.reshape(1, -1)
        positive_raw_actions = torch.as_tensor(positive_actions, dtype=torch.float32, device=device)

    collision_update = collision_event is not None
    effective_epochs = max(1, args.epochs)
    effective_lr_scale = 1.0
    if collision_update:
        effective_epochs = min(effective_epochs, max(1, args.collision_epoch_cap))
        effective_lr_scale = float(np.clip(args.collision_lr_scale, 0.05, 1.0))
    effective_lr_policy = args.lr_policy * effective_lr_scale
    effective_lr_world_model = args.lr_world_model * effective_lr_scale

    optimizer_pi = torch.optim.Adam(policy.parameters(), lr=effective_lr_policy)
    if isinstance(ckpt.get("optimizer_pi"), dict):
        try:
            optimizer_pi.load_state_dict(ckpt["optimizer_pi"])
        except Exception:
            pass
    for group in optimizer_pi.param_groups:
        group["lr"] = effective_lr_policy
    optimizer_wm = torch.optim.Adam(world_model.parameters(), lr=effective_lr_world_model)
    rng = np.random.default_rng(int(time.time()) % (2**32 - 1))

    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []
    wm_losses: List[float] = []
    positive_anchor_losses: List[float] = []
    for _ in range(effective_epochs):
        for idx in minibatches(n, max(1, args.batch_size), rng):
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
            new_log_probs, entropy, values = policy.evaluate(tensors["policy_states"][idx_t], tensors["raw_actions"][idx_t])
            ratio = torch.exp(new_log_probs - tensors["old_log_probs"][idx_t])
            surr1 = ratio * tensors["advantages"][idx_t]
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * tensors["advantages"][idx_t]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, tensors["returns"][idx_t])
            entropy_loss = -entropy.mean()
            loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss
            if positive_policy_states is not None and positive_raw_actions is not None:
                positive_count = int(positive_policy_states.shape[0])
                positive_indices = rng.choice(
                    positive_count,
                    size=min(len(idx), positive_count),
                    replace=False,
                )
                positive_idx_t = torch.as_tensor(positive_indices, dtype=torch.long, device=device)
                positive_mean, _, _ = policy.forward(positive_policy_states[positive_idx_t])
                positive_anchor_loss = nn.functional.smooth_l1_loss(
                    positive_mean,
                    positive_raw_actions[positive_idx_t],
                )
                loss = loss + args.positive_anchor_coef * positive_anchor_loss
                positive_anchor_losses.append(float(positive_anchor_loss.item()))
            optimizer_pi.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer_pi.step()

            next_hat, risk_hat, progress_hat = world_model(tensors["states"][idx_t], tensors["chosen_actions"][idx_t])
            # CARLA world coordinates make raw MSE brittle after a single
            # failure. Smooth L1 keeps the exact risk target learnable without
            # allowing one impact episode to dominate the world model.
            wm_state = nn.functional.smooth_l1_loss(next_hat, tensors["next_states"][idx_t])
            wm_risk = nn.functional.mse_loss(risk_hat, tensors["risk_targets"][idx_t])
            wm_progress = nn.functional.mse_loss(progress_hat, tensors["progress_targets"][idx_t])
            wm_loss = wm_state + wm_risk + wm_progress
            optimizer_wm.zero_grad()
            wm_loss.backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), args.max_grad_norm)
            optimizer_wm.step()

            policy_losses.append(float(loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.mean().item()))
            wm_losses.append(float(wm_loss.item()))

    next_episode = int(ckpt.get("episode", 0)) + 1
    saved = {
        **{k: v for k, v in ckpt.items() if k not in ("policy", "world_model", "optimizer_pi", "optimizer_wm")},
        "episode": next_episode,
        "policy": policy.state_dict(),
        "world_model": world_model.state_dict(),
        "policy_state_mean": policy_state_mean.detach().float().cpu().numpy(),
        "policy_state_std": policy_state_std.detach().float().cpu().numpy(),
        "optimizer_pi": optimizer_pi.state_dict(),
        "optimizer_wm": optimizer_wm.state_dict(),
        "policy_action_semantics": ckpt.get(
            "policy_action_semantics",
            "simlingo_residual_with_learned_gate_v1",
        ),
        "policy_role": "online_rl_complement_to_simlingo",
        "policy_input_semantics": ckpt.get("policy_input_semantics", "world_state_v1"),
        "online_rl_update_count": int(ckpt.get("online_rl_update_count", 0)) + 1,
        "online_rl": {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trace": str(trace_path),
            "transitions": n,
            "reward_mean": float(np.mean(rollout["rewards"])),
            "reward_sum": float(np.sum(rollout["rewards"])),
            "reward_parts": rollout["reward_parts"],
            "metrics": metrics,
            "no_guard": True,
            "complement_to_simlingo": True,
            "action_semantics": ckpt.get(
                "policy_action_semantics",
                "simlingo_residual_with_learned_gate_v1",
            ),
            "mean_intervention_strength": float(np.mean(rollout["intervention_strengths"])),
            "failure_signal_used": bool(catastrophic_reason),
            "failure_reason": catastrophic_reason,
            "reward_schema": REWARD_SCHEMA,
            "collision_event": collision_event,
            "original_rows": original_row_count,
            "truncated_rows": len(rows),
            "post_impact_rows_excluded": max(0, original_row_count - len(rows)),
            "policy_observation_migration": observation_migration,
            "positive_replay_sources": positive_sources,
            "positive_replay_samples": int(positive_states.shape[0]),
            "positive_anchor_coefficient": float(args.positive_anchor_coef),
            "effective_epochs": effective_epochs,
            "effective_lr_policy": effective_lr_policy,
            "effective_lr_world_model": effective_lr_world_model,
        },
    }
    save_checkpoint(output_checkpoint, saved)

    summary = {
        "status": "updated",
        "trace": str(trace_path),
        "checkpoint": str(output_checkpoint),
        "device": str(device),
        "transitions": n,
        "episode": next_episode,
        "reward_mean": float(np.mean(rollout["rewards"])),
        "reward_sum": float(np.sum(rollout["rewards"])),
        "reward_parts": rollout["reward_parts"],
        "mean_intervention_strength": float(np.mean(rollout["intervention_strengths"])),
        "failure_signal_used": bool(catastrophic_reason),
        "failure_reason": catastrophic_reason,
        "reward_schema": REWARD_SCHEMA,
        "collision_event": collision_event,
        "original_rows": original_row_count,
        "truncated_rows": len(rows),
        "post_impact_rows_excluded": max(0, original_row_count - len(rows)),
        "policy_observation_migration": observation_migration,
        "positive_replay_sources": positive_sources,
        "positive_replay_samples": int(positive_states.shape[0]),
        "positive_anchor_loss": float(np.mean(positive_anchor_losses)) if positive_anchor_losses else 0.0,
        "effective_epochs": effective_epochs,
        "effective_lr_policy": effective_lr_policy,
        "effective_lr_world_model": effective_lr_world_model,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "world_model_loss": float(np.mean(wm_losses)) if wm_losses else 0.0,
        "metrics": metrics,
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
