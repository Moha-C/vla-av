"""Sequential Bench2Drive replay contract for residual DreamerV3.

Only physically executed CARLA transitions are accepted.  No counterfactual
label or synthetic collision is fabricated by this module.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ResidualDreamerConfig


OBSERVATION_FEATURES: Tuple[str, ...] = (
    "ego_speed", "ego_acceleration", "native_steer", "native_throttle",
    "native_brake", "progress_delta", "lane_edge_distance",
    "lane_center_offset", "left_clearance", "right_clearance",
    "left_front_distance", "left_rear_distance", "right_front_distance",
    "right_rear_distance", "front_obstacle_distance", "front_clearance",
    "front_relative_speed", "current_oncoming_distance",
    "current_oncoming_closing_speed", "current_oncoming_ttc",
    "left_oncoming_distance", "left_oncoming_ttc", "right_oncoming_distance",
    "right_oncoming_ttc", "left_lane_available", "right_lane_available",
    "nearest_vru_distance", "blocked_fraction", "overtake_phase",
    "return_distance", "traffic_light_state", "route_curvature",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: %s" % (path, line_number, exc))
            if isinstance(value, dict):
                rows.append(value)
    return rows


def discover_traces(patterns: Sequence[str]) -> List[Path]:
    paths = set()
    for pattern in patterns:
        for value in glob.glob(pattern, recursive=True):
            path = Path(value)
            if path.is_file() and path.stat().st_size:
                paths.add(path.resolve())
    return sorted(paths)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("status")
    return value if isinstance(value, Mapping) else {}


def _timestamp(row: Mapping[str, Any]) -> float:
    status = _status(row)
    return _number(status.get("timestamp"), _number(row.get("collector_time")))


def _observation(row: Mapping[str, Any]) -> np.ndarray:
    payload = row.get("observation")
    if not isinstance(payload, Mapping):
        raise ValueError("trace is missing the captured named 32D observation")
    values = np.asarray([_number(payload.get(key), float("nan")) for key in OBSERVATION_FEATURES], dtype=np.float32)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("trace contains an incomplete or non-finite observation")
    return values


def _action(status: Mapping[str, Any]) -> np.ndarray:
    payload = status.get("final_action") or status.get("applied_control") or status.get("base_action")
    if not isinstance(payload, Mapping):
        raise ValueError("trace row has no physically executed control")
    return np.asarray(
        [
            np.clip(_number(payload.get("steer")), -1.0, 1.0),
            np.clip(_number(payload.get("throttle")), 0.0, 1.0),
            np.clip(_number(payload.get("brake")), 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def _policy_source(statuses: Sequence[Mapping[str, Any]]) -> str:
    for status in statuses:
        mode = str(status.get("mode", "")).lower()
        source = str(status.get("policy_source", "")).lower()
        variant = str(status.get("variant", "")).lower()
        alpha = _number(status.get("alpha", status.get("dreamer_weight", 0.0)))
        native = (
            mode in ("native", "simlingo_native")
            or source in ("native", "simlingo_native")
            or "simlingo_native" in variant
            or "report_native_collect" in variant
        )
        if not native or alpha > 1.0e-8 or bool(status.get("applied", False)):
            return "non_native_or_unknown"
    return "simlingo_native"


def _event_flags(path: Path, rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray, str]:
    transitions = max(0, len(rows) - 1)
    collision = np.zeros(transitions, dtype=np.float32)
    offroad = np.zeros(transitions, dtype=np.float32)
    quality = "none"
    events_path = path.parent / "collision_events.jsonl"
    if events_path.exists() and transitions:
        times = [_timestamp(row) for row in rows]
        for event in read_jsonl(events_path):
            event_time = _number(event.get("wall_time", event.get("timestamp")), -1.0)
            if event_time < 0.0:
                continue
            index = next((i for i, value in enumerate(times[1:]) if value >= event_time), transitions - 1)
            collision[max(0, min(index, transitions - 1))] = 1.0
            quality = "synchronized_event"
    for index, row in enumerate(rows[1:]):
        status = _status(row)
        if bool(status.get("collision", status.get("collided", False))):
            collision[index] = 1.0
            quality = "synchronized_event"
        if bool(status.get("offroad", status.get("lane_departure", status.get("outside_route", False)))):
            offroad[index] = 1.0
            quality = "synchronized_event"
    episode = _read_json(path.parent / "episode.json")
    metrics = episode.get("metrics") if isinstance(episode.get("metrics"), Mapping) else {}
    if transitions and collision.sum() == 0.0 and _number(metrics.get("collisions")) > 0.0:
        collision[-1] = 1.0
        quality = "terminal_proxy"
    if transitions and offroad.sum() == 0.0 and _number(metrics.get("offroad")) > 0.0:
        offroad[-1] = 1.0
        quality = "terminal_proxy" if quality == "none" else quality
    return collision, offroad, quality


def _position(status: Mapping[str, Any]) -> Optional[np.ndarray]:
    state = status.get("state_vector")
    if not isinstance(state, Sequence) or len(state) < 2:
        return None
    point = np.asarray([_number(state[0], float("nan")), _number(state[1], float("nan"))], dtype=np.float32)
    return point if np.isfinite(point).all() else None


def _risk(observation: np.ndarray, collision: float, offroad: float) -> float:
    if collision:
        return 1.0
    front = max(0.0, 1.0 - float(observation[15]) / (15.0 / 80.0))
    current_ttc = max(0.0, 1.0 - float(observation[19]) / (8.0 / 20.0))
    left_ttc = max(0.0, 1.0 - float(observation[21]) / (8.0 / 20.0))
    right_ttc = max(0.0, 1.0 - float(observation[23]) / (8.0 / 20.0))
    vru = max(0.0, 1.0 - float(observation[26]) / (12.0 / 80.0))
    return float(np.clip(max(front, current_ttc, left_ttc, right_ttc, vru, 0.9 * offroad), 0.0, 1.0))


@dataclass
class Episode:
    key: str
    seed: str
    path: Path
    metadata: Dict[str, Any]
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    continuation: np.ndarray
    collision: np.ndarray
    offroad: np.ndarray
    risk: np.ndarray
    progress: np.ndarray

    @property
    def transitions(self) -> int:
        return int(self.actions.shape[0])


def build_episode(path: Path, config: ResidualDreamerConfig) -> Optional[Episode]:
    rows = sorted(read_jsonl(path), key=_timestamp)
    if len(rows) < 3:
        return None
    statuses = [_status(row) for row in rows]
    source = _policy_source(statuses)
    if config.data.require_native_simlingo and source != "simlingo_native":
        raise ValueError("policy source is not native SimLingo")
    episode_payload = _read_json(path.parent / "episode.json")
    metrics = episode_payload.get("metrics") if isinstance(episode_payload.get("metrics"), Mapping) else {}
    ground_truth = bool(episode_payload.get("bench2drive_ground_truth", metrics.get("bench2drive_ground_truth", False)))
    if config.data.require_bench2drive_ground_truth and not ground_truth:
        raise ValueError("Bench2Drive ground truth is missing")
    observations = np.stack([_observation(row) for row in rows])
    actions = np.stack([_action(status) for status in statuses[:-1]])
    collision, offroad, timing_quality = _event_flags(path, rows)
    progress = []
    for previous, current in zip(statuses, statuses[1:]):
        first, second = _position(previous), _position(current)
        distance = 0.0 if first is None or second is None else float(np.linalg.norm(second - first))
        progress.append(float(np.clip(distance, 0.0, config.data.maximum_step_progress_m)))
    progress_array = np.asarray(progress, dtype=np.float32)
    risk = np.asarray(
        [_risk(observations[index + 1], collision[index], offroad[index]) for index in range(len(actions))],
        dtype=np.float32,
    )
    continuation = np.ones(len(actions), dtype=np.float32)
    continuation[-1] = 0.0
    first_incident = np.flatnonzero((collision > 0.0) | (offroad > 0.0))
    if first_incident.size:
        keep = int(first_incident[0]) + 1
        observations = observations[: keep + 1]
        actions = actions[:keep]
        collision = collision[:keep]
        offroad = offroad[:keep]
        progress_array = progress_array[:keep]
        risk = risk[:keep]
        continuation = continuation[:keep]
        continuation[-1] = 0.0
    rewards = []
    previous_action = actions[0]
    for index, action in enumerate(actions):
        control_change = float(np.linalg.norm(action - previous_action))
        safe = 1.0 - 2.0 * float(risk[index])
        reward = (
            config.reward.progress_scale * float(progress_array[index])
            + config.reward.safe_scale * safe
            - config.reward.collision_penalty * float(collision[index])
            - config.reward.offroad_penalty * float(offroad[index])
            - config.reward.control_change_penalty * control_change
        )
        rewards.append(reward)
        previous_action = action
    route_completion = _number(metrics.get("route_completion"), 0.0)
    completed_cleanly = route_completion >= 99.99 and not collision.any() and not offroad.any()
    if rewards and completed_cleanly:
        rewards[-1] += config.reward.completion_bonus
    first = rows[0]
    metadata = {
        "route_id": str(first.get("route_id", "unknown")),
        "seed": str(first.get("seed", "unknown")),
        "town": str(first.get("town", "unknown")),
        "scenario": str(first.get("scenario", "unknown")),
        "route_file": str(first.get("route_file", "")),
        "policy_source": source,
        "bench2drive_ground_truth": ground_truth,
        "event_timing_quality": timing_quality,
        "route_completion": route_completion,
        "driving_score": _number(metrics.get("driving_score"), float("nan")),
        "trace_sha256": sha256_file(path),
    }
    seed = metadata["seed"]
    return Episode(
        key="%s:%s" % (metadata["route_id"], seed),
        seed=seed,
        path=path,
        metadata=metadata,
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        continuation=continuation.astype(np.float32),
        collision=collision.astype(np.float32),
        offroad=offroad.astype(np.float32),
        risk=risk.astype(np.float32),
        progress=progress_array.astype(np.float32),
    )


@dataclass
class Splits:
    train: List[Episode]
    validation: List[Episode]
    test: List[Episode]

    def seed_sets(self) -> Dict[str, List[str]]:
        return {
            "train": sorted({item.seed for item in self.train}),
            "validation": sorted({item.seed for item in self.validation}),
            "test": sorted({item.seed for item in self.test}),
        }

    def verify(self, config: ResidualDreamerConfig) -> None:
        seeds = {key: set(value) for key, value in self.seed_sets().items()}
        if seeds["train"] & seeds["validation"] or seeds["train"] & seeds["test"] or seeds["validation"] & seeds["test"]:
            raise RuntimeError("seed leakage between dataset splits")
        minimums = {
            "train": config.data.minimum_train_seeds,
            "validation": config.data.minimum_validation_seeds,
            "test": config.data.minimum_test_seeds,
        }
        for key, minimum in minimums.items():
            if len(seeds[key]) < int(minimum):
                raise RuntimeError("%s split has fewer than %d seeds" % (key, minimum))


def stratified_seed_split(episodes: Sequence[Episode], config: ResidualDreamerConfig) -> Splits:
    by_seed: Dict[str, List[Episode]] = {}
    for episode in episodes:
        by_seed.setdefault(episode.seed, []).append(episode)
    units = list(by_seed.values())
    if len(units) < (
        config.data.minimum_train_seeds
        + config.data.minimum_validation_seeds
        + config.data.minimum_test_seeds
    ):
        raise RuntimeError("not enough distinct seeds for the frozen split")
    groups: Dict[Tuple[str, str, str], List[List[Episode]]] = {}
    for unit in units:
        first = unit[0]
        key = (
            str(first.metadata.get("town", "unknown")),
            str(first.metadata.get("scenario", "unknown")),
            str(first.metadata.get("route_id", "unknown")),
        )
        groups.setdefault(key, []).append(unit)
    rng = random.Random(config.data.split_seed)
    ordered_groups = sorted(groups.items())
    for _, values in ordered_groups:
        rng.shuffle(values)
    total = len(units)
    targets = {
        "train": int(round(total * config.data.train_ratio)),
        "validation": int(round(total * config.data.validation_ratio)),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    targets["train"] = max(targets["train"], config.data.minimum_train_seeds)
    targets["validation"] = max(targets["validation"], config.data.minimum_validation_seeds)
    targets["test"] = max(targets["test"], config.data.minimum_test_seeds)
    while sum(targets.values()) > total:
        key = max(targets, key=lambda item: targets[item] - getattr(config.data, "minimum_%s_seeds" % item))
        targets[key] -= 1
    assigned: Dict[str, List[List[Episode]]] = {"train": [], "validation": [], "test": []}
    leftovers: List[List[Episode]] = []
    # Every route/scenario stratum contributes to training whenever possible.
    for _, values in ordered_groups:
        assigned["train"].append(values[0])
        leftovers.extend(values[1:])
    rng.shuffle(leftovers)
    for unit in leftovers:
        deficits = {key: targets[key] - len(assigned[key]) for key in assigned}
        destination = max((key for key in assigned if deficits[key] > 0), key=lambda key: (deficits[key], key), default="train")
        assigned[destination].append(unit)
    result = Splits(
        train=[episode for unit in assigned["train"] for episode in unit],
        validation=[episode for unit in assigned["validation"] for episode in unit],
        test=[episode for unit in assigned["test"] for episode in unit],
    )
    result.verify(config)
    return result


class SequenceDataset(Dataset):
    def __init__(self, episodes: Sequence[Episode], length: int):
        self.episodes = list(episodes)
        self.length = int(length)
        self.indices: List[Tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            for start in range(episode.transitions - self.length + 1):
                self.indices.append((episode_index, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_index, start = self.indices[index]
        episode = self.episodes[episode_index]
        end = start + self.length
        return {
            "observations": torch.from_numpy(episode.observations[start : end + 1]),
            "actions": torch.from_numpy(episode.actions[start:end]),
            "rewards": torch.from_numpy(episode.rewards[start:end]),
            "continuation": torch.from_numpy(episode.continuation[start:end]),
            "risk": torch.from_numpy(episode.risk[start:end]),
            "collision": torch.from_numpy(episode.collision[start:end]),
            "offroad": torch.from_numpy(episode.offroad[start:end]),
        }

    def sample_weights(
        self,
        event_weight: float,
        danger_weight: float,
        danger_threshold: float,
    ) -> torch.Tensor:
        weights = []
        for episode_index, start in self.indices:
            episode = self.episodes[episode_index]
            end = start + self.length
            has_event = bool(
                np.any(episode.collision[start:end] > 0.5)
                or np.any(episode.offroad[start:end] > 0.5)
            )
            has_danger = bool(np.any(episode.risk[start:end] >= danger_threshold))
            if has_event:
                weight = event_weight
            elif has_danger:
                weight = danger_weight
            else:
                weight = 1.0
            weights.append(float(weight))
        return torch.as_tensor(weights, dtype=torch.double)


def split_manifest(splits: Splits, config: ResidualDreamerConfig) -> Dict[str, Any]:
    splits.verify(config)

    def summarize(episodes: Sequence[Episode]) -> Dict[str, Any]:
        return {
            "episodes": len(episodes),
            "transitions": int(sum(item.transitions for item in episodes)),
            "seeds": sorted({item.seed for item in episodes}),
            "towns": sorted({str(item.metadata["town"]) for item in episodes}),
            "scenarios": sorted({str(item.metadata["scenario"]) for item in episodes}),
            "routes": sorted({str(item.metadata["route_id"]) for item in episodes}),
            "collisions": int(sum(item.collision.sum() for item in episodes)),
            "offroad": int(sum(item.offroad.sum() for item in episodes)),
            "traces": [
                {
                    "path": str(item.path),
                    "sha256": item.metadata["trace_sha256"],
                    "seed": item.seed,
                    "route_id": item.metadata["route_id"],
                    "town": item.metadata["town"],
                    "scenario": item.metadata["scenario"],
                    "transitions": item.transitions,
                    "event_timing_quality": item.metadata["event_timing_quality"],
                }
                for item in episodes
            ],
        }

    return {
        "schema_version": "residual_dreamerv3_dataset_v1",
        "split_seed": config.data.split_seed,
        "feature_order": list(OBSERVATION_FEATURES),
        "seed_sets": splits.seed_sets(),
        "train": summarize(splits.train),
        "validation": summarize(splits.validation),
        "test": summarize(splits.test),
    }
