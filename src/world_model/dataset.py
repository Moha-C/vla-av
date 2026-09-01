"""Ordered Bench2Drive traces and seed-disjoint Dreamer datasets."""

from __future__ import annotations

import glob
import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DreamerConfig
from .observation import (
    DREAMER_OBSERVATION_FEATURES,
    DreamerObservationBuilder,
)
from .reward import DreamerReward, RewardSignals


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: invalid JSON: %s" % (path, line_number, exc))
            if isinstance(item, dict):
                rows.append(item)
    return rows


def discover_traces(patterns: Sequence[str]) -> List[Path]:
    paths = set()
    for pattern in patterns:
        for item in glob.glob(pattern, recursive=True):
            path = Path(item)
            if path.is_file() and path.stat().st_size > 0:
                paths.add(path.resolve())
    return sorted(paths)


def route_metadata(route_file: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "route_file": route_file or "unknown",
        "town": "unknown",
        "route_xml_id": "unknown",
        "scenario": "unknown",
        "scenario_name": "unknown",
        "weather": "unknown",
    }
    path = Path(route_file) if route_file else None
    if path is None or not path.exists():
        return metadata
    try:
        root = ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError):
        return metadata
    route = root.find(".//route")
    if route is not None:
        metadata["town"] = route.attrib.get("town", "unknown")
        metadata["route_xml_id"] = route.attrib.get("id", "unknown")
    scenario = root.find(".//scenario")
    if scenario is not None:
        metadata["scenario"] = scenario.attrib.get("type", "unknown")
        metadata["scenario_name"] = scenario.attrib.get("name", "unknown")
    weather = root.find(".//weather")
    if weather is not None:
        metadata["weather"] = dict(sorted(weather.attrib.items()))
    return metadata


def _action_dict(status: Mapping[str, Any], key: str) -> Optional[Mapping[str, Any]]:
    value = status.get(key)
    return value if isinstance(value, Mapping) else None


def native_action(status: Mapping[str, Any]) -> np.ndarray:
    action = _action_dict(status, "base_action") or _action_dict(status, "native_action") or {}
    return np.asarray(
        [
            _number(action.get("steer")),
            _number(action.get("throttle")),
            _number(action.get("brake")),
        ],
        dtype=np.float32,
    )


def final_action(status: Mapping[str, Any]) -> np.ndarray:
    action = _action_dict(status, "final_action") or _action_dict(status, "applied_control")
    if action is None:
        if bool(status.get("applied", False)):
            action = _action_dict(status, "chosen_action")
        action = action or _action_dict(status, "base_action") or {}
    return np.asarray(
        [
            np.clip(_number(action.get("steer")), -1.0, 1.0),
            np.clip(_number(action.get("throttle")), 0.0, 1.0),
            np.clip(_number(action.get("brake")), 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def authority_alpha(status: Mapping[str, Any]) -> float:
    return float(
        np.clip(
            _number(
                status.get(
                    "alpha",
                    status.get(
                        "dreamer_weight",
                        status.get("rl_intervention_strength", 0.0),
                    ),
                )
            ),
            0.0,
            1.0,
        )
    )


def _stored_observation(row: Mapping[str, Any]) -> Optional[np.ndarray]:
    payload = row.get("observation")
    if not isinstance(payload, Mapping):
        return None
    values = np.asarray(
        [_number(payload.get(name), float("nan")) for name in DREAMER_OBSERVATION_FEATURES],
        dtype=np.float32,
    )
    if values.shape != (len(DREAMER_OBSERVATION_FEATURES),):
        return None
    return values if np.isfinite(values).all() else None


def _context(status: Mapping[str, Any], progress_m: float) -> Dict[str, Any]:
    result = dict(status)
    state = status.get("state_vector")
    if isinstance(state, Sequence) and len(state) >= 3:
        result.setdefault("ego_speed_mps", _number(state[2]))
    result["route_progress_m"] = float(progress_m)
    front_id = int(_number(status.get("front_vehicle_id"), -1))
    nearby = status.get("nearby_vehicles")
    if isinstance(nearby, list):
        for vehicle in nearby:
            if not isinstance(vehicle, Mapping):
                continue
            if int(_number(vehicle.get("id"), -2)) == front_id:
                result.setdefault(
                    "front_relative_speed_mps",
                    _number(vehicle.get("closing_speed_mps")),
                )
                break
    return result


def _position(status: Mapping[str, Any]) -> Optional[np.ndarray]:
    state = status.get("state_vector")
    if not isinstance(state, Sequence) or len(state) < 2:
        return None
    point = np.asarray([_number(state[0]), _number(state[1])], dtype=np.float32)
    return point if np.isfinite(point).all() else None


def _status_timestamp(row: Mapping[str, Any]) -> float:
    status = row.get("status") if isinstance(row.get("status"), Mapping) else {}
    return _number(status.get("timestamp"), _number(row.get("collector_time")))


def _event_flags(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, bool]:
    collision = np.zeros(max(0, len(rows) - 1), dtype=np.float32)
    offroad = np.zeros_like(collision)
    ground_truth = False
    collision_path = path.parent / "collision_events.jsonl"
    if collision_path.exists() and collision.size:
        ground_truth = True
        events = read_jsonl(collision_path)
        times = [_status_timestamp(row) for row in rows]
        for event in events:
            timestamp = _number(event.get("timestamp", event.get("wall_time")), -1.0)
            if timestamp < 0.0:
                continue
            index = next((i for i, value in enumerate(times[1:]) if value >= timestamp), len(collision) - 1)
            collision[max(0, min(index, len(collision) - 1))] = 1.0
    for index, row in enumerate(rows[1:]):
        status = row.get("status") if isinstance(row.get("status"), Mapping) else {}
        if "collision" in status or "collided" in status:
            ground_truth = True
        if "offroad" in status or "lane_departure" in status or "outside_route" in status:
            ground_truth = True
        if bool(status.get("collision", status.get("collided", False))):
            collision[index] = 1.0
        if bool(status.get("offroad", status.get("lane_departure", status.get("outside_route", False)))):
            offroad[index] = 1.0
    episode_path = path.parent / "episode.json"
    if episode_path.exists() and collision.size:
        try:
            payload = json.loads(episode_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        metrics = payload.get("metrics") if isinstance(payload, Mapping) else {}
        metrics = metrics if isinstance(metrics, Mapping) else {}
        ground_truth = bool(
            payload.get("bench2drive_ground_truth", False)
            or metrics.get("bench2drive_ground_truth", False)
        ) or ground_truth
        if collision.sum() == 0 and _number(metrics.get("collisions", metrics.get("collision", 0))) > 0:
            collision[-1] = 1.0
        if offroad.sum() == 0 and _number(metrics.get("offroad", metrics.get("outside_route_lanes", 0))) > 0:
            offroad[-1] = 1.0
    return collision, offroad, ground_truth


def policy_source(statuses: Sequence[Mapping[str, Any]]) -> str:
    """Classify who actually controlled the recorded CARLA transitions.

    A report shadow model is accepted as native because alpha is exactly zero
    and therefore its actions cannot affect the environment. Unknown or mixed
    traces are deliberately not guessed to be native.
    """

    if not statuses:
        return "unknown"
    native_markers = (
        "simlingo_native",
        "native_simlingo",
        "report_native_collect",
    )
    for status in statuses:
        mode = str(status.get("mode", "")).strip().lower()
        variant = str(status.get("variant", "")).strip().lower()
        alpha = _number(
            status.get(
                "alpha",
                status.get(
                    "dreamer_weight", status.get("rl_intervention_strength", 0.0)
                ),
            )
        )
        applied = bool(status.get("applied", False))
        explicit_native = mode in ("native", "simlingo_native") or any(
            marker in variant for marker in native_markers
        )
        report_shadow = (
            "report_aligned" in variant
            and bool(status.get("shadow", False))
            and alpha <= 1.0e-8
            and not applied
        )
        if not (explicit_native or report_shadow):
            return "non_native_or_unknown"
        if alpha > 1.0e-8 or applied:
            return "non_native_or_unknown"
    return "simlingo_native"


def geometric_risk(status: Mapping[str, Any], collision: float, offroad: float) -> float:
    if collision:
        return 1.0
    ttcs = [
        _number(status.get("current_oncoming_ttc_s"), 99.0),
        _number(status.get("left_oncoming_ttc_s"), 99.0),
        _number(status.get("right_oncoming_ttc_s"), 99.0),
        _number(status.get("left_ttc_s"), 99.0),
        _number(status.get("right_ttc_s"), 99.0),
    ]
    minimum_ttc = max(0.0, min(ttcs))
    ttc_risk = max(0.0, 1.0 - minimum_ttc / 8.0)
    front = _number(status.get("front_vehicle_clearance_m", status.get("front_vehicle_m")), 80.0)
    front_risk = max(0.0, 1.0 - max(0.0, front) / 15.0)
    vru = min(
        _number(status.get("nearest_walker_m"), 80.0),
        _number(status.get("nearest_bike_m"), 80.0),
    )
    vru_risk = max(0.0, 1.0 - max(0.0, vru) / 12.0)
    return float(np.clip(max(ttc_risk, front_risk, vru_risk, 0.9 * offroad), 0.0, 1.0))


@dataclass
class DreamerEpisode:
    key: str
    seed: str
    metadata: Dict[str, Any]
    observations: np.ndarray
    actions: np.ndarray
    alpha: np.ndarray
    progress: np.ndarray
    risk: np.ndarray
    continuation: np.ndarray
    collision: np.ndarray
    offroad: np.ndarray
    reward: np.ndarray
    value: np.ndarray

    @property
    def transitions(self) -> int:
        return int(self.actions.shape[0])


def build_episode(path: Path, config: DreamerConfig) -> Optional[DreamerEpisode]:
    rows = read_jsonl(path)
    rows.sort(key=_status_timestamp)
    if len(rows) < 3:
        return None
    statuses = [
        row.get("status") if isinstance(row.get("status"), Mapping) else {}
        for row in rows
    ]
    progress = [0.0]
    for previous, current in zip(statuses, statuses[1:]):
        first = _position(previous)
        second = _position(current)
        distance = 0.0 if first is None or second is None else float(np.linalg.norm(second - first))
        progress.append(progress[-1] + min(distance, config.observation.max_progress_delta_m))
    stored_observations = [_stored_observation(row) for row in rows]
    if all(item is not None for item in stored_observations):
        observations = [item for item in stored_observations if item is not None]
    else:
        builder = DreamerObservationBuilder(config.observation)
        observations = [
            builder.build(_context(status, distance), native_action(status)).as_array()
            for status, distance in zip(statuses, progress)
        ]
    actions = np.asarray([final_action(status) for status in statuses[:-1]], dtype=np.float32)
    alphas = np.asarray(
        [authority_alpha(status) for status in statuses[:-1]], dtype=np.float32
    )
    progress_delta = np.diff(np.asarray(progress, dtype=np.float32))
    collision, offroad, event_ground_truth = _event_flags(path, rows)
    risk = np.asarray(
        [
            # Action[index] moves the world from statuses[index] to
            # statuses[index + 1].  The action-conditioned prior must learn
            # the resulting risk, not the risk that existed before the action.
            geometric_risk(status, collision[index], offroad[index])
            for index, status in enumerate(statuses[1:])
        ],
        dtype=np.float32,
    )
    continuation = np.ones(len(actions), dtype=np.float32)
    continuation[-1] = 0.0
    if collision.any() and config.reward.collision_terminal:
        first_collision = int(np.argmax(collision > 0))
        keep = first_collision + 1
        observations = observations[: keep + 1]
        actions = actions[:keep]
        alphas = alphas[:keep]
        progress_delta = progress_delta[:keep]
        collision = collision[:keep]
        offroad = offroad[:keep]
        risk = risk[:keep]
        continuation = continuation[:keep]
        continuation[-1] = 0.0
    reward_function = DreamerReward(config.reward)
    rewards = []
    previous_action = actions[0]
    for index, action in enumerate(actions):
        result = reward_function(
            RewardSignals(
                progress_delta=float(progress_delta[index]),
                safety=DreamerReward.safety_from_risk(float(risk[index])),
                collision=float(collision[index]),
                offroad=float(offroad[index]),
                jerk=DreamerReward.action_jerk(action, previous_action),
                alpha=float(alphas[index]),
            )
        )
        rewards.append(result.total)
        previous_action = action
    rewards_array = np.asarray(rewards, dtype=np.float32)
    values = np.zeros_like(rewards_array)
    running = 0.0
    discount = float(config.evaluator.continuation_discount)
    for index in range(len(values) - 1, -1, -1):
        running = float(rewards_array[index]) + discount * float(continuation[index]) * running
        values[index] = running

    first_row = rows[0]
    route_file = str(first_row.get("route_file", ""))
    metadata = route_metadata(route_file)
    metadata.update(
        {
            "trace_path": str(path),
            "route_id": str(first_row.get("route_id", Path(route_file).stem.replace("bench2drive_", "") or "unknown")),
            "seed": str(first_row.get("seed", "unknown")),
            "town": str(first_row.get("town") or metadata.get("town") or "unknown"),
            "policy_source": policy_source(statuses),
            "event_ground_truth": bool(event_ground_truth),
        }
    )
    return DreamerEpisode(
        key="%s:%s:%s" % (metadata["route_id"], metadata["seed"], path.name),
        seed=metadata["seed"],
        metadata=metadata,
        observations=np.asarray(observations, dtype=np.float32),
        actions=actions,
        alpha=alphas,
        progress=progress_delta.astype(np.float32),
        risk=risk.astype(np.float32),
        continuation=continuation.astype(np.float32),
        collision=collision.astype(np.float32),
        offroad=offroad.astype(np.float32),
        reward=rewards_array,
        value=(np.sign(values) * np.log1p(np.abs(values))).astype(np.float32),
    )


@dataclass
class DatasetSplits:
    train: List[DreamerEpisode]
    validation: List[DreamerEpisode]
    test: List[DreamerEpisode]
    seed_sets: Dict[str, List[str]]

    def verify(self) -> None:
        train = set(self.seed_sets["train"])
        validation = set(self.seed_sets["validation"])
        test = set(self.seed_sets["test"])
        if train & validation or train & test or validation & test:
            raise RuntimeError("seed leakage detected between train/validation/test")


def split_by_seed(episodes: Sequence[DreamerEpisode], config: DreamerConfig) -> DatasetSplits:
    grouped: Dict[str, List[DreamerEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.seed, []).append(episode)
    seeds = sorted(grouped)
    minimum_train = int(config.training.minimum_train_seeds)
    minimum_validation = int(config.training.minimum_validation_seeds)
    minimum_test = int(config.training.minimum_test_seeds)
    minimum_total = minimum_train + minimum_validation + minimum_test
    if len(seeds) < minimum_total:
        raise RuntimeError(
            "at least %d distinct seeds are required (%d train, %d validation, %d test)"
            % (minimum_total, minimum_train, minimum_validation, minimum_test)
        )
    random.Random(config.training.split_seed).shuffle(seeds)
    count = len(seeds)
    validation_count = max(
        minimum_validation,
        int(round(count * config.training.validation_ratio)),
    )
    test_count = max(minimum_test, int(round(count * config.training.test_ratio)))
    while validation_count + test_count > count - minimum_train:
        if validation_count >= test_count and validation_count > minimum_validation:
            validation_count -= 1
        elif test_count > minimum_test:
            test_count -= 1
        else:
            raise RuntimeError("seed split cannot satisfy configured minimums")
    test_seeds = sorted(seeds[:test_count])
    validation_seeds = sorted(seeds[test_count : test_count + validation_count])
    train_seeds = sorted(seeds[test_count + validation_count :])
    split = DatasetSplits(
        train=[episode for seed in train_seeds for episode in grouped[seed]],
        validation=[episode for seed in validation_seeds for episode in grouped[seed]],
        test=[episode for seed in test_seeds for episode in grouped[seed]],
        seed_sets={"train": train_seeds, "validation": validation_seeds, "test": test_seeds},
    )
    split.verify()
    return split


class SequenceDataset(Dataset):
    def __init__(self, episodes: Sequence[DreamerEpisode], sequence_length: int):
        self.episodes = list(episodes)
        self.sequence_length = int(sequence_length)
        self.indices: List[Tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            for start in range(0, episode.transitions - self.sequence_length + 1):
                self.indices.append((episode_index, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_index, start = self.indices[index]
        episode = self.episodes[episode_index]
        end = start + self.sequence_length
        return {
            "observations": torch.from_numpy(episode.observations[start : end + 1]),
            "actions": torch.from_numpy(episode.actions[start:end]),
            "alpha": torch.from_numpy(episode.alpha[start:end]),
            "progress": torch.from_numpy(episode.progress[start:end]),
            "risk": torch.from_numpy(episode.risk[start:end]),
            "continuation": torch.from_numpy(episode.continuation[start:end]),
            "collision": torch.from_numpy(episode.collision[start:end]),
            "offroad": torch.from_numpy(episode.offroad[start:end]),
            "reward": torch.from_numpy(episode.reward[start:end]),
            "value": torch.from_numpy(episode.value[start:end]),
        }


def dataset_summary(episodes: Sequence[DreamerEpisode]) -> Dict[str, Any]:
    def statistics(values: np.ndarray) -> Dict[str, Any]:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if not flat.size:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "minimum": None,
                "maximum": None,
            }
        return {
            "count": int(flat.size),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "minimum": float(np.min(flat)),
            "maximum": float(np.max(flat)),
        }

    nonempty_actions = [item.actions for item in episodes if item.actions.size]
    controls = (
        np.concatenate(nonempty_actions, axis=0)
        if nonempty_actions
        else np.empty((0, 3), dtype=np.float32)
    )
    nonempty_alpha = [item.alpha for item in episodes if item.alpha.size]
    alpha = (
        np.concatenate(nonempty_alpha)
        if nonempty_alpha
        else np.empty(0, dtype=np.float32)
    )
    nonempty_reward = [item.reward for item in episodes if item.reward.size]
    reward = (
        np.concatenate(nonempty_reward)
        if nonempty_reward
        else np.empty(0, dtype=np.float32)
    )
    ground_truth_count = int(
        sum(bool(item.metadata.get("event_ground_truth", False)) for item in episodes)
    )
    return {
        "episodes": len(episodes),
        "transitions": int(sum(item.transitions for item in episodes)),
        "seeds": sorted({item.seed for item in episodes}),
        "routes": sorted({str(item.metadata.get("route_id", "unknown")) for item in episodes}),
        "towns": sorted({str(item.metadata.get("town", "unknown")) for item in episodes}),
        "scenarios": sorted({str(item.metadata.get("scenario", "unknown")) for item in episodes}),
        "policy_sources": sorted(
            {str(item.metadata.get("policy_source", "unknown")) for item in episodes}
        ),
        "ground_truth_episode_count": ground_truth_count,
        "ground_truth_episode_fraction": (
            ground_truth_count / float(len(episodes)) if episodes else None
        ),
        "collisions": int(sum(float(item.collision.sum()) for item in episodes)),
        "offroad": int(sum(float(item.offroad.sum()) for item in episodes)),
        "physical_action_statistics": {
            "steer": statistics(controls[:, 0]),
            "throttle": statistics(controls[:, 1]),
            "brake": statistics(controls[:, 2]),
        },
        "authority_alpha_statistics": statistics(alpha),
        "reward_statistics": statistics(reward),
    }


def split_manifest(splits: DatasetSplits) -> Dict[str, Any]:
    splits.verify()
    return {
        "seed_sets": splits.seed_sets,
        "train": dataset_summary(splits.train),
        "validation": dataset_summary(splits.validation),
        "test": dataset_summary(splits.test),
    }
