"""Build a leakage-resistant temporal index for the DeepAccident dataset.

DeepAccident stores each sensor in a separate directory for four instrumented
vehicles.  This module indexes the front camera of the actor named in the
official ``colliding agents`` metadata and the same actor in the paired normal
run.  Accident timing is weakly supervised: the official simulator stops an
accident scenario at impact, so the final recorded frame is treated as the
event boundary.  The assumption is recorded in every generated audit and must
not be confused with a control or expert-action label.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FRAME_NUMBER = re.compile(r"(\d+)(?=\.[^.]+$)")
TOWN_NAME = re.compile(r"(Town\d+(?:HD)?)", re.IGNORECASE)
COLLIDING_AGENTS = re.compile(r"colliding\s+agents:\s+(\S+)\s+(\S+)", re.IGNORECASE)
AGENT_TO_VEHICLE_ROLE = {
    "ego": "ego_vehicle",
    "other": "other_vehicle",
    "ego_behind": "ego_vehicle_behind",
    "other_behind": "other_vehicle_behind",
}
VEHICLE_ROLES = frozenset(AGENT_TO_VEHICLE_ROLE.values())


@dataclass(frozen=True)
class DeepAccidentIndexConfig:
    fps: float = 10.0
    prediction_horizon_s: float = 2.0
    split_seed: int = 230401168
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    require_labels: bool = True

    def validate(self) -> None:
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        if self.prediction_horizon_s <= 0.0:
            raise ValueError("prediction_horizon_s must be positive")
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError("split ratios must sum to 1")


def _frame_number(path: Path) -> int:
    match = FRAME_NUMBER.search(path.name)
    if match is None:
        raise ValueError("frame filename has no numeric suffix: %s" % path)
    return int(match.group(1))


def _camera_context(camera_dir: Path) -> Tuple[Path, str]:
    # <category>/<vehicle_role>/Camera_Front/<scenario>
    if camera_dir.parent.name != "Camera_Front":
        raise ValueError("expected a Camera_Front scenario directory: %s" % camera_dir)
    vehicle_dir = camera_dir.parent.parent
    if vehicle_dir.name not in VEHICLE_ROLES:
        raise ValueError("unknown DeepAccident vehicle role: %s" % vehicle_dir.name)
    return vehicle_dir.parent, vehicle_dir.name


def _scenario_directories(root: Path) -> List[Path]:
    directories = []
    for camera_root in root.rglob("Camera_Front"):
        if not camera_root.is_dir() or camera_root.parent.name not in VEHICLE_ROLES:
            continue
        directories.extend(path for path in camera_root.iterdir() if path.is_dir())
    return sorted(set(path.resolve() for path in directories))


def _matching_label(
    category: Path, vehicle_role: str, scenario_id: str, image: Path
) -> Optional[Path]:
    candidate = category / vehicle_role / "label" / scenario_id / (image.stem + ".txt")
    return candidate if candidate.exists() else None


def _meta_files(category: Path, scenario_id: str) -> List[Path]:
    meta = category / "meta"
    if not meta.exists():
        return []
    exact = meta / (scenario_id + ".txt")
    if exact.exists():
        return [exact]
    return sorted(meta.glob(scenario_id + "*.txt"))


def _accident_category(category: Path) -> Path:
    if "_normal" in category.name:
        return category.with_name(category.name.replace("_normal", "_accident"))
    return category


def _colliding_vehicle_roles(category: Path, scenario_id: str) -> Tuple[str, ...]:
    roles = set()
    for meta_file in _meta_files(_accident_category(category), scenario_id):
        try:
            content = meta_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = COLLIDING_AGENTS.search(content)
        if match is None:
            continue
        for agent in match.groups():
            agent = agent.lower()
            if agent == "none":
                continue
            role = AGENT_TO_VEHICLE_ROLE.get(agent)
            if role is None:
                raise ValueError("unknown colliding agent %r in %s" % (agent, meta_file))
            roles.add(role)
    return tuple(sorted(roles))


def _parse_ego_velocity(label: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    if label is None:
        return None, None
    try:
        first = label.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        values = first.split()
        x_value = float(values[0])
        y_value = float(values[1])
    except (OSError, IndexError, ValueError):
        return None, None
    if not (math.isfinite(x_value) and math.isfinite(y_value)):
        return None, None
    return x_value, y_value


def _town(scenario_id: str) -> str:
    match = TOWN_NAME.search(scenario_id)
    return match.group(1) if match else "unknown"


def _scenario_key(category: Path, scenario_id: str, vehicle_role: str) -> str:
    return "%s/%s/%s" % (category.name, vehicle_role, scenario_id)


def _split_scenarios(
    scenarios: Sequence[Mapping[str, Any]], config: DeepAccidentIndexConfig
) -> Dict[str, str]:
    """Assign complete accident/normal scenario pairs to one split.

    DeepAccident records a normal counterpart for an accident run under the
    same ``scenario_id``.  Splitting those counterparts independently leaks
    scene geometry while also making validation depend on whether it happened
    to receive the normal or accident half of a pair.  The scenario id is
    therefore the atomic split group; unpaired runs remain valid one-item
    groups.
    """

    groups: Dict[str, List[str]] = defaultdict(list)
    for scenario in scenarios:
        groups[str(scenario["scenario_id"])].append(str(scenario["scenario_key"]))

    result: Dict[str, str] = {}
    group_ids = sorted(groups)
    rng = random.Random(config.split_seed)
    rng.shuffle(group_ids)
    count = len(group_ids)
    train_end = int(round(count * config.train_ratio))
    validation_end = train_end + int(round(count * config.validation_ratio))
    if count >= 3:
        train_end = min(max(1, train_end), count - 2)
        validation_end = min(max(train_end + 1, validation_end), count - 1)
    for index, group_id in enumerate(group_ids):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        for key in groups[group_id]:
            result[key] = split
    return result


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_index(
    dataset_root: Path,
    output_dir: Path,
    config: Optional[DeepAccidentIndexConfig] = None,
) -> Dict[str, Any]:
    config = config or DeepAccidentIndexConfig()
    config.validate()
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError("DeepAccident root not found: %s" % dataset_root)

    scenario_rows: List[Dict[str, Any]] = []
    frame_rows_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    rejected: List[Dict[str, Any]] = []
    rejected_sources = set()

    for camera_dir in _scenario_directories(dataset_root):
        try:
            category, vehicle_role = _camera_context(camera_dir)
        except ValueError as exc:
            rejected.append({"path": str(camera_dir), "reason": str(exc)})
            continue
        scenario_id = camera_dir.name
        try:
            colliding_vehicle_roles = _colliding_vehicle_roles(category, scenario_id)
        except ValueError as exc:
            source_key = (category.name, scenario_id)
            if source_key not in rejected_sources:
                rejected.append({"path": str(category / "meta"), "reason": str(exc)})
                rejected_sources.add(source_key)
            continue
        if not colliding_vehicle_roles:
            source_key = (category.name, scenario_id)
            if source_key not in rejected_sources:
                rejected.append(
                    {
                        "path": str(category / "meta" / (scenario_id + ".txt")),
                        "reason": "no_colliding_actor_in_accident_metadata",
                        "scenario_id": scenario_id,
                        "category": category.name,
                    }
                )
                rejected_sources.add(source_key)
            continue
        if vehicle_role not in colliding_vehicle_roles:
            continue
        key = _scenario_key(category, scenario_id, vehicle_role)
        try:
            images = sorted(camera_dir.glob("*.jpg"), key=_frame_number)
            numbers = [_frame_number(path) for path in images]
        except ValueError as exc:
            rejected.append({"path": str(camera_dir), "reason": str(exc)})
            continue
        if not images:
            rejected.append({"path": str(camera_dir), "reason": "no_front_camera_frames"})
            continue
        if len(numbers) != len(set(numbers)):
            rejected.append({"path": str(camera_dir), "reason": "duplicate_frame_numbers"})
            continue

        is_accident = "accident" in category.name.lower() and "normal" not in category.name.lower()
        missing_labels = 0
        invalid_velocity_labels = 0
        rows = []
        final_index = len(images) - 1
        for position, image in enumerate(images):
            label = _matching_label(category, vehicle_role, scenario_id, image)
            if label is None:
                missing_labels += 1
            velocity_x, velocity_y = _parse_ego_velocity(label)
            if label is not None and velocity_x is None:
                invalid_velocity_labels += 1
            frames_to_event = final_index - position if is_accident else -1
            seconds_to_event = frames_to_event / config.fps if is_accident else -1.0
            event_within_horizon = bool(
                is_accident and seconds_to_event <= config.prediction_horizon_s
            )
            rows.append(
                {
                    "scenario_key": key,
                    "scenario_id": scenario_id,
                    "category": category.name,
                    "vehicle_role": vehicle_role,
                    "colliding_vehicle_roles": list(colliding_vehicle_roles),
                    "town": _town(scenario_id),
                    "frame_number": numbers[position],
                    "frame_position": position,
                    "image_path": str(image.relative_to(dataset_root)),
                    "label_path": str(label.relative_to(dataset_root)) if label else "",
                    "ego_velocity_x": velocity_x,
                    "ego_velocity_y": velocity_y,
                    "is_accident_scenario": is_accident,
                    "event_within_horizon": event_within_horizon,
                    "seconds_to_event": seconds_to_event,
                    "target_source": (
                        "terminal_frame_proxy_for_colliding_actor"
                        if is_accident
                        else "paired_normal_colliding_actor_viewpoint"
                    ),
                }
            )

        if config.require_labels and missing_labels:
            rejected.append(
                {
                    "path": str(camera_dir),
                    "reason": "missing_labels",
                    "missing_labels": missing_labels,
                    "frames": len(images),
                }
            )
            continue
        scenario_rows.append(
            {
                "scenario_key": key,
                "scenario_id": scenario_id,
                "category": category.name,
                "vehicle_role": vehicle_role,
                "colliding_vehicle_roles": list(colliding_vehicle_roles),
                "town": _town(scenario_id),
                "is_accident_scenario": is_accident,
                "frames": len(images),
                "first_frame_number": numbers[0],
                "last_frame_number": numbers[-1],
                "missing_labels": missing_labels,
                "invalid_velocity_labels": invalid_velocity_labels,
                "meta_files": [str(path.relative_to(dataset_root)) for path in _meta_files(category, scenario_id)],
                "collision_meta_files": [
                    str(path.relative_to(dataset_root))
                    for path in _meta_files(_accident_category(category), scenario_id)
                ],
                "content_sha256": hashlib.sha256(
                    (key + ":" + ",".join(str(value) for value in numbers)).encode("utf-8")
                ).hexdigest(),
            }
        )
        frame_rows_by_scenario[key] = rows

    if not scenario_rows:
        raise RuntimeError("no valid DeepAccident colliding-actor front-camera scenarios found")

    split_lookup = _split_scenarios(scenario_rows, config)
    for scenario in scenario_rows:
        scenario["split_group"] = scenario["scenario_id"]
        scenario["split"] = split_lookup[scenario["scenario_key"]]
    frame_rows: List[Dict[str, Any]] = []
    for scenario in sorted(scenario_rows, key=lambda item: item["scenario_key"]):
        split = split_lookup[scenario["scenario_key"]]
        for row in frame_rows_by_scenario[scenario["scenario_key"]]:
            row["split"] = split
            frame_rows.append(row)

    split_counts = Counter(str(row["split"]) for row in scenario_rows)
    split_group_counts = Counter(
        str(row["split"]) for row in {row["scenario_id"]: row for row in scenario_rows}.values()
    )
    group_classes: Dict[str, set] = defaultdict(set)
    for row in scenario_rows:
        group_classes[str(row["scenario_id"])].add(bool(row["is_accident_scenario"]))
    split_accident_counts = Counter(
        "%s:%s" % (row["split"], "accident" if row["is_accident_scenario"] else "normal")
        for row in scenario_rows
    )
    audit = {
        "schema_version": 2,
        "dataset_root": str(dataset_root),
        "target_semantics": {
            "event": (
                "collision proxy at the final recorded frame from the official "
                "colliding actor's front camera"
            ),
            "viewpoint": "front camera of each actor listed in colliding agents metadata",
            "warning": "DeepAccident provides prediction/perception labels, not expert control actions",
            "prediction_horizon_s": config.prediction_horizon_s,
            "fps": config.fps,
        },
        "config": asdict(config),
        "scenarios": len(scenario_rows),
        "source_scenario_groups": len({str(row["scenario_id"]) for row in scenario_rows}),
        "frames": len(frame_rows),
        "accident_scenarios": sum(bool(row["is_accident_scenario"]) for row in scenario_rows),
        "normal_scenarios": sum(not bool(row["is_accident_scenario"]) for row in scenario_rows),
        "positive_horizon_frames": sum(bool(row["event_within_horizon"]) for row in frame_rows),
        "towns": dict(sorted(Counter(str(row["town"]) for row in scenario_rows).items())),
        "categories": dict(sorted(Counter(str(row["category"]) for row in scenario_rows).items())),
        "vehicle_roles": dict(
            sorted(Counter(str(row["vehicle_role"]) for row in scenario_rows).items())
        ),
        "split_scenarios": dict(sorted(split_counts.items())),
        "split_groups": dict(sorted(split_group_counts.items())),
        "paired_split_groups": sum(
            classes == {False, True} for classes in group_classes.values()
        ),
        "rejected_source_groups": len(
            {str(row.get("scenario_id", row["path"])) for row in rejected}
        ),
        "split_class_counts": dict(sorted(split_accident_counts.items())),
        "rejected": rejected,
    }
    _jsonl_write(output_dir / "frames.jsonl", frame_rows)
    _jsonl_write(output_dir / "scenarios.jsonl", scenario_rows)
    _json_write(output_dir / "audit.json", audit)
    return audit
