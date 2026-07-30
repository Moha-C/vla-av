#!/usr/bin/env python
"""Build an Alpamayo/action-adapter manifest from pure CARLA RGB captures."""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **_kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else []

try:
    import numpy as np
except ModuleNotFoundError:  # Allows metadata-only enrichment on machines without NumPy.
    np = None  # type: ignore[assignment]

try:
    import cv2
except ModuleNotFoundError:  # Allows metadata-only enrichment on machines without OpenCV.
    cv2 = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


CITYSCAPES_COLORS: dict[str, tuple[int, int, int]] = {
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),
    "traffic_light": (250, 170, 30),
    "traffic_sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "rider": (255, 0, 0),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "train": (0, 80, 100),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
}

VRU_LABELS = ("person", "rider", "motorcycle", "bicycle")
VEHICLE_LABELS = ("car", "truck", "bus", "train", "motorcycle", "bicycle")

DISPLAY_LABELS = {
    "person": "pedestrians",
    "rider": "riders",
    "motorcycle": "motorcycles",
    "bicycle": "bicycles",
    "car": "cars",
    "truck": "trucks",
    "bus": "buses",
    "train": "trains",
    "traffic_light": "traffic lights",
    "traffic_sign": "traffic signs",
}


DEFAULT_TRAINING_PROMPT = (
    "Role: autonomous urban driving VLA planner and control head. "
    "Input: front camera frame, ego-motion history, traffic-rule metadata, semantic "
    "scene evidence, and expert CARLA autopilot label for the exact frame. "
    "Reasoning style: learn a Chain-of-Causation before acting: perception evidence "
    "-> traffic law and right-of-way -> risk to vulnerable road users/vehicles -> "
    "safe maneuver intent -> steering, throttle, brake. "
    "Safety hierarchy: 1. avoid collision with pedestrians, cyclists, scooters, "
    "motorcyclists, vehicles, and obstacles; 2. obey red/yellow lights, stop signs, "
    "stop lines, crosswalks, blocked junctions, and unsafe merges; 3. yield to VRUs, "
    "vehicles already in conflict, parked vehicles pulling out, and traffic with "
    "priority; 4. stay centered in the drivable lane, follow markings/arrows, and "
    "avoid crossing solid lines unless needed for safety; 5. keep motion smooth, "
    "progressive, and comfortable. "
    "Decision rules: red light means brake to a full stop before the stop line; "
    "green light means proceed only when the crosswalk, junction, and ego path are "
    "clear; stop sign means complete stop, hold about 3 seconds, then proceed only "
    "when priority is clear; VRU near or entering the ego corridor means slow or "
    "stop and yield; vehicle conflict means yield and preserve gap; no hazard means "
    "track the lane at a safe speed. "
    "Training target: imitate the expert label while preserving the causal reason "
    "for the action; output stable CARLA controls and a future trajectory consistent "
    "with the Chain-of-Causation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="data/synthetic/transferred_real")
    parser.add_argument("--run-glob", default="carla_b2008_base_g*_*")
    parser.add_argument("--run-dir", action="append", default=None)
    parser.add_argument("--metadata-name", default="episode.jsonl")
    parser.add_argument("--source-video-name", default="carla_rgb.mp4")
    parser.add_argument("--seg-video-name", default="carla_seg.mp4")
    parser.add_argument("--output-dir", default="data/alpamayo_carla_dataset_b2008_base_combined")
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--jpeg-quality", type=int, default=97)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-run", type=int, default=None)
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--future-steps", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--training-prompt", default=DEFAULT_TRAINING_PROMPT)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append to an existing manifest and skip runs already present in it.",
    )
    parser.add_argument(
        "--skip-bad-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip missing/corrupt CARLA videos instead of aborting the full dataset build.",
    )
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [Path(path).expanduser().resolve() for path in args.run_dir]
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    return sorted(path for path in runs_dir.glob(args.run_glob) if path.is_dir())


def load_processed_runs(manifest_path: Path) -> tuple[set[str], int]:
    """Return source_run values already written in an existing manifest."""

    if not manifest_path.exists():
        return set(), 0

    processed_runs: set[str] = set()
    rows = 0
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed manifest row %s in %s", line_number, manifest_path)
                continue
            source_run = row.get("source_run")
            if source_run:
                processed_runs.add(str(source_run))
            rows += 1
    return processed_runs, rows


def record_timestamp(record: dict[str, Any], fallback: float) -> float:
    try:
        return float(record.get("timestamp", fallback))
    except (TypeError, ValueError):
        return fallback


def ego_location(record: dict[str, Any]) -> tuple[float, float, float]:
    ego_state = record.get("ego_state") or {}
    location = ego_state.get("location") or (0.0, 0.0, 0.0)
    return (float(location[0]), float(location[1]), float(location[2]))


def ego_yaw_rad(record: dict[str, Any]) -> float:
    ego_state = record.get("ego_state") or {}
    rotation = ego_state.get("rotation") or (0.0, 0.0, 0.0)
    return math.radians(float(rotation[1]))


def nearest_record_index(timestamps: list[float], target_timestamp: float) -> int:
    insertion = bisect.bisect_left(timestamps, target_timestamp)
    if insertion <= 0:
        return 0
    if insertion >= len(timestamps):
        return len(timestamps) - 1
    before = insertion - 1
    after = insertion
    if abs(timestamps[after] - target_timestamp) < abs(timestamps[before] - target_timestamp):
        return after
    return before


def local_xyz_sequence(
    records: list[dict[str, Any]],
    timestamps: list[float],
    center_idx: int,
    offsets_seconds: Iterable[float],
) -> list[list[float]]:
    center = records[center_idx]
    center_time = timestamps[center_idx]
    cx, cy, cz = ego_location(center)
    yaw = ego_yaw_rad(center)
    forward = (math.cos(yaw), math.sin(yaw))
    left = (-math.sin(yaw), math.cos(yaw))

    sequence: list[list[float]] = []
    for offset in offsets_seconds:
        target_idx = nearest_record_index(timestamps, center_time + float(offset))
        tx, ty, tz = ego_location(records[target_idx])
        dx = tx - cx
        dy = ty - cy
        sequence.append(
            [
                float(dx * forward[0] + dy * forward[1]),
                float(dx * left[0] + dy * left[1]),
                float(tz - cz),
            ]
        )
    return sequence


def output_image_path(output_dir: Path, run_name: str, frame_idx: int, image_format: str) -> tuple[Path, str]:
    relative = Path("images") / run_name / f"frame_{frame_idx:06d}.{image_format}"
    return output_dir / relative, str(relative)


def selected_frame(frame_idx: int, written: int, args: argparse.Namespace) -> bool:
    if frame_idx % max(1, int(args.frame_stride)) != 0:
        return False
    if args.max_frames_per_run is not None and written >= int(args.max_frames_per_run):
        return False
    return True


def semantic_stats(seg_frame_bgr: np.ndarray | None) -> dict[str, Any]:
    if seg_frame_bgr is None:
        return {
            "available": False,
            "pixel_fraction": {},
            "label_details": {},
            "dominant_labels": [],
            "vru_visible": False,
            "vru_pixel_fraction": 0.0,
            "vru_labels_visible": [],
            "vru_in_ego_corridor": False,
            "vru_near_path": False,
            "vehicle_visible": False,
            "vehicle_pixel_fraction": 0.0,
            "vehicle_labels_visible": [],
            "vehicle_in_ego_corridor": False,
            "traffic_control_visible": False,
            "traffic_control_pixel_fraction": 0.0,
            "traffic_control_labels_visible": [],
        }
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required to compute semantic stats from segmentation frames.")
    frame_rgb = cv2.cvtColor(seg_frame_bgr, cv2.COLOR_BGR2RGB)
    total_pixels = max(1, int(frame_rgb.shape[0] * frame_rgb.shape[1]))
    fractions: dict[str, float] = {}
    label_details: dict[str, dict[str, Any]] = {}
    for label, color in CITYSCAPES_COLORS.items():
        color_array = np.asarray(color, dtype=np.uint8)
        mask = np.all(frame_rgb == color_array, axis=-1)
        pixels = int(mask.sum())
        fraction = float(pixels) / float(total_pixels)
        if fraction > 0.0:
            fractions[label] = fraction
            label_details[label] = label_geometry(mask, pixels=pixels, total_pixels=total_pixels)

    vru_fraction = sum(float(fractions.get(label, 0.0)) for label in VRU_LABELS)
    vehicle_fraction = sum(float(fractions.get(label, 0.0)) for label in VEHICLE_LABELS)
    traffic_control_fraction = float(fractions.get("traffic_light", 0.0)) + float(fractions.get("traffic_sign", 0.0))
    vru_labels_visible = [
        label for label in VRU_LABELS if float(fractions.get(label, 0.0)) > 0.00002
    ]
    vehicle_labels_visible = [
        label for label in VEHICLE_LABELS if float(fractions.get(label, 0.0)) > 0.0005
    ]
    traffic_control_labels_visible = [
        label for label in ("traffic_light", "traffic_sign") if float(fractions.get(label, 0.0)) > 0.00001
    ]
    vru_in_ego_corridor = any(
        bool(label_details.get(label, {}).get("in_ego_corridor", False))
        for label in vru_labels_visible
    )
    vru_near_path = any(
        bool(label_details.get(label, {}).get("near_ego_path", False))
        for label in vru_labels_visible
    )
    vehicle_in_ego_corridor = any(
        bool(label_details.get(label, {}).get("in_ego_corridor", False))
        for label in vehicle_labels_visible
    )
    return {
        "available": True,
        "pixel_fraction": fractions,
        "label_details": label_details,
        "dominant_labels": [
            label for label, _fraction in sorted(fractions.items(), key=lambda item: item[1], reverse=True)[:8]
        ],
        "vru_visible": vru_fraction > 0.00002,
        "vru_pixel_fraction": vru_fraction,
        "vru_labels_visible": vru_labels_visible,
        "vru_in_ego_corridor": vru_in_ego_corridor,
        "vru_near_path": vru_near_path,
        "vehicle_visible": vehicle_fraction > 0.0005,
        "vehicle_pixel_fraction": vehicle_fraction,
        "vehicle_labels_visible": vehicle_labels_visible,
        "vehicle_in_ego_corridor": vehicle_in_ego_corridor,
        "traffic_control_visible": traffic_control_fraction > 0.00001,
        "traffic_control_pixel_fraction": traffic_control_fraction,
        "traffic_control_labels_visible": traffic_control_labels_visible,
    }


def label_geometry(mask: np.ndarray, *, pixels: int, total_pixels: int) -> dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy is required to compute semantic label geometry.")
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return {"pixel_fraction": 0.0}
    height, width = mask.shape[:2]
    x1 = float(xs.min()) / float(max(1, width - 1))
    x2 = float(xs.max()) / float(max(1, width - 1))
    y1 = float(ys.min()) / float(max(1, height - 1))
    y2 = float(ys.max()) / float(max(1, height - 1))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    overlaps_ego_corridor = x2 >= 0.38 and x1 <= 0.62
    bottom_near = y2 >= 0.45
    return {
        "pixel_fraction": float(pixels) / float(max(1, total_pixels)),
        "pixel_count": int(pixels),
        "bbox_norm": [x1, y1, x2, y2],
        "center_norm": [cx, cy],
        "lateral_zone": lateral_zone(cx),
        "depth_zone": depth_zone(y2),
        "in_ego_corridor": bool(overlaps_ego_corridor),
        "near_ego_path": bool(overlaps_ego_corridor and bottom_near),
    }


def lateral_zone(x_norm: float) -> str:
    if x_norm < 0.35:
        return "left"
    if x_norm > 0.65:
        return "right"
    return "center"


def depth_zone(bottom_y_norm: float) -> str:
    if bottom_y_norm >= 0.70:
        return "near"
    if bottom_y_norm >= 0.45:
        return "mid"
    return "far"


def behavior_tags(record: dict[str, Any], semantic: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    light_state = str(record.get("traffic_light_state", "None")).lower()
    hazards = [str(item).lower() for item in (record.get("hazards") or [])]
    speed = float(record.get("speed_kmh", 0.0) or 0.0)
    brake = float(record.get("brake", 0.0) or 0.0)
    throttle = float(record.get("throttle", 0.0) or 0.0)
    steer = float(record.get("steering", 0.0) or 0.0)

    if bool(record.get("at_traffic_light", False)):
        tags.append(f"traffic_light_{light_state}")
    if light_state in {"red", "yellow"} or any(item.startswith("traffic_light_") for item in hazards):
        tags.append("must_yield_to_signal")
    if light_state == "green":
        tags.append("green_light_proceed_if_clear")
    if bool(record.get("near_stop_sign", False)):
        tags.append("stop_sign_near")
        distance = record.get("near_stop_sign_distance_m")
        if distance is not None and float(distance) < 8.0:
            tags.append("stop_sign_close")
    if semantic.get("vru_visible"):
        tags.append("vru_visible")
        for label in semantic.get("vru_labels_visible") or []:
            tags.append(f"vru_{label}_visible")
    if semantic.get("vru_in_ego_corridor"):
        tags.append("vru_in_ego_corridor")
    if semantic.get("vru_near_path"):
        tags.append("vru_near_path")
    if semantic.get("vehicle_visible"):
        tags.append("vehicles_visible")
        for label in semantic.get("vehicle_labels_visible") or []:
            tags.append(f"vehicle_{label}_visible")
    if semantic.get("vehicle_in_ego_corridor"):
        tags.append("vehicle_in_ego_corridor")
    if semantic.get("traffic_control_visible"):
        tags.append("traffic_control_visible")
    if brake > 0.2:
        tags.append("expert_braking")
    if speed < 1.0 and brake > 0.1:
        tags.append("expert_stopped")
    if throttle > 0.15 and brake < 0.05:
        tags.append("expert_accelerating")
    if abs(steer) > 0.15:
        tags.append("expert_turning")
    if not tags:
        tags.append("lane_following")
    return sorted(set(tags))


def rule_context(record: dict[str, Any], semantic: dict[str, Any], tags: list[str]) -> str:
    parts: list[str] = []
    light_state = str(record.get("traffic_light_state", "None"))
    if bool(record.get("at_traffic_light", False)):
        if light_state.lower() == "red":
            parts.append("A red traffic light controls the ego path; the correct behavior is to brake and stop before the stop line.")
        elif light_state.lower() == "yellow":
            parts.append("A yellow traffic light is present; prepare to stop unless already committed safely through the intersection.")
        elif light_state.lower() == "green":
            parts.append("A green traffic light is present; proceed only if the intersection and crosswalk are clear.")
        else:
            parts.append(f"A traffic light is present with state {light_state}; obey its signal and preserve right-of-way.")
    if bool(record.get("near_stop_sign", False)):
        distance = record.get("near_stop_sign_distance_m")
        if distance is None:
            parts.append("A stop sign is near; come to a complete stop before proceeding when clear.")
        else:
            parts.append(
                f"A stop sign is approximately {float(distance):.1f} m ahead; stop completely, wait about 3 seconds, then proceed only when priority is clear."
            )
    if semantic.get("vru_visible"):
        vru_phrase = visible_label_phrase(semantic, "vru_labels_visible", fallback="vulnerable road users")
        if semantic.get("vru_near_path"):
            parts.append(
                f"{vru_phrase.capitalize()} are near the ego driving corridor; slow or stop and yield until the path is clear."
            )
        elif semantic.get("vru_in_ego_corridor"):
            parts.append(
                f"{vru_phrase.capitalize()} overlap the ego corridor; anticipate crossing/conflict and prepare to brake."
            )
        else:
            parts.append(
                f"{vru_phrase.capitalize()} are visible; monitor them and yield if they approach the ego path or a crossing."
            )
    if semantic.get("vehicle_visible"):
        vehicle_phrase = visible_label_phrase(semantic, "vehicle_labels_visible", fallback="other vehicles")
        if semantic.get("vehicle_in_ego_corridor"):
            parts.append(
                f"{vehicle_phrase.capitalize()} are in or near the ego corridor; maintain gap, avoid cutting priority, and brake if conflict develops."
            )
        else:
            parts.append(
                f"{vehicle_phrase.capitalize()} are visible; maintain safe distance, respect priority, and do not cut across their right-of-way."
            )
    if "expert_braking" in tags:
        parts.append("The expert autopilot is braking in this frame; learn the visual/contextual reason for slowing or stopping.")
    if "expert_stopped" in tags:
        parts.append("The ego vehicle is stopped or nearly stopped; hold position until the traffic rule or obstacle allows motion.")
    if not parts:
        parts.append("No immediate traffic-rule hazard is labeled; follow the lane smoothly, keep centered, and maintain safe speed.")
    return " ".join(parts)


def visible_label_phrase(semantic: dict[str, Any], key: str, *, fallback: str) -> str:
    labels = [DISPLAY_LABELS.get(str(label), str(label).replace("_", " ")) for label in (semantic.get(key) or [])]
    if not labels:
        return fallback
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def action_interpretation(record: dict[str, Any]) -> str:
    brake = float(record.get("brake", 0.0) or 0.0)
    throttle = float(record.get("throttle", 0.0) or 0.0)
    steer = float(record.get("steering", 0.0) or 0.0)
    speed = float(record.get("speed_kmh", 0.0) or 0.0)
    if speed < 1.0 and brake > 0.1:
        longitudinal = "hold a full or near-full stop"
    elif brake > 0.2:
        longitudinal = "decelerate/brake"
    elif throttle > 0.15 and brake < 0.05:
        longitudinal = "accelerate or maintain forward motion"
    elif throttle <= 0.05 and brake <= 0.05:
        longitudinal = "coast gently"
    else:
        longitudinal = "blend speed smoothly"
    if steer > 0.15:
        lateral = "steer right"
    elif steer < -0.15:
        lateral = "steer left"
    else:
        lateral = "keep lane-centered steering"
    return f"{longitudinal} while {lateral}"


def chain_of_causation(record: dict[str, Any], semantic: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    perception: list[str] = []
    rules: list[str] = []
    risks: list[str] = []

    light_state = str(record.get("traffic_light_state", "None"))
    speed = float(record.get("speed_kmh", 0.0) or 0.0)
    hazards = [str(item) for item in (record.get("hazards") or [])]

    if semantic.get("available"):
        dominant = semantic.get("dominant_labels") or []
        if dominant:
            perception.append("semantic scene contains " + ", ".join(str(label) for label in dominant[:6]))
    if semantic.get("vru_visible"):
        details = label_detail_phrases(semantic, semantic.get("vru_labels_visible") or [])
        perception.append("VRU evidence: " + "; ".join(details or [visible_label_phrase(semantic, "vru_labels_visible", fallback="vulnerable road users")]))
        if semantic.get("vru_near_path"):
            risks.append("VRU is near/in the ego corridor, so collision risk dominates the maneuver.")
        elif semantic.get("vru_in_ego_corridor"):
            risks.append("VRU overlaps the ego corridor; prepare to yield before conflict.")
        else:
            risks.append("VRU is visible but not clearly in the ego corridor; monitor and preserve a stopping option.")
    if semantic.get("vehicle_visible"):
        details = label_detail_phrases(semantic, semantic.get("vehicle_labels_visible") or [])
        perception.append("vehicle evidence: " + "; ".join(details or [visible_label_phrase(semantic, "vehicle_labels_visible", fallback="vehicles")]))
        if semantic.get("vehicle_in_ego_corridor"):
            risks.append("vehicle occupies or overlaps the ego corridor; preserve distance and right-of-way.")
    if bool(record.get("at_traffic_light", False)):
        perception.append(f"traffic light state is {light_state}")
        if light_state.lower() == "red":
            rules.append("red light requires stopping before the stop line")
        elif light_state.lower() == "yellow":
            rules.append("yellow light requires preparing to stop unless already safely committed")
        elif light_state.lower() == "green":
            rules.append("green light allows motion only after verifying crosswalk, junction, and path are clear")
    if bool(record.get("near_stop_sign", False)):
        distance = record.get("near_stop_sign_distance_m")
        if distance is None:
            perception.append("stop sign is near")
        else:
            perception.append(f"stop sign is {float(distance):.1f} m ahead")
        rules.append("stop sign requires complete stop, brief hold, and priority check before proceeding")
    if hazards:
        risks.append("hazard labels from CARLA: " + ", ".join(hazards))
    if not perception:
        perception.append("open lane-following scene without labeled immediate traffic-control hazard")
    if not rules:
        rules.append("default rule is lane following with safe speed and continuous right-of-way monitoring")
    if not risks:
        risks.append("no immediate high-priority conflict is labeled; keep smooth centered progress")

    decision = action_interpretation(record)
    return {
        "perception_evidence": perception,
        "traffic_rule_evaluation": rules,
        "risk_assessment": risks,
        "expert_decision": decision,
        "expert_action": {
            "steering": float(record.get("steering", 0.0) or 0.0),
            "throttle": float(record.get("throttle", 0.0) or 0.0),
            "brake": float(record.get("brake", 0.0) or 0.0),
            "speed_kmh": speed,
        },
        "supervision_note": "The action label is expert imitation; the causal trace explains why this label is safe for this frame.",
    }


def label_detail_phrases(semantic: dict[str, Any], labels: list[str]) -> list[str]:
    details = semantic.get("label_details") or {}
    phrases: list[str] = []
    for label in labels:
        info = details.get(label) or {}
        display = DISPLAY_LABELS.get(str(label), str(label).replace("_", " "))
        if not info:
            phrases.append(display)
            continue
        phrases.append(
            f"{display} at {info.get('lateral_zone', 'unknown')}/"
            f"{info.get('depth_zone', 'unknown')} "
            f"(ego_corridor={bool(info.get('in_ego_corridor', False))})"
        )
    return phrases


def reasoning_trace_text(trace: dict[str, Any]) -> str:
    return (
        "Chain-of-Causation: "
        "Perception: " + " ".join(trace["perception_evidence"]) + " "
        "Rule evaluation: " + " ".join(trace["traffic_rule_evaluation"]) + " "
        "Risk: " + " ".join(trace["risk_assessment"]) + " "
        "Expert decision: " + str(trace["expert_decision"]) + "."
    )


def situational_instruction(
    record: dict[str, Any],
    semantic: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, str, list[str], dict[str, Any], str]:
    tags = behavior_tags(record, semantic)
    context = rule_context(record, semantic, tags)
    trace = chain_of_causation(record, semantic, tags)
    reasoning = reasoning_trace_text(trace)
    action_text = (
        "Expert CARLA autopilot action for this exact frame: "
        f"steer={float(record.get('steering', 0.0)):+.3f}, "
        f"throttle={float(record.get('throttle', 0.0)):.3f}, "
        f"brake={float(record.get('brake', 0.0)):.3f}, "
        f"speed={float(record.get('speed_kmh', 0.0)):.1f} km/h."
    )
    return f"{args.training_prompt} Situation: {context} {reasoning} {action_text}", context, tags, trace, reasoning


def make_manifest_record(
    *,
    source_record: dict[str, Any],
    image_relpath: str,
    source_run: str,
    source_video: Path,
    frame_idx: int,
    history_xyz: list[list[float]],
    future_xyz: list[list[float]],
    semantic: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    steering = float(source_record.get("steering", 0.0))
    throttle = float(source_record.get("throttle", 0.0))
    brake = float(source_record.get("brake", 0.0))
    frame_instruction, context, tags, trace, reasoning = situational_instruction(source_record, semantic, args)
    record = dict(source_record)
    record.update(
        {
            "format": "carla_rgb_alpamayo_v1",
            "source": "carla_rgb",
            "source_run": source_run,
            "source_video": str(source_video),
            "frame_index": int(frame_idx),
            "image_path": image_relpath,
            "photoreal_frame_path": image_relpath,
            "carla_rgb_frame_path": image_relpath,
            "camera_indices": [int(args.camera_index)],
            "training_prompt": args.training_prompt,
            "navigation_prompt": args.training_prompt,
            "situational_instruction": frame_instruction,
            "rule_context": context,
            "driving_policy_tags": tags,
            "semantic_context": semantic,
            "chain_of_causation": trace,
            "reasoning_trace": reasoning,
            "reasoning_format": "perception -> rule/right-of-way -> risk -> expert maneuver/action",
            "ego_history_xyz": history_xyz,
            "ego_future_xyz": future_xyz,
            "ego_history_seconds": [
                float(-(args.history_steps - 1 - idx) * args.dt)
                for idx in range(args.history_steps)
            ],
            "ego_future_seconds": [
                float((idx + 1) * args.dt)
                for idx in range(args.future_steps)
            ],
            "action": {
                "steering": steering,
                "throttle": throttle,
                "brake": brake,
            },
        }
    )
    return record


def process_run(run_dir: Path, *, output_dir: Path, manifest_file: Any, args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = run_dir / args.metadata_name
    source_video = run_dir / args.source_video_name
    seg_video = run_dir / args.seg_video_name
    if not metadata_path.exists():
        LOGGER.warning("Skipping %s: missing %s", run_dir.name, metadata_path.name)
        return {"run": run_dir.name, "frames": 0, "skipped": "missing_metadata"}
    if not source_video.exists():
        LOGGER.warning("Skipping %s: missing %s", run_dir.name, source_video.name)
        return {"run": run_dir.name, "frames": 0, "skipped": "missing_source_video"}
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required to decode CARLA videos. Run inside the project conda env.")

    metadata = load_jsonl(metadata_path)
    if not metadata:
        LOGGER.warning("Skipping %s: empty metadata", run_dir.name)
        return {"run": run_dir.name, "frames": 0, "skipped": "empty_metadata"}
    timestamps = [
        record_timestamp(record, fallback=float(idx) * args.dt)
        for idx, record in enumerate(metadata)
    ]

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        message = f"Could not open CARLA RGB video: {source_video}"
        if args.skip_bad_videos:
            LOGGER.warning("Skipping %s: %s", run_dir.name, message)
            return {
                "run": run_dir.name,
                "frames": 0,
                "skipped": "bad_source_video",
                "source_video": str(source_video),
            }
        raise RuntimeError(message)
    seg_capture = cv2.VideoCapture(str(seg_video)) if seg_video.exists() else None
    if seg_capture is not None and not seg_capture.isOpened():
        LOGGER.warning("Could not open CARLA segmentation video: %s", seg_video)
        seg_capture.release()
        seg_capture = None

    written = 0
    decoded = 0
    total = int(min(capture.get(cv2.CAP_PROP_FRAME_COUNT) or len(metadata), len(metadata)))
    progress = tqdm(total=total, desc=run_dir.name, unit="frame", leave=False, dynamic_ncols=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok or decoded >= len(metadata):
                break
            seg_frame = None
            if seg_capture is not None:
                seg_ok, candidate_seg = seg_capture.read()
                if seg_ok:
                    seg_frame = candidate_seg
            frame_idx = decoded
            decoded += 1
            progress.update(1)
            if not selected_frame(frame_idx, written, args):
                continue
            semantic = semantic_stats(seg_frame)

            image_path, image_relpath = output_image_path(output_dir, run_dir.name, frame_idx, args.image_format)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.dry_run:
                if args.image_format == "jpg":
                    cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
                else:
                    cv2.imwrite(str(image_path), frame)

            history_offsets = [
                -(args.history_steps - 1 - idx) * args.dt
                for idx in range(args.history_steps)
            ]
            future_offsets = [
                (idx + 1) * args.dt
                for idx in range(args.future_steps)
            ]
            manifest_record = make_manifest_record(
                source_record=metadata[frame_idx],
                image_relpath=image_relpath,
                source_run=run_dir.name,
                source_video=source_video,
                frame_idx=frame_idx,
                history_xyz=local_xyz_sequence(metadata, timestamps, frame_idx, history_offsets),
                future_xyz=local_xyz_sequence(metadata, timestamps, frame_idx, future_offsets),
                semantic=semantic,
                args=args,
            )
            if not args.dry_run:
                manifest_file.write(json.dumps(manifest_record) + "\n")
            written += 1
    finally:
        progress.close()
        capture.release()
        if seg_capture is not None:
            seg_capture.release()

    LOGGER.info("Prepared %s CARLA RGB frames from %s", written, run_dir.name)
    return {
        "run": run_dir.name,
        "frames": written,
        "metadata": str(metadata_path),
        "source_video": str(source_video),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    run_dirs = discover_run_dirs(args)
    if not run_dirs:
        raise RuntimeError("No CARLA run directories matched the requested inputs.")

    processed_runs, existing_frames = load_processed_runs(manifest_path) if args.resume else (set(), 0)
    if args.resume and processed_runs:
        LOGGER.info(
            "Resume enabled: keeping %s existing manifest rows from %s processed runs.",
            existing_frames,
            len(processed_runs),
        )

    summaries: list[dict[str, Any]] = []
    mode = "a" if args.resume else "w"
    with manifest_path.open(mode, encoding="utf-8") as manifest_file:
        for run_dir in tqdm(run_dirs, desc="runs", unit="run", dynamic_ncols=True):
            if run_dir.name in processed_runs:
                summaries.append({"run": run_dir.name, "frames": 0, "skipped": "already_processed"})
                continue
            summaries.append(process_run(run_dir, output_dir=output_dir, manifest_file=manifest_file, args=args))

    summary = {
        "format": "carla_rgb_alpamayo_v1",
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "total_frames": existing_frames + sum(int(item.get("frames", 0)) for item in summaries),
        "existing_frames_before_resume": int(existing_frames),
        "resume": bool(args.resume),
        "skip_bad_videos": bool(args.skip_bad_videos),
        "runs": summaries,
        "history_steps": int(args.history_steps),
        "future_steps": int(args.future_steps),
        "dt": float(args.dt),
        "frame_stride": int(args.frame_stride),
        "training_prompt": args.training_prompt,
        "dry_run": bool(args.dry_run),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Wrote summary: %s", summary_path)


if __name__ == "__main__":
    main()
