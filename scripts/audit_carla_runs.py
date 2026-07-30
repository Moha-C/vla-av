#!/usr/bin/env python
"""Audit CARLA capture runs and export preview frames with labels."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="data/synthetic/transferred_real")
    parser.add_argument("--run-glob", default="carla_b2008_base_g*_*")
    parser.add_argument("--output-dir", default="artifacts/carla_b2008_audit")
    parser.add_argument("--metadata-name", default="episode.jsonl")
    parser.add_argument("--video-name", default="carla_rgb.mp4")
    parser.add_argument("--samples-per-category", type=int, default=12)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def categories(record: dict[str, Any]) -> list[str]:
    result: list[str] = []
    light = str(record.get("traffic_light_state", "None")).lower()
    hazards = [str(item).lower() for item in (record.get("hazards") or [])]
    brake = float(record.get("brake", 0.0) or 0.0)
    throttle = float(record.get("throttle", 0.0) or 0.0)
    speed = float(record.get("speed_kmh", 0.0) or 0.0)
    steer = float(record.get("steering", 0.0) or 0.0)
    if bool(record.get("at_traffic_light", False)):
        result.append(f"traffic_light_{light}")
    if light in {"red", "yellow"}:
        result.append("must_stop_for_light")
    if light == "green":
        result.append("green_light")
    if bool(record.get("near_stop_sign", False)):
        result.append("stop_sign")
        distance = record.get("near_stop_sign_distance_m")
        if distance is not None and float(distance) < 8.0:
            result.append("stop_sign_close")
    for hazard in hazards:
        result.append(f"hazard_{hazard}")
    if brake > 0.2:
        result.append("expert_braking")
    if speed < 1.0 and brake > 0.1:
        result.append("expert_stopped")
    if throttle > 0.15 and brake < 0.05:
        result.append("expert_accelerating")
    if abs(steer) > 0.15:
        result.append("expert_turning")
    if not result:
        result.append("lane_following")
    return sorted(set(result))


def draw_overlay(frame_bgr: np.ndarray, record: dict[str, Any], run_name: str, frame_idx: int) -> np.ndarray:
    image = frame_bgr.copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 170), (0, 0, 0), thickness=-1)
    image = cv2.addWeighted(overlay, 0.58, image, 0.42, 0.0)
    lines = [
        f"{run_name} frame={frame_idx}",
        f"action steer={float(record.get('steering', 0.0)):+.3f} throttle={float(record.get('throttle', 0.0)):.3f} brake={float(record.get('brake', 0.0)):.3f} speed={float(record.get('speed_kmh', 0.0)):.1f} km/h",
        f"light at={bool(record.get('at_traffic_light', False))} state={record.get('traffic_light_state', 'None')} stop={bool(record.get('near_stop_sign', False))} stop_dist={record.get('near_stop_sign_distance_m')}",
        f"hazards={record.get('hazards') or []}",
        f"tags={categories(record)}",
    ]
    y = 28
    for line in lines:
        cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    return image


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    counts = collections.Counter()
    actions = {
        "frames": 0,
        "brake_gt_02": 0,
        "stopped": 0,
        "throttle_gt_015": 0,
        "turning": 0,
    }
    selected: dict[str, list[tuple[Path, int, dict[str, Any]]]] = collections.defaultdict(list)
    run_summaries: list[dict[str, Any]] = []

    run_dirs = sorted(path for path in runs_dir.glob(args.run_glob) if path.is_dir())
    for run_dir in tqdm(run_dirs, desc="audit runs", unit="run", dynamic_ncols=True):
        metadata_path = run_dir / args.metadata_name
        if not metadata_path.exists():
            run_summaries.append({"run": run_dir.name, "skipped": "missing_metadata"})
            continue
        rows = load_jsonl(metadata_path)
        run_counts = collections.Counter()
        for idx, row in enumerate(rows):
            actions["frames"] += 1
            if float(row.get("brake", 0.0) or 0.0) > 0.2:
                actions["brake_gt_02"] += 1
            if float(row.get("speed_kmh", 0.0) or 0.0) < 1.0 and float(row.get("brake", 0.0) or 0.0) > 0.1:
                actions["stopped"] += 1
            if float(row.get("throttle", 0.0) or 0.0) > 0.15:
                actions["throttle_gt_015"] += 1
            if abs(float(row.get("steering", 0.0) or 0.0)) > 0.15:
                actions["turning"] += 1
            for category in categories(row):
                counts[category] += 1
                run_counts[category] += 1
                if len(selected[category]) < int(args.samples_per_category):
                    selected[category].append((run_dir, idx, row))
        run_summaries.append({"run": run_dir.name, "frames": len(rows), "categories": dict(run_counts)})

    for category, samples in selected.items():
        category_dir = preview_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx, (run_dir, frame_idx, row) in enumerate(samples):
            video_path = run_dir / args.video_name
            frame = read_frame(video_path, frame_idx)
            if frame is None:
                continue
            preview = draw_overlay(frame, row, run_dir.name, frame_idx)
            cv2.imwrite(str(category_dir / f"{sample_idx:03d}_{run_dir.name}_f{frame_idx:06d}.jpg"), preview)

    summary = {
        "runs_dir": str(runs_dir),
        "run_glob": args.run_glob,
        "run_count": len(run_dirs),
        "category_counts": dict(counts.most_common()),
        "action_counts": actions,
        "runs": run_summaries,
        "preview_dir": str(preview_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_count": len(run_dirs), "frames": actions["frames"], "category_counts": dict(counts.most_common(20)), "preview_dir": str(preview_dir)}, indent=2))


if __name__ == "__main__":
    main()
