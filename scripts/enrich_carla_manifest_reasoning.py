#!/usr/bin/env python
"""Add Chain-of-Causation reasoning fields to an existing CARLA manifest."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.prepare_alpamayo_carla_dataset import (  # noqa: E402
    DEFAULT_TRAINING_PROMPT,
    VEHICLE_LABELS,
    VRU_LABELS,
    behavior_tags,
    rule_context,
    semantic_stats,
    situational_instruction,
)

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **_kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/alpamayo_carla_dataset_b2008_base_combined/manifest.jsonl")
    parser.add_argument("--output", default=None, help="Output manifest. Defaults to in-place update.")
    parser.add_argument("--training-prompt", default=DEFAULT_TRAINING_PROMPT)
    parser.add_argument("--recompute-semantic-from-seg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--recompute-semantic-filter",
        default="all",
        choices=("all", "vru", "safety"),
        help=(
            "When recomputing segmentation geometry, limit expensive video reads. "
            "'vru' recomputes only rows that already have VRU pixels/tags; 'safety' "
            "also includes braking/stopped, traffic lights, stop signs, and vehicles."
        ),
    )
    parser.add_argument(
        "--semantic-max-width",
        type=int,
        default=0,
        help="Resize segmentation frames with nearest-neighbor before mask geometry. 640 is much faster for QA tags.",
    )
    parser.add_argument("--runs-dir", default="data/synthetic/transferred_real")
    parser.add_argument("--seg-video-name", default="carla_seg.mp4")
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def backfill_semantic_context(semantic: dict[str, Any]) -> dict[str, Any]:
    semantic = dict(semantic or {})
    fractions = semantic.get("pixel_fraction") or {}

    vru_labels = [
        label for label in VRU_LABELS if float(fractions.get(label, 0.0) or 0.0) > 0.00002
    ]
    vehicle_labels = [
        label for label in VEHICLE_LABELS if float(fractions.get(label, 0.0) or 0.0) > 0.0005
    ]
    traffic_labels = [
        label
        for label in ("traffic_light", "traffic_sign")
        if float(fractions.get(label, 0.0) or 0.0) > 0.00001
    ]

    vru_fraction = sum(float(fractions.get(label, 0.0) or 0.0) for label in VRU_LABELS)
    vehicle_fraction = sum(float(fractions.get(label, 0.0) or 0.0) for label in VEHICLE_LABELS)
    traffic_fraction = sum(float(fractions.get(label, 0.0) or 0.0) for label in ("traffic_light", "traffic_sign"))

    semantic.setdefault("available", bool(fractions))
    semantic.setdefault("label_details", {})
    semantic.setdefault(
        "dominant_labels",
        [label for label, _value in sorted(fractions.items(), key=lambda item: item[1], reverse=True)[:8]],
    )
    semantic["vru_pixel_fraction"] = float(semantic.get("vru_pixel_fraction", vru_fraction) or 0.0)
    semantic["vehicle_pixel_fraction"] = float(semantic.get("vehicle_pixel_fraction", vehicle_fraction) or 0.0)
    semantic["traffic_control_pixel_fraction"] = float(
        semantic.get("traffic_control_pixel_fraction", traffic_fraction) or 0.0
    )
    semantic["vru_visible"] = bool(semantic.get("vru_visible", vru_fraction > 0.00002))
    semantic["vehicle_visible"] = bool(semantic.get("vehicle_visible", vehicle_fraction > 0.0005))
    semantic["traffic_control_visible"] = bool(
        semantic.get("traffic_control_visible", traffic_fraction > 0.00001)
    )
    semantic["vru_labels_visible"] = list(semantic.get("vru_labels_visible") or vru_labels)
    semantic["vehicle_labels_visible"] = list(semantic.get("vehicle_labels_visible") or vehicle_labels)
    semantic["traffic_control_labels_visible"] = list(
        semantic.get("traffic_control_labels_visible") or traffic_labels
    )
    semantic.setdefault("vru_in_ego_corridor", False)
    semantic.setdefault("vru_near_path", False)
    semantic.setdefault("vehicle_in_ego_corridor", False)
    return semantic


def enrich_record(record: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    semantic = backfill_semantic_context(record.get("semantic_context") or {})
    prompt_args = SimpleNamespace(training_prompt=args.training_prompt)
    frame_instruction, context, tags, trace, reasoning = situational_instruction(
        record,
        semantic,
        prompt_args,
    )
    enriched = dict(record)
    enriched.update(
        {
            "training_prompt": args.training_prompt,
            "navigation_prompt": args.training_prompt,
            "situational_instruction": frame_instruction,
            "rule_context": context,
            "driving_policy_tags": tags,
            "semantic_context": semantic,
            "chain_of_causation": trace,
            "reasoning_trace": reasoning,
            "reasoning_format": "perception -> rule/right-of-way -> risk -> expert maneuver/action",
        }
    )
    return enriched


class SegmentationReader:
    def __init__(self, *, runs_dir: Path, seg_video_name: str, semantic_max_width: int = 0) -> None:
        self.runs_dir = runs_dir
        self.seg_video_name = seg_video_name
        self.semantic_max_width = max(0, int(semantic_max_width))
        self._run_name: str | None = None
        self._capture: Any = None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._run_name = None

    def semantic_for_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        source_run = str(record.get("source_run") or "")
        frame_idx = int(record.get("frame_index", record.get("sample_index", 0)) or 0)
        if not source_run:
            return None
        if source_run != self._run_name:
            self.close()
            seg_video = self.runs_dir / source_run / self.seg_video_name
            if not seg_video.exists():
                return None
            import cv2

            capture = cv2.VideoCapture(str(seg_video))
            if not capture.isOpened():
                capture.release()
                return None
            self._capture = capture
            self._run_name = source_run
        if self._capture is None:
            return None
        current_pos = int(self._capture.get(1))
        if current_pos != frame_idx:
            self._capture.set(1, frame_idx)
        ok, frame = self._capture.read()
        if not ok:
            return None
        if self.semantic_max_width > 0 and int(frame.shape[1]) > self.semantic_max_width:
            import cv2

            scale = float(self.semantic_max_width) / float(frame.shape[1])
            target_size = (int(self.semantic_max_width), max(1, int(round(frame.shape[0] * scale))))
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
        return semantic_stats(frame)


def should_recompute_semantic(record: dict[str, Any], *, mode: str) -> bool:
    if mode == "all":
        return True
    tags = set(str(tag) for tag in (record.get("driving_policy_tags") or []))
    semantic = record.get("semantic_context") or {}
    hazards = [str(item).lower() for item in (record.get("hazards") or [])]
    has_vru = bool(semantic.get("vru_visible")) or "vru_visible" in tags
    if mode == "vru":
        return has_vru
    if mode == "safety":
        return (
            has_vru
            or bool(semantic.get("vehicle_visible"))
            or "vehicles_visible" in tags
            or "expert_braking" in tags
            or "expert_stopped" in tags
            or bool(record.get("at_traffic_light", False))
            or bool(record.get("near_stop_sign", False))
            or any(item.startswith("traffic_light_") for item in hazards)
        )
    return True


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else manifest
    temp_output = output.with_suffix(output.suffix + ".tmp")
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    seg_reader = (
        SegmentationReader(
            runs_dir=runs_dir,
            seg_video_name=str(args.seg_video_name),
            semantic_max_width=int(args.semantic_max_width),
        )
        if args.recompute_semantic_from_seg
        else None
    )

    counts = collections.Counter()
    rows = 0
    recomputed_semantic = 0
    skipped_recompute = 0
    try:
        with manifest.open("r", encoding="utf-8") as src:
            raw_lines = [line for line in src if line.strip()]
        with temp_output.open("w", encoding="utf-8") as dst:
            for raw_line in tqdm(raw_lines, desc="enrich manifest", unit="frame", dynamic_ncols=True):
                line = raw_line.strip()
                if not line:
                    continue
                raw_record = json.loads(line)
                if seg_reader is not None and should_recompute_semantic(
                    raw_record,
                    mode=str(args.recompute_semantic_filter),
                ):
                    recomputed = seg_reader.semantic_for_record(raw_record)
                    if recomputed is not None:
                        raw_record["semantic_context"] = recomputed
                        recomputed_semantic += 1
                elif seg_reader is not None:
                    skipped_recompute += 1
                record = enrich_record(raw_record, args=args)
                rows += 1
                counts.update(record.get("driving_policy_tags") or [])
                dst.write(json.dumps(record) + "\n")
    finally:
        if seg_reader is not None:
            seg_reader.close()

    if output == manifest and args.backup:
        backup_path = manifest.with_suffix(manifest.suffix + ".bak")
        shutil.copy2(manifest, backup_path)
        print(f"backup: {backup_path}")
    temp_output.replace(output)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "rows": rows,
                "recomputed_semantic": recomputed_semantic,
                "skipped_recompute": skipped_recompute,
                "top_tags": counts.most_common(30),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
