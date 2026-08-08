#!/usr/bin/env python3
"""Preserve a reviewed SimLingo/Dreamer run as a curated RL reference.

The command deliberately distinguishes a clean evaluation artifact from an
RL-training rollout. A video and a Bench2Drive result are enough to retain a
human-reviewed success, but PPO eligibility additionally requires a JSONL
trace containing per-step RL state/action data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "dreamer_rl_curated" / "positive_overtakes"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = payload.get("_checkpoint") or {}
    records = checkpoint.get("records") or []
    if not records:
        raise ValueError("Bench2Drive result contains no route record")
    return records[0]


def infraction_count(value: Any) -> float:
    if isinstance(value, list):
        return float(len(value))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_result(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = first_record(payload)
    infractions = record.get("infractions") or {}
    scores = record.get("scores") or {}
    collisions = sum(
        infraction_count(infractions.get(key))
        for key in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
    )
    offroad = infraction_count(infractions.get("outside_route_lanes"))
    red_lights = infraction_count(infractions.get("red_light"))
    blocked = infraction_count(infractions.get("vehicle_blocked"))
    metrics = {
        "status": record.get("status"),
        "route_completion": float(scores.get("score_route", 0.0) or 0.0),
        "driving_score": float(scores.get("score_composed", 0.0) or 0.0),
        "infraction_penalty": float(scores.get("score_penalty", 0.0) or 0.0),
        "collisions": collisions,
        "offroad": offroad,
        "red_lights": red_lights,
        "blocked": blocked,
        "min_speed_infractions": infraction_count(infractions.get("min_speed_infractions")),
        "duration_game_s": float((record.get("meta") or {}).get("duration_game", 0.0) or 0.0),
        "duration_system_s": float((record.get("meta") or {}).get("duration_system", 0.0) or 0.0),
    }
    return record, metrics


def validate_clean_overtake(metrics: Dict[str, Any]) -> None:
    failures = []
    if metrics["status"] != "Completed":
        failures.append(f"status={metrics['status']!r}")
    if metrics["route_completion"] < 99.9:
        failures.append(f"route_completion={metrics['route_completion']}")
    for key in ("collisions", "offroad", "red_lights", "blocked"):
        if metrics[key] != 0.0:
            failures.append(f"{key}={metrics[key]}")
    if failures:
        raise ValueError("run is not a clean positive overtake: " + ", ".join(failures))


def parse_launch(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    started_match = re.search(r"^\[dashboard\] started_at=(.+)$", text, re.MULTILINE)
    route_match = re.search(r"^\[dashboard\] route=(\S+) town=(\S+) scenario=(.+)$", text, re.MULTILINE)
    mode_match = re.search(r"^\[dashboard\] mode=(\S+) dreamer=(\S+) cot=(\S+) seed=(\d+)$", text, re.MULTILINE)
    checkpoint_match = re.search(r"^\[simlingo-eval\] model=(.+)$", text, re.MULTILINE)
    no_guard = "SIMLINGO_DREAMER_RL_NOGUARD enabled" in text
    return {
        "started_at_local": started_match.group(1).strip() if started_match else None,
        "route_id": route_match.group(1) if route_match else None,
        "town": route_match.group(2) if route_match else None,
        "scenario": route_match.group(3).strip() if route_match else None,
        "run_mode": mode_match.group(1) if mode_match else None,
        "dreamer_mode": mode_match.group(2) if mode_match else None,
        "cot_mode": mode_match.group(3) if mode_match else None,
        "seed": int(mode_match.group(4)) if mode_match else None,
        "simlingo_model": checkpoint_match.group(1).strip() if checkpoint_match else None,
        "runtime_no_guard": no_guard,
    }


def rl_trace_rows(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                status = (json.loads(line).get("status") or {})
            except (json.JSONDecodeError, AttributeError):
                continue
            if status.get("mode") == "rl_noguard" and isinstance(status.get("rl_raw_action"), list):
                count += 1
    return count


def preserve(source: Path, destination: Path, copy: bool = False) -> Dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    storage = "copy"
    if not copy:
        try:
            os.link(source, destination)
            storage = "hardlink"
        except OSError:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)
    return {
        "source": str(source),
        "preserved": str(destination.resolve()),
        "storage": storage,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def update_registry(path: Path, manifest: Dict[str, Any]) -> None:
    existing = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("id") != manifest["id"]:
                    existing.append(row)
    existing.append(manifest)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in existing:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def optional_artifacts(items: Iterable[Tuple[str, Optional[Path]]]) -> Iterable[Tuple[str, Path]]:
    for name, path in items:
        if path is not None:
            yield name, path


def slug(value: Any) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "unknown")).strip("_")
    return normalized.lower() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--launch-log", type=Path, required=True)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--carla-log", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--rl-trace", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--visual-note", default="Human-reviewed clean overtake")
    args = parser.parse_args()

    result_path = args.result.expanduser().resolve()
    launch_path = args.launch_log.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    record, metrics = parse_result(result_path)
    validate_clean_overtake(metrics)
    launch = parse_launch(launch_path)
    route_id = launch.get("route_id") or "unknown"
    seed = launch.get("seed") or "unknown"
    started_at = launch.get("started_at_local") or datetime.fromtimestamp(
        result_path.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")
    stamp = re.sub(r"\D", "", str(started_at))[:14]
    run_id = (
        f"{slug(launch.get('town'))}_{slug(launch.get('scenario'))}_"
        f"route_{slug(route_id)}_seed_{slug(seed)}_{stamp}"
    )
    output_dir = args.output_root.expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    trace_path = args.rl_trace.expanduser().resolve() if args.rl_trace else None
    trace_rows = rl_trace_rows(trace_path)
    artifacts: Dict[str, Any] = {}
    for name, source in optional_artifacts(
        (
            ("bench2drive_result", result_path),
            ("launch_log", launch_path),
            ("simlingo_run_log", args.run_log.expanduser().resolve() if args.run_log else None),
            ("carla_log", args.carla_log.expanduser().resolve() if args.carla_log else None),
            ("replay_video", args.video.expanduser().resolve() if args.video else None),
            ("rl_trace", trace_path),
        )
    ):
        artifacts[name] = preserve(source, output_dir / source.name)
    artifacts["policy_checkpoint"] = preserve(
        checkpoint_path,
        output_dir / "policy_checkpoint_at_evaluation.pt",
        copy=True,
    )

    manifest = {
        "schema_version": 1,
        "id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": "positive_clean_overtake",
        "human_review": {
            "status": "accepted",
            "visual_note": args.visual_note,
        },
        "run": {
            **launch,
            "scenario_name": record.get("scenario_name"),
            "weather_id": record.get("weather_id"),
        },
        "metrics": metrics,
        "training_eligibility": {
            "validation_reference": True,
            "ppo_update": trace_rows >= 64,
            "rl_trace_rows": trace_rows,
            "reason": (
                "complete no-guard RL rollout available"
                if trace_rows >= 64
                else "evaluation run only; replay with online RL collection before PPO update"
            ),
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_registry(args.output_root.expanduser().resolve() / "registry.jsonl", manifest)
    print(json.dumps({
        "ok": True,
        "id": run_id,
        "output_dir": str(output_dir),
        "ppo_update_eligible": manifest["training_eligibility"]["ppo_update"],
        "rl_trace_rows": trace_rows,
        "metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
