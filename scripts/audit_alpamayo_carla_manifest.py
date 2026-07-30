#!/usr/bin/env python
"""Audit a CARLA/Alpamayo manifest before training."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/alpamayo_carla_dataset_b2008_base_combined/manifest.jsonl")
    parser.add_argument("--output-json", default="artifacts/alpamayo_carla_manifest_audit.json")
    parser.add_argument("--examples-jsonl", default="artifacts/alpamayo_carla_manifest_audit_examples.jsonl")
    parser.add_argument("--check-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-image-checks", type=int, default=5000)
    parser.add_argument("--max-examples-per-bucket", type=int, default=20)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            rows.append(row)
    return rows


def action(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("action") or row
    return {
        "steering": float(raw.get("steering", row.get("steering", 0.0)) or 0.0),
        "throttle": float(raw.get("throttle", row.get("throttle", 0.0)) or 0.0),
        "brake": float(raw.get("brake", row.get("brake", 0.0)) or 0.0),
    }


def speed(row: dict[str, Any]) -> float:
    return float(row.get("speed_kmh", 0.0) or 0.0)


def tags(row: dict[str, Any]) -> set[str]:
    return set(str(tag) for tag in (row.get("driving_policy_tags") or []))


def semantic(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("semantic_context") or {}


def is_braking(row: dict[str, Any]) -> bool:
    return action(row)["brake"] > 0.2


def is_stopped(row: dict[str, Any]) -> bool:
    return speed(row) < 1.0 and action(row)["brake"] > 0.1


def is_accelerating(row: dict[str, Any]) -> bool:
    a = action(row)
    return a["throttle"] > 0.15 and a["brake"] < 0.05


def is_red(row: dict[str, Any]) -> bool:
    return bool(row.get("at_traffic_light", False)) and str(row.get("traffic_light_state", "")).lower() == "red"


def is_yellow(row: dict[str, Any]) -> bool:
    return bool(row.get("at_traffic_light", False)) and str(row.get("traffic_light_state", "")).lower() == "yellow"


def is_green(row: dict[str, Any]) -> bool:
    return bool(row.get("at_traffic_light", False)) and str(row.get("traffic_light_state", "")).lower() == "green"


def stop_distance(row: dict[str, Any]) -> float | None:
    raw = row.get("near_stop_sign_distance_m")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def is_stop_close(row: dict[str, Any]) -> bool:
    dist = stop_distance(row)
    return bool(row.get("near_stop_sign", False)) and dist is not None and dist < 8.0


def has_vru(row: dict[str, Any]) -> bool:
    sem = semantic(row)
    return bool(sem.get("vru_visible")) or "vru_visible" in tags(row)


def has_vru_near(row: dict[str, Any]) -> bool:
    sem = semantic(row)
    row_tags = tags(row)
    return bool(sem.get("vru_near_path")) or "vru_near_path" in row_tags


def has_vehicle(row: dict[str, Any]) -> bool:
    sem = semantic(row)
    return bool(sem.get("vehicle_visible")) or "vehicles_visible" in tags(row)


def percent(numer: int | float, denom: int | float) -> float:
    return 0.0 if not denom else float(numer) * 100.0 / float(denom)


def bucket_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {
            "frames": 0,
            "braking_pct": 0.0,
            "stopped_pct": 0.0,
            "accelerating_pct": 0.0,
            "mean_speed_kmh": 0.0,
            "mean_throttle": 0.0,
            "mean_brake": 0.0,
        }
    speeds = [speed(row) for row in rows]
    throttles = [action(row)["throttle"] for row in rows]
    brakes = [action(row)["brake"] for row in rows]
    return {
        "frames": count,
        "braking_pct": percent(sum(is_braking(row) for row in rows), count),
        "stopped_pct": percent(sum(is_stopped(row) for row in rows), count),
        "accelerating_pct": percent(sum(is_accelerating(row) for row in rows), count),
        "mean_speed_kmh": float(statistics.fmean(speeds)),
        "mean_throttle": float(statistics.fmean(throttles)),
        "mean_brake": float(statistics.fmean(brakes)),
    }


def add_example(
    examples: dict[str, list[dict[str, Any]]],
    bucket: str,
    row: dict[str, Any],
    *,
    max_examples: int,
    reason: str,
) -> None:
    if len(examples[bucket]) >= max_examples:
        return
    examples[bucket].append(
        {
            "bucket": bucket,
            "reason": reason,
            "source_run": row.get("source_run"),
            "frame_index": row.get("frame_index"),
            "image_path": row.get("image_path"),
            "action": action(row),
            "speed_kmh": speed(row),
            "traffic_light_state": row.get("traffic_light_state"),
            "near_stop_sign_distance_m": row.get("near_stop_sign_distance_m"),
            "driving_policy_tags": row.get("driving_policy_tags") or [],
            "rule_context": row.get("rule_context"),
            "reasoning_trace": row.get("reasoning_trace"),
        }
    )


def run_level_compliance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_run[str(row.get("source_run", ""))].append(row)

    red_runs = stop_runs = vru_near_runs = 0
    red_runs_with_stop = stop_runs_with_stop = vru_near_runs_with_yield = 0
    for run_rows in by_run.values():
        run_rows.sort(key=lambda row: int(row.get("frame_index", row.get("sample_index", 0)) or 0))
        if any(is_red(row) for row in run_rows):
            red_runs += 1
            if any(is_red(row) and (is_braking(row) or is_stopped(row)) for row in run_rows):
                red_runs_with_stop += 1
        if any(is_stop_close(row) for row in run_rows):
            stop_runs += 1
            if any(is_stop_close(row) and is_stopped(row) for row in run_rows):
                stop_runs_with_stop += 1
        if any(has_vru_near(row) for row in run_rows):
            vru_near_runs += 1
            if any(has_vru_near(row) and (is_braking(row) or is_stopped(row)) for row in run_rows):
                vru_near_runs_with_yield += 1

    return {
        "runs": len(by_run),
        "red_light_runs": red_runs,
        "red_light_runs_with_brake_or_stop": red_runs_with_stop,
        "red_light_run_compliance_pct": percent(red_runs_with_stop, red_runs),
        "stop_sign_close_runs": stop_runs,
        "stop_sign_close_runs_with_full_stop": stop_runs_with_stop,
        "stop_sign_run_compliance_pct": percent(stop_runs_with_stop, stop_runs),
        "vru_near_path_runs": vru_near_runs,
        "vru_near_path_runs_with_brake_or_stop": vru_near_runs_with_yield,
        "vru_near_path_run_yield_pct": percent(vru_near_runs_with_yield, vru_near_runs),
    }


def audit(rows: list[dict[str, Any]], *, dataset_root: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    row_tags = collections.Counter()
    source_runs = set()
    missing_coc = 0
    bad_history = 0
    bad_future = 0
    missing_prompt = 0
    checked_images = 0
    missing_images = 0

    red_rows: list[dict[str, Any]] = []
    yellow_rows: list[dict[str, Any]] = []
    green_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []
    stop_close_rows: list[dict[str, Any]] = []
    vru_rows: list[dict[str, Any]] = []
    vru_near_rows: list[dict[str, Any]] = []
    vehicle_rows: list[dict[str, Any]] = []

    for row in rows:
        source_runs.add(str(row.get("source_run", "")))
        row_tags.update(row.get("driving_policy_tags") or [])
        if not row.get("chain_of_causation") or not row.get("reasoning_trace"):
            missing_coc += 1
            add_example(examples, "missing_coc", row, max_examples=args.max_examples_per_bucket, reason="No Chain-of-Causation fields")
        if len(row.get("ego_history_xyz") or []) != 16:
            bad_history += 1
        if len(row.get("ego_future_xyz") or []) != 64:
            bad_future += 1
        if not row.get("training_prompt") or not row.get("navigation_prompt"):
            missing_prompt += 1
        if args.check_images and checked_images < int(args.max_image_checks):
            checked_images += 1
            image_path = dataset_root / str(row.get("image_path", ""))
            if not image_path.exists():
                missing_images += 1
                add_example(examples, "missing_image", row, max_examples=args.max_examples_per_bucket, reason=str(image_path))

        if is_red(row):
            red_rows.append(row)
            if is_accelerating(row) and not is_braking(row):
                add_example(examples, "red_light_accelerating", row, max_examples=args.max_examples_per_bucket, reason="Red light but expert label accelerates")
        if is_yellow(row):
            yellow_rows.append(row)
        if is_green(row):
            green_rows.append(row)
        if bool(row.get("near_stop_sign", False)):
            stop_rows.append(row)
        if is_stop_close(row):
            stop_close_rows.append(row)
            if stop_distance(row) is not None and stop_distance(row) < 5.0 and is_accelerating(row):
                add_example(examples, "stop_close_accelerating", row, max_examples=args.max_examples_per_bucket, reason="Stop sign close but expert label accelerates")
        if has_vru(row):
            vru_rows.append(row)
        if has_vru_near(row):
            vru_near_rows.append(row)
            if speed(row) > 15.0 and is_accelerating(row):
                add_example(examples, "vru_near_fast_accelerating", row, max_examples=args.max_examples_per_bucket, reason="VRU near path but speed/throttle remain high")
        if has_vehicle(row):
            vehicle_rows.append(row)

    quality = {
        "manifest_rows": len(rows),
        "source_runs": len(source_runs),
        "missing_chain_of_causation": missing_coc,
        "bad_ego_history_len": bad_history,
        "bad_ego_future_len": bad_future,
        "missing_prompt_fields": missing_prompt,
        "checked_images": checked_images,
        "missing_images": missing_images,
    }
    coverage = {
        "top_tags": row_tags.most_common(80),
        "red_light": bucket_stats(red_rows),
        "yellow_light": bucket_stats(yellow_rows),
        "green_light": bucket_stats(green_rows),
        "stop_sign_any": bucket_stats(stop_rows),
        "stop_sign_close": bucket_stats(stop_close_rows),
        "vru_visible": bucket_stats(vru_rows),
        "vru_near_path": bucket_stats(vru_near_rows),
        "vehicles_visible": bucket_stats(vehicle_rows),
    }
    risk_examples = {key: len(value) for key, value in examples.items()}
    summary = {
        "quality": quality,
        "coverage": coverage,
        "run_level": run_level_compliance(rows),
        "warning_example_counts": risk_examples,
        "go_no_go": go_no_go(quality, coverage, run_level_compliance(rows), risk_examples),
    }
    return summary, examples


def go_no_go(
    quality: dict[str, Any],
    coverage: dict[str, Any],
    run_level: dict[str, Any],
    risk_examples: dict[str, int],
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if quality["manifest_rows"] < 100000:
        warnings.append("Dataset has fewer than 100k frames.")
    if quality["missing_chain_of_causation"] > 0:
        blockers.append("Some rows are missing Chain-of-Causation fields.")
    if quality["bad_ego_history_len"] > 0 or quality["bad_ego_future_len"] > 0:
        blockers.append("Some rows do not match Alpamayo 16-history/64-future trajectory shape.")
    if coverage["red_light"]["frames"] < 10000:
        warnings.append("Red-light coverage is low.")
    if coverage["green_light"]["frames"] < 1000:
        warnings.append("Green-light coverage is low.")
    if coverage["stop_sign_close"]["frames"] < 1000:
        warnings.append("Close stop-sign coverage is low.")
    if coverage["vru_visible"]["frames"] < 10000:
        warnings.append("VRU-visible coverage is low.")
    if coverage["vru_near_path"]["frames"] < 500:
        warnings.append("VRU-near-path coverage is still low; consider more pedestrian/cyclist crossing captures.")
    if run_level["red_light_runs"] and run_level["red_light_run_compliance_pct"] < 80.0:
        warnings.append("Many red-light runs do not include clear braking/stopped labels.")
    if run_level["stop_sign_close_runs"] and run_level["stop_sign_run_compliance_pct"] < 50.0:
        warnings.append("Many stop-sign-close runs do not include a full stop label.")
    if risk_examples.get("red_light_accelerating", 0) > 0:
        warnings.append("Found examples where red-light rows accelerate; inspect whether they are false positives or bad labels.")
    return {
        "status": "GO" if not blockers else "NO_GO",
        "blockers": blockers,
        "warnings": warnings,
    }


def print_report(summary: dict[str, Any]) -> None:
    quality = summary["quality"]
    coverage = summary["coverage"]
    run_level = summary["run_level"]
    print("\n=== QUALITY ===")
    for key, value in quality.items():
        print(f"{key}: {value}")
    print("\n=== KEY COVERAGE ===")
    for key in [
        "red_light",
        "yellow_light",
        "green_light",
        "stop_sign_any",
        "stop_sign_close",
        "vru_visible",
        "vru_near_path",
        "vehicles_visible",
    ]:
        print(f"{key}: {json.dumps(coverage[key], sort_keys=True)}")
    print("\n=== RUN LEVEL ===")
    for key, value in run_level.items():
        print(f"{key}: {value}")
    print("\n=== TOP TAGS ===")
    for tag, count in coverage["top_tags"][:50]:
        print(f"{tag}: {count}")
    print("\n=== WARNINGS ===")
    print(json.dumps(summary["go_no_go"], indent=2))
    print("\nwarning_example_counts:", summary["warning_example_counts"])


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    examples_jsonl = Path(args.examples_jsonl).expanduser().resolve()
    rows = read_manifest(manifest)
    summary, examples = audit(rows, dataset_root=manifest.parent, args=args)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    examples_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with examples_jsonl.open("w", encoding="utf-8") as file:
        for bucket, bucket_examples in examples.items():
            for example in bucket_examples:
                file.write(json.dumps({"bucket": bucket, **example}) + "\n")

    print_report(summary)
    print(f"\nsummary_json: {output_json}")
    print(f"examples_jsonl: {examples_jsonl}")


if __name__ == "__main__":
    main()
