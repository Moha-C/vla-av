#!/usr/bin/env python3
"""Run paired native/Report-Dreamer CARLA evaluations on identical route seeds."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from summarize_report_dreamer_run import bench2drive_summary
except ImportError:  # Imported as scripts.run_report_dreamer_ab_campaign.
    from scripts.summarize_report_dreamer_run import bench2drive_summary

ROOT = Path(__file__).resolve().parents[1]
SIMLINGO_LOGS = ROOT / "logs" / "simlingo_eval"


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def expected_paths(route_id: str, seed: str) -> Dict[str, Path]:
    label = "bench2drive_%s" % route_id
    return {
        "result": SIMLINGO_LOGS / ("results_%s_seed_%s.json" % (label, seed)),
        "log": SIMLINGO_LOGS / ("run_%s_seed_%s.log" % (label, seed)),
    }


def snapshot_mtimes(paths: Dict[str, Path]) -> Dict[str, int]:
    return {
        key: path.stat().st_mtime_ns if path.exists() else -1
        for key, path in paths.items()
    }


def aggregate(rows: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition and row.get("eligible")]
    count = len(selected)
    if not count:
        return {"runs": 0}

    def numeric_values(key: str) -> List[float]:
        result = []
        for row in selected:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                result.append(float(value))
        return result

    def stats(key: str) -> Dict[str, Any]:
        values = numeric_values(key)
        if not values:
            return {"count": 0, "mean": None, "std": None}
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    driving = stats("driving_score")
    completion = stats("route_completion")
    collisions = stats("collisions")
    offroad = stats("offroad_infractions")
    success_values = [1.0 if bool(row.get("success")) else 0.0 for row in selected]
    return {
        "runs": count,
        "driving_score_count": driving["count"],
        "driving_score_mean": driving["mean"],
        "driving_score_std": driving["std"],
        "route_completion_count": completion["count"],
        "route_completion_mean": completion["mean"],
        "route_completion_std": completion["std"],
        "collisions_total": sum(collisions_values := numeric_values("collisions")),
        "collisions_mean": collisions["mean"],
        "collisions_std": collisions["std"],
        "offroad_total": sum(offroad_values := numeric_values("offroad_infractions")),
        "offroad_mean": offroad["mean"],
        "offroad_std": offroad["std"],
        "success_rate": statistics.fmean(success_values),
        "success_rate_std": (
            statistics.pstdev(success_values) if len(success_values) > 1 else 0.0
        ),
        "missing_collision_metrics": count - len(collisions_values),
        "missing_offroad_metrics": count - len(offroad_values),
    }


def paired_deltas(
    rows: List[Dict[str, Any]], candidate_condition: str
) -> Dict[str, Any]:
    by_key: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        if not row.get("eligible") or row.get("condition") not in ("A", candidate_condition):
            continue
        key = (row.get("route"), row.get("seed"), row.get("weather"))
        by_key.setdefault(key, {})[row["condition"]] = row

    metrics = (
        "driving_score",
        "route_completion",
        "collisions",
        "offroad_infractions",
    )
    deltas: Dict[str, List[float]] = {metric: [] for metric in metrics}
    complete_pairs = 0
    for pair in by_key.values():
        if "A" not in pair or candidate_condition not in pair:
            continue
        complete_pairs += 1
        native = pair["A"]
        candidate = pair[candidate_condition]
        for metric in metrics:
            left = native.get(metric)
            right = candidate.get(metric)
            if all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in (left, right)
            ):
                deltas[metric].append(float(right) - float(left))

    payload: Dict[str, Any] = {"paired_runs": complete_pairs}
    for metric, values in deltas.items():
        payload[metric] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),
            "values": values,
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ablation", choices=("C", "D", "E"), default="D")
    parser.add_argument("--routes", default="55,57")
    parser.add_argument("--seeds", default="20260818,20260819,20260820")
    parser.add_argument("--weather", default="day")
    parser.add_argument("--quality", default="Low")
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    routes = parse_list(args.routes)
    seeds = parse_list(args.seeds)
    if not routes or len(seeds) < 2:
        raise ValueError("at least one route and two distinct seeds are required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds are not allowed")
    campaign_id = time.strftime("%Y%m%d_%H%M%S")
    output = (
        Path(args.output).resolve()
        if args.output
        else ROOT / "logs" / "report_dreamer_campaigns" / campaign_id
    )
    output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    matrix = [
        {"route": route, "seed": seed, "weather": args.weather}
        for route in routes
        for seed in seeds
    ]
    for item in matrix:
        for condition in ("A", args.ablation):
            run_dir = output / ("condition_%s_route_%s_seed_%s" % (condition, item["route"], item["seed"]))
            run_dir.mkdir(parents=True, exist_ok=True)
            paths = expected_paths(item["route"], item["seed"])
            before = snapshot_mtimes(paths)
            env = os.environ.copy()
            env.update(
                {
                    "REPORT_DREAMER_ABLATION": condition,
                    "REPORT_DREAMER_SHADOW": "0",
                    "REPORT_DREAMER_CHECKPOINT": str(checkpoint),
                    "REPORT_DREAMER_RUN_DIR": str(run_dir),
                    "ROUTE_ID": item["route"],
                    "SEED": item["seed"],
                    "SIMLINGO_VISUAL_WEATHER": item["weather"],
                    "CARLA_QUALITY": args.quality,
                    "SIMLINGO_RECORD": "0",
                    "SIMLINGO_PLAYBACK_AFTER": "0",
                }
            )
            log_path = run_dir / "launcher.log"
            started = time.time()
            with log_path.open("w", encoding="utf-8") as handle:
                try:
                    completed = subprocess.run(
                        ["bash", str(ROOT / "scripts" / "run_report_dreamer_live_test.sh")],
                        cwd=str(ROOT),
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                        check=False,
                    )
                    exit_code = completed.returncode
                except subprocess.TimeoutExpired:
                    exit_code = 124
            copied = {}
            for key, path in paths.items():
                if path.exists() and path.stat().st_mtime_ns > before[key]:
                    destination = run_dir / path.name
                    shutil.copy2(path, destination)
                    copied[key] = destination
            row: Dict[str, Any] = {
                "condition": condition,
                **item,
                "checkpoint": None if condition == "A" else str(checkpoint),
                "exit_code": exit_code,
                "elapsed_seconds": time.time() - started,
                "eligible": "result" in copied,
                "exclusion_reason": None if "result" in copied else "no fresh Bench2Drive result",
                "run_dir": str(run_dir),
            }
            if "result" in copied:
                bench = bench2drive_summary(copied["result"])
                row.update(bench)
                row["eligible"] = bool(bench.get("complete_result"))
                row["exclusion_reason"] = bench.get("exclusion_reason")
            rows.append(row)
            partial = {
                "protocol_version": "report_dreamer_paired_ab_v1",
                "candidate_ablation": args.ablation,
                "checkpoint": str(checkpoint),
                "matrix": matrix,
                "runs": rows,
                "complete": False,
            }
            (output / "closed_loop_ab_summary.json").write_text(
                json.dumps(partial, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    native = aggregate(rows, "A")
    candidate = aggregate(rows, args.ablation)
    paired_keys = {
        (row["route"], row["seed"], row["weather"])
        for row in rows
        if row.get("eligible") and row["condition"] == "A"
    } & {
        (row["route"], row["seed"], row["weather"])
        for row in rows
        if row.get("eligible") and row["condition"] == args.ablation
    }
    payload = {
        "protocol_version": "report_dreamer_paired_ab_v1",
        "candidate_ablation": args.ablation,
        "checkpoint": str(checkpoint),
        "carla_quality": args.quality,
        "matrix": matrix,
        "runs": rows,
        "paired_run_count": len(paired_keys),
        "aggregate": {"A": native, args.ablation: candidate},
        "paired_candidate_minus_native": paired_deltas(rows, args.ablation),
        "complete": len(paired_keys) == len(matrix),
        "selection_note": (
            "Only fresh, paired A/candidate Bench2Drive results on identical "
            "route, seed and weather keys are eligible."
        ),
    }
    destination = output / "closed_loop_ab_summary.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
