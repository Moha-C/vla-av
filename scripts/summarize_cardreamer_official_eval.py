#!/usr/bin/env python3
"""Aggregate the finite official CarDreamer overtake evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.input_dir.glob("seed_*.json"))
    if not files:
        raise RuntimeError(f"No seed_*.json results in {args.input_dir}")
    runs = [json.loads(path.read_text()) for path in files]
    episodes = [episode for run in runs for episode in run["episodes"]]
    total = len(episodes)

    def rate(key: str) -> float:
        return sum(bool(ep[key]) for ep in episodes) / total if total else 0.0

    clean_rate = rate("clean_overtake")
    completion_rate = rate("overtake_completed")
    collision_rate = rate("collision")
    offroad_rate = rate("out_of_lane")
    gates = {
        "minimum_episodes": {"threshold": 10, "value": total, "pass": total >= 10},
        "clean_overtake_rate": {"threshold": 0.80, "value": clean_rate, "pass": clean_rate >= 0.80},
        "overtake_completion_rate": {
            "threshold": 0.80,
            "value": completion_rate,
            "pass": completion_rate >= 0.80,
        },
        "maximum_collision_rate": {"threshold": 0.10, "value": collision_rate, "pass": collision_rate <= 0.10},
        "maximum_out_of_lane_rate": {"threshold": 0.10, "value": offroad_rate, "pass": offroad_rate <= 0.10},
    }
    payload = {
        "schema_version": 1,
        "protocol": "cardreamer_official_overtake_multiseed_acceptance",
        "source_files": [str(path) for path in files],
        "seeds": [run["seed"] for run in runs],
        "episodes": total,
        "checkpoint_sha256": sorted({run["checkpoint_sha256"] for run in runs}),
        "metrics": {
            "clean_overtake_rate": clean_rate,
            "overtake_completion_rate": completion_rate,
            "collision_rate": collision_rate,
            "out_of_lane_rate": offroad_rate,
            "mean_return": sum(ep["return"] for ep in episodes) / total if total else 0.0,
        },
        "gates": gates,
        "accepted_for_shadow_transfer": all(item["pass"] for item in gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["accepted_for_shadow_transfer"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
