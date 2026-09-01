#!/usr/bin/env python3
"""Promote only a prediction-tested and paired-CARLA-evaluated candidate."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_LOSS_KEYS = (
    "observation",
    "progress",
    "risk",
    "continuation",
    "value",
    "collision",
    "offroad",
    "prior_observation",
    "prior_progress",
    "prior_risk",
    "prior_continuation",
    "prior_value",
    "prior_collision",
    "prior_offroad",
    "prior_prediction_total",
    "action_contrastive",
    "action_safety_risk",
    "action_safety_collision",
    "action_safety_speed",
    "action_safety_progress",
    "action_safety_hazard_fraction",
    "action_safety_monotonic",
    "dynamics_kl",
    "representation_kl",
    "total",
)

ACTION_SENSITIVITY_SCHEMA = "report_action_sensitivity_v1"
MINIMUM_SENSITIVITY_STATES = 128
MINIMUM_HAZARD_STATES = 32
MINIMUM_TRANSITION_SPREAD = 1.0e-2
MINIMUM_OUTPUT_SPREAD = 5.0e-3
MINIMUM_PROGRESS_SPREAD = 5.0e-3
MINIMUM_RISK_SPREAD = 1.0e-3
MAXIMUM_COLLAPSED_OUTPUT_FRACTION = 0.25
MINIMUM_BRAKE_RISK_ADVANTAGE = 5.0e-3
MINIMUM_BRAKE_COLLISION_ADVANTAGE = 0.0
MINIMUM_THROTTLE_PROGRESS_ADVANTAGE = 1.0e-3


def load(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON root must be an object: %s" % path)
    return payload


def finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("missing/non-numeric metric: %s" % label) from exc
    if not math.isfinite(result):
        raise RuntimeError("non-finite metric: %s" % label)
    return result


def validate_prediction_metrics(prediction: Dict[str, Any]) -> None:
    seed_count = int(prediction.get("test_seed_count", 0))
    if seed_count < 2:
        raise RuntimeError("prediction evaluation needs at least two frozen test seeds")
    aggregate = prediction.get("aggregate_prediction_losses")
    dispersion = prediction.get("dispersion")
    per_seed = prediction.get("per_seed")
    if not isinstance(aggregate, dict) or not isinstance(dispersion, dict):
        raise RuntimeError("prediction evaluation is missing aggregate/dispersion data")
    if not isinstance(per_seed, dict) or len(per_seed) != seed_count:
        raise RuntimeError("prediction per-seed metrics do not match test_seed_count")
    for key in PREDICTION_LOSS_KEYS:
        finite_number(aggregate.get(key), "aggregate_prediction_losses.%s" % key)
        row = dispersion.get(key)
        if not isinstance(row, dict) or int(row.get("seed_count", 0)) != seed_count:
            raise RuntimeError("prediction dispersion is incomplete for %s" % key)
        finite_number(row.get("mean_across_seeds"), "dispersion.%s.mean" % key)
        finite_number(row.get("std_across_seeds"), "dispersion.%s.std" % key)
        for seed, metrics in per_seed.items():
            if not isinstance(metrics, dict):
                raise RuntimeError("invalid per-seed prediction row for %s" % seed)
            finite_number(metrics.get(key), "per_seed.%s.%s" % (seed, key))
    sensitivity = prediction.get("action_sensitivity")
    if not isinstance(sensitivity, dict):
        raise RuntimeError("prediction evaluation is missing action sensitivity")
    if sensitivity.get("schema_version") != ACTION_SENSITIVITY_SCHEMA:
        raise RuntimeError("unknown action-sensitivity schema")
    if int(sensitivity.get("states", 0)) < MINIMUM_SENSITIVITY_STATES:
        raise RuntimeError("action sensitivity was measured on too few states")
    if int(sensitivity.get("hazard_states", 0)) < MINIMUM_HAZARD_STATES:
        raise RuntimeError("action sensitivity contains too few hazard states")
    minimum_metrics = (
        ("mean_transition_spread", MINIMUM_TRANSITION_SPREAD),
        ("mean_output_spread", MINIMUM_OUTPUT_SPREAD),
        ("mean_progress_spread", MINIMUM_PROGRESS_SPREAD),
        ("mean_risk_spread", MINIMUM_RISK_SPREAD),
        ("hazard_brake_risk_advantage", MINIMUM_BRAKE_RISK_ADVANTAGE),
        (
            "hazard_brake_collision_advantage",
            MINIMUM_BRAKE_COLLISION_ADVANTAGE,
        ),
        (
            "hazard_throttle_progress_advantage",
            MINIMUM_THROTTLE_PROGRESS_ADVANTAGE,
        ),
    )
    for key, minimum in minimum_metrics:
        value = finite_number(sensitivity.get(key), "action_sensitivity.%s" % key)
        if value < minimum:
            raise RuntimeError(
                "action-sensitivity sanity check failed: %s=%.6g < %.6g"
                % (key, value, minimum)
            )
    collapsed = finite_number(
        sensitivity.get("collapsed_output_fraction_1e-4"),
        "action_sensitivity.collapsed_output_fraction_1e-4",
    )
    if collapsed > MAXIMUM_COLLAPSED_OUTPUT_FRACTION:
        raise RuntimeError(
            "action-conditioned outputs collapse too often: %.3f > %.3f"
            % (collapsed, MAXIMUM_COLLAPSED_OUTPUT_FRACTION)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prediction-metrics", required=True)
    parser.add_argument("--closed-loop-summary", required=True)
    parser.add_argument("--destination", default="")
    parser.add_argument("--minimum-paired-runs", type=int, default=3)
    args = parser.parse_args()
    candidate = Path(args.candidate).resolve()
    prediction_path = Path(args.prediction_metrics).resolve()
    closed_loop_path = Path(args.closed_loop_summary).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    checkpoint = torch.load(str(candidate), map_location="cpu")
    if checkpoint.get("checkpoint_version") != "report_aligned_dreamer_v2":
        raise RuntimeError("candidate is not a report-aligned Dreamer checkpoint")
    prediction = load(prediction_path)
    if Path(prediction.get("checkpoint", "")).resolve() != candidate:
        raise RuntimeError("prediction metrics were produced for a different checkpoint")
    validate_prediction_metrics(prediction)
    closed = load(closed_loop_path)
    if closed.get("protocol_version") != "report_dreamer_paired_ab_v1":
        raise RuntimeError("unknown closed-loop protocol")
    if Path(closed.get("checkpoint", "")).resolve() != candidate:
        raise RuntimeError("closed-loop campaign used a different checkpoint")
    if not closed.get("complete"):
        raise RuntimeError("closed-loop campaign is incomplete")
    paired = int(closed.get("paired_run_count", 0))
    if paired < args.minimum_paired_runs:
        raise RuntimeError("not enough paired closed-loop runs")
    ablation = str(closed.get("candidate_ablation", "D"))
    if ablation not in ("D", "E"):
        raise RuntimeError("only learned-authority ablation D or E can be promoted")
    aggregates = closed.get("aggregate") or {}
    native = aggregates.get("A") or {}
    experimental = aggregates.get(ablation) or {}
    if int(native.get("runs", 0)) != int(experimental.get("runs", -1)):
        raise RuntimeError("native and candidate run counts differ")
    if int(native.get("runs", 0)) != paired:
        raise RuntimeError("aggregate run counts do not match paired_run_count")
    if int(native.get("missing_collision_metrics", 1)) or int(experimental.get("missing_collision_metrics", 1)):
        raise RuntimeError("closed-loop campaign has missing collision metrics")
    if int(native.get("missing_offroad_metrics", 1)) or int(experimental.get("missing_offroad_metrics", 1)):
        raise RuntimeError("closed-loop campaign has missing off-road metrics")
    native_collision = finite_number(native.get("collisions_total"), "native.collisions_total")
    experimental_collision = finite_number(experimental.get("collisions_total"), "candidate.collisions_total")
    native_offroad = finite_number(native.get("offroad_total"), "native.offroad_total")
    experimental_offroad = finite_number(experimental.get("offroad_total"), "candidate.offroad_total")
    native_score = finite_number(native.get("driving_score_mean"), "native.driving_score_mean")
    experimental_score = finite_number(experimental.get("driving_score_mean"), "candidate.driving_score_mean")
    if experimental_collision > native_collision:
        raise RuntimeError("candidate has more collisions than native SimLingo")
    if experimental_offroad > native_offroad:
        raise RuntimeError("candidate has more off-road infractions than native SimLingo")
    if experimental_score <= native_score:
        raise RuntimeError("candidate does not improve mean driving score")
    default_name = (
        "report_dreamer_pairwise.pt" if ablation == "E" else "report_dreamer.pt"
    )
    destination = (
        Path(args.destination).resolve()
        if args.destination
        else ROOT / "checkpoints" / "report_aligned_dreamer" / "production" / default_name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, destination)
    manifest = {
        "checkpoint": str(destination),
        "source_candidate": str(candidate),
        "prediction_metrics": str(prediction_path),
        "closed_loop_summary": str(closed_loop_path),
        "paired_runs": paired,
        "ablation": ablation,
        "native_aggregate": native,
        "candidate_aggregate": experimental,
        "promotion_rule": (
            "frozen prediction test >=2 seeds; action-conditioned prior passes "
            "directional brake/throttle sanity checks; complete paired A/candidate "
            "CARLA matrix; no collision/off-road regression; strictly higher mean "
            "driving score"
        ),
        "action_sensitivity": prediction.get("action_sensitivity"),
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
