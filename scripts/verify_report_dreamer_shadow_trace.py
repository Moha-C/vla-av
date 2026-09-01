#!/usr/bin/env python3
"""Verify that a report-Dreamer shadow trace was control-inert.

This check is intentionally stricter than a visual inspection: every recorded
tick must keep alpha at zero, leave the native SimLingo action bit-identical,
and contain finite RSSM observations/predictions.  It validates integration,
not closed-loop driving performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from src.world_model.observation import DREAMER_OBSERVATION_FEATURES


PREDICTION_KEYS = (
    "native_predicted_progress",
    "native_predicted_risk",
    "selected_predicted_progress",
    "selected_predicted_risk",
    "selected_predicted_continuation",
    "selected_predicted_value",
    "inference_latency_ms",
)


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s is not numeric" % label) from exc
    if not math.isfinite(result):
        raise ValueError("%s is not finite" % label)
    return result


def _action(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("%s must be a three-value sequence" % label)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("%s must contain three finite controls" % label)
    return result


def load_trace(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d is invalid JSON" % (path, line_number)) from exc
            if not isinstance(row, dict):
                raise ValueError("%s:%d is not a JSON object" % (path, line_number))
            rows.append(row)
    return rows


def verify_shadow_rows(
    rows: Sequence[Mapping[str, Any]], minimum_ticks: int = 20
) -> Dict[str, Any]:
    if len(rows) < int(minimum_ticks):
        raise ValueError(
            "shadow trace has %d ticks; at least %d are required"
            % (len(rows), int(minimum_ticks))
        )

    latencies: List[float] = []
    selected_indices: List[int] = []
    selected_kinds = set()
    metadata_keys = ("map", "route", "scenario", "seed", "ablation")
    reference_metadata = {key: rows[0].get(key) for key in metadata_keys}
    maximum_action_difference = 0.0

    for index, row in enumerate(rows):
        prefix = "tick %d" % index
        if row.get("shadow") is not True:
            raise ValueError("%s is not marked shadow=true" % prefix)
        if row.get("applied") is not False:
            raise ValueError("%s reports an applied Dreamer action" % prefix)
        alpha = _finite_number(row.get("alpha"), "%s alpha" % prefix)
        if alpha != 0.0:
            raise ValueError("%s alpha is not exactly zero" % prefix)

        native = _action(row.get("native_action"), "%s native_action" % prefix)
        final = _action(row.get("final_action"), "%s final_action" % prefix)
        difference = float(np.max(np.abs(native - final)))
        maximum_action_difference = max(maximum_action_difference, difference)
        if not np.array_equal(native, final):
            raise ValueError(
                "%s final action differs from native SimLingo action" % prefix
            )

        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("%s has no named observation" % prefix)
        missing = [
            name for name in DREAMER_OBSERVATION_FEATURES if name not in observation
        ]
        if missing:
            raise ValueError("%s observation misses %s" % (prefix, missing[0]))
        for name in DREAMER_OBSERVATION_FEATURES:
            _finite_number(observation[name], "%s observation.%s" % (prefix, name))

        for key in PREDICTION_KEYS:
            value = _finite_number(row.get(key), "%s %s" % (prefix, key))
            if key == "inference_latency_ms":
                if value < 0.0:
                    raise ValueError("%s latency is negative" % prefix)
                latencies.append(value)

        kinds = row.get("candidate_kinds")
        features = row.get("candidate_features")
        utilities = row.get("candidate_utilities")
        if not isinstance(kinds, list) or not kinds:
            raise ValueError("%s has no candidate list" % prefix)
        if not isinstance(features, list) or len(features) != len(kinds):
            raise ValueError("%s candidate feature count is inconsistent" % prefix)
        if not isinstance(utilities, list) or len(utilities) != len(kinds):
            raise ValueError("%s candidate utility count is inconsistent" % prefix)
        for candidate_index, candidate_features in enumerate(features):
            feature_array = np.asarray(candidate_features, dtype=np.float64)
            if feature_array.shape != (5,) or not np.isfinite(feature_array).all():
                raise ValueError(
                    "%s candidate %d does not have five finite features"
                    % (prefix, candidate_index)
                )
        utility_array = np.asarray(utilities, dtype=np.float64)
        if utility_array.shape != (len(kinds),) or not np.isfinite(utility_array).all():
            raise ValueError("%s candidate utilities are not finite" % prefix)
        selected = int(row.get("selected_index", -1))
        if not 0 <= selected < len(kinds):
            raise ValueError("%s selected candidate index is invalid" % prefix)
        if "selected_kind" in row and str(row["selected_kind"]) != str(kinds[selected]):
            raise ValueError("%s selected kind does not match its index" % prefix)
        selected_indices.append(selected)
        selected_kinds.add(str(kinds[selected]))

        for key, expected in reference_metadata.items():
            if row.get(key) != expected:
                raise ValueError("%s changes run metadata field %s" % (prefix, key))

    latency_array = np.asarray(latencies, dtype=np.float64)
    proposals = int(np.count_nonzero(np.asarray(selected_indices) != 0))
    return {
        "valid": True,
        "verification_kind": "report_dreamer_shadow_control_invariance",
        "ticks": len(rows),
        "minimum_ticks": int(minimum_ticks),
        "metadata": reference_metadata,
        "all_shadow": True,
        "all_alpha_zero": True,
        "all_applied_false": True,
        "native_final_bit_exact": True,
        "maximum_native_final_absolute_difference": maximum_action_difference,
        "nonnative_proposal_ticks": proposals,
        "nonnative_proposal_rate": proposals / float(len(rows)),
        "selected_kinds_observed": sorted(selected_kinds),
        "inference_latency_ms": {
            "count": len(latencies),
            "median": float(np.median(latency_array)),
            "p95": float(np.percentile(latency_array, 95)),
            "p99": float(np.percentile(latency_array, 99)),
            "maximum": float(np.max(latency_array)),
        },
        "closed_loop_performance_claim": False,
        "note": (
            "This verifies shadow integration and exact native-control preservation. "
            "It does not establish a driving-performance improvement."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--minimum-ticks", type=int, default=20)
    args = parser.parse_args()
    trace_path = Path(args.trace).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    summary = verify_shadow_rows(load_trace(trace_path), args.minimum_ticks)
    summary["trace_path"] = str(trace_path)
    destination = (
        Path(args.output).resolve()
        if args.output
        else trace_path.with_name("shadow_verification.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
