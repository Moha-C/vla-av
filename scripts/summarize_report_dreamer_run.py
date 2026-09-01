#!/usr/bin/env python3
"""Join report-Dreamer runtime traces with final Bench2Drive ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def number(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON root must be an object: %s" % path)
    return payload


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError("%s:%d is not an object" % (path, line_number))
            rows.append(item)
    return rows


def infraction_count(value: Any) -> float:
    if isinstance(value, list):
        return float(len(value))
    return float(number(value, 0.0) or 0.0)


def bench2drive_summary(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    checkpoint = payload.get("_checkpoint") or {}
    records = checkpoint.get("records") or []
    record = records[0] if records else checkpoint.get("global_record") or {}
    global_record = checkpoint.get("global_record") or {}
    progress = checkpoint.get("progress") or []
    complete = bool(
        records
        and payload.get("entry_status") == "Finished"
        and payload.get("eligible") is True
        and isinstance(progress, list)
        and len(progress) >= 2
        and int(number(progress[1], 0.0) or 0) > 0
        and int(number(progress[0], -1.0) or -1)
        == int(number(progress[1], 0.0) or 0)
        and all(
            isinstance(item, dict) and item.get("status") == "Completed"
            for item in records
        )
    )
    status = (
        record.get("status")
        or global_record.get("status")
        or payload.get("entry_status")
    )
    if not complete:
        return {
            "result_path": str(path.resolve()),
            "status": status,
            "complete_result": False,
            "exclusion_reason": (
                "Bench2Drive result is not a finished eligible route record"
            ),
            "driving_score": None,
            "route_completion": None,
            "score_penalty": None,
            "collisions": None,
            "offroad_infractions": None,
            "red_light_infractions": None,
            "stop_infractions": None,
            "blocked_infractions": None,
            "success": None,
        }
    scores = record.get("scores") or record.get("scores_mean") or global_record.get("scores_mean") or {}
    infractions = record.get("infractions") or global_record.get("infractions") or {}
    collisions = sum(
        infraction_count(infractions.get(key))
        for key in (
            "collisions_vehicle",
            "collisions_pedestrian",
            "collisions_layout",
        )
    )
    offroad = infraction_count(infractions.get("outside_route_lanes"))
    return {
        "result_path": str(path.resolve()),
        "status": status,
        "complete_result": True,
        "exclusion_reason": None,
        "driving_score": number(scores.get("score_composed")),
        "route_completion": number(scores.get("score_route")),
        "score_penalty": number(scores.get("score_penalty")),
        "collisions": collisions,
        "offroad_infractions": offroad,
        "red_light_infractions": infraction_count(infractions.get("red_light")),
        "stop_infractions": infraction_count(infractions.get("stop_infraction")),
        "blocked_infractions": infraction_count(infractions.get("vehicle_blocked")),
        "success": bool(
            (number(scores.get("score_route"), 0.0) or 0.0) >= 99.0
            and collisions == 0.0
            and offroad == 0.0
        ),
    }


def values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    result = []
    for row in rows:
        value = number(row.get(key))
        if value is not None:
            result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    trace_path = Path(args.trace)
    result_path = Path(args.result)
    rows = load_jsonl(trace_path)
    if not rows:
        raise RuntimeError("empty Dreamer trace: %s" % trace_path)
    alphas = values(rows, "alpha")
    latencies = values(rows, "inference_latency_ms")
    selected = [int(row.get("selected_index", 0)) for row in rows]
    applied = [
        bool(row.get("applied"))
        if "applied" in row
        else abs(number(row.get("alpha"), 0.0) or 0.0) > 1.0e-8
        for row in rows
    ]
    kinds = sorted({str(row.get("selected_kind", "native")) for row in rows})
    risk_gain = []
    progress_gain = []
    harmful_proposal_proxy = 0
    harmful_intervention_proxy = 0
    for row, was_applied in zip(rows, applied):
        native_risk = number(row.get("native_predicted_risk"))
        selected_risk = number(row.get("selected_predicted_risk"))
        native_progress = number(row.get("native_predicted_progress"))
        selected_progress = number(row.get("selected_predicted_progress"))
        if native_risk is not None and selected_risk is not None:
            risk_gain.append(native_risk - selected_risk)
        if native_progress is not None and selected_progress is not None:
            progress_gain.append(selected_progress - native_progress)
        harmful = int(row.get("selected_index", 0)) != 0 and (
            (native_risk is not None and selected_risk is not None and selected_risk > native_risk)
            or (
                native_progress is not None
                and selected_progress is not None
                and selected_progress < native_progress
            )
        )
        if harmful:
            harmful_proposal_proxy += 1
            if was_applied:
                harmful_intervention_proxy += 1
    bench = bench2drive_summary(result_path)
    summary = {
        "trace_path": str(trace_path.resolve()),
        "ticks": len(rows),
        "map": rows[0].get("map"),
        "route": rows[0].get("route"),
        "scenario": rows[0].get("scenario"),
        "seed": rows[0].get("seed"),
        "ablation": rows[0].get("ablation"),
        "shadow": bool(rows[0].get("shadow")),
        "proposal_rate": float(np.mean(np.asarray(selected) != 0)),
        "intervention_rate": float(np.mean(applied)),
        "proposal_ticks": int(np.count_nonzero(np.asarray(selected) != 0)),
        "applied_ticks": int(np.count_nonzero(applied)),
        "alpha_mean": float(np.mean(alphas)) if alphas else None,
        "alpha_std": float(np.std(alphas)) if alphas else None,
        "alpha_max": float(np.max(alphas)) if alphas else None,
        "inference_latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "inference_latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "candidate_indices_observed": sorted(set(selected)),
        "selected_kinds_observed": kinds,
        "predicted_risk_gain_mean": float(np.mean(risk_gain)) if risk_gain else None,
        "predicted_progress_gain_mean": float(np.mean(progress_gain)) if progress_gain else None,
        "potentially_harmful_proposal_proxy_count": harmful_proposal_proxy,
        "potentially_harmful_intervention_proxy_count": harmful_intervention_proxy,
        "bench2drive": bench,
        "per_tick_reward_available": False,
        "per_tick_incident_labels_available": False,
        "scientific_note": (
            "Collision/off-road/rule outcomes are authoritative result-level Bench2Drive "
            "criteria. Per-tick reward or incident attribution is not inferred when the "
            "runtime trace lacks synchronized ground-truth events."
        ),
    }
    destination = (
        Path(args.output)
        if args.output
        else trace_path.with_name("run_summary.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
