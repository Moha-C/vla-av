#!/usr/bin/env python3
"""Attach authoritative Bench2Drive route labels to one native trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def _items(value: Any) -> Iterable[Any]:
    return value if isinstance(value, list) else ()


def _number(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_trace_result_binding(trace: Path, result_path: Path) -> Dict[str, Any]:
    first_row: Optional[Dict[str, Any]] = None
    with trace.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "native trace starts with invalid JSON at line %d" % line_number
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError("native trace first row is not an object")
            first_row = payload
            break
    if first_row is None:
        raise RuntimeError("native trace is empty: %s" % trace)
    recorded_result = str(first_row.get("result_path", "")).strip()
    if not recorded_result:
        raise RuntimeError("native trace does not record its expected result path")
    if Path(recorded_result).resolve() != result_path:
        raise RuntimeError(
            "native trace/result mismatch: recorded %s, received %s"
            % (Path(recorded_result).resolve(), result_path)
        )
    if result_path.stat().st_mtime_ns < trace.stat().st_mtime_ns:
        raise RuntimeError("Bench2Drive result is older than the native trace")
    return first_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    trace = Path(args.trace).resolve()
    result_path = Path(args.result).resolve()
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError("native trace is empty: %s" % trace)
    trace_metadata = validate_trace_result_binding(trace, result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = result.get("_checkpoint") if isinstance(result, Mapping) else {}
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    records = list(_items(checkpoint.get("records")))
    if not records:
        raise RuntimeError("Bench2Drive result has no route record: %s" % result_path)
    progress = checkpoint.get("progress")
    progress = progress if isinstance(progress, list) else []
    finished = (
        str(result.get("entry_status", "")) == "Finished"
        and bool(result.get("eligible", False))
        and len(progress) >= 2
        and int(_number(progress[1], 0.0) or 0) > 0
        and int(_number(progress[0], -1.0) or -1)
        == int(_number(progress[1], 0.0) or 0)
        and all(
            isinstance(record, Mapping)
            and str(record.get("status", "")) == "Completed"
            for record in records
        )
    )
    if not finished:
        raise RuntimeError(
            "Bench2Drive result is incomplete or not eligible; native trace "
            "must not enter training: %s" % result_path
        )
    collisions = 0
    offroad = 0
    red_lights = 0
    stop_signs = 0
    for record in records:
        infractions = record.get("infractions") if isinstance(record, Mapping) else {}
        infractions = infractions if isinstance(infractions, Mapping) else {}
        collisions += sum(
            len(list(_items(infractions.get(key))))
            for key in (
                "collisions_layout",
                "collisions_pedestrian",
                "collisions_vehicle",
            )
        )
        offroad += len(list(_items(infractions.get("outside_route_lanes"))))
        red_lights += len(list(_items(infractions.get("red_light"))))
        stop_signs += len(list(_items(infractions.get("stop_infraction"))))
    global_record = checkpoint.get("global_record")
    global_record = global_record if isinstance(global_record, Mapping) else {}
    scores = global_record.get("scores_mean")
    scores = scores if isinstance(scores, Mapping) else {}
    payload: Dict[str, Any] = {
        "schema_version": "report_native_episode_v1",
        "trace": str(trace),
        "bench2drive_result": str(result_path),
        "route_id": trace_metadata.get("route_id"),
        "town": trace_metadata.get("town"),
        "scenario": trace_metadata.get("scenario"),
        "seed": trace_metadata.get("seed"),
        "weather": trace_metadata.get("weather"),
        "bench2drive_ground_truth": True,
        "entry_status": result.get("entry_status"),
        "terminal_validation": {
            "entry_status": result.get("entry_status"),
            "eligible": result.get("eligible"),
            "progress": progress,
            "record_statuses": [record.get("status") for record in records],
            "accepted": True,
        },
        "metrics": {
            "bench2drive_ground_truth": True,
            "collisions": collisions,
            "offroad": offroad,
            "red_light": red_lights,
            "stop_infraction": stop_signs,
            "driving_score": _number(scores.get("score_composed")),
            "route_completion": _number(scores.get("score_route")),
            "infraction_penalty": _number(scores.get("score_penalty")),
        },
        "records": records,
        "note": (
            "Incident totals and route scores are copied from the authoritative "
            "Bench2Drive result. Per-tick collision timing is only available when "
            "collision_events.jsonl was emitted during the route."
        ),
    }
    destination = (
        Path(args.output).resolve()
        if args.output
        else trace.parent / "episode.json"
    )
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
