#!/usr/bin/env python3
"""Summarize read-only CarDreamer shadow traces and enforce transfer gates.

The scorer recomputes safety labels from raw geometry.  This keeps old traces
comparable when the diagnostic thresholds evolve and avoids trusting labels
that may have been produced by an earlier sidecar revision.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--min-decisions", type=int, default=100)
    parser.add_argument("--min-opportunities", type=int, default=3)
    parser.add_argument("--min-overtake-proposal-rate", type=float, default=0.60)
    parser.add_argument("--max-unsafe-rate", type=float, default=0.05)
    parser.add_argument("--blocked-distance", type=float, default=18.0)
    parser.add_argument("--blocked-vehicle-speed", type=float, default=1.5)
    parser.add_argument("--minimum-clearance", type=float, default=5.0)
    parser.add_argument("--minimum-oncoming-ttc", type=float, default=7.0)
    parser.add_argument("--minimum-rear-ttc", type=float, default=5.0)
    return parser.parse_args()


def read_trace(path):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("state") == "observing":
            rows.append(row)
    return rows


def reassess(row, blocked_ticks, args):
    front_distance = float(row.get("front_vehicle_m", 80.0))
    front_speed = float(row.get("front_vehicle_speed_mps", 80.0))
    blocked = front_distance <= args.blocked_distance and front_speed <= args.blocked_vehicle_speed
    left_safe = (
        bool(row.get("left_lane_available"))
        and float(row.get("left_clear_m", 0.0)) >= args.minimum_clearance
        and float(row.get("left_oncoming_ttc_s", 0.0)) >= args.minimum_oncoming_ttc
        and float(row.get("left_rear_ttc_s", 0.0)) >= args.minimum_rear_ttc
    )
    right_safe = (
        bool(row.get("right_lane_available"))
        and float(row.get("right_clear_m", 0.0)) >= args.minimum_clearance
        and float(row.get("right_oncoming_ttc_s", 0.0)) >= args.minimum_oncoming_ttc
        and float(row.get("right_rear_ttc_s", 0.0)) >= args.minimum_rear_ttc
    )
    opportunity = bool(blocked and left_safe)
    maneuver = row.get("maneuver", "unknown")
    steer = float(row.get("proposed_control", {}).get("steer", 0.0))
    strong_lateral_intent = abs(steer) >= 0.45
    unsafe = False
    label = "coherent_lane_follow"
    if blocked and maneuver == "left" and strong_lateral_intent and not left_safe:
        unsafe = True
        label = "unsafe_left_proposal"
    elif blocked and maneuver == "right" and strong_lateral_intent and not right_safe:
        unsafe = True
        label = "unsafe_right_proposal"
    elif opportunity and maneuver == "left":
        label = "coherent_overtake_proposal"
    elif opportunity and blocked_ticks >= 5:
        label = "missed_safe_overtake_opportunity"
    elif blocked and not left_safe and maneuver in {"brake", "straight_coast"}:
        label = "coherent_wait_for_gap"
    elif blocked and not left_safe and maneuver == "straight_accelerate":
        unsafe = True
        label = "unsafe_acceleration_into_blockage"
    return {
        **row,
        "recomputed_blocked": blocked,
        "recomputed_left_safe": left_safe,
        "recomputed_right_safe": right_safe,
        "recomputed_safe_overtake_opportunity": opportunity,
        "recomputed_unsafe": unsafe,
        "recomputed_coherence_label": label,
    }


def close_event(event, events):
    if not event:
        return
    opportunities = [row for row in event if row["recomputed_safe_overtake_opportunity"]]
    events.append(
        {
            "decisions": len(event),
            "safe_opportunity_observed": bool(opportunities),
            "left_proposal_in_safe_gap": any(row.get("maneuver") == "left" for row in opportunities),
            "unsafe_proposals": sum(bool(row["recomputed_unsafe"]) for row in event),
            "first_game_time": event[0].get("game_time"),
            "last_game_time": event[-1].get("game_time"),
        }
    )


def evaluate_trace(path, args):
    raw_rows = read_trace(path)
    rows = []
    events = []
    current_event = []
    blocked_ticks = 0
    for raw in raw_rows:
        front_distance = float(raw.get("front_vehicle_m", 80.0))
        front_speed = float(raw.get("front_vehicle_speed_mps", 80.0))
        is_blocked = front_distance <= args.blocked_distance and front_speed <= args.blocked_vehicle_speed
        blocked_ticks = blocked_ticks + 1 if is_blocked else 0
        row = reassess(raw, blocked_ticks, args)
        rows.append(row)
        if row["recomputed_blocked"]:
            current_event.append(row)
        elif current_event:
            close_event(current_event, events)
            current_event = []
    close_event(current_event, events)

    opportunity_events = [event for event in events if event["safe_opportunity_observed"]]
    successful_events = [event for event in opportunity_events if event["left_proposal_in_safe_gap"]]
    blocked_rows = [row for row in rows if row["recomputed_blocked"]]
    safe_opportunity_observed = any(
        row["recomputed_safe_overtake_opportunity"] for row in rows
    )
    left_proposal_in_safe_gap = any(
        row["recomputed_safe_overtake_opportunity"] and row.get("maneuver") == "left"
        for row in rows
    )
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "events": events,
        "report": {
            "path": str(path.resolve()),
            "decisions": len(rows),
            "blocked_decisions": len(blocked_rows),
            "unsafe_proposals": sum(bool(row["recomputed_unsafe"]) for row in rows),
            "blockage_events": len(events),
            "safe_overtake_opportunity_events": len(opportunity_events),
            "successful_overtake_proposal_events": len(successful_events),
            "safe_overtake_opportunity_observed": safe_opportunity_observed,
            "left_proposal_in_safe_gap": left_proposal_in_safe_gap,
            "lateral_adapters": dict(Counter(row.get("lateral_adapter", "native") for row in rows)),
            "blocked_maneuvers": dict(Counter(row.get("maneuver", "unknown") for row in blocked_rows)),
        },
    }


def main():
    args = parse_args()
    evaluated = [evaluate_trace(path, args) for path in args.traces]
    all_rows = [row for trace in evaluated for row in trace["rows"]]
    all_events = [event for trace in evaluated for event in trace["events"]]
    opportunity_traces = [
        trace for trace in evaluated if trace["report"]["safe_overtake_opportunity_observed"]
    ]
    successful_traces = [
        trace for trace in opportunity_traces if trace["report"]["left_proposal_in_safe_gap"]
    ]
    unsafe = [row for row in all_rows if row["recomputed_unsafe"]]
    authority_violations = [row for row in all_rows if row.get("control_authority") != "none"]
    decisions = len(all_rows)
    blocked_rows = [row for row in all_rows if row["recomputed_blocked"]]
    proposal_rate = len(successful_traces) / len(opportunity_traces) if opportunity_traces else 0.0
    unsafe_rate = len(unsafe) / len(blocked_rows) if blocked_rows else 1.0
    gates = {
        "enough_decisions": decisions >= args.min_decisions,
        "enough_safe_overtake_opportunity_runs": len(opportunity_traces) >= args.min_opportunities,
        "safe_overtake_run_proposal_rate": proposal_rate >= args.min_overtake_proposal_rate,
        "unsafe_proposal_rate_during_blockage": unsafe_rate <= args.max_unsafe_rate,
        "zero_control_authority": not authority_violations,
    }
    payload = {
        "schema_version": 2,
        "protocol": "cardreamer_town10hd_read_only_shadow",
        "scientific_scope": "privileged-information transfer; not camera-only",
        "thresholds": {
            "blocked_distance_m": args.blocked_distance,
            "blocked_vehicle_speed_mps": args.blocked_vehicle_speed,
            "minimum_clearance_m": args.minimum_clearance,
            "minimum_oncoming_ttc_s": args.minimum_oncoming_ttc,
            "minimum_rear_ttc_s": args.minimum_rear_ttc,
        },
        "summary": {
            "decisions": decisions,
            "blockage_events": len(all_events),
            "blocked_decisions": len(blocked_rows),
            "safe_overtake_opportunity_runs": len(opportunity_traces),
            "successful_overtake_proposal_runs": len(successful_traces),
            "safe_overtake_run_proposal_rate": proposal_rate,
            "unsafe_proposals": len(unsafe),
            "unsafe_proposal_rate_during_blockage": unsafe_rate,
            "lateral_adapters": dict(Counter(row.get("lateral_adapter", "native") for row in all_rows)),
            "blocked_maneuvers": dict(Counter(row.get("maneuver", "unknown") for row in blocked_rows)),
            "coherence_labels": dict(
                Counter(row["recomputed_coherence_label"] for row in all_rows)
            ),
        },
        "gates": gates,
        "accepted_for_residual_integration": all(gates.values()),
        "traces": [trace["report"] for trace in evaluated],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["accepted_for_residual_integration"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
