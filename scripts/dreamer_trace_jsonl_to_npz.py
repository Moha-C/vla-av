#!/usr/bin/env python3
"""Convert live Dreamer status traces into offline world-model transitions.

The dashboard's "Action Dreaming collect" mode records rows produced by
scripts/action_dreaming_collect_normal.py.  Each row contains a Dreamer status
snapshot.  When snapshots include the 28D SimLingo/Dreamer state vector, two
successive rows form a transition:

    state_t + action_t -> state_t+1

This is intentionally simple and conservative.  It lets us bootstrap a
checkpoint from real SimLingo closed-loop runs without touching CARLA during
offline training.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _action_from_status(status: Dict[str, Any], key: str) -> np.ndarray:
    action = status.get(key) or status.get("chosen_action") or status.get("base_action") or {}
    steer = np.clip(_as_float(action.get("steer")), -1.0, 1.0)
    throttle = np.clip(_as_float(action.get("throttle")), 0.0, 1.0)
    brake = np.clip(_as_float(action.get("brake")), 0.0, 1.0)
    stop_continue = 0.0 if brake > 0.5 else 1.0
    return np.asarray([steer, throttle, brake, stop_continue], dtype=np.float32)


def _state_from_status(status: Dict[str, Any], expected_dim: int) -> np.ndarray | None:
    raw = status.get("state_vector")
    if not isinstance(raw, list) or len(raw) < expected_dim:
        return None
    state = np.asarray(raw[:expected_dim], dtype=np.float32)
    if not np.all(np.isfinite(state)):
        return None
    return state


def _risk_from_status(status: Dict[str, Any]) -> float:
    model_risk = max(_as_float(status.get("base_risk")), _as_float(status.get("chosen_risk")))
    front_m = _as_float(status.get("front_vehicle_m"), 80.0)
    shield = 1.0 if status.get("collision_shield_active") else 0.0
    blocked = min(_as_float(status.get("blocked_ticks")) / 40.0, 1.0)
    front_risk = max(0.0, min(1.0, (18.0 - front_m) / 18.0))
    return float(np.clip(max(model_risk, front_risk, shield, blocked * 0.35), 0.0, 1.0))


def _target_side_ttc(status: Dict[str, Any], side: int) -> float:
    if side < 0:
        return _as_float(status.get("left_ttc_s"), 99.0)
    if side > 0:
        return _as_float(status.get("right_ttc_s"), 99.0)
    return min(
        _as_float(status.get("left_ttc_s"), 99.0),
        _as_float(status.get("right_ttc_s"), 99.0),
    )


def _unsafe_recovery_teacher_reason(status: Dict[str, Any], args: argparse.Namespace) -> str:
    """Reject only unsafe overtake/commit teacher actions, not useful hold examples."""
    kind = str(status.get("chosen_kind") or "")
    if "recovery" not in kind or "hold" in kind:
        return ""

    action = status.get("chosen_action") or {}
    brake = _as_float(action.get("brake"))
    throttle = _as_float(action.get("throttle"))
    steer = abs(_as_float(action.get("steer")))
    if brake > 0.5 and throttle < 0.15 and steer < 0.1:
        return ""

    base_risk = _as_float(status.get("base_risk"))
    chosen_risk = _as_float(status.get("chosen_risk"), base_risk)
    if chosen_risk > args.max_recovery_risk:
        return "risk_too_high"
    if chosen_risk - base_risk > args.max_recovery_risk_increase:
        return "risk_increase"

    base_progress = _as_float(status.get("base_progress"))
    chosen_progress = _as_float(status.get("chosen_progress"), base_progress)
    if chosen_progress < base_progress - args.max_recovery_progress_drop:
        return "progress_drop"

    side = int(round(_as_float(status.get("chosen_side"))))
    ttc = _target_side_ttc(status, side)
    if side != 0 and ttc < args.min_recovery_ttc:
        return "low_ttc"

    return ""


def _progress_from_states(a: np.ndarray, b: np.ndarray) -> float:
    # Use travelled distance as a route-progress proxy.  The offline trainer
    # normalizes this target, so meters are acceptable here.
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    dist = math.sqrt(dx * dx + dy * dy)
    if not math.isfinite(dist):
        return 0.0
    return float(max(0.0, min(dist, 20.0)))


def load_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows.sort(key=lambda row: (
        row.get("route_file", ""),
        str(row.get("seed", "")),
        _as_float(row.get("collector_time")),
    ))
    return rows


def same_run(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        a.get("route_file") == b.get("route_file")
        and str(a.get("seed", "")) == str(b.get("seed", ""))
    )


def _repeat_factor(status: Dict[str, Any], args: argparse.Namespace) -> int:
    kind = str(status.get("chosen_kind") or "")
    if "recovery" in kind or status.get("gap_recovery_sides"):
        return max(1, args.recovery_oversample)
    if kind in ("base", "model_steer_delta") and _as_float(status.get("front_vehicle_m"), 80.0) < 16.0:
        return max(1, args.near_hazard_oversample)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="one or more collected JSONL traces")
    parser.add_argument("--output", required=True, help="output .npz")
    parser.add_argument("--state-dim", type=int, default=28)
    parser.add_argument("--action-key", default="chosen_action", choices=["chosen_action", "base_action"])
    parser.add_argument("--max-stationary-per-run", type=int, default=80,
                        help="keep only this many near-identical stopped/hold transitions per run")
    parser.add_argument("--stationary-progress-eps", type=float, default=0.02,
                        help="distance/progress below this is considered stationary")
    parser.add_argument("--recovery-oversample", type=int, default=1,
                        help="duplicate recovery/overtake transitions to reduce hold/base imbalance")
    parser.add_argument("--near-hazard-oversample", type=int, default=1,
                        help="duplicate non-recovery transitions near a front hazard")
    parser.add_argument("--max-recovery-risk", type=float, default=0.92,
                        help="drop recovery/overtake teacher actions above this chosen risk")
    parser.add_argument("--max-recovery-risk-increase", type=float, default=0.08,
                        help="drop recovery/overtake actions that increase risk by more than this")
    parser.add_argument("--min-recovery-ttc", type=float, default=2.8,
                        help="drop side recovery/overtake actions with lower target-side TTC")
    parser.add_argument("--max-recovery-progress-drop", type=float, default=0.02,
                        help="drop recovery/overtake actions that lose this much progress vs base")
    args = parser.parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.input]
    rows = load_rows(paths)

    states, actions, next_states, risks, progresses = [], [], [], [], []
    run_ids, route_ids, towns, chosen_kinds = [], [], [], []
    skipped = Counter()
    kept_kinds = Counter()
    stationary_per_run = Counter()

    for a, b in zip(rows, rows[1:]):
        if not same_run(a, b):
            skipped["run_boundary"] += 1
            continue
        status_a = a.get("status") or {}
        status_b = b.get("status") or {}
        unsafe_reason = _unsafe_recovery_teacher_reason(status_a, args)
        if unsafe_reason:
            skipped[f"unsafe_recovery_teacher_{unsafe_reason}"] += 1
            continue

        state = _state_from_status(status_a, args.state_dim)
        next_state = _state_from_status(status_b, args.state_dim)
        if state is None or next_state is None:
            skipped["missing_state_vector"] += 1
            continue

        action = _action_from_status(status_a, args.action_key)
        progress = _progress_from_states(state, next_state)
        run_key = f"{a.get('route_id','')}_{a.get('seed','')}"
        is_stationary_hold = (
            progress <= args.stationary_progress_eps
            and float(state[2]) <= 0.25
            and float(next_state[2]) <= 0.25
            and (status_a.get("chosen_kind") in ("collision_shield_hold", "recovery_hold", "hold")
                 or bool(status_a.get("collision_shield_active")))
        )
        if is_stationary_hold:
            stationary_per_run[run_key] += 1
            if stationary_per_run[run_key] > args.max_stationary_per_run:
                skipped["stationary_hold_cap"] += 1
                continue

        kind = str(status_a.get("chosen_kind") or "unknown")
        repeat = _repeat_factor(status_a, args)
        for _ in range(repeat):
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            risks.append(_risk_from_status(status_a))
            progresses.append(progress)
            run_ids.append(run_key)
            route_ids.append(str(a.get("route_id", "")))
            towns.append(str(a.get("town", "")))
            chosen_kinds.append(kind)
            kept_kinds[kind] += 1

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        risk_targets=np.asarray(risks, dtype=np.float32),
        progress_targets=np.asarray(progresses, dtype=np.float32),
        run_ids=np.asarray(run_ids),
        route_ids=np.asarray(route_ids),
        towns=np.asarray(towns),
        chosen_kinds=np.asarray(chosen_kinds),
    )

    print(f"input_rows={len(rows)} transitions={len(states)} output={output}")
    print(f"skipped={dict(skipped)}")
    print(f"chosen_kinds={dict(kept_kinds)}")
    print(f"routes={dict(Counter(route_ids))}")
    print(f"towns={dict(Counter(towns))}")


if __name__ == "__main__":
    main()
