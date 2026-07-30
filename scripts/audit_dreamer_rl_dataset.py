#!/usr/bin/env python3
"""Audit a Dreamer RL transition dataset before training.

The converter can produce a dataset from any trace rows that have state vectors.
This audit is deliberately stricter than "file exists": it checks diversity,
stationary/hold imbalance, action validity, and recovery coverage so we do not
train an RL variant on a collapsed or unsafe collection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _counter(values: np.ndarray, limit: int = 20) -> Dict[str, int]:
    counts = Counter(str(v) for v in values.tolist())
    return dict(counts.most_common(limit))


def _stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"min": 0.0, "p50": 0.0, "mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _kind_mask(kinds: np.ndarray, *tokens: str) -> np.ndarray:
    lowered = np.char.lower(kinds.astype(str))
    mask = np.zeros(lowered.shape, dtype=bool)
    for token in tokens:
        mask |= np.char.find(lowered, token.lower()) >= 0
    return mask


def audit(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    raw = np.load(path, allow_pickle=True)
    required = ["states", "actions", "next_states", "risk_targets", "progress_targets"]
    missing = [key for key in required if key not in raw.files]
    if missing:
        return {"ok": False, "errors": [f"missing required arrays: {', '.join(missing)}"]}

    states = np.asarray(raw["states"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    next_states = np.asarray(raw["next_states"], dtype=np.float32)
    risks = np.asarray(raw["risk_targets"], dtype=np.float32).reshape(-1)
    progress = np.asarray(raw["progress_targets"], dtype=np.float32).reshape(-1)
    kinds = np.asarray(raw["chosen_kinds"] if "chosen_kinds" in raw.files else np.array(["unknown"] * len(states)))
    route_ids = np.asarray(raw["route_ids"] if "route_ids" in raw.files else np.array(["unknown"] * len(states)))
    towns = np.asarray(raw["towns"] if "towns" in raw.files else np.array(["unknown"] * len(states)))
    run_ids = np.asarray(raw["run_ids"] if "run_ids" in raw.files else np.array(["unknown"] * len(states)))

    errors = []
    warnings = []
    n = int(states.shape[0])
    state_dim = int(states.shape[1]) if states.ndim == 2 else 0
    action_dim = int(actions.shape[1]) if actions.ndim == 2 else 0

    finite_ok = (
        np.all(np.isfinite(states))
        and np.all(np.isfinite(actions))
        and np.all(np.isfinite(next_states))
        and np.all(np.isfinite(risks))
        and np.all(np.isfinite(progress))
    )
    if not finite_ok:
        errors.append("non-finite values detected")
    if n < args.min_transitions:
        errors.append(f"too few transitions: {n} < {args.min_transitions}")
    if len(set(run_ids.astype(str).tolist())) < args.min_runs:
        errors.append(f"too few runs: {len(set(run_ids.astype(str).tolist()))} < {args.min_runs}")
    if len(set(route_ids.astype(str).tolist())) < args.min_routes:
        errors.append(f"too few routes: {len(set(route_ids.astype(str).tolist()))} < {args.min_routes}")
    if state_dim != args.expected_state_dim:
        errors.append(f"unexpected state_dim: {state_dim} != {args.expected_state_dim}")
    if action_dim != args.expected_action_dim:
        errors.append(f"unexpected action_dim: {action_dim} != {args.expected_action_dim}")

    action_bounds_ok = (
        action_dim >= 4
        and np.all(actions[:, 0] >= -1.0001)
        and np.all(actions[:, 0] <= 1.0001)
        and np.all(actions[:, 1:] >= -0.0001)
        and np.all(actions[:, 1:] <= 1.0001)
    )
    if not action_bounds_ok:
        errors.append("actions outside expected bounds")

    throttle_brake_conflict = np.zeros(n, dtype=bool)
    if action_dim >= 3 and n:
        throttle_brake_conflict = (actions[:, 1] > 0.45) & (actions[:, 2] > 0.45)
    conflict_fraction = float(throttle_brake_conflict.mean()) if n else 0.0
    if conflict_fraction > args.max_throttle_brake_conflict_fraction:
        errors.append(
            "too many throttle+brake conflicts: "
            f"{conflict_fraction:.3f} > {args.max_throttle_brake_conflict_fraction:.3f}"
        )

    speed = states[:, 2] if state_dim >= 3 and n else np.zeros(n, dtype=np.float32)
    stationary = (progress <= args.stationary_progress_eps) & (speed <= args.stationary_speed_eps)
    stationary_fraction = float(stationary.mean()) if n else 0.0
    moving_fraction = 1.0 - stationary_fraction if n else 0.0
    if stationary_fraction > args.max_stationary_fraction:
        errors.append(
            f"stationary/hold imbalance too high: {stationary_fraction:.3f} > {args.max_stationary_fraction:.3f}"
        )
    if moving_fraction < args.min_moving_fraction:
        errors.append(f"moving fraction too low: {moving_fraction:.3f} < {args.min_moving_fraction:.3f}")

    recovery_mask = _kind_mask(kinds, "recovery", "overtake", "gap_commit", "finish_pass")
    recovery_fraction = float(recovery_mask.mean()) if n else 0.0
    if recovery_fraction < args.min_recovery_fraction:
        warnings.append(
            f"low recovery/overtake coverage: {recovery_fraction:.3f} < {args.min_recovery_fraction:.3f}"
        )

    steer_std = float(np.std(actions[:, 0])) if action_dim >= 1 and n else 0.0
    if steer_std < args.min_steer_std:
        errors.append(f"steer variation too low: {steer_std:.4f} < {args.min_steer_std:.4f}")

    summary = {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "path": str(path),
        "n_transitions": n,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "finite_ok": bool(finite_ok),
        "n_runs": int(len(set(run_ids.astype(str).tolist()))),
        "n_routes": int(len(set(route_ids.astype(str).tolist()))),
        "towns": _counter(towns),
        "routes": _counter(route_ids),
        "chosen_kinds": _counter(kinds),
        "risk": _stats(risks),
        "progress": _stats(progress),
        "speed": _stats(speed),
        "actions": {
            "steer": _stats(actions[:, 0] if action_dim >= 1 else np.array([])),
            "throttle": _stats(actions[:, 1] if action_dim >= 2 else np.array([])),
            "brake": _stats(actions[:, 2] if action_dim >= 3 else np.array([])),
            "stop_continue": _stats(actions[:, 3] if action_dim >= 4 else np.array([])),
            "steer_std": steer_std,
            "throttle_brake_conflict_fraction": conflict_fraction,
        },
        "stationary_fraction": stationary_fraction,
        "moving_fraction": moving_fraction,
        "recovery_fraction": recovery_fraction,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dreamer transition .npz")
    parser.add_argument("--json-output", default="", help="write audit summary JSON")
    parser.add_argument("--expected-state-dim", type=int, default=28)
    parser.add_argument("--expected-action-dim", type=int, default=4)
    parser.add_argument("--min-transitions", type=int, default=1000)
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument("--min-routes", type=int, default=2)
    parser.add_argument("--stationary-progress-eps", type=float, default=0.02)
    parser.add_argument("--stationary-speed-eps", type=float, default=0.25)
    parser.add_argument("--max-stationary-fraction", type=float, default=0.65)
    parser.add_argument("--min-moving-fraction", type=float, default=0.25)
    parser.add_argument("--min-recovery-fraction", type=float, default=0.02)
    parser.add_argument("--max-throttle-brake-conflict-fraction", type=float, default=0.05)
    parser.add_argument("--min-steer-std", type=float, default=0.015)
    parser.add_argument("--soft", action="store_true", help="print failures but exit 0")
    args = parser.parse_args()

    path = Path(args.dataset).expanduser().resolve()
    summary = audit(path, args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)

    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")

    if not summary.get("ok") and not args.soft:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
