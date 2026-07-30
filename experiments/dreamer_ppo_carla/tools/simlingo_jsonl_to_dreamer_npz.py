#!/usr/bin/env python3
"""Convert SimLingo Action Dreaming JSONL into Dreamer-PPO transition arrays.

The upstream Dreamer-PPO repo uses a compact 28D vector state. This adapter
keeps that shape but fills it from the fields collected by our SimLingo
Action Dreaming pipeline. It is intentionally offline-only: no CARLA server is
needed and no SimLingo files are modified.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


LIGHT_TO_FLOAT = {
    "red": 0.0,
    "yellow": 1.0,
    "green": 2.0,
    "none": 2.0,
    None: 2.0,
}


def _as_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_state(sample):
    ego = sample.get("ego_state") or {}
    scene = sample.get("scene_context") or {}
    route = sample.get("route_context") or {}
    loc = ego.get("location") or [0.0, 0.0, 0.0]
    target = route.get("target_point") or [0.0, 0.0]
    target_next = route.get("target_point_next") or target

    state = np.zeros(28, dtype=np.float32)

    # ego: x, y, speed, heading, acc_x, acc_y
    state[0] = _as_float(loc[0] if len(loc) > 0 else 0.0)
    state[1] = _as_float(loc[1] if len(loc) > 1 else 0.0)
    state[2] = _as_float(ego.get("speed_mps"))
    state[3] = np.deg2rad(_as_float(ego.get("yaw_deg")))
    state[4] = _as_float(ego.get("accel_mps2"))
    state[5] = 0.0

    # lane/context approximations from target point geometry.
    state[6] = _as_float(target[1] if len(target) > 1 else 0.0)  # lateral offset proxy
    state[7] = 3.5
    dx1 = _as_float(target[0] if len(target) > 0 else 0.0)
    dy1 = _as_float(target[1] if len(target) > 1 else 0.0)
    dx2 = _as_float(target_next[0] if len(target_next) > 0 else dx1)
    dy2 = _as_float(target_next[1] if len(target_next) > 1 else dy1)
    state[8] = np.arctan2(dy2 - dy1, max(abs(dx2 - dx1), 1e-3))
    state[9] = 0.0

    # traffic/route.
    state[10] = LIGHT_TO_FLOAT.get(str(scene.get("traffic_light_state")).lower(), 2.0)
    state[11] = 50.0 if scene.get("traffic_light_state") in (None, "none") else 20.0
    state[12] = _as_float(route.get("route_progress_percent")) / 100.0

    # nearest/front vehicle.
    front_dist = _as_float(scene.get("front_vehicle_m"), 80.0)
    nearest_vehicle = _as_float(scene.get("nearest_vehicle_m"), front_dist)
    state[13] = min(front_dist, nearest_vehicle)
    state[14] = max(0.0, state[2] + _as_float(scene.get("front_vehicle_rel_speed_mps")))
    state[15] = 0.0
    state[16] = state[13]
    state[17] = 0.0

    # VRU slots: walker then bike.
    walker = _as_float(scene.get("nearest_walker_m"), 80.0)
    bike = _as_float(scene.get("nearest_bike_m"), 80.0)
    for base, dist in ((18, walker), (23, bike)):
        state[base] = dist
        state[base + 1] = 0.0
        state[base + 2] = 0.0
        state[base + 3] = dist
        state[base + 4] = 0.0

    return state


def build_action(sample, action_key):
    action = sample.get(action_key) or sample.get("action_star") or sample.get("executed_action") or {}
    brake = np.clip(_as_float(action.get("brake")), 0.0, 1.0)
    throttle = np.clip(_as_float(action.get("throttle")), 0.0, 1.0)
    steer = np.clip(_as_float(action.get("steer")), -1.0, 1.0)
    stop_continue = 0.0 if brake > 0.5 else 1.0
    return np.asarray([steer, throttle, brake, stop_continue], dtype=np.float32)


def risk_target(sample):
    risk = sample.get("risk_targets") or {}
    return float(np.clip(max(
        _as_float(risk.get("collision_risk")),
        _as_float(risk.get("vru_risk")),
        _as_float(risk.get("rule_risk")),
        1.0 if risk.get("offroad") else 0.0,
    ), 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="filtered SimLingo JSONL")
    parser.add_argument("--output", required=True, help="output .npz")
    parser.add_argument("--action-key", default="action_star",
                        choices=["action_star", "executed_action"])
    args = parser.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rows.sort(key=lambda o: (o.get("run_id", ""), o.get("sample_index", 0), o.get("wall_time", 0.0)))

    states, actions, next_states, risks, progresses = [], [], [], [], []
    scenarios, towns, run_ids = [], [], []
    per_run = Counter()
    skipped = Counter()

    for a, b in zip(rows, rows[1:]):
        if a.get("run_id") != b.get("run_id"):
            skipped["run_boundary"] += 1
            continue
        s = build_state(a)
        ns = build_state(b)
        progress = max(0.0, float(ns[12] - s[12]))
        states.append(s)
        actions.append(build_action(a, args.action_key))
        next_states.append(ns)
        risks.append(risk_target(a))
        progresses.append(progress)
        scenarios.append(a.get("scenario_type", "unknown"))
        towns.append(a.get("town", "unknown"))
        run_ids.append(a.get("run_id", "unknown"))
        per_run[a.get("run_id", "unknown")] += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        risk_targets=np.asarray(risks, dtype=np.float32),
        progress_targets=np.asarray(progresses, dtype=np.float32),
        scenarios=np.asarray(scenarios),
        towns=np.asarray(towns),
        run_ids=np.asarray(run_ids),
    )

    print(f"input_rows={len(rows)} transitions={len(states)} output={out}")
    print(f"runs={len(per_run)} skipped={dict(skipped)}")
    print(f"towns={dict(Counter(towns))}")
    print(f"scenarios={dict(Counter(scenarios))}")


if __name__ == "__main__":
    main()
