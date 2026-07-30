"""Standalone tests for the VRU-aware reward function.

Run with:  python tests/test_rewards.py
"""
import os
import sys

# Make the project root importable when run as `python tests/test_rewards.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Ensure UTF-8 output so the ✅ marks render on Windows consoles (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np

from configs.config import Config
from rewards.vru_reward import (
    compute_reward, EGO_SPEED, ROUTE_PROGRESS, VRU1_DIST, VRU2_DIST,
)

ZERO_ACTION = np.zeros(4, dtype=np.float32)


def make_state(route_progress=0.0, ego_speed=0.0, vru_dist=50.0):
    """Build a dim-28 state with the fields the reward cares about."""
    s = np.zeros(28, dtype=np.float32)
    s[ROUTE_PROGRESS] = route_progress
    s[EGO_SPEED] = ego_speed
    s[VRU1_DIST] = vru_dist
    s[VRU2_DIST] = vru_dist
    return s


def ok(name, result=""):
    print(f"✅ {name} : {result}")


# ---------------------------------------------------------------------- #
# Scenarios
# ---------------------------------------------------------------------- #
def scenario_safe(cfg, verbose=True):
    state = make_state(route_progress=0.3, ego_speed=10.0, vru_dist=50.0)
    nxt = make_state(route_progress=0.5, ego_speed=10.0, vru_dist=50.0)
    total, c = compute_reward(state, nxt, ZERO_ACTION, ZERO_ACTION, {}, cfg)
    assert abs(c["vru_risk"]) < 1e-2, c["vru_risk"]   # far VRU -> ~0 risk
    assert c["collision"] == 0.0
    assert total > 0, total                            # progressing, no penalties
    if verbose:
        ok("safe scenario", f"total={total:.3f}")
    return total


def scenario_dangerous(cfg, verbose=True):
    # VRU at dist 1.0, ego_speed 2.0 -> ttc = 0.5s (< tau_ttc).
    state = make_state(route_progress=0.3, ego_speed=2.0, vru_dist=1.0)
    nxt = make_state(route_progress=0.3, ego_speed=2.0, vru_dist=1.0)
    total, c = compute_reward(state, nxt, ZERO_ACTION, ZERO_ACTION, {}, cfg)
    assert c["vru_risk"] < -1.0, c["vru_risk"]         # large negative risk
    assert total < 0, total
    if verbose:
        ok("dangerous scenario", f"total={total:.3f}")
    return total


def scenario_collision(cfg, verbose=True):
    state = make_state(route_progress=0.3, ego_speed=5.0, vru_dist=30.0)
    nxt = make_state(route_progress=0.3, ego_speed=5.0, vru_dist=30.0)
    info = {"collision": True}
    total, c = compute_reward(state, nxt, ZERO_ACTION, ZERO_ACTION, info, cfg)
    assert c["collision"] == -10.0, c["collision"]
    if verbose:
        ok("collision penalty", f"r_collision={c['collision']:.1f}")
    return c


def scenario_red_light(cfg, verbose=True):
    state = make_state(route_progress=0.3, ego_speed=5.0, vru_dist=30.0)
    nxt = make_state(route_progress=0.3, ego_speed=5.0, vru_dist=30.0)
    info = {"red_light_violation": True}
    total, c = compute_reward(state, nxt, ZERO_ACTION, ZERO_ACTION, info, cfg)
    assert c["rules"] == -2.0, c["rules"]
    if verbose:
        ok("red light penalty", f"r_rules={c['rules']:.1f}")
    return c


def scenario_comfort(cfg, verbose=True):
    state = make_state(route_progress=0.3, ego_speed=5.0, vru_dist=30.0)
    nxt = make_state(route_progress=0.3, ego_speed=5.0, vru_dist=30.0)
    action = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    prev = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    total, c = compute_reward(state, nxt, action, prev, {}, cfg)
    assert c["comfort"] < -1.0, c["comfort"]
    if verbose:
        ok("comfort penalty", f"r_comfort={c['comfort']:.3f}")
    return c


def scenario_component_signs(cfg, verbose=True):
    # A messy step: progressing, VRU close, collision + lane departure,
    # jerky action, and a rule violation.
    state = make_state(route_progress=0.3, ego_speed=2.0, vru_dist=1.0)
    nxt = make_state(route_progress=0.5, ego_speed=2.0, vru_dist=1.0)
    action = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    prev = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    info = {
        "collision": True,
        "lane_departure": True,
        "red_light_violation": True,
        "stop_sign_violation": True,
        "crosswalk_conflict": True,
        "general_risk": 0.2,
    }
    total, c = compute_reward(state, nxt, action, prev, info, cfg)
    assert c["progress"] > 0, c["progress"]
    assert c["vru_risk"] <= 0, c["vru_risk"]
    assert c["collision"] <= 0, c["collision"]
    assert c["lane_depart"] <= 0, c["lane_depart"]
    assert c["comfort"] <= 0, c["comfort"]
    assert c["rules"] <= 0, c["rules"]
    if verbose:
        ok("component signs", "all correct")
    return c


# ---------------------------------------------------------------------- #
def run_all(verbose=True):
    """Run all 6 reward scenarios; raises AssertionError on failure."""
    cfg = Config()
    scenario_safe(cfg, verbose)
    scenario_dangerous(cfg, verbose)
    scenario_collision(cfg, verbose)
    scenario_red_light(cfg, verbose)
    scenario_comfort(cfg, verbose)
    scenario_component_signs(cfg, verbose)
    return 6


def main():
    print("Running VRU-aware reward tests...\n")
    run_all(verbose=True)
    print("\n✅ ALL REWARD TESTS PASSED")


if __name__ == "__main__":
    main()
