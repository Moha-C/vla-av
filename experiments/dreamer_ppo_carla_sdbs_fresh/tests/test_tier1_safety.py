"""Tier-1 traffic-safety validation — runs with NO CARLA installed.

Covers the expanded 48-dim state (rear/side vehicle awareness), the vehicle
safety reward terms, and the lane-change blind-spot mandate.

Run with:  python tests/test_tier1_safety.py

Note: the layout the spec describes is 48-dimensional (28 + four new 5-dim
vehicle blocks), not 42 — so these tests assert 48.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np

from configs.config import Config
from configs.sdbs_config import SDBSConfig
from env.carla_env import CarlaEnv, classify_vehicles
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from planning.sdbs_planner import SDBSPlanner, LANE_CHANGE_STEER_CLAMP
from rewards.vru_reward import (
    compute_reward, EGO_SPEED,
    VEHICLE_AHEAD_DIST, VEHICLE_BEHIND_DIST, VEHICLE_LEFT_DIST, VEHICLE_RIGHT_DIST,
    VRU1_DIST, VRU2_DIST,
)

ZERO = np.zeros(4, dtype=np.float32)
# Tier-1 rear/side vehicle awareness is opt-in; enable it explicitly to test
# the expanded 28 -> 48 layout (VRU block moves to 38/43, constants above).
TIER1 = dict(enable_tier1_state=True)
DIM = Config(**TIER1).state_dim          # 48


def tier1_config():
    return Config(**TIER1)


def ok(name, result=""):
    print(f"✅ {name} : {result}")


def safe_state():
    """48-dim state with green light far away, far vehicles, far VRUs."""
    s = np.zeros(DIM, dtype=np.float32)
    s[7] = 3.5            # lane width
    s[10] = 2.0           # green light
    s[11] = 50.0          # light far
    for idx in (VEHICLE_AHEAD_DIST, VEHICLE_BEHIND_DIST,
                VEHICLE_LEFT_DIST, VEHICLE_RIGHT_DIST):
        s[idx] = 100.0
    s[VRU1_DIST] = 50.0
    s[VRU2_DIST] = 50.0
    return s


def make_planner():
    # SDBSConfig adds tau_safe + the mandate parameters; Tier-1 gives 48-dim.
    cfg = SDBSConfig(**TIER1)
    policy = ActorCritic(cfg.state_dim, cfg.action_dim, cfg.hidden)
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    return SDBSPlanner(policy, wm, policy, cfg), cfg


# 1 ------------------------------------------------------------------- #
def test_state_vector_size():
    env = CarlaEnv(mock=True, config=tier1_config())
    state = env.reset()
    assert state.shape == (48,), f"state is {state.shape}, expected (48,)"
    env.close()
    ok("state_vector_size", "48 dimensions (Tier-1 enabled)")


# 2 ------------------------------------------------------------------- #
def test_vehicle_detection():
    vehicles = [
        {"x": 10.0, "y": 0.0, "speed": 5.0, "heading": 0.0},   # ahead
        {"x": -5.0, "y": 0.0, "speed": 5.0, "heading": 0.0},   # behind
        {"x": 0.0, "y": 3.0, "speed": 5.0, "heading": 0.0},    # left
        {"x": 0.0, "y": -3.0, "speed": 5.0, "heading": 0.0},   # right
    ]
    blocks = classify_vehicles(0.0, 0.0, 0.0, lane_width=3.5, vehicles=vehicles)
    assert abs(blocks["ahead"][0] - 10.0) < 1e-6, blocks["ahead"]
    assert abs(blocks["behind"][0] - 5.0) < 1e-6, blocks["behind"]
    assert abs(blocks["left"][0] - 3.0) < 1e-6, blocks["left"]
    assert abs(blocks["right"][0] - 3.0) < 1e-6, blocks["right"]
    assert abs(blocks["nearest"][0] - 3.0) < 1e-6, blocks["nearest"]
    ok("vehicle_detection", "all directions detected correctly")


# 3 ------------------------------------------------------------------- #
def test_vehicle_reward():
    cfg = tier1_config()
    state = safe_state()
    state[VEHICLE_AHEAD_DIST] = 2.0           # vehicle very close ahead
    state[EGO_SPEED] = 5.0
    nxt = state.copy()
    nxt[VEHICLE_AHEAD_DIST] = 1.8             # even closer
    info = {"collision_with_vehicle": False}
    reward, c = compute_reward(state, nxt, ZERO, ZERO, info, cfg)
    assert c["vehicle_proximity"] < 0, c["vehicle_proximity"]
    assert reward < 0, reward
    ok("vehicle_reward", "proximity penalties applied")


# 4 ------------------------------------------------------------------- #
def test_rear_collision_risk():
    cfg = tier1_config()
    state = safe_state()
    state[EGO_SPEED] = 8.0
    state[VEHICLE_BEHIND_DIST] = 5.0          # 5 m behind
    state[VEHICLE_BEHIND_DIST + 1] = 15.0     # closing at 15 m/s (faster)
    nxt = state.copy()
    reward, c = compute_reward(state, nxt, ZERO, ZERO, {}, cfg)
    # TTC = 5 / (15 - 8) = 0.71 s  ->  harsh penalty
    assert c["rear_risk"] < -1.0, c["rear_risk"]
    ok("rear_collision_risk", f"critical TTC penalized (rear_risk={c['rear_risk']:.2f})")


# 5 ------------------------------------------------------------------- #
def test_lane_change_mandate():
    planner, _ = make_planner()
    state = safe_state()
    state[VEHICLE_RIGHT_DIST] = 1.5           # vehicle only 1.5 m to the right
    state[VEHICLE_RIGHT_DIST + 1] = 5.0
    state[EGO_SPEED] = 5.0
    info = {"requested_action_steering": 0.5}   # steering right = lane change
    mandate = planner.evaluate_mandated_safety(state, info)
    assert mandate["mandate"] == "stay_in_lane", mandate
    assert "unsafe right" in mandate["reason"], mandate["reason"]
    ok("lane_change_mandate", "unsafe change blocked")


# 6 ------------------------------------------------------------------- #
def test_lane_change_allowed():
    planner, _ = make_planner()
    state = safe_state()
    state[VEHICLE_RIGHT_DIST] = 50.0          # vehicle far away
    state[EGO_SPEED] = 5.0
    info = {"requested_action_steering": 0.5}
    mandate = planner.evaluate_mandated_safety(state, info)
    assert mandate["mandate"] is None, mandate
    ok("lane_change_allowed", "safe change permitted")


# 7 ------------------------------------------------------------------- #
def test_full_mandate_pipeline():
    planner, cfg = make_planner()
    cfg.compute_budget = 30
    env = CarlaEnv(mock=True, config=cfg)
    env.reset()
    for _ in range(5):
        env.step(np.random.uniform(-1, 1, size=env.action_dim).astype(np.float32))

    # A vehicle is close on the right while the policy wants to steer right.
    state = safe_state()
    state[VEHICLE_RIGHT_DIST] = 1.5
    state[VEHICLE_RIGHT_DIST + 1] = 12.0
    state[EGO_SPEED] = 6.0
    info = {"requested_action_steering": 0.6}
    action, _plan, meta = planner.plan(state, info)
    assert meta["mandate"] == "stay_in_lane", meta["mandate"]
    assert abs(float(action[0])) <= LANE_CHANGE_STEER_CLAMP + 1e-6, float(action[0])
    env.close()
    ok("full_mandate_pipeline", "safety checks active")


def main():
    print("Running Tier-1 safety tests (no CARLA needed)...\n")
    np.random.seed(0)
    test_state_vector_size()
    test_vehicle_detection()
    test_vehicle_reward()
    test_rear_collision_risk()
    test_lane_change_mandate()
    test_lane_change_allowed()
    test_full_mandate_pipeline()
    print("\n✅ ALL TIER 1 SAFETY TESTS PASSED")


if __name__ == "__main__":
    main()
