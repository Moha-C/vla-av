"""Tier-2 generalization validation — runs with NO CARLA installed.

Covers map-agnostic features, the augmented-state wrapper, defensive-driving
mode, and unknown-map detection.

Run with:  python tests/test_tier2_generalization.py

Note: the augmented state is 55-dim (48 base + 7 features), not 49 — the brief's
"42 -> 49" assumed the pre-Tier-1 state.
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
from env.carla_env import CarlaEnv
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from planning.sdbs_planner import SDBSPlanner
from planning.defensive_driving import DefensiveDrivingController
from training.map_agnostic_state import MapAgnosticStateWrapper
from utils.map_agnostic_features import compute_map_agnostic_features
from utils.map_detection import is_unknown_map, activate_defensive_mode_if_unknown

# Tier-2 (map-agnostic features) builds on Tier-1 state. Enable both explicitly
# so the augmented state is 48 (Tier-1 base) -> 55 (with the 7 features).
TIER1 = dict(enable_tier1_state=True)
TIER12 = dict(enable_tier1_state=True, enable_tier2_state=True)
DIM = Config(**TIER1).state_dim           # 48 (Tier-1 base)
AUG = Config(**TIER12).state_dim          # 55 (Tier-1 + Tier-2)


def ok(name, result=""):
    print(f"✅ {name} : {result}")


# 1 ------------------------------------------------------------------- #
def test_map_agnostic_features():
    state = np.zeros(DIM, dtype=np.float32)
    state[6] = 0.0          # centered
    state[7] = 4.0          # 4 m lane
    state[8] = 0.005        # straight
    state[9] = 0            # not an intersection
    info = {"weather": "clear", "vru_list": [], "vehicle_list": []}
    feats = compute_map_agnostic_features(state, info, Config(**TIER1))
    assert feats["in_lane_center"] > 0.9, feats["in_lane_center"]
    assert feats["road_type"] == "straight", feats["road_type"]
    assert feats["visibility"] > 0.8, feats["visibility"]
    ok("map_agnostic_features", "all features computed")


# 2 ------------------------------------------------------------------- #
def test_state_augmentation():
    # The wrapper appends the 7 map features to a Tier-1 (48-dim) base -> 55.
    wrapper = MapAgnosticStateWrapper(Config(**TIER1))
    state = np.zeros(DIM, dtype=np.float32)
    state[7] = 4.0
    augmented = wrapper.augment_state(state, {"weather": "clear"})
    assert augmented.shape == (AUG,), f"expected ({AUG},), got {augmented.shape}"
    assert 0.0 <= augmented[DIM] <= 1.0, augmented[DIM]   # in_lane_center
    ok("state_augmentation", f"{DIM} dims -> {AUG} dims correct")


# 3 ------------------------------------------------------------------- #
def test_defensive_mode_activation():
    cfg = SDBSConfig()
    controller = DefensiveDrivingController(cfg)
    base_w_vru, base_w_vehicle = cfg.w_vru, cfg.w_vehicle
    controller.activate_defensive_mode()
    assert cfg.w_vru > base_w_vru, (cfg.w_vru, base_w_vru)
    assert cfg.w_vehicle > base_w_vehicle, (cfg.w_vehicle, base_w_vehicle)
    assert controller.defensive_mode_active is True
    controller.deactivate_defensive_mode()
    assert cfg.w_vru == base_w_vru                      # restored in place
    ok("defensive_mode_activation", "weights scaled correctly")


# 4 ------------------------------------------------------------------- #
def test_risky_action_detection():
    controller = DefensiveDrivingController(SDBSConfig())
    controller.activate_defensive_mode()
    action = np.array([0.8, 0.5, 0.0, 1.0], dtype=np.float32)   # hard turn + throttle
    state = np.zeros(DIM, dtype=np.float32)
    state[9] = 1            # is_junction
    assert controller.is_risky_action(action, state, {}) is True
    ok("risky_action_detection", "dangerous actions identified")


# 5 ------------------------------------------------------------------- #
def test_safe_action_filtering():
    cfg = SDBSConfig()
    policy = ActorCritic(cfg.state_dim, cfg.action_dim, cfg.hidden)
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    planner = SDBSPlanner(policy, wm, policy, cfg)
    planner.defensive_controller.activate_defensive_mode()

    state = np.zeros(DIM, dtype=np.float32)
    state[9] = 1            # intersection -> big steering is risky
    actions = [
        np.array([0.8, 0.5, 0, 1], dtype=np.float32),   # risky
        np.array([0.1, 0.3, 0, 1], dtype=np.float32),   # safe
        np.array([0.7, 0.6, 0, 1], dtype=np.float32),   # risky
    ]
    safe = planner.filter_risky_actions(actions, state, {})
    assert len(safe) >= 1
    assert all(not planner.defensive_controller.is_risky_action(a, state, {})
               for a in safe)
    ok("safe_action_filtering", "risky actions removed")


# 6 ------------------------------------------------------------------- #
def test_map_detection():
    trained_maps = [{"name": "Town01", "center": (0, 0), "radius": 100}]
    assert is_unknown_map((10, 10), trained_maps) is False
    assert is_unknown_map((500, 500), trained_maps) is True
    ok("map_detection", "known vs unknown maps")


# 7 ------------------------------------------------------------------- #
def test_unknown_map_defensive_activation():
    controller = DefensiveDrivingController(SDBSConfig())
    trained_maps = [{"name": "Town01", "center": (0, 0), "radius": 100}]
    activate_defensive_mode_if_unknown((500, 500), trained_maps, controller)
    assert controller.defensive_mode_active is True
    ok("unknown_map_defensive_activation", "mode activated automatically")


# 8 ------------------------------------------------------------------- #
def test_augmented_state_pipeline():
    # With both tiers enabled the ENV itself emits the full 55-dim state
    # (map-agnostic features appended) — no external wrapper needed.
    cfg = Config(**TIER12)
    env = CarlaEnv(mock=True, config=cfg)
    obs = env.reset()
    assert obs.shape == (AUG,), obs.shape
    for _ in range(5):
        obs, _r, _done, _info = env.step(
            np.random.uniform(-1, 1, size=env.action_dim).astype(np.float32))
        assert obs.shape == (AUG,), obs.shape
    env.close()
    ok("augmented_state_pipeline", "env emits full 55-dim state end-to-end")


def main():
    print("Running Tier-2 generalization tests (no CARLA needed)...\n")
    np.random.seed(0)
    test_map_agnostic_features()
    test_state_augmentation()
    test_defensive_mode_activation()
    test_risky_action_detection()
    test_safe_action_filtering()
    test_map_detection()
    test_unknown_map_defensive_activation()
    test_augmented_state_pipeline()
    print("\n✅ ALL TIER 2 GENERALIZATION TESTS PASSED")


if __name__ == "__main__":
    main()
