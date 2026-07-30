"""Tier-3 interpretability validation — runs with NO CARLA installed.

Covers the per-episode SafetyTracker, the LaneChangeExplainer, and the enhanced
Logger CSV schema.

Run with:  python tests/test_tier3_interpretability.py
"""
import csv
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np

from configs.config import Config
from training.logger import Logger
from utils.safety_tracker import SafetyTracker
from planning.lane_change_explainer import LaneChangeExplainer

# Lane-change interpretability concerns rear/side vehicles (Tier-1 state), so
# use the Tier-1 layout where index 18 is the rear-vehicle block.
DIM = Config(enable_tier1_state=True).state_dim          # 48


def ok(name, result=""):
    print(f"✅ {name} : {result}")


# 1 ------------------------------------------------------------------- #
def test_safety_tracker_init():
    tracker = SafetyTracker()
    assert tracker.vru_collisions == 0
    assert tracker.lane_changes_attempted == 0
    assert tracker.min_ttc_vru == float("inf")
    ok("safety_tracker_init", "initialized correctly")


# 2 ------------------------------------------------------------------- #
def test_safety_tracker_vru():
    tracker = SafetyTracker()
    tracker.record_vru_observation(distance=5.0, speed=1.5, ttc=2.0)
    tracker.record_vru_observation(distance=10.0, speed=0.5, ttc=5.0)
    summary = tracker.summarize()
    assert summary["avg_distance_to_vru"] == 7.5, summary["avg_distance_to_vru"]
    assert summary["min_ttc_vru"] == 2.0, summary["min_ttc_vru"]
    ok("safety_tracker_vru", "metrics computed correctly")


# 3 ------------------------------------------------------------------- #
def test_safety_tracker_near_miss():
    tracker = SafetyTracker()
    tracker.record_vru_observation(5.0, 2.0, 1.5)    # TTC < 2.5 -> near-miss
    tracker.record_vru_observation(5.0, 2.0, 3.0)    # TTC > 2.5 -> safe
    assert tracker.summarize()["vru_near_misses"] == 1
    ok("safety_tracker_near_miss", "detected correctly")


# 4 ------------------------------------------------------------------- #
def test_safety_tracker_rear_incident():
    tracker = SafetyTracker()
    tracker.record_vehicle_observation(3.0, 1.0, 1.5, direction="rear")
    assert tracker.summarize()["rear_incidents"] == 1
    ok("safety_tracker_rear_incident", "rear TTC tracked")


# 5 ------------------------------------------------------------------- #
def test_lane_change_explainer():
    explainer = LaneChangeExplainer()
    state = np.zeros(DIM, dtype=np.float32)
    state[18] = 5.0          # rear vehicle at 5 m
    explainer.record_decision(
        timestep=100,
        action=np.array([0.5, 0.3, 0, 1], dtype=np.float32),   # right lane change
        state=state, info={}, mandate=None,
        is_safe=True, reason="avoid_front_vehicle")
    assert len(explainer.lane_change_log) == 1
    rec = explainer.lane_change_log[0]
    assert rec["direction"] == "right"
    assert rec["is_safe"] is True
    ok("lane_change_explainer", "decisions recorded")


# 6 ------------------------------------------------------------------- #
def test_lane_change_explainer_mandate():
    explainer = LaneChangeExplainer()
    mandate = {"mandate": "stay_in_lane", "direction": "right",
               "reason": "unsafe right lane change (vehicle at 1.5m)"}
    explainer.record_decision(
        timestep=50,
        action=np.array([0.6, 0.2, 0, 1], dtype=np.float32),
        state=np.zeros(DIM, dtype=np.float32), info={}, mandate=mandate,
        is_safe=False, reason="blocked_unsafe_lane_change")
    rec = explainer.lane_change_log[0]
    assert rec["blocked_by_mandate"] is True
    assert "unsafe right" in rec["mandate_reason"]
    ok("lane_change_explainer_mandate", "blocking recorded")


# 7 ------------------------------------------------------------------- #
def test_logger_csv_format():
    tmp = tempfile.mkdtemp(prefix="tier3_log_")
    logger = Logger(tmp, filename="t.csv")
    logger.log(0, {
        "episode": 0, "return": -100.5,
        "vru_collisions": 0, "vehicle_collisions": 1,
        "lane_changes_attempted": 5, "lane_changes_safe": 4,
        "min_ttc_vru": 1.8, "rear_incidents": 2,
    })
    logger.close()
    with open(logger.csv_path, newline="", encoding="utf-8") as f:
        cols = csv.DictReader(f).fieldnames
    for c in ("vru_collisions", "vehicle_collisions", "lane_changes_safe",
              "min_ttc_vru", "rear_incidents"):
        assert c in cols, c
    ok("logger_csv_format", "all columns present")


# 8 ------------------------------------------------------------------- #
def test_full_tracking_pipeline():
    tmp = tempfile.mkdtemp(prefix="tier3_pipe_")
    tracker = SafetyTracker()
    explainer = LaneChangeExplainer()
    logger = Logger(tmp, filename="t.csv")
    for t in range(5):
        tracker.step()
        if t == 2:
            tracker.record_vru_observation(5.0, 1.5, 1.8)       # near-miss
        if t == 3:
            explainer.record_decision(
                timestep=t,
                action=np.array([0.4, 0.2, 0, 1], dtype=np.float32),
                state=np.zeros(DIM, dtype=np.float32), info={}, mandate=None,
                is_safe=True, reason="avoid_front")
    summary = tracker.summarize()
    logger.log(0, summary)
    logger.create_summary_table()      # exercises the console summary
    logger.close()
    assert summary["vru_near_misses"] == 1
    assert len(explainer.lane_change_log) == 1
    assert os.path.exists(logger.csv_path)
    ok("full_tracking_pipeline", "all components integrated")


def main():
    print("Running Tier-3 interpretability tests (no CARLA needed)...\n")
    test_safety_tracker_init()
    test_safety_tracker_vru()
    test_safety_tracker_near_miss()
    test_safety_tracker_rear_incident()
    test_lane_change_explainer()
    test_lane_change_explainer_mandate()
    test_logger_csv_format()
    test_full_tracking_pipeline()
    print("\n✅ ALL TIER 3 INTERPRETABILITY TESTS PASSED")


if __name__ == "__main__":
    main()
