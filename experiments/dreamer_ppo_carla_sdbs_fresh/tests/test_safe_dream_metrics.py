"""SAFE-DREAM metrics validation suite — runs with NO CARLA installed.

Run with:  python tests/test_safe_dream_metrics.py

Uses small synthetic in-memory CSVs (written to a temp dir) for the metric
math, plus a short mock training run for the end-to-end pipeline check.
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

from evaluation.safe_dream_metrics import SafeDreamEvaluator


def ok(name, result=""):
    print(f"✅ {name} : {result}")


def _write_csv(path, columns):
    """Write a CSV from a {column: [values]} dict (all lists equal length)."""
    fieldnames = list(columns.keys())
    n = len(columns[fieldnames[0]]) if fieldnames else 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({k: columns[k][i] for k in fieldnames})
    return path


# 1 ------------------------------------------------------------------- #
def test_collision_rate(tmp):
    vru = [1, 0, 2, 0, 0, 1, 0, 0, 3, 0]     # sum 7
    veh = [0, 1, 0, 0, 1, 0, 0, 2, 0, 0]     # sum 4
    path = _write_csv(os.path.join(tmp, "cr.csv"),
                      {"vru_collisions": vru, "vehicle_collisions": veh})
    ev = SafeDreamEvaluator(path, "sdbs", distance_per_episode_km=0.15)
    total = sum(vru) + sum(veh)              # 11
    expected = total / (0.15 * 10) * 1e6
    got = ev.collision_rate()
    assert abs(got - expected) < 1e-3, (got, expected)
    ok("collision_rate", f"matches expected value ({got:,.0f}/Mkm)")


# 2 ------------------------------------------------------------------- #
def test_near_collision_rate(tmp):
    min_ttc = [1.0, 2.0, 0.5, 3.0, 1.4, 1.6, 1.5, 0.9, 5.0, 1.49]
    # Below 1.5: 1.0, 0.5, 1.4, 0.9, 1.49  -> 5 episodes (1.5 is NOT below).
    path = _write_csv(os.path.join(tmp, "ncr.csv"),
                      {"min_ttc_this_episode": min_ttc})
    ev = SafeDreamEvaluator(path, "sdbs", distance_per_episode_km=0.15)
    expected = 5 / (0.15 * 10) * 1e6
    got = ev.near_collision_rate(ttc_threshold=1.5)
    assert abs(got - expected) < 1e-3, (got, expected)
    ok("near_collision_rate", "threshold applied correctly")


# 3 ------------------------------------------------------------------- #
def test_min_ttc_stats(tmp):
    vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    path = _write_csv(os.path.join(tmp, "ttc.csv"),
                      {"min_ttc_this_episode": vals})
    ev = SafeDreamEvaluator(path, "sdbs")
    stats = ev.min_ttc_stats()
    assert stats["min"] == 0.5, stats
    assert abs(stats["p5"] - float(np.percentile(vals, 5))) < 1e-9, stats
    ok("min_ttc_stats", f"min={stats['min']}, p5={stats['p5']:.3f} correct")


# 4 ------------------------------------------------------------------- #
def test_counterfactual_coverage_baseline(tmp):
    # Baseline: no beam search -> n_candidates_evaluated all zero.
    path = _write_csv(os.path.join(tmp, "baseline_cc.csv"),
                      {"n_candidates_evaluated": [0, 0, 0],
                       "vru_collisions": [0, 1, 0]})
    ev = SafeDreamEvaluator(path, "baseline")
    result = ev.counterfactual_coverage()
    assert result is None, result
    ok("counterfactual_coverage", "correctly N/A for baseline")


# 5 ------------------------------------------------------------------- #
def test_future_diversity(tmp):
    entropy = [0.5, 1.0, 1.5, 2.0]           # mean 1.25
    path = _write_csv(os.path.join(tmp, "fd.csv"),
                      {"candidate_score_entropy": entropy,
                       "n_candidates_evaluated": [4, 4, 4, 4]})
    ev = SafeDreamEvaluator(path, "sdbs")
    got = ev.future_diversity()
    assert abs(got - 1.25) < 1e-9, got
    ok("future_diversity", f"entropy averaged correctly ({got:.3f})")


# 6 ------------------------------------------------------------------- #
def test_umrr(tmp):
    n_eval = [10, 20, 30, 40]                # sum 100
    n_rej = [1, 2, 3, 4]                     # sum 10
    path = _write_csv(os.path.join(tmp, "umrr.csv"),
                      {"n_candidates_evaluated": n_eval,
                       "n_candidates_rejected_unsafe": n_rej})
    ev = SafeDreamEvaluator(path, "sdbs")
    got = ev.unsafe_maneuver_rejection_rate()
    assert abs(got - 10.0) < 1e-9, got
    ok("unsafe_maneuver_rejection_rate", f"percentage correct ({got:.1f}%)")


# 7 ------------------------------------------------------------------- #
def test_safety_gain_sign(tmp):
    n = 10
    safe = _write_csv(os.path.join(tmp, "sg_safe.csv"),
                      {"vru_collisions": [0] * n, "vehicle_collisions": [0] * n,
                       "min_ttc_this_episode": [5.0] * n})
    risky = _write_csv(os.path.join(tmp, "sg_risky.csv"),
                       {"vru_collisions": [5] * n, "vehicle_collisions": [2] * n,
                        "min_ttc_this_episode": [0.5] * n})
    ev_safe = SafeDreamEvaluator(safe, "sdbs")
    ev_risky = SafeDreamEvaluator(risky, "baseline")

    sg = ev_safe.safety_gain(ev_risky)       # safe vs risky -> positive
    assert sg > 0, sg
    sg_rev = ev_risky.safety_gain(ev_safe)   # risky vs safe -> negative
    assert sg_rev < 0, sg_rev
    ok("safety_gain", "sign convention matches paper (positive = safer)")


# 8 ------------------------------------------------------------------- #
def test_generate_report(tmp):
    n = 5
    path = _write_csv(os.path.join(tmp, "report.csv"), {
        "vru_collisions": [0, 1, 0, 0, 2],
        "vehicle_collisions": [0, 0, 1, 0, 0],
        "min_ttc_this_episode": [3.0, 1.2, 2.0, 4.0, 0.8],
        "min_safety_distance_this_episode": [5.0, 1.5, 3.0, 6.0, 0.9],
        "safety_envelope_violations": [1, 3, 0, 0, 4],
        "disengagement_count": [0, 1, 0, 0, 2],
        "jerk_mean": [0.5, 0.6, 0.4, 0.7, 0.9],
        "jerk_max": [1.0, 1.5, 0.8, 1.2, 2.0],
        "n_candidates_evaluated": [12, 12, 12, 12, 12],
        "n_candidates_rejected_unsafe": [1, 2, 0, 1, 3],
        "candidate_score_entropy": [1.0, 1.1, 0.9, 1.2, 1.3],
        "chosen_risk_prediction": [0.2, 0.6, 0.3, 0.1, 0.8],
        "r_vru": [-0.1, -0.5, -0.2, -0.05, -0.9],
    })
    ev = SafeDreamEvaluator(path, "sdbs")
    report = ev.generate_report()

    expected_keys = {
        "variant", "n_episodes", "distance_per_episode_km", "distance_assumption",
        "collision_rate", "near_collision_rate", "min_ttc", "min_safety_distance",
        "safety_envelope_violations", "disengagement_rate",
        "traffic_rule_compliance", "driving_comfort",
        "counterfactual_coverage", "future_diversity",
        "unsafe_maneuver_rejection_rate", "risk_prediction_accuracy",
        "hazard_anticipation_horizon", "not_applicable",
    }
    missing = expected_keys - set(report.keys())
    assert not missing, f"missing keys: {missing}"
    assert isinstance(report["not_applicable"], list)
    assert len(report["not_applicable"]) == 6, report["not_applicable"]
    ok("generate_report", "all keys present, skipped dimensions documented")


# 9 ------------------------------------------------------------------- #
def test_full_pipeline(tmp):
    from configs.config import Config
    from configs.sdbs_config import SDBSConfig
    from training.ppo_baseline import train_baseline
    from training.dreamer_ppo import train_sdbs

    # --- baseline (no beam) ---
    base_cfg = Config()
    base_cfg.rollout_size = 128
    base_cfg.update_epochs = 1
    base_cfg.max_episode_steps = 20
    train_baseline(base_cfg, mock=True, num_episodes=3, verbose=False,
                   log_dir=tmp, log_name="baseline.csv")

    # --- sdbs (real beam-search diagnostics) ---
    sdbs_cfg = SDBSConfig()
    sdbs_cfg.rollout_size = 128
    sdbs_cfg.update_epochs = 1
    sdbs_cfg.batch_size = 64
    sdbs_cfg.max_episode_steps = 20
    sdbs_cfg.wm_warmup_steps = 0
    sdbs_cfg.wm_batch_size = 64
    sdbs_cfg.beam_width_max = 6
    sdbs_cfg.horizon_max = 2
    sdbs_cfg.num_groups_max = 2
    sdbs_cfg.compute_budget = 30
    sdbs_cfg.scenarios_per_stage = 2
    sdbs_cfg.max_scenarios_per_episode = 2
    train_sdbs(sdbs_cfg, mock=True, num_episodes=3, verbose=False,
               log_dir=tmp, ckpt_dir=None, log_name="sdbs.csv")

    base_ev = SafeDreamEvaluator(os.path.join(tmp, "baseline.csv"), "baseline")
    sdbs_ev = SafeDreamEvaluator(os.path.join(tmp, "sdbs.csv"), "sdbs")

    base_report = base_ev.generate_report()
    sdbs_report = sdbs_ev.generate_report()
    assert base_report["counterfactual_coverage"] is None      # baseline N/A
    assert sdbs_report["counterfactual_coverage"] is not None   # beam ran

    sg = sdbs_ev.safety_gain(base_ev)
    assert isinstance(sg, float), type(sg)
    ok("full_pipeline", "mock data flows through cleanly end-to-end")


def main():
    print("Running SAFE-DREAM metrics tests (no CARLA needed)...\n")
    np.random.seed(0)
    tmp = tempfile.mkdtemp(prefix="safe_dream_")
    test_collision_rate(tmp)
    test_near_collision_rate(tmp)
    test_min_ttc_stats(tmp)
    test_counterfactual_coverage_baseline(tmp)
    test_future_diversity(tmp)
    test_umrr(tmp)
    test_safety_gain_sign(tmp)
    test_generate_report(tmp)
    test_full_pipeline(tmp)
    print("\n✅ ALL SAFE-DREAM METRICS TESTS PASSED")


if __name__ == "__main__":
    main()
