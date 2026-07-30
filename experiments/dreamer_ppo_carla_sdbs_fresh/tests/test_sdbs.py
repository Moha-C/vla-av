"""S-DBS validation suite — runs with NO CARLA installed.

Run with:  python tests/test_sdbs.py
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
import torch

from configs.sdbs_config import SDBSConfig
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from planning.sdbs_core import (
    Plan, BeamState, compute_conflict_cells, jaccard_diversity,
)
from planning.sdbs_planner import SDBSPlanner
from planning.curriculum import (
    ScenarioBank, SumTree, PrioritizedScenarioReplayer, RiskAwareCurriculum,
)
from rewards.vru_reward import EGO_SPEED, resolve_layout


def ok(name, result=""):
    print(f"✅ {name} : {result}")


def _mock_state(config, vru_dists=(30.0, 30.0), speed=5.0, x=0.0, y=0.0):
    # Place the VRU blocks at the layout-resolved position so this helper works
    # at any dimension (28 default, 48/55 with tiers enabled).
    lay = resolve_layout(config)
    s = np.zeros(config.state_dim, dtype=np.float32)
    s[0], s[1] = x, y
    s[EGO_SPEED] = speed
    s[7] = 3.5
    s[10] = 2.0          # traffic light: green (0 = red)
    s[11] = 50.0         # dist to light (far)
    s[lay.vru0] = vru_dists[0]
    s[lay.vru1] = vru_dists[1]
    return s


# 1 ------------------------------------------------------------------- #
def test_conflict_cells():
    # A straight trajectory stepping 1.0 m in x each step, cell size 0.5 m.
    traj = []
    for t in range(4):
        s = np.zeros(28, dtype=np.float32)
        s[0] = t * 1.0
        s[1] = 0.0
        traj.append(s)
    cells = compute_conflict_cells(traj, discretization=0.5)
    assert len(cells) == 4
    assert (0, 0, 0) in cells          # x=0   -> cell 0
    assert (2, 0, 1) in cells          # x=1.0 -> cell 2 at time 1
    assert (4, 0, 2) in cells          # x=2.0 -> cell 4 at time 2
    ok("conflict_cells", "discretization correct")


# 2 ------------------------------------------------------------------- #
def test_jaccard():
    a = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    b = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    ca = compute_conflict_cells([_pos(p) for p in a])
    cb = compute_conflict_cells([_pos(p) for p in b])
    assert abs(jaccard_diversity(ca, cb) - 1.0) < 1e-9      # identical -> 1
    far = compute_conflict_cells([_pos(np.array([100.0, 100.0])),
                                  _pos(np.array([101.0, 100.0]))])
    assert jaccard_diversity(ca, far) == 0.0                # disjoint -> 0
    ok("jaccard_diversity", "overlap computation correct")


def _pos(xy):
    s = np.zeros(28, dtype=np.float32)
    s[0], s[1] = float(xy[0]), float(xy[1])
    return s


# 3 ------------------------------------------------------------------- #
def test_beam_state():
    beam = BeamState(depth=0, num_groups=2)
    for i in range(10):
        p = Plan(actions=[i], imagined_states=[], group_id=i % 2)
        p.score = float(i)
        beam.add_plan(p, i % 2)
    beam.get_top_per_group(2)
    for g in beam.groups:
        assert len(g) <= 2
        scores = [p.score for p in g]
        assert scores == sorted(scores, reverse=True)
    assert max(p.score for p in beam.groups[0]) == 8.0   # even indices
    assert max(p.score for p in beam.groups[1]) == 9.0   # odd indices
    ok("beam_state", "trim and rank correct")


# 4 ------------------------------------------------------------------- #
def test_difficulty():
    config = SDBSConfig()
    policy = ActorCritic(config.state_dim, config.action_dim, config.hidden)
    wm = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)
    planner = SDBSPlanner(policy, wm, policy, config)

    calm = _mock_state(config, vru_dists=(40.0, 40.0), speed=2.0)
    danger = _mock_state(config, vru_dists=(2.0, 3.0), speed=12.0)
    d_calm = planner.estimate_scene_difficulty(calm, {})
    d_danger = planner.estimate_scene_difficulty(
        danger, {"occlusion_flag": 1.0, "vru_risk": 0.9})
    assert 0.0 <= d_calm <= 1.0 and 0.0 <= d_danger <= 1.0
    assert d_danger > d_calm
    B, H, G = planner.get_search_params(d_danger)
    assert config.beam_width_min <= B <= config.beam_width_max
    assert config.horizon_min <= H <= config.horizon_max
    ok("difficulty_estimation", f"calm={d_calm:.2f} danger={d_danger:.2f}")


# 5 ------------------------------------------------------------------- #
def test_mandated_safety():
    config = SDBSConfig()
    policy = ActorCritic(config.state_dim, config.action_dim, config.hidden)
    wm = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)
    planner = SDBSPlanner(policy, wm, policy, config)

    # VRU 1.5 m ahead at 12 m/s -> TTC ~0.12 s < tau_safe -> hard stop.
    danger = _mock_state(config, vru_dists=(1.5, 40.0), speed=12.0)
    mandate = planner.evaluate_mandated_safety(danger, {})
    assert mandate["mandate"] == "stop"
    assert mandate["clamped_controls"][2] == 1.0          # full brake
    safe = _mock_state(config, vru_dists=(40.0, 40.0), speed=2.0)
    assert planner.evaluate_mandated_safety(safe, {})["mandate"] is None
    ok("mandated_safety", "hard constraints recognized")


# 6 ------------------------------------------------------------------- #
def test_full_planning():
    config = SDBSConfig()
    config.compute_budget = 60
    policy = ActorCritic(config.state_dim, config.action_dim, config.hidden)
    wm = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)
    planner = SDBSPlanner(policy, wm, policy, config)

    state = _mock_state(config, vru_dists=(8.0, 12.0), speed=6.0)
    best_action, plan, meta = planner.plan(state, {"vru_risk": 0.4})
    assert best_action.shape == (config.action_dim,)
    assert isinstance(plan, Plan)
    for key in ("lookahead", "beam_width", "groups", "difficulty",
                "conflict_cells_best", "serendipity_bonus_used",
                "planning_latency_ms", "first_raw_action"):
        assert key in meta, key
    assert meta["first_raw_action"].shape == (config.action_dim,)
    ok("sdbs_planning", "full pipeline works")


# 7 ------------------------------------------------------------------- #
def test_curriculum_unlock():
    config = SDBSConfig()
    config.scenarios_per_stage = 4
    curriculum = RiskAwareCurriculum(config)
    assert curriculum.current_stage() == 1
    stage1 = curriculum.stage_scenarios[0]
    # Record a 95%+ success rate across all stage-1 scenarios.
    for sid in stage1:
        for _ in range(20):
            curriculum.record_rollout(
                sid, {"vru_collisions": 0, "route_completion": 1.0})
    assert curriculum.should_advance_stage()
    curriculum.advance_to_next_stage()
    assert curriculum.current_stage() == 2
    active = curriculum.get_active_scenarios()
    assert any(s.startswith("stage2_") for s in active)
    ok("curriculum", "stage advancement correct")


# 8 ------------------------------------------------------------------- #
def test_per_sampling():
    bank = ScenarioBank()
    for i in range(100):
        bank.add(f"sc_{i:03d}", difficulty=0.5)
    replayer = PrioritizedScenarioReplayer(bank, capacity=128)
    # Scenario i fails with i collisions -> priority increases with i.
    for i in range(100):
        replayer.record_episode(f"sc_{i:03d}", n_collisions=i,
                                n_near_misses=0, n_ttc_violations=0,
                                progress_deficit=0.0)
    samples = replayer.sample_scenarios(4000)
    idxs = [int(s.split("_")[1]) for s in samples]
    mean_idx = float(np.mean(idxs))
    # Higher-priority (higher-index) scenarios should dominate; a uniform
    # sampler would give ~49.5, priority-weighted (p ~ i^alpha) gives ~61.
    assert mean_idx > 56, mean_idx
    ok("per_sampling", f"priorities respected (mean idx={mean_idx:.1f})")


def main():
    print("Running S-DBS tests (no CARLA needed)...\n")
    torch.manual_seed(0)
    np.random.seed(0)
    test_conflict_cells()
    test_jaccard()
    test_beam_state()
    test_difficulty()
    test_mandated_safety()
    test_full_planning()
    test_curriculum_unlock()
    test_per_sampling()
    print("\n✅ ALL S-DBS TESTS PASSED")


if __name__ == "__main__":
    main()
