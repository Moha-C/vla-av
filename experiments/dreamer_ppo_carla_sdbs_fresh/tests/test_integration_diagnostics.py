"""Integration diagnostics that mirror the reported downstream failure.

The S-DBS planner was integrated into a separate CARLA adapter and every
variant scored 0% route completion, with S-DBS worse than baseline. The
suspected causes were (a) a state-vector dimension/order mismatch between what
the policy was trained on and what it is fed at inference, (b) no way to verify
the multi-step dreaming horizon incrementally, and (c) no early sanity checks.

These tests exercise the guardrails added for exactly those failure modes so
any future integration can be validated in isolation, without CARLA.

Run with:  python tests/test_integration_diagnostics.py
"""
import os
import sys
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from configs.config import Config
from configs.sdbs_config import SDBSConfig
from env.carla_env import (
    CarlaEnv, expected_state_dim, state_layout, validate_state_vector,
)
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from rewards.vru_reward import resolve_layout, BASE_STATE_DIM
from training.dreamer_ppo import train, train_sdbs, select_action_with_dreaming
from planning.sdbs_planner import SDBSPlanner
from utils.checkpoint_check import check_checkpoint_compatibility
from utils.progress_monitor import ProgressMonitor


def ok(name, result=""):
    print(f"✅ {name} : {result}")


def _small_sdbs_config(horizon=1, groups=1, beam_width=4):
    cfg = SDBSConfig()
    cfg.rollout_size = 128
    cfg.update_epochs = 1
    cfg.batch_size = 64
    cfg.max_episode_steps = 20
    cfg.wm_warmup_steps = 0
    cfg.wm_batch_size = 64
    cfg.compute_budget = 30
    cfg.scenarios_per_stage = 2
    cfg.max_scenarios_per_episode = 2
    cfg.sdbs_force_fixed_params = True
    cfg.sdbs_fixed_horizon = horizon
    cfg.sdbs_fixed_groups = groups
    cfg.sdbs_fixed_beam_width = beam_width
    return cfg


# 1 ------------------------------------------------------------------- #
def test_state_dimension_contract():
    """A correctly-built state passes; a wrong-shaped one fails loudly.

    The expected dimension is read from the config (28 default, 55 with both
    tiers) — never a hardcoded constant.
    """
    # Default (v1) layout: the env emits exactly 28 dims.
    cfg = Config()
    env = CarlaEnv(mock=True, config=cfg)
    state = env.reset()
    env.close()
    exp = expected_state_dim(cfg)
    assert exp == 28, exp
    assert state.shape[-1] == exp, state.shape
    validate_state_vector(state, exp)              # correct shape -> no raise

    # Deliberately wrong-shaped input MUST raise (never silently reshape).
    raised = False
    try:
        validate_state_vector(np.zeros(42, dtype=np.float32), exp)
    except ValueError:
        raised = True
    assert raised, "validate_state_vector accepted a 42-dim vector"

    # With both tiers enabled the env emits 55 dims, and the check follows.
    full_cfg = Config(enable_tier1_state=True, enable_tier2_state=True)
    full_env = CarlaEnv(mock=True, config=full_cfg)
    full_state = full_env.reset()
    full_env.close()
    assert full_state.shape[-1] == expected_state_dim(full_cfg) == 55
    validate_state_vector(full_state, expected_state_dim(full_cfg))
    ok("state_dimension_contract", "validated correctly (28 default, 55 tiers)")


# 2 ------------------------------------------------------------------- #
def test_checkpoint_compatibility():
    """A saved checkpoint's input dim is detected, and a mismatch is caught."""
    ckpt_dir = tempfile.mkdtemp(prefix="diag_ckpt_")
    try:
        cfg = Config()
        cfg.rollout_size = 128
        cfg.update_epochs = 1
        cfg.batch_size = 64
        cfg.max_episode_steps = 20
        # One episode of mock training writes checkpoints/episode_0000.pt.
        train(cfg, mock=True, num_episodes=1, verbose=False,
              eval_interval=0, ckpt_dir=ckpt_dir)
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        assert ckpts, "training produced no checkpoint"
        ckpt_path = os.path.join(ckpt_dir, ckpts[0])

        # Correct state dim -> PASS.
        assert check_checkpoint_compatibility(
            ckpt_path, cfg.state_dim, cfg.action_dim) is True
        # Deliberately wrong state dim -> FAIL (does not raise, returns False).
        assert check_checkpoint_compatibility(
            ckpt_path, 42, cfg.action_dim) is False
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    ok("checkpoint_compatibility_check", "correctly detects mismatch")


# 3 ------------------------------------------------------------------- #
def test_sdbs_h1_g1_equivalence():
    """Fixed H=1,G=1 S-DBS collapses to the same single-step path as dreaming."""
    cfg = _small_sdbs_config(horizon=1, groups=1, beam_width=4)  # base 28-dim
    policy = ActorCritic(cfg.state_dim, cfg.action_dim, cfg.hidden)
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    planner = SDBSPlanner(policy, wm, policy, cfg)

    # Forcing fixed params bypasses difficulty scaling: (B, H, G) == fixed.
    B, H, G = planner.get_search_params(difficulty=0.9)
    assert (B, H, G) == (4, 1, 1), (B, H, G)

    state = np.zeros(cfg.state_dim, dtype=np.float32)
    state[7] = 3.5
    state[10] = 2.0          # green light
    state[11] = 50.0

    sdbs_action, plan, meta = planner.plan(state)
    assert sdbs_action.shape == (cfg.action_dim,), sdbs_action.shape
    assert meta["lookahead"] == 1 and meta["groups"] == 1
    assert len(plan.actions) == 1          # a single imagined step

    # The plain one-step dreaming path produces the same-shaped single action.
    dream_action, raw, log_prob, value, scores = select_action_with_dreaming(
        policy, wm, torch.as_tensor(state), k=cfg.dream_k)
    assert dream_action.shape == (cfg.action_dim,), dream_action.shape
    ok("sdbs_h1_g1_equivalence", "matches one-step dreaming path")


# 4 ------------------------------------------------------------------- #
def test_progress_monitor():
    """A never-advancing route_progress is flagged; a rising one is not."""
    stalled = ProgressMonitor(stall_threshold_steps=200)
    for t in range(250):
        stalled.record(t, 0.3)                     # identical every step
    result = stalled.check_stalled()
    assert result["stalled"] is True, result
    assert "wiring bug" in result["reason"]

    moving = ProgressMonitor(stall_threshold_steps=200)
    for t in range(250):
        moving.record(t, 0.001 * t)                # steadily increasing
    assert moving.check_stalled()["stalled"] is False
    ok("progress_monitor", "stall detection works correctly")


# 5 ------------------------------------------------------------------- #
def test_incremental_horizon():
    """H=1,2,3 (fixed) all run cleanly with no shape errors or NaN losses."""
    for h in (1, 2, 3):
        cfg = _small_sdbs_config(horizon=h, groups=h, beam_width=max(4, 2 * h))
        history = train_sdbs(cfg, mock=True, num_episodes=2, verbose=False,
                             log_dir=None, ckpt_dir=None)
        assert len(history) == 2, (h, len(history))
        for rec in history:
            for key in ("return", "ppo_loss", "loss_wm", "aux_loss"):
                assert np.isfinite(rec[key]), (h, key, rec[key])
    ok("incremental_horizon", "H=1,2,3 all run cleanly")


# 6 ------------------------------------------------------------------- #
def test_default_config_matches_v1_layout():
    """A fresh default Config() is the original v1 28-dim layout, in order.

    Guards against silently defaulting to an expanded state that was never
    validated against the downstream adapter.
    """
    cfg = Config()
    assert cfg.state_dim == BASE_STATE_DIM == 28, cfg.state_dim
    assert cfg.enable_tier1_state is False and cfg.enable_tier2_state is False

    lay = resolve_layout(cfg)
    assert lay.dim == 28 and lay.vru0 == 18, (lay.dim, lay.vru0)

    # Field order is exactly ego/lane/traffic/vehicle_ahead/vru — no rear/side
    # vehicle blocks and no map-agnostic block in the default layout.
    ranges = state_layout(cfg)
    assert list(ranges.keys()) == [
        'ego', 'lane', 'traffic', 'vehicle_ahead', 'vru'], list(ranges.keys())
    assert ranges['ego'] == (0, 6)
    assert ranges['lane'] == (6, 10)
    assert ranges['traffic'] == (10, 13)
    assert ranges['vehicle_ahead'] == (13, 18)
    assert ranges['vru'] == (18, 28)
    ok("default_config_matches_v1_layout", "28-dim ego/lane/traffic/veh/vru")


def main():
    print("Running S-DBS integration diagnostics (no CARLA needed)...\n")
    torch.manual_seed(0)
    np.random.seed(0)
    test_state_dimension_contract()
    test_checkpoint_compatibility()
    test_sdbs_h1_g1_equivalence()
    test_progress_monitor()
    test_incremental_horizon()
    test_default_config_matches_v1_layout()
    print("\n✅ ALL INTEGRATION DIAGNOSTIC TESTS PASSED")


if __name__ == "__main__":
    main()
