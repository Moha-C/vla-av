"""Full end-to-end pipeline test that runs with NO CARLA installed.

Run with:  python tests/test_mock.py
"""
import os
import sys

# Make the project root importable when run as `python tests/test_mock.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Ensure UTF-8 output so the ✅ marks render on Windows consoles (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from configs.config import Config
from env.carla_env import CarlaEnv
from models.world_model import WorldModel
from models.actor_critic import ActorCritic
from models.rssm import RSSM
from training.rollout_buffer import RolloutBuffer
from training.ppo import update_ppo, update_world_model
from training.dreamer_ppo import select_action_with_dreaming, train
from training.ppo_baseline import train_baseline
from training.world_model_trainer import WorldModelTrainer

# tests/ dir on path so the standalone reward suite is importable.
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))
from test_rewards import run_all as run_reward_scenarios


def ok(name, result=""):
    print(f"✅ {name} : {result}")


def test_env(config):
    env = CarlaEnv(mock=True, config=config)
    state = env.reset()
    assert state.shape == (config.state_dim,), state.shape
    for _ in range(10):
        action = np.random.uniform(-1, 1, size=config.action_dim).astype(np.float32)
        next_state, reward, done, info = env.step(action)
        assert next_state.shape == (config.state_dim,)
        assert np.isscalar(reward) or np.ndim(reward) == 0
        assert isinstance(done, bool)
        assert "vru_risk" in info and "progress" in info
    env.close()
    ok("CarlaEnv(mock=True)", "reset + 10 steps, shapes correct")


def test_world_model(config):
    wm = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)
    B = 8
    state = torch.randn(B, config.state_dim)
    action = torch.randn(B, config.action_dim)
    ns, risk, prog = wm(state, action)
    assert ns.shape == (B, config.state_dim)
    assert risk.shape == (B, 1)
    assert prog.shape == (B, 1)
    assert torch.all((risk >= 0) & (risk <= 1))
    ok("WorldModel.forward", f"shapes {tuple(ns.shape)}, {tuple(risk.shape)}, {tuple(prog.shape)}")


def test_rssm(config):
    rssm = RSSM(config.state_dim, config.action_dim)
    B = 8
    state = torch.randn(B, config.state_dim)
    action = torch.randn(B, config.action_dim)
    ns, risk, prog, hidden = rssm(state, action)
    assert ns.shape == (B, config.state_dim)
    assert risk.shape == (B, 1)
    assert prog.shape == (B, 1)
    ns2, _, _, _ = rssm(state, action, hidden)
    assert ns2.shape == (B, config.state_dim)
    ok("RSSM.forward", "recurrent step shapes correct")


def test_actor_critic(config):
    ac = ActorCritic(config.state_dim, config.action_dim, config.hidden)
    B = 8
    state = torch.randn(B, config.state_dim)
    action, log_prob, value, raw_action = ac.act(state)
    assert action.shape == (B, config.action_dim)
    assert log_prob.shape == (B,)
    assert value.shape == (B,)
    # bounds: steering in [-1,1], throttle/brake/stop in [0,1]
    assert torch.all(action[:, 0] >= -1) and torch.all(action[:, 0] <= 1)
    assert torch.all(action[:, 1:] >= 0) and torch.all(action[:, 1:] <= 1)
    lp, ent, val = ac.evaluate(state, raw_action)
    assert lp.shape == (B,) and ent.shape == (B,) and val.shape == (B,)
    ok("ActorCritic.act/evaluate", "shapes + action bounds correct")


def test_rollout_buffer(config):
    size = config.rollout_size
    buf = RolloutBuffer(size, config.state_dim, config.action_dim,
                        gamma=config.gamma, lam=config.lam)
    for _ in range(size):
        buf.store(
            state=np.random.randn(config.state_dim).astype(np.float32),
            action=np.random.randn(config.action_dim).astype(np.float32),
            reward=float(np.random.randn()),
            done=False,
            value=float(np.random.randn()),
            log_prob=float(np.random.randn()),
            next_state=np.random.randn(config.state_dim).astype(np.float32),
            risk_target=float(np.random.rand()),
            progress_target=float(np.random.randn()),
        )
    assert buf.is_full()
    buf.finish_path(last_value=0.0)
    batch = buf.get()
    assert batch["states"].shape == (size, config.state_dim)
    assert batch["advantages"].shape == (size,)
    assert batch["returns"].shape == (size,)
    # advantages normalized
    adv = batch["advantages"]
    assert abs(float(adv.mean())) < 1e-4
    assert abs(float(adv.std()) - 1.0) < 1e-2
    ok("RolloutBuffer", f"stored {size}, GAE + normalized advantages")


def test_ppo_update(config):
    ac = ActorCritic(config.state_dim, config.action_dim, config.hidden)
    opt = torch.optim.Adam(ac.parameters(), lr=config.lr_policy)
    B = 256
    batch = {
        "states": torch.randn(B, config.state_dim),
        "actions": torch.randn(B, config.action_dim),
        "log_probs": torch.randn(B),
        "advantages": torch.randn(B),
        "returns": torch.randn(B),
    }
    out = update_ppo(ac, opt, batch, clip_eps=config.clip_eps,
                     ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                     max_grad_norm=config.max_grad_norm)
    assert np.isscalar(out["loss"])
    ok("update_ppo", f"loss={out['loss']:.4f} (scalar)")


def test_world_model_update(config):
    wm = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)
    opt = torch.optim.Adam(wm.parameters(), lr=config.lr_wm)
    B = 256
    batch = {
        "states": torch.randn(B, config.state_dim),
        "actions": torch.randn(B, config.action_dim),
        "next_states": torch.randn(B, config.state_dim),
        "risk_targets": torch.rand(B, 1),
        "progress_targets": torch.randn(B, 1),
    }
    out = update_world_model(wm, opt, batch, max_grad_norm=config.max_grad_norm)
    assert np.isscalar(out["loss"])
    ok("update_world_model", f"loss={out['loss']:.4f} (scalar)")


def test_dreaming(config):
    ac = ActorCritic(config.state_dim, config.action_dim, config.hidden)
    wm = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)
    state = torch.randn(config.state_dim)
    action, raw_action, log_prob, value, scores = select_action_with_dreaming(
        ac, wm, state, k=config.dream_k,
        w_progress=config.w_progress, w_risk=config.w_risk, w_value=config.w_value,
    )
    assert action.shape == (config.action_dim,), action.shape
    assert raw_action.shape == (config.action_dim,)
    assert len(scores) == config.dream_k
    ok("select_action_with_dreaming", f"action shape {tuple(action.shape)}")


def test_training_loop(config):
    # Small rollout so 3 episodes run quickly; no eval/checkpoint side effects.
    small = Config()
    small.rollout_size = 256
    small.update_epochs = 2
    small.batch_size = 64
    small.max_episode_steps = 50
    history = train(small, mock=True, num_episodes=3, verbose=False,
                    eval_interval=0, ckpt_dir=None)
    assert len(history) == 3
    for h in history:
        assert np.isscalar(h["return"])
    ok("train(mock=True)", f"3 episodes, returns "
       f"{[round(h['return'], 1) for h in history]}")


def test_ppo_baseline(config):
    # Small rollout so 3 episodes run quickly; no logging side effects.
    small = Config()
    small.rollout_size = 256
    small.update_epochs = 2
    small.batch_size = 64
    small.max_episode_steps = 50
    history = train_baseline(small, mock=True, num_episodes=3, verbose=False,
                             log_dir=None)
    assert len(history) == 3
    for h in history:
        assert np.isscalar(h["return"])
        assert "vru_collisions" in h and "ppo_loss" in h
    ok("ppo_baseline", "3 episodes completed")


def test_vru_rewards(config):
    n = run_reward_scenarios(verbose=False)
    assert n == 6
    ok("vru_rewards", "all 6 scenarios passed")


def test_world_model_trainer(config):
    wm = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)
    trainer = WorldModelTrainer(wm, config)
    # A fixed batch: repeated updates should drive the loss down.
    batch = {
        "states": torch.randn(256, config.state_dim),
        "actions": torch.randn(256, config.action_dim),
        "next_states": torch.randn(256, config.state_dim),
        "risk_targets": torch.rand(256),
        "progress_targets": torch.randn(256),
    }
    losses = [trainer.update(batch)["loss_wm"] for _ in range(5)]
    assert losses[-1] < losses[0], losses
    ok("world_model_trainer", "loss decreased over 5 updates")


def test_dreamer_ppo_full(config):
    import shutil
    import tempfile
    ckpt = tempfile.mkdtemp(prefix="dppo_ckpt_")
    try:
        small = Config()
        small.rollout_size = 256
        small.update_epochs = 2
        small.batch_size = 64
        small.max_episode_steps = 50
        small.wm_warmup_steps = 0          # dreaming gates ON from episode 0
        history = train(small, mock=True, num_episodes=3, verbose=False,
                        eval_interval=0, ckpt_dir=ckpt)
        assert len(history) == 3
        assert sum(h["dreaming_steps"] for h in history) > 0
        assert all(h["dreaming_active"] == 1 for h in history)
        assert os.path.isdir(ckpt)
        assert any(f.endswith(".pt") for f in os.listdir(ckpt))
    finally:
        shutil.rmtree(ckpt, ignore_errors=True)
    ok("dreamer_ppo_full", "3 episodes, dreaming gates correctly")


def main():
    config = Config()
    print("Running Dreamer-PPO mock pipeline tests (no CARLA needed)...\n")
    test_env(config)
    test_world_model(config)
    test_rssm(config)
    test_actor_critic(config)
    test_rollout_buffer(config)
    test_ppo_update(config)
    test_world_model_update(config)
    test_dreaming(config)
    test_training_loop(config)
    test_ppo_baseline(config)
    test_vru_rewards(config)
    test_world_model_trainer(config)
    test_dreamer_ppo_full(config)
    print("\n✅ ALL TESTS PASSED")


if __name__ == "__main__":
    main()
