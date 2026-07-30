"""Standalone tests for dreaming action selection and evaluation.

Run with:  python tests/test_dreaming.py
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

import torch

from configs.config import Config
from env.carla_env import CarlaEnv
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from training.world_model_trainer import WorldModelTrainer
from training.evaluator import Evaluator
from training.dreamer_ppo import select_action_with_dreaming, _select_action, train


def ok(name, result=""):
    print(f"✅ {name} : {result}")


# ---------------------------------------------------------------------- #
# Mocks for the safety test
# ---------------------------------------------------------------------- #
class MockPolicy:
    """Cycles candidate throttles so some are 'risky' (>0.5) and some safe."""
    def __init__(self):
        self._throttles = [0.8, 0.2, 0.9, 0.1, 0.7]
        self._i = 0

    def act(self, state):
        t = self._throttles[self._i % len(self._throttles)]
        self._i += 1
        action = torch.tensor([[0.0, t, 0.0, 1.0]], dtype=torch.float32)
        raw = action.clone()
        log_prob = torch.tensor([0.0])
        value = torch.tensor([0.0])
        return action, log_prob, value, raw


class MockWorldModel:
    """Predicts high risk for throttle > 0.5, low risk otherwise."""
    def __call__(self, state, action):
        throttle = action[..., 1]
        risk = torch.where(throttle > 0.5,
                           torch.tensor(0.9), torch.tensor(0.1)).reshape(-1, 1)
        progress = torch.zeros(action.shape[0], 1)
        return state, risk, progress


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #
def test_dreaming_basic():
    cfg = Config()
    policy = ActorCritic(cfg.state_dim, cfg.action_dim, cfg.hidden)
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    state = torch.randn(cfg.state_dim)
    action, raw_action, log_prob, value, scores = select_action_with_dreaming(
        policy, wm, state, k=5)
    assert action.shape == (cfg.action_dim,), action.shape
    assert raw_action.shape == (cfg.action_dim,)
    assert len(scores) == 5
    ok("dreaming_basic", "action shape correct, 5 scores")


def test_dreaming_safety():
    state = torch.randn(28)
    action, raw_action, log_prob, value, scores = select_action_with_dreaming(
        MockPolicy(), MockWorldModel(), state,
        k=5, w_progress=1.0, w_risk=2.0, w_value=0.5)
    # Risk dominates the score, so the chosen action must be a low-throttle one.
    assert float(action[1]) < 0.5, float(action[1])
    ok("dreaming_safety", "selected low-risk action")


def test_dreaming_gate():
    cfg = Config()
    policy = ActorCritic(cfg.state_dim, cfg.action_dim, cfg.hidden)
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    trainer = WorldModelTrainer(wm, cfg)
    state = torch.randn(cfg.state_dim)

    trainer.train_steps = 0                  # below warmup -> not ready
    assert trainer.is_ready() is False
    *_, dreaming_used = _select_action(policy, wm, trainer, state, cfg, "cpu")
    assert dreaming_used is False

    trainer.train_steps = cfg.wm_warmup_steps + 1   # above warmup -> ready
    assert trainer.is_ready() is True
    *_, dreaming_used = _select_action(policy, wm, trainer, state, cfg, "cpu")
    assert dreaming_used is True
    ok("dreaming_gate", "warmup gate works correctly")


def test_evaluator():
    cfg = Config()
    cfg.max_episode_steps = 20
    env = CarlaEnv(mock=True, config=cfg)
    policy = ActorCritic(cfg.state_dim, cfg.action_dim, cfg.hidden)
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    evaluator = Evaluator(env, policy, wm, cfg)
    stats = evaluator.evaluate(n_episodes=2)
    for key in ("eval_return", "eval_vru_collisions", "eval_lane_departures",
                "eval_route_completion", "eval_near_misses"):
        assert key in stats, key
    assert 0.0 <= stats["eval_route_completion"] <= 1.0, stats["eval_route_completion"]
    env.close()
    ok("evaluator", "2 episodes, metrics correct")


def test_full_dreaming_loop():
    cfg = Config()
    cfg.rollout_size = 256
    cfg.update_epochs = 2
    cfg.batch_size = 64
    cfg.max_episode_steps = 50
    cfg.wm_warmup_steps = 0                   # force dreaming ON from episode 0
    history = train(cfg, mock=True, num_episodes=3, verbose=False,
                    eval_interval=0, ckpt_dir=None)
    assert all(h["dreaming_steps"] > 0 for h in history), history
    ok("full_dreaming_loop", "dreaming active in all episodes")


def main():
    print("Running dreaming tests...\n")
    test_dreaming_basic()
    test_dreaming_safety()
    test_dreaming_gate()
    test_evaluator()
    test_full_dreaming_loop()
    print("\n✅ ALL DREAMING TESTS PASSED")


if __name__ == "__main__":
    main()
