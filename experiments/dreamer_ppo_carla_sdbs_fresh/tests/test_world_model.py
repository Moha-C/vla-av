"""Standalone tests for the world-model trainer and replay buffer.

Run with:  python tests/test_world_model.py
"""
import os
import sys

# Make the project root importable when run as `python tests/test_world_model.py`.
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
from models.world_model import WorldModel
from training.wm_buffer import WorldModelBuffer
from training.world_model_trainer import WorldModelTrainer


def ok(name, result=""):
    print(f"✅ {name} : {result}")


def fake_batch(n=256, state_dim=28, action_dim=4):
    return {
        "states": torch.randn(n, state_dim),
        "actions": torch.randn(n, action_dim),
        "next_states": torch.randn(n, state_dim),
        "risk_targets": torch.rand(n),
        "progress_targets": torch.randn(n),
    }


def test_wm_buffer():
    buf = WorldModelBuffer(capacity=50_000, state_dim=28, action_dim=4)
    for _ in range(500):
        buf.add(np.random.randn(28), np.random.randn(4), np.random.randn(28),
                np.random.rand(), np.random.randn())
    assert len(buf) == 500, len(buf)

    for _ in range(60_000):
        buf.add(np.random.randn(28), np.random.randn(4), np.random.randn(28),
                np.random.rand(), np.random.randn())
    assert len(buf) == 50_000, len(buf)            # circular, capped at capacity

    batch = buf.sample(64)
    assert batch["states"].shape == (64, 28)
    assert batch["actions"].shape == (64, 4)
    assert batch["next_states"].shape == (64, 28)
    assert batch["risk_targets"].shape == (64,)
    assert batch["progress_targets"].shape == (64,)
    ok("wm_buffer", "capacity and sampling correct")


def test_wm_update():
    model = WorldModel(28, 4, 256)
    trainer = WorldModelTrainer(model, Config())
    batch = fake_batch(256)

    before = [p.detach().clone() for p in model.parameters()]
    stats = trainer.update(batch)
    changed = any(not torch.equal(b, p)
                  for b, p in zip(before, model.parameters()))

    assert np.isscalar(stats["loss_wm"]) and stats["loss_wm"] > 0
    assert changed, "world-model parameters did not change after update"
    ok("wm_trainer.update", f"loss={stats['loss_wm']:.4f}")


def test_wm_evaluate():
    model = WorldModel(28, 4, 256)
    trainer = WorldModelTrainer(model, Config())
    errs = trainer.evaluate(fake_batch(256))
    for key in ("state_pred_error", "risk_pred_error", "progress_pred_error"):
        assert key in errs, key
        assert isinstance(errs[key], float) and errs[key] >= 0.0
    ok("wm_trainer.evaluate", f"state_err={errs['state_pred_error']:.4f}")


def test_wm_learning():
    # Learnable pattern: next_state = state, except the first 4 dims add action.
    torch.manual_seed(0)
    model = WorldModel(28, 4, 256)
    trainer = WorldModelTrainer(model, Config())
    buf = WorldModelBuffer(capacity=10_000, state_dim=28, action_dim=4)
    rng = np.random.default_rng(0)
    for _ in range(5_000):
        state = (0.5 * rng.standard_normal(28)).astype(np.float32)
        action = (0.25 * rng.standard_normal(4)).astype(np.float32)
        next_state = state.copy()
        next_state[:4] += action
        buf.add(state, action, next_state, 0.0, 0.0)

    err_before = trainer.evaluate(buf.sample(512), n_samples=512)["state_pred_error"]
    for _ in range(200):
        trainer.update(buf.sample(256))

    err_after = trainer.evaluate(buf.sample(512), n_samples=512)["state_pred_error"]
    assert err_after < 0.1, err_after
    assert err_after < err_before, (err_before, err_after)
    ok("wm_learning",
       f"error dropped to {err_after:.4f} after 200 steps")


def test_wm_ready_flag():
    buf = WorldModelBuffer(capacity=2_000, state_dim=28, action_dim=4)
    for _ in range(999):
        buf.add(np.zeros(28), np.zeros(4), np.zeros(28), 0.0, 0.0)
    assert buf.is_ready(1000) is False
    buf.add(np.zeros(28), np.zeros(4), np.zeros(28), 0.0, 0.0)
    assert buf.is_ready(1000) is True
    ok("wm_ready_flag", "correct")


def main():
    print("Running world-model tests...\n")
    test_wm_buffer()
    test_wm_update()
    test_wm_evaluate()
    test_wm_learning()
    test_wm_ready_flag()
    print("\n✅ ALL WORLD MODEL TESTS PASSED")


if __name__ == "__main__":
    main()
