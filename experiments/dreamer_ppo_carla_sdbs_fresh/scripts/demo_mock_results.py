"""Quick, visible demo of the learnable components on mock/synthetic data.

Mock CARLA states are random noise, so RL returns never trend. But the
*learnable* pieces do show real curves, which this script makes visible:

  1. world_model_loss.png  - world-model loss decreasing on structured data

Run:  python scripts/demo_mock_results.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import Config
from models.world_model import WorldModel
from training.world_model_trainer import WorldModelTrainer

OUT = os.path.join(ROOT, "demo_plots")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------- #
def demo_world_model():
    print("[1/1] World-model loss on a structured (learnable) batch...")
    cfg = Config()
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    trainer = WorldModelTrainer(wm, cfg)
    rng = torch.Generator().manual_seed(0)
    states = torch.randn(256, cfg.state_dim, generator=rng)
    actions = torch.randn(256, cfg.action_dim, generator=rng)
    # Deterministic targets so the model has a real function to fit.
    next_states = torch.tanh(states + 0.1)
    batch = {
        "states": states, "actions": actions, "next_states": next_states,
        "risk_targets": torch.sigmoid(states[:, :1].squeeze(-1)),
        "progress_targets": states[:, 12],
    }
    losses = [trainer.update(batch)["loss_wm"] for _ in range(200)]
    print(f"      WM loss: {losses[0]:.3f} -> {losses[-1]:.3f}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Total loss")
    plt.title("World-model loss decreasing on structured data")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "world_model_loss.png"), dpi=120)
    plt.close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    torch.manual_seed(0)
    np.random.seed(0)
    demo_world_model()
    print("Done.")
    print(f"\nPlots written to: {OUT}")
    for f in sorted(os.listdir(OUT)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
