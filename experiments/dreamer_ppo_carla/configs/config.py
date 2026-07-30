"""All hyperparameters for the Dreamer-PPO CARLA project."""
from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    state_dim: int = 28
    action_dim: int = 4
    host: str = "localhost"
    port: int = 2000
    town: str = "Town01"
    fps: int = 10
    max_episode_steps: int = 1000

    # World model
    wm_hidden: int = 256
    lr_wm: float = 3e-4
    wm_batch_size: int = 256
    wm_warmup_steps: int = 1000   # transitions trained before dreaming starts

    # PPO
    hidden: int = 256
    lr_policy: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_size: int = 2048
    update_epochs: int = 10
    batch_size: int = 64
    num_episodes: int = 1000

    # Dreaming
    dream_k: int = 5
    w_progress: float = 1.0
    w_risk: float = 2.0
    w_value: float = 0.5

    # Reward weights
    w_prog: float = 1.0
    w_vru: float = 2.0
    w_safe: float = 1.0
    w_comfort: float = 0.1
    w_rules: float = 0.5
    sigma_d: float = 5.0
    lambda_ttc: float = 0.5
    tau_ttc: float = 2.0
    lambda_cross: float = 1.0
    eta1: float = 0.5  # lane departure weight in safety term
    eta2: float = 0.3  # general risk weight in safety term
