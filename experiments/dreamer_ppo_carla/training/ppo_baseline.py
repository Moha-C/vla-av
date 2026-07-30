"""Step 2 baseline: plain PPO with NO dreaming.

Same training structure as ``dreamer_ppo.py`` but the action is taken
directly from the policy (``policy.act``) — no candidate sampling, no
imagined rollouts. The policy is *never* conditioned on the world model.

The world model is still trained in the background from the collected
transitions (Step 4), so that by the time dreaming is enabled it is already
well-trained — but it does not influence the baseline policy in any way.
"""
import argparse

import numpy as np
import torch

from configs.config import Config
from env.carla_env import CarlaEnv
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from training.rollout_buffer import RolloutBuffer
from training.ppo import update_ppo
from training.logger import Logger
from training.wm_buffer import WorldModelBuffer
from training.world_model_trainer import WorldModelTrainer
from rewards.vru_reward import ROUTE_PROGRESS


# ---------------------------------------------------------------------- #
# Plain action selection (no dreaming)
# ---------------------------------------------------------------------- #
def select_action(policy, state):
    """Sample one action straight from the policy for rollout collection.

    Returns (action, raw_action, log_prob, value) where `action` is the
    bounded action sent to the environment and `raw_action` is the
    pre-squash sample stored for the PPO update.
    """
    if not torch.is_tensor(state):
        state = torch.as_tensor(np.asarray(state, dtype=np.float32))
    if state.dim() == 1:
        state = state.unsqueeze(0)
    with torch.no_grad():
        action, log_prob, value, raw_action = policy.act(state)
    return (
        action.squeeze(0).numpy(),
        raw_action.squeeze(0).numpy(),
        float(log_prob.item()),
        float(value.item()),
    )


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def train_baseline(config=None, mock=False, num_episodes=None, verbose=True,
                   log_dir=None):
    config = config or Config()
    if num_episodes is not None:
        config.num_episodes = num_episodes

    env = CarlaEnv(mock=mock, config=config)
    policy = ActorCritic(config.state_dim, config.action_dim, config.hidden)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=config.lr_policy)

    buffer = RolloutBuffer(
        config.rollout_size, config.state_dim, config.action_dim,
        gamma=config.gamma, lam=config.lam,
    )
    # World model trained in the background (does not affect the policy).
    world_model = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)
    wm_buffer = WorldModelBuffer(capacity=50_000, state_dim=config.state_dim,
                                 action_dim=config.action_dim)
    wm_trainer = WorldModelTrainer(world_model, config)

    logger = Logger(log_dir) if log_dir else None
    history = []
    try:
        for episode in range(config.num_episodes):
            buffer.clear()
            obs = env.reset()
            ep_return = 0.0
            ep_collisions = 0
            ep_lane_departures = 0
            ep_components = {k: 0.0 for k in
                             ("progress", "vru_risk", "collision", "comfort", "rules")}
            route_completion = float(obs[ROUTE_PROGRESS])
            done = False

            # Fill the rollout buffer (may span several env episodes).
            while not buffer.is_full():
                action, raw_action, log_prob, value = select_action(policy, obs)
                next_obs, reward, done, info = env.step(action)

                risk_target = float(info.get("vru_risk", 0.0))
                progress_target = float(info.get("progress", 0.0))

                buffer.store(obs, raw_action, reward, done, value, log_prob,
                             next_obs, risk_target, progress_target)
                wm_buffer.add(obs, action, next_obs, risk_target, progress_target)
                ep_return += reward
                ep_collisions += int(bool(info.get("collision", False)))
                ep_lane_departures += int(bool(info.get("lane_departure", False)))
                comp = info.get("reward_components", {})
                for key in ep_components:
                    ep_components[key] += float(comp.get(key, 0.0))
                route_completion = float(next_obs[ROUTE_PROGRESS])
                obs = next_obs

                if done:
                    buffer.finish_path(last_value=0.0)
                    obs = env.reset()

            # Bootstrap the value of the final (unfinished) state.
            if buffer.path_start < buffer.ptr:
                with torch.no_grad():
                    _, _, last_v = policy.forward(
                        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    )
                buffer.finish_path(last_value=float(last_v.item()))

            # ---- PPO update (policy only) ---- #
            batch = buffer.get()
            n = batch["states"].shape[0]
            stats = {}
            for _ in range(config.update_epochs):
                idx = np.random.permutation(n)
                for start in range(0, n, config.batch_size):
                    mb_idx = idx[start:start + config.batch_size]
                    mb = {key: val[mb_idx] for key, val in batch.items()}
                    stats = update_ppo(
                        policy, opt_pi, mb, clip_eps=config.clip_eps,
                        ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                        max_grad_norm=config.max_grad_norm,
                    )

            # ---- background world-model update (does not affect policy) ---- #
            wm_stats, wm_eval = {}, {}
            if wm_buffer.is_ready(min_size=1000):
                wm_batch = wm_buffer.sample(config.wm_batch_size)
                wm_stats = wm_trainer.update(wm_batch)
                wm_eval = wm_trainer.evaluate(wm_batch)
                if verbose and episode % 10 == 0:
                    print(f"  WM eval: state_err={wm_eval['state_pred_error']:.4f} "
                          f"risk_err={wm_eval['risk_pred_error']:.4f}")

            record = {
                "episode": episode,
                "return": ep_return,
                "r_progress": ep_components["progress"],
                "r_vru": ep_components["vru_risk"],
                "r_collision": ep_components["collision"],
                "r_comfort": ep_components["comfort"],
                "r_rules": ep_components["rules"],
                "ppo_loss": stats.get("loss", 0.0),
                "vf_loss": stats.get("value_loss", 0.0),
                "entropy": stats.get("entropy", 0.0),
                "loss_wm": wm_stats.get("loss_wm", 0.0),
                "wm_state_err": wm_eval.get("state_pred_error", 0.0),
                "wm_risk_err": wm_eval.get("risk_pred_error", 0.0),
                "vru_collisions": ep_collisions,
                "lane_departures": ep_lane_departures,
                "route_completion": route_completion,
            }
            history.append(record)
            if logger is not None:
                logger.log(episode, record)
            if verbose:
                print(
                    f"Episode {episode:04d} | return={ep_return:.2f} | "
                    f"ppo_loss={stats.get('loss', 0.0):.4f} | "
                    f"vru_collisions={ep_collisions}"
                )
    finally:
        env.close()
        if logger is not None:
            logger.close()

    return history


def main():
    parser = argparse.ArgumentParser(description="PPO baseline (Step 2) for CARLA")
    parser.add_argument("--mock", action="store_true",
                        help="run without CARLA installed")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    cfg = Config(num_episodes=args.episodes)
    train_baseline(cfg, mock=args.mock, log_dir="logs")


if __name__ == "__main__":
    main()
