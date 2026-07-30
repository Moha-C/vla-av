"""Step 5: full Dreamer-PPO training with world-model dreaming.

The policy is trained with PPO. Once the world model is ready
(``wm_trainer.is_ready()``), action selection switches from direct policy
sampling to *dreaming*: sample ``k`` candidate actions, imagine each one
step ahead with the world model, and execute the best-scored candidate.
Before the world model is ready, the gate falls back to ``policy.act``.
"""
import argparse
import os
import sys
from collections import defaultdict

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
from training.evaluator import Evaluator
from rewards.vru_reward import ROUTE_PROGRESS


# ---------------------------------------------------------------------- #
# Dreaming action selection
# ---------------------------------------------------------------------- #
@torch.no_grad()
def select_action_with_dreaming(policy, world_model, state, k=5, w_progress=1.0,
                                w_risk=2.0, w_value=0.5, device="cpu"):
    """Sample k candidate actions, score each with the world model, pick best.

    Returns ``(best_action, best_raw_action, best_log_prob, best_value, scores)``:
      - ``best_action``     bounded action tensor, shape (action_dim,)
      - ``best_raw_action`` pre-squash sample (kept so the PPO ratio stays
                            consistent between rollout collection and update)
      - ``best_log_prob``   scalar tensor
      - ``best_value``      scalar tensor
      - ``scores``          list of the k candidate scores (for logging)
    """
    if not torch.is_tensor(state):
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device)

    candidates = []
    for _ in range(k):
        action, log_prob, value, raw_action = policy.act(state.unsqueeze(0))
        action = action.squeeze(0)
        raw_action = raw_action.squeeze(0)
        log_prob = log_prob.squeeze(0)
        value = value.squeeze(0)

        # World model imagines the immediate future of this candidate.
        next_state_hat, risk_hat, progress_hat = world_model(
            state.unsqueeze(0), action.unsqueeze(0)
        )
        risk = risk_hat.item()
        progress = progress_hat.item()
        val = value.item()

        score = w_progress * progress - w_risk * risk + w_value * val
        candidates.append((score, action, raw_action, log_prob, value))

    best = max(candidates, key=lambda c: c[0])
    scores = [c[0] for c in candidates]
    return best[1], best[2], best[3], best[4], scores


def _select_action(policy, world_model, wm_trainer, state, config, device):
    """Gate: dream if the world model is ready, otherwise sample the policy.

    Returns (action_np, raw_action_np, log_prob, value, scores, dreaming_used).
    """
    if wm_trainer.is_ready():
        action, raw_action, log_prob, value, scores = select_action_with_dreaming(
            policy, world_model, state, k=config.dream_k,
            w_progress=config.w_progress, w_risk=config.w_risk,
            w_value=config.w_value, device=device,
        )
        dreaming_used = True
    else:
        action, log_prob, value, raw_action = policy.act(state.unsqueeze(0))
        action = action.squeeze(0)
        raw_action = raw_action.squeeze(0)
        log_prob = log_prob.squeeze(0)
        value = value.squeeze(0)
        scores = None
        dreaming_used = False

    return (action.cpu().numpy(), raw_action.cpu().numpy(),
            float(log_prob.item()), float(value.item()), scores, dreaming_used)


def _save_checkpoint(path, episode, policy, world_model, optimizer_pi, eval_return):
    torch.save({
        "episode": episode,
        "policy": policy.state_dict(),
        "world_model": world_model.state_dict(),
        "optimizer_pi": optimizer_pi.state_dict(),
        "eval_return": eval_return,
    }, path)


def _load_world_model_checkpoint(path, world_model, device):
    """Warm-start the world model from a guarded or RL checkpoint."""
    if not path:
        return
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("world_model")
    if state_dict is None:
        raise KeyError(f"{path} must contain either 'model' or 'world_model'.")
    world_model.load_state_dict(state_dict, strict=True)
    print(f"Loaded initial world model from {path}", flush=True)


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def train(config=None, mock=False, num_episodes=None, verbose=True, log_dir=None,
          device="cpu", eval_interval=50, ckpt_dir="checkpoints",
          init_world_model_path=None, log_name="training_log.csv"):
    config = config or Config()
    if num_episodes is not None:
        config.num_episodes = num_episodes
    device = torch.device(device)

    env = CarlaEnv(mock=mock, config=config)
    policy = ActorCritic(config.state_dim, config.action_dim, config.hidden).to(device)
    world_model = WorldModel(config.state_dim, config.action_dim,
                             config.wm_hidden).to(device)
    _load_world_model_checkpoint(init_world_model_path, world_model, device)
    optimizer_pi = torch.optim.Adam(policy.parameters(), lr=config.lr_policy)

    buffer = RolloutBuffer(
        config.rollout_size, config.state_dim, config.action_dim,
        gamma=config.gamma, lam=config.lam,
    )
    wm_buffer = WorldModelBuffer(capacity=50_000, state_dim=config.state_dim,
                                 action_dim=config.action_dim)
    wm_trainer = WorldModelTrainer(world_model, config)
    evaluator = Evaluator(env, policy, world_model, config, device=device)

    logger = Logger(log_dir, log_name) if log_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    best_eval_return = -float("inf")

    history = []
    try:
        for episode in range(config.num_episodes):
            buffer.clear()
            obs = env.reset()
            ep_return = 0.0
            ep_stats = defaultdict(float)
            ep_collisions = 0
            ep_lane_departures = 0
            route_completion = float(obs[ROUTE_PROGRESS])
            done = False
            info = {}

            # ---- collect a rollout (may span several env episodes) ---- #
            while not buffer.is_full():
                state = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_np, raw_action_np, log_prob, value, _scores, dreaming_used = \
                    _select_action(policy, world_model, wm_trainer, state,
                                   config, device)
                if dreaming_used:
                    ep_stats["dreaming_steps"] += 1

                next_obs, reward, done, info = env.step(action_np)
                risk_target = float(info.get("vru_risk", 0.0))
                progress_target = float(info.get("progress", 0.0))

                buffer.store(obs, raw_action_np, reward, done, value, log_prob,
                             next_obs, risk_target, progress_target)
                wm_buffer.add(obs, action_np, next_obs, risk_target, progress_target)

                for comp_key, comp_val in info.get("reward_components", {}).items():
                    ep_stats[f"r_{comp_key}"] += comp_val

                ep_return += reward
                ep_collisions += int(info.get("vru_collisions", 0))
                ep_lane_departures += int(info.get("lane_departures", 0))
                route_completion = float(next_obs[ROUTE_PROGRESS])
                obs = next_obs

                if done:
                    buffer.finish_path(last_value=0.0)
                    obs = env.reset()

            # Bootstrap the value of the final (unfinished) state.
            if buffer.path_start < buffer.ptr:
                with torch.no_grad():
                    _, _, last_v = policy.forward(
                        torch.as_tensor(obs, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    )
                buffer.finish_path(last_value=float(last_v.item()))

            # ---- PPO update ---- #
            batch = buffer.get()
            n = batch["states"].shape[0]
            ppo_stats = {}
            for _ in range(config.update_epochs):
                idx = np.random.permutation(n)
                for start in range(0, n, config.batch_size):
                    mb_idx = idx[start:start + config.batch_size]
                    mb = {key: val[mb_idx] for key, val in batch.items()}
                    ppo_stats = update_ppo(
                        policy, optimizer_pi, mb, clip_eps=config.clip_eps,
                        ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                        max_grad_norm=config.max_grad_norm,
                    )

            # ---- world-model update (from its own replay) ---- #
            if wm_buffer.is_ready(min_size=1000):
                wm_stats = wm_trainer.update(wm_buffer.sample(config.wm_batch_size))
                wm_eval = wm_trainer.evaluate(wm_buffer.sample(256))
            else:
                wm_stats = {"loss_wm": 0.0, "loss_state": 0.0,
                            "loss_risk": 0.0, "loss_progress": 0.0}
                wm_eval = {"state_pred_error": 0.0, "risk_pred_error": 0.0,
                           "progress_pred_error": 0.0}

            if verbose and episode % 10 == 0 and wm_buffer.is_ready(1000):
                print(f"  WM eval | state_err={wm_eval['state_pred_error']:.4f} "
                      f"risk_err={wm_eval['risk_pred_error']:.4f} "
                      f"progress_err={wm_eval['progress_pred_error']:.4f}")

            dreaming_active = wm_trainer.is_ready()

            # ---- periodic greedy evaluation + best-model checkpoint ---- #
            eval_stats = {}
            if eval_interval and episode % eval_interval == 0:
                eval_stats = evaluator.evaluate(n_episodes=5)
                if verbose:
                    print("\n" + "=" * 50)
                    print(f"EVAL episode {episode}")
                    print(f"  return:          {eval_stats['eval_return']:.2f}")
                    print(f"  vru_collisions:  {eval_stats['eval_vru_collisions']:.2f}")
                    print(f"  near_misses:     {eval_stats['eval_near_misses']:.2f}")
                    print(f"  route_completion:{eval_stats['eval_route_completion']:.2f}")
                    print("=" * 50 + "\n")
                if ckpt_dir and eval_stats["eval_return"] > best_eval_return:
                    best_eval_return = eval_stats["eval_return"]
                    _save_checkpoint(os.path.join(ckpt_dir, "best_model.pt"),
                                     episode, policy, world_model, optimizer_pi,
                                     best_eval_return)
                    if verbose:
                        print(f"  💾 New best model saved "
                              f"(return={best_eval_return:.2f})")

            # ---- periodic checkpoint regardless of performance ---- #
            if ckpt_dir and episode % 100 == 0:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"episode_{episode:04d}.pt"),
                    episode, policy, world_model, optimizer_pi,
                    eval_stats.get("eval_return", best_eval_return),
                )

            if verbose:
                print(f"Episode {episode:04d} | return={ep_return:.2f} | "
                      f"dreaming={'ON' if dreaming_active else 'OFF'} | "
                      f"ppo_loss={ppo_stats.get('loss', 0.0):.4f} | "
                      f"wm_loss={wm_stats['loss_wm']:.4f} | "
                      f"vru_collisions={ep_collisions}")

            record = {
                "episode": episode,
                "return": ep_return,
                "r_progress": ep_stats.get("r_progress", 0.0),
                "r_vru": ep_stats.get("r_vru_risk", 0.0),
                "r_collision": ep_stats.get("r_collision", 0.0),
                "r_comfort": ep_stats.get("r_comfort", 0.0),
                "r_rules": ep_stats.get("r_rules", 0.0),
                "ppo_loss": ppo_stats.get("loss", 0.0),
                "vf_loss": ppo_stats.get("value_loss", 0.0),
                "entropy": ppo_stats.get("entropy", 0.0),
                "loss_wm": wm_stats.get("loss_wm", 0.0),
                "wm_state_err": wm_eval.get("state_pred_error", 0.0),
                "wm_risk_err": wm_eval.get("risk_pred_error", 0.0),
                "dreaming_active": int(dreaming_active),
                "dreaming_steps": ep_stats.get("dreaming_steps", 0.0),
                "vru_collisions": ep_collisions,
                "lane_departures": ep_lane_departures,
                "route_completion": route_completion,
            }
            record.update(eval_stats)
            history.append(record)
            if logger is not None:
                logger.log(episode, record)
    finally:
        env.close()
        if logger is not None:
            logger.close()

    return history


def main():
    # Ensure UTF-8 stdout so the 💾 marker renders on Windows consoles (cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Dreamer-PPO for CARLA")
    parser.add_argument("--mock", action="store_true",
                        help="run without CARLA installed")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--town", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--rollout-size", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--log-name", default="training_log.csv")
    parser.add_argument("--init-world-model", default=None)
    args = parser.parse_args()

    config = Config()
    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port
    if args.town is not None:
        config.town = args.town
    if args.max_episode_steps is not None:
        config.max_episode_steps = args.max_episode_steps
    if args.rollout_size is not None:
        config.rollout_size = args.rollout_size
    train(config, mock=args.mock, num_episodes=args.episodes,
          device=args.device, log_dir=args.log_dir, ckpt_dir=args.ckpt_dir,
          eval_interval=args.eval_interval,
          init_world_model_path=args.init_world_model,
          log_name=args.log_name)


if __name__ == "__main__":
    main()
