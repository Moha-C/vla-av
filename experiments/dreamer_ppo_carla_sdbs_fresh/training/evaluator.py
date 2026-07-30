"""Greedy evaluation of the policy (no training, no exploration).

Runs a handful of deterministic episodes to measure true performance —
called periodically from the training loop. The deterministic action is the
*mean* of the policy's Gaussian, squashed to the bounded action space (no
sampling), so the metric is reproducible.
"""
import numpy as np
import torch

from rewards.vru_reward import EGO_SPEED, ROUTE_PROGRESS, resolve_layout


class Evaluator:
    def __init__(self, env, policy, world_model, config, device="cpu"):
        self.env = env
        self.policy = policy
        self.world_model = world_model
        self.config = config
        self.layout = resolve_layout(config)
        self.device = torch.device(device)

    def _near_misses(self, state):
        """Count VRUs whose time-to-collision is under tau_ttc this step."""
        ego_speed = float(state[EGO_SPEED])
        count = 0
        for idx in self.layout.vru_indices:
            ttc = float(state[idx]) / (ego_speed + 1e-6)
            if ttc < self.config.tau_ttc:
                count += 1
        return count

    @torch.no_grad()
    def evaluate(self, n_episodes=5):
        returns, collisions, lane_departures = [], [], []
        completions, near_misses = [], []

        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            steps = 0
            ep_return = 0.0
            ep_collisions = 0
            ep_lane = 0
            ep_near = 0
            completion = float(obs[ROUTE_PROGRESS])

            while not done and steps < self.config.max_episode_steps:
                state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                mean, _std, _value = self.policy.forward(state)
                action = self.policy._squash(mean)        # deterministic, bounded
                next_obs, reward, done, info = self.env.step(action.cpu().numpy())

                ep_return += reward
                ep_collisions += int(info.get("vru_collisions", 0))
                ep_lane += int(info.get("lane_departures", 0))
                ep_near += self._near_misses(next_obs)
                completion = float(next_obs[ROUTE_PROGRESS])
                obs = next_obs
                steps += 1

            returns.append(ep_return)
            collisions.append(ep_collisions)
            lane_departures.append(ep_lane)
            completions.append(completion)
            near_misses.append(ep_near)

        return {
            "eval_return": float(np.mean(returns)),
            "eval_vru_collisions": float(np.mean(collisions)),
            "eval_lane_departures": float(np.mean(lane_departures)),
            "eval_route_completion": float(np.mean(completions)),
            "eval_near_misses": float(np.mean(near_misses)),
        }
