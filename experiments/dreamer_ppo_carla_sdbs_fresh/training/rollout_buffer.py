"""GAE rollout buffer for PPO + world-model training."""
import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, size, state_dim, action_dim, gamma=0.99, lam=0.95):
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clear()

    # ------------------------------------------------------------------ #
    def clear(self):
        self.states = np.zeros((self.size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, self.action_dim), dtype=np.float32)
        self.next_states = np.zeros((self.size, self.state_dim), dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.log_probs = np.zeros(self.size, dtype=np.float32)
        self.risk_targets = np.zeros(self.size, dtype=np.float32)
        self.progress_targets = np.zeros(self.size, dtype=np.float32)

        self.advantages = np.zeros(self.size, dtype=np.float32)
        self.returns = np.zeros(self.size, dtype=np.float32)

        self.ptr = 0          # current write index
        self.path_start = 0   # start index of the current episode

    # ------------------------------------------------------------------ #
    def store(self, state, action, reward, done, value, log_prob, next_state,
              risk_target=0.0, progress_target=0.0):
        assert self.ptr < self.size, "RolloutBuffer overflow"
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.next_states[i] = next_state
        self.risk_targets[i] = risk_target
        self.progress_targets[i] = progress_target
        self.ptr += 1

    def finish_path(self, last_value=0.0):
        """Compute GAE advantages and returns for the just-finished segment."""
        sl = slice(self.path_start, self.ptr)
        rewards = self.rewards[sl]
        values = self.values[sl]
        dones = self.dones[sl]
        n = len(rewards)

        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            last_gae = delta + self.gamma * self.lam * non_terminal * last_gae
            adv[t] = last_gae

        self.advantages[sl] = adv
        self.returns[sl] = adv + values
        self.path_start = self.ptr

    # ------------------------------------------------------------------ #
    def get(self):
        """Return all stored transitions as tensors with normalized advantages."""
        assert self.ptr > 0, "Buffer is empty"
        n = self.ptr
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return {
            "states": torch.as_tensor(self.states[:n]),
            "actions": torch.as_tensor(self.actions[:n]),
            "next_states": torch.as_tensor(self.next_states[:n]),
            "log_probs": torch.as_tensor(self.log_probs[:n]),
            "values": torch.as_tensor(self.values[:n]),
            "advantages": torch.as_tensor(adv),
            "returns": torch.as_tensor(self.returns[:n]),
            "risk_targets": torch.as_tensor(self.risk_targets[:n]).unsqueeze(-1),
            "progress_targets": torch.as_tensor(self.progress_targets[:n]).unsqueeze(-1),
        }

    def is_full(self):
        return self.ptr >= self.size

    def __len__(self):
        return self.ptr
