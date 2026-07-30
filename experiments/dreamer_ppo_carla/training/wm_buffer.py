"""Long-lived circular replay buffer for world-model training.

The PPO rollout buffer is short (a couple thousand steps, cleared after
every policy update). The world model benefits from a much larger, stable
pool of past transitions — this buffer keeps up to `capacity` of them and
overwrites the oldest once full.
"""
import numpy as np
import torch


class WorldModelBuffer:
    def __init__(self, capacity=50_000, state_dim=28, action_dim=4):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.risk_targets = np.zeros(capacity, dtype=np.float32)
        self.progress_targets = np.zeros(capacity, dtype=np.float32)

        self._ptr = 0       # next write index
        self._size = 0       # number of valid entries

    def add(self, state, action, next_state, risk_target, progress_target):
        i = self._ptr
        self.states[i] = state
        self.actions[i] = action
        self.next_states[i] = next_state
        self.risk_targets[i] = float(risk_target)
        self.progress_targets[i] = float(progress_target)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random minibatch as a dict of tensors."""
        n = min(batch_size, self._size)
        idx = np.random.randint(0, self._size, size=n)
        return {
            "states": torch.as_tensor(self.states[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "next_states": torch.as_tensor(self.next_states[idx]),
            "risk_targets": torch.as_tensor(self.risk_targets[idx]),
            "progress_targets": torch.as_tensor(self.progress_targets[idx]),
        }

    def is_ready(self, min_size=1000):
        return self._size >= min_size

    def __len__(self):
        return self._size
