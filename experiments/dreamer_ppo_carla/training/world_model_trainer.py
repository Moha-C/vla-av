"""Dedicated trainer for the world model.

Trains `WorldModel` on transitions collected during PPO rollouts, with its
own AdamW optimizer, loss tracking, and an evaluation that reports mean
absolute prediction errors so we can monitor whether the model is actually
learning to predict next states / risk / progress.
"""
import numpy as np
import torch
import torch.nn as nn


class WorldModelTrainer:
    def __init__(self, world_model, config):
        self.model = world_model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            world_model.parameters(), lr=config.lr_wm, weight_decay=1e-4
        )
        self.loss_history = []
        self.train_steps = 0   # number of transitions trained on

    # ------------------------------------------------------------------ #
    @staticmethod
    def _unpack(batch):
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        risk_targets = batch["risk_targets"].reshape(-1)
        progress_targets = batch["progress_targets"].reshape(-1)
        return states, actions, next_states, risk_targets, progress_targets

    def update(self, batch):
        """One gradient step on a (mini)batch. Returns a dict of scalar losses."""
        states, actions, next_states, risk_targets, progress_targets = \
            self._unpack(batch)

        next_state_hat, risk_hat, progress_hat = self.model(states, actions)

        loss_state = nn.functional.mse_loss(next_state_hat, next_states)
        loss_risk = nn.functional.mse_loss(risk_hat.squeeze(-1), risk_targets)
        loss_progress = nn.functional.mse_loss(
            progress_hat.squeeze(-1), progress_targets
        )
        loss_total = loss_state + loss_risk + loss_progress

        self.optimizer.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += states.shape[0]
        stats = {
            "loss_wm": float(loss_total.item()),
            "loss_state": float(loss_state.item()),
            "loss_risk": float(loss_risk.item()),
            "loss_progress": float(loss_progress.item()),
        }
        self.loss_history.append(stats["loss_wm"])
        return stats

    @torch.no_grad()
    def evaluate(self, batch, n_samples=100):
        """Mean absolute prediction error on up to `n_samples` rows."""
        states, actions, next_states, risk_targets, progress_targets = \
            self._unpack(batch)

        n = states.shape[0]
        if n > n_samples:
            idx = torch.randperm(n)[:n_samples]
            states, actions, next_states = states[idx], actions[idx], next_states[idx]
            risk_targets, progress_targets = risk_targets[idx], progress_targets[idx]

        next_state_hat, risk_hat, progress_hat = self.model(states, actions)
        return {
            "state_pred_error": float(
                (next_state_hat - next_states).abs().mean().item()
            ),
            "risk_pred_error": float(
                (risk_hat.squeeze(-1) - risk_targets).abs().mean().item()
            ),
            "progress_pred_error": float(
                (progress_hat.squeeze(-1) - progress_targets).abs().mean().item()
            ),
        }

    def is_ready(self):
        """True once the model has trained on >= wm_warmup_steps transitions."""
        return self.train_steps >= self.config.wm_warmup_steps
