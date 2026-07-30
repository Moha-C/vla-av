"""PPO clipped-surrogate update and world-model update."""
import torch
import torch.nn as nn


def update_ppo(policy, optimizer, batch, clip_eps=0.2, ent_coef=0.01,
               vf_coef=0.5, max_grad_norm=0.5):
    """One PPO update over a (mini)batch. Returns a dict of scalar losses."""
    states = batch["states"]
    actions = batch["actions"]          # raw (pre-squash) actions
    old_log_probs = batch["log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    new_log_probs, entropy, values = policy.evaluate(states, actions)

    # Clipped surrogate objective.
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss.
    value_loss = nn.functional.mse_loss(values, returns)

    # Entropy bonus (negative because we add it to a minimized loss).
    entropy_loss = -entropy.mean()

    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.mean().item()),
    }


def update_world_model(world_model, optimizer, batch, max_grad_norm=0.5):
    """One world-model regression update. Returns a dict of scalar losses."""
    states = batch["states"]
    actions = batch["actions"]
    next_states = batch["next_states"]
    risk_targets = batch["risk_targets"]
    progress_targets = batch["progress_targets"]

    next_state_hat, risk_hat, progress_hat = world_model(states, actions)

    state_loss = nn.functional.mse_loss(next_state_hat, next_states)
    risk_loss = nn.functional.mse_loss(risk_hat, risk_targets)
    progress_loss = nn.functional.mse_loss(progress_hat, progress_targets)
    loss = state_loss + risk_loss + progress_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(world_model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "state_loss": float(state_loss.item()),
        "risk_loss": float(risk_loss.item()),
        "progress_loss": float(progress_loss.item()),
    }
