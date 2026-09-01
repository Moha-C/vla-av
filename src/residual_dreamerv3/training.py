"""World-model and imagined residual actor/critic training."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from .baselines import (
    Normalization,
    PersistenceBaseline,
    RidgeDynamicsBaseline,
    action_sensitivity,
    build_gate_report,
    evaluate_baseline,
    evaluate_world_model,
    write_json,
)
from .config import ResidualDreamerConfig
from .data import Episode, SequenceDataset, Splits
from .model import RSSMState, ResidualDreamerV3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _limited(dataset: Dataset, maximum: int, seed: int) -> Dataset:
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices)


def loader_for(
    episodes: Sequence[Episode],
    config: ResidualDreamerConfig,
    shuffle: bool,
    maximum_windows: Optional[int] = None,
) -> DataLoader:
    base_dataset = SequenceDataset(episodes, config.data.sequence_length)
    dataset: Dataset = base_dataset
    maximum = config.training.maximum_windows if maximum_windows is None else int(maximum_windows)
    dataset = _limited(dataset, maximum, config.data.split_seed + int(shuffle))
    sampler = None
    if shuffle and len(dataset):
        weights = base_dataset.sample_weights(
            config.data.event_window_weight,
            config.data.danger_window_weight,
            config.data.danger_risk_threshold,
        )
        if isinstance(dataset, Subset):
            weights = weights[torch.as_tensor(dataset.indices, dtype=torch.long)]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(config.data.split_seed),
        )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        drop_last=False,
        num_workers=0,
    )


def batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in batch.items()}


@torch.no_grad()
def world_model_losses(
    model: ResidualDreamerV3,
    loader: DataLoader,
    config: ResidualDreamerConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.world_model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = batch_to_device(raw, device)
        targets = {key: batch[key] for key in ("rewards", "continuation", "risk", "collision", "offroad")}
        _, losses = model.world_model.loss(
            batch["observations"], batch["actions"], targets, config.loss
        )
        size = int(batch["observations"].shape[0])
        count += size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * size
    return {key: value / max(1, count) for key, value in totals.items()}


def train_world_model(
    splits: Splits,
    config: ResidualDreamerConfig,
    output: Path,
    device: torch.device,
    epochs: Optional[int] = None,
    maximum_windows: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.data.split_seed)
    model = ResidualDreamerV3(config).to(device)
    optimizer = torch.optim.AdamW(
        model.world_model.parameters(),
        lr=config.training.world_model_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    training = loader_for(splits.train, config, True, maximum_windows)
    validation = loader_for(splits.validation, config, False, maximum_windows)
    if len(training.dataset) == 0 or len(validation.dataset) == 0:
        raise RuntimeError("sequence dataset is empty; collect longer episodes")
    best = float("inf")
    checkpoint = output / "world_model_candidate.pt"
    history: List[Dict[str, Any]] = []
    total_epochs = int(epochs or config.training.world_model_epochs)
    for epoch in range(1, total_epochs + 1):
        model.world_model.train()
        running = 0.0
        count = 0
        for raw in training:
            batch = batch_to_device(raw, device)
            targets = {key: batch[key] for key in ("rewards", "continuation", "risk", "collision", "offroad")}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.world_model.loss(
                batch["observations"], batch["actions"], targets, config.loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite world-model loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.world_model.parameters(), config.training.gradient_clip)
            optimizer.step()
            size = int(batch["observations"].shape[0])
            running += float(loss.detach().cpu()) * size
            count += size
        validation_metrics = world_model_losses(model, validation, config, device)
        row = {
            "epoch": epoch,
            "train_total": running / max(1, count),
            "validation": validation_metrics,
        }
        history.append(row)
        print(
            "[residual-dreamerv3/world] epoch=%d train=%.6f validation=%.6f"
            % (epoch, row["train_total"], validation_metrics["total"]),
            flush=True,
        )
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            torch.save(
                {
                    "schema_version": config.schema_version,
                    "kind": "residual_dreamerv3_world_model",
                    "config": config.to_dict(),
                    "world_model_state": model.world_model.state_dict(),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "seed_sets": splits.seed_sets(),
                    "metadata": {"status": "world_model_candidate"},
                },
                str(checkpoint),
            )
    report = {"best_validation_total": best, "history": history}
    write_json(output / "world_model_history.json", report)
    return checkpoint, report


def load_world_model(
    checkpoint: Path,
    config: ResidualDreamerConfig,
    device: torch.device,
) -> ResidualDreamerV3:
    payload = torch.load(str(checkpoint), map_location=device)
    model = ResidualDreamerV3(config).to(device)
    model.world_model.load_state_dict(payload["world_model_state"])
    model.world_model.eval()
    return model


def validate_world_model(
    model: ResidualDreamerV3,
    splits: Splits,
    config: ResidualDreamerConfig,
    output: Path,
    device: torch.device,
    split_name: str = "validation",
) -> Dict[str, Any]:
    train = splits.train
    evaluation = getattr(splits, split_name)
    normalization = Normalization.fit(train)
    persistence = PersistenceBaseline(train)
    ridge = RidgeDynamicsBaseline().fit(train)
    ridge.save(output / "action_conditioned_ridge.npz")
    baselines = [
        evaluate_baseline(persistence, evaluation, normalization, config.gate.horizons),
        evaluate_baseline(ridge, evaluation, normalization, config.gate.horizons),
    ]
    model_metrics = evaluate_world_model(
        model.world_model, evaluation, normalization, config.gate.horizons, device
    )
    sensitivity = action_sensitivity(
        model.world_model, evaluation, normalization, device
    )
    report = build_gate_report(
        model_metrics, baselines, sensitivity, config.gate, split_name
    )
    report["normalization"] = normalization.to_dict()
    write_json(output / ("world_model_gate_%s.json" % split_name), report)
    return report


def _last_posterior(
    model: ResidualDreamerV3,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> RSSMState:
    with torch.no_grad():
        posteriors, _ = model.world_model.observe_sequence(
            observations, actions, deterministic=False
        )
    if not posteriors:
        return model.world_model.observe_initial(observations[:, 0]).detach()
    return posteriors[-1].detach()


def lambda_returns(
    rewards: torch.Tensor,
    continuation: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    discount: float,
    lambda_: float,
) -> torch.Tensor:
    next_values = torch.cat((values[:, 1:], bootstrap[:, None]), dim=1)
    accumulator = bootstrap
    result: List[torch.Tensor] = []
    for index in range(rewards.shape[1] - 1, -1, -1):
        mixed = (1.0 - lambda_) * next_values[:, index] + lambda_ * accumulator
        accumulator = rewards[:, index] + discount * continuation[:, index] * mixed
        result.append(accumulator)
    return torch.stack(list(reversed(result)), dim=1)


@dataclass
class ImaginationResult:
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    objective: torch.Tensor
    authority: torch.Tensor
    residual_norm: torch.Tensor
    continuation: torch.Tensor


def imagine_actor_critic(
    model: ResidualDreamerV3,
    start: RSSMState,
    initial_observation: torch.Tensor,
    deterministic: bool = False,
) -> ImaginationResult:
    config = model.config
    state = start
    observation = initial_observation
    previous_action = observation[:, 2:5]
    rewards: List[torch.Tensor] = []
    continuations: List[torch.Tensor] = []
    features: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    authorities: List[torch.Tensor] = []
    residual_norms: List[torch.Tensor] = []
    for _ in range(config.actor.imagination_horizon):
        feature = model.world_model.feature(state)
        actor = model.actor(feature, observation, deterministic=deterministic)
        state = model.world_model.imagine_step(state, actor.final_action, deterministic=False)
        prediction = model.world_model.prediction(state, observation)
        next_feature = model.world_model.feature(state)
        normalized_residual = torch.stack(
            (
                actor.residual[:, 0] / config.actor.maximum_steer_residual,
                actor.residual[:, 1] / config.actor.maximum_longitudinal_residual,
            ),
            dim=-1,
        )
        residual_norm = torch.linalg.vector_norm(normalized_residual, dim=-1)
        action_change = torch.linalg.vector_norm(actor.final_action - previous_action, dim=-1)
        learned_reward = prediction.reward.clamp(
            config.model.reward_low, config.model.reward_high
        )
        reward = (
            learned_reward
            - config.reward.intervention_penalty * actor.authority * residual_norm
            - config.reward.residual_change_penalty * action_change
        )
        rewards.append(reward)
        continuations.append(prediction.continuation.clamp(0.0, 1.0))
        features.append(next_feature)
        entropies.append(actor.entropy)
        authorities.append(actor.authority)
        residual_norms.append(residual_norm)
        previous_action = actor.final_action
        observation = prediction.observation
    reward_tensor = torch.stack(rewards, dim=1)
    continuation_tensor = torch.stack(continuations, dim=1)
    feature_tensor = torch.stack(features, dim=1)
    with torch.no_grad():
        slow_values = model.slow_critic(feature_tensor.reshape(-1, feature_tensor.shape[-1])).reshape(
            feature_tensor.shape[:2]
        )
        bootstrap = model.slow_critic(model.world_model.feature(state))
    returns = lambda_returns(
        reward_tensor,
        continuation_tensor,
        slow_values,
        bootstrap,
        config.actor.discount,
        config.actor.lambda_return,
    )
    weights = torch.ones_like(continuation_tensor)
    if weights.shape[1] > 1:
        weights[:, 1:] = torch.cumprod(
            config.actor.discount * continuation_tensor[:, :-1].detach(), dim=1
        )
    detached_returns = returns.detach()
    low = torch.quantile(detached_returns, 0.05)
    high = torch.quantile(detached_returns, 0.95)
    scale = (high - low).clamp_min(1.0)
    centered = (returns - detached_returns.mean()) / scale
    entropy = torch.stack(entropies, dim=1)
    actor_loss = -(
        (weights * centered).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0e-6)
    ).mean() - config.actor.entropy_scale * entropy.mean()
    critic_features = feature_tensor.detach()
    critic_logits = model.critic.logits(critic_features.reshape(-1, critic_features.shape[-1]))
    critic_loss = model.critic.loss(
        critic_features.reshape(-1, critic_features.shape[-1]), detached_returns.reshape(-1)
    ).mean()
    with torch.no_grad():
        slow_target = model.slow_critic(
            critic_features.reshape(-1, critic_features.shape[-1])
        )
    critic_values = model.critic(
        critic_features.reshape(-1, critic_features.shape[-1])
    )
    critic_loss = critic_loss + config.actor.slow_critic_regularization * torch.mean(
        (critic_values - slow_target) ** 2
    )
    return ImaginationResult(
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        objective=(weights * returns).sum() / weights.sum().clamp_min(1.0e-6),
        authority=torch.stack(authorities, dim=1).mean(),
        residual_norm=torch.stack(residual_norms, dim=1).mean(),
        continuation=continuation_tensor.mean(),
    )


@torch.no_grad()
def policy_metrics(
    model: ResidualDreamerV3,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {"objective": 0.0, "authority": 0.0, "residual_norm": 0.0, "continuation": 0.0}
    count = 0
    for raw in loader:
        batch = batch_to_device(raw, device)
        start = _last_posterior(model, batch["observations"], batch["actions"])
        result = imagine_actor_critic(
            model, start, batch["observations"][:, -1], deterministic=True
        )
        size = int(batch["observations"].shape[0])
        count += size
        for key in totals:
            totals[key] += float(getattr(result, key).detach().cpu()) * size
    return {key: value / max(1, count) for key, value in totals.items()}


def train_actor_critic(
    model: ResidualDreamerV3,
    splits: Splits,
    config: ResidualDreamerConfig,
    output: Path,
    device: torch.device,
    world_gate: Mapping[str, Any],
    epochs: Optional[int] = None,
    maximum_windows: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    if not bool(world_gate.get("passed", False)):
        raise RuntimeError("world-model validation gate failed; actor training is forbidden")
    for parameter in model.world_model.parameters():
        parameter.requires_grad_(False)
    training = loader_for(splits.train, config, True, maximum_windows)
    validation = loader_for(splits.validation, config, False, maximum_windows)
    actor_optimizer = torch.optim.AdamW(
        model.actor.parameters(),
        lr=config.training.actor_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        model.critic.parameters(),
        lr=config.training.critic_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: List[Dict[str, Any]] = []
    best = -float("inf")
    checkpoint = output / "actor_candidate.pt"
    total_epochs = int(epochs or config.training.actor_epochs)
    for epoch in range(1, total_epochs + 1):
        model.actor.train()
        model.critic.train()
        actor_total = 0.0
        critic_total = 0.0
        count = 0
        for raw in training:
            batch = batch_to_device(raw, device)
            start = _last_posterior(model, batch["observations"], batch["actions"])
            result = imagine_actor_critic(model, start, batch["observations"][:, -1])
            actor_optimizer.zero_grad(set_to_none=True)
            result.actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), config.training.gradient_clip)
            actor_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            result.critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), config.training.gradient_clip)
            critic_optimizer.step()
            model.update_slow_critic()
            size = int(batch["observations"].shape[0])
            count += size
            actor_total += float(result.actor_loss.detach().cpu()) * size
            critic_total += float(result.critic_loss.detach().cpu()) * size
        metrics = policy_metrics(model, validation, device)
        row = {
            "epoch": epoch,
            "actor_loss": actor_total / max(1, count),
            "critic_loss": critic_total / max(1, count),
            "validation": metrics,
        }
        history.append(row)
        print(
            "[residual-dreamerv3/actor] epoch=%d actor=%.6f critic=%.6f objective=%.6f authority=%.4f"
            % (epoch, row["actor_loss"], row["critic_loss"], metrics["objective"], metrics["authority"]),
            flush=True,
        )
        finite = all(np.isfinite(value) for value in metrics.values())
        if finite and metrics["objective"] > best:
            best = metrics["objective"]
            torch.save(
                {
                    "schema_version": config.schema_version,
                    "kind": "residual_dreamerv3_actor_critic",
                    "config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "world_model_gate": dict(world_gate),
                    "epoch": epoch,
                    "validation": metrics,
                    "seed_sets": splits.seed_sets(),
                    "metadata": {
                        "status": "candidate",
                        "control_allowed": False,
                        "guards_active": False,
                        "policy_role": "continuous_residual_complement_to_simlingo",
                    },
                },
                str(checkpoint),
            )
    report = {"best_validation_objective": best, "history": history}
    write_json(output / "actor_history.json", report)
    return checkpoint, report


def closed_loop_promotion_checks(report: Mapping[str, Any]) -> Dict[str, bool]:
    baseline = report.get("baseline", {})
    candidate = report.get("candidate", {})
    seeds = report.get("seeds", [])
    return {
        "schema": report.get("schema_version") == "residual_dreamerv3_closed_loop_eval_v1",
        "paired_evaluation": report.get("paired_evaluation") is True,
        "enough_seeds": isinstance(seeds, (list, tuple)) and len(set(str(item) for item in seeds)) >= 6,
        "driving_score_not_worse": float(candidate.get("driving_score", -1.0)) >= float(baseline.get("driving_score", 0.0)),
        "route_completion_not_worse": float(candidate.get("route_completion", -1.0)) >= float(baseline.get("route_completion", 0.0)),
        "collisions_not_worse": float(candidate.get("collisions_per_km", float("inf"))) <= float(baseline.get("collisions_per_km", 0.0)),
        "offroad_not_worse": float(candidate.get("offroad_rate", float("inf"))) <= float(baseline.get("offroad_rate", 0.0)),
    }


def promote_checkpoint(candidate: Path, evaluation_report: Path, output: Path) -> Path:
    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    checks = closed_loop_promotion_checks(report)
    if not all(checks.values()):
        raise RuntimeError("closed-loop promotion gate failed: %s" % checks)
    payload = torch.load(str(candidate), map_location="cpu")
    payload["metadata"] = dict(payload.get("metadata", {}))
    payload["metadata"].update(
        {
            "status": "promoted",
            "control_allowed": True,
            "promotion_checks": checks,
            "closed_loop_report": str(evaluation_report.resolve()),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output))
    return output
