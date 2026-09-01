#!/usr/bin/env python3
"""Train an RSSM-conditioned SimLingo complement without touching production.

The V2 checkpoint contains a temporal world model, but its migrated PPO actor
was never trained to use the RSSM feature.  This trainer keeps the world model
frozen, distils validated clean Dreamer-v1 interventions into the complete
actor, and adds a conservative differentiable imagination objective.  A held
out teacher route and a normal-driving trust gate decide whether a candidate
is written.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from external.simlingo.team_code.dreamer_guard import ActorCritic
from external.simlingo.team_code.dreamer_world_models import (
    RSSMConfig,
    RSSMState,
    TemporalRSSMWorldModel,
    symexp,
)
from scripts import train_dreamer_rssm_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "external/simlingo/checkpoints/dreamer_ppo_rssm_v2/candidate_model.pt"
)
DEFAULT_OUTPUT = ROOT / "external/simlingo/checkpoints/dreamer_ppo_rssm_v3"


@dataclass
class ActorSamples:
    inputs: torch.Tensor
    targets: torch.Tensor
    base_actions: torch.Tensor
    world_observations: torch.Tensor
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor
    routes: List[str]

    def index(self, indices: torch.Tensor) -> "ActorSamples":
        cpu_indices = indices.detach().cpu()
        return ActorSamples(
            inputs=self.inputs[cpu_indices],
            targets=self.targets[cpu_indices],
            base_actions=self.base_actions[cpu_indices],
            world_observations=self.world_observations[cpu_indices],
            deter=self.deter[cpu_indices],
            stoch=self.stoch[cpu_indices],
            logits=self.logits[cpu_indices],
            routes=[self.routes[int(index)] for index in cpu_indices.tolist()],
        )

    def to(self, device: torch.device) -> "ActorSamples":
        return ActorSamples(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            base_actions=self.base_actions.to(device),
            world_observations=self.world_observations.to(device),
            deter=self.deter.to(device),
            stoch=self.stoch.to(device),
            logits=self.logits.to(device),
            routes=self.routes,
        )

    def __len__(self) -> int:
        return int(self.inputs.shape[0])


def empty_samples(input_dim: int, config: RSSMConfig) -> ActorSamples:
    return ActorSamples(
        inputs=torch.empty(0, input_dim),
        targets=torch.empty(0, 4),
        base_actions=torch.empty(0, 3),
        world_observations=torch.empty(0, config.observation_dim),
        deter=torch.empty(0, config.deter_dim),
        stoch=torch.empty(0, config.stochastic_size),
        logits=torch.empty(0, config.stoch_dim, config.classes),
        routes=[],
    )


@torch.no_grad()
def extract_samples(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[v2.Episode],
    checkpoint: Dict[str, Any],
    device: torch.device,
    teacher_only: bool,
    anchor_stride: int = 4,
) -> ActorSamples:
    policy_mean = np.asarray(checkpoint["policy_state_mean"], dtype=np.float32)
    policy_std = np.maximum(
        np.asarray(checkpoint["policy_state_std"], dtype=np.float32), 1e-6
    )
    world_mean = np.asarray(checkpoint["world_observation_mean"], dtype=np.float32)
    world_std = np.maximum(
        np.asarray(checkpoint["world_observation_std"], dtype=np.float32), 1e-6
    )
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.maximum(
        np.asarray(checkpoint["action_std"], dtype=np.float32), 1e-6
    )
    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    base_actions: List[torch.Tensor] = []
    world_observations: List[torch.Tensor] = []
    deter: List[torch.Tensor] = []
    stoch: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    routes: List[str] = []
    model.eval()
    for episode in episodes:
        policy_obs = torch.from_numpy(
            ((episode.observations - policy_mean) / policy_std).astype(np.float32)
        ).to(device)
        world_obs = torch.from_numpy(
            ((episode.observations - world_mean) / world_std).astype(np.float32)
        ).to(device)
        actions = torch.from_numpy(
            ((episode.actions - action_mean) / action_std).astype(np.float32)
        ).to(device)
        posterior = model.observe_initial(world_obs[0:1], deterministic=True)
        for step in range(episode.transitions):
            include = (
                episode.teacher_mask[step] > 0.5
                if teacher_only
                else step % max(1, anchor_stride) == 0
            )
            if include:
                inputs.append(torch.cat([
                    policy_obs[step], model.feature(posterior)[0]
                ], dim=-1).cpu())
                targets.append(torch.from_numpy(episode.teacher_targets[step]).float())
                base_actions.append(torch.from_numpy(
                    episode.observations[step, 28:31]
                ).float())
                world_observations.append(world_obs[step].cpu())
                deter.append(posterior.deter[0].cpu())
                stoch.append(posterior.stoch[0].cpu())
                logits.append(posterior.logits[0].cpu())
                routes.append(episode.route_id)
            posterior, _ = model.obs_step(
                posterior,
                actions[step:step + 1],
                world_obs[step + 1:step + 2],
                deterministic=True,
            )
    if not inputs:
        return empty_samples(v2.OBSERVATION_DIM + model.feature_dim, model.config)
    return ActorSamples(
        inputs=torch.stack(inputs),
        targets=torch.stack(targets),
        base_actions=torch.stack(base_actions),
        world_observations=torch.stack(world_observations),
        deter=torch.stack(deter),
        stoch=torch.stack(stoch),
        logits=torch.stack(logits),
        routes=routes,
    )


def decoded_actor_output(actor: ActorCritic, inputs: torch.Tensor) -> torch.Tensor:
    mean, _, _ = actor(inputs)
    steering = torch.tanh(mean[:, 0:1])
    longitudinal = torch.tanh(mean[:, 1:2] - mean[:, 2:3])
    throttle = torch.relu(longitudinal)
    brake = torch.relu(-longitudinal)
    gate = torch.sigmoid(mean[:, 3:4])
    return torch.cat([steering, throttle, brake, gate], dim=-1)


def blend_with_simlingo(
    base_action: torch.Tensor,
    decoded_action: torch.Tensor,
) -> torch.Tensor:
    """Apply the checkpoint's signed-longitudinal learned gate semantics."""
    gate = decoded_action[:, 3:4]
    steering = base_action[:, 0:1] + gate * (
        decoded_action[:, 0:1] - base_action[:, 0:1]
    )
    base_longitudinal = base_action[:, 1:2] - base_action[:, 2:3]
    target_longitudinal = decoded_action[:, 1:2] - decoded_action[:, 2:3]
    longitudinal = base_longitudinal + gate * (
        target_longitudinal - base_longitudinal
    )
    return torch.cat([
        steering.clamp(-1.0, 1.0),
        torch.relu(longitudinal).clamp(0.0, 1.0),
        torch.relu(-longitudinal).clamp(0.0, 1.0),
        gate,
    ], dim=-1)


@torch.no_grad()
def actor_metrics(
    actor: ActorCritic,
    samples: ActorSamples,
    reference: ActorCritic | None = None,
) -> Dict[str, float]:
    if len(samples) == 0:
        return {
            "samples": 0,
            "control_mae_active": math.inf,
            "gate_accuracy": 0.0,
            "anchor_deviation": math.inf,
            "latent_sensitivity": 0.0,
        }
    device = next(actor.parameters()).device
    inputs = samples.inputs.to(device)
    targets = samples.targets.to(device)
    predicted = decoded_actor_output(actor, inputs)
    active = targets[:, 3] >= 0.5
    control_mae = (
        float((predicted[active, :3] - targets[active, :3]).abs().mean().cpu())
        if bool(active.any()) else 0.0
    )
    zero_latent = inputs.clone()
    zero_latent[:, v2.OBSERVATION_DIM:] = 0.0
    zero_prediction = decoded_actor_output(actor, zero_latent)
    anchor_deviation = 0.0
    if reference is not None:
        reference_prediction = decoded_actor_output(reference, inputs)
        anchor_deviation = float(
            (predicted - reference_prediction).abs().mean().cpu()
        )
    return {
        "samples": len(samples),
        "active_samples": int(active.sum().cpu()),
        "control_mae_active": control_mae,
        "gate_accuracy": float(
            ((predicted[:, 3] >= 0.5) == active).float().mean().cpu()
        ),
        "mean_gate": float(predicted[:, 3].mean().cpu()),
        "mean_throttle": float(predicted[:, 1].mean().cpu()),
        "mean_brake": float(predicted[:, 2].mean().cpu()),
        "anchor_deviation": anchor_deviation,
        "latent_sensitivity": float(
            (predicted - zero_prediction).abs().mean().cpu()
        ),
    }


def imagined_actor_loss(
    actor: ActorCritic,
    model: TemporalRSSMWorldModel,
    samples: ActorSamples,
    checkpoint: Dict[str, Any],
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Differentiate a conservative short rollout through the frozen RSSM."""
    device = next(actor.parameters()).device
    batch = samples.to(device)
    state = RSSMState(batch.deter, batch.stoch, batch.logits)
    world_observation = batch.world_observations
    world_mean = torch.as_tensor(
        checkpoint["world_observation_mean"], dtype=torch.float32, device=device
    ).reshape(1, -1)
    world_std = torch.as_tensor(
        checkpoint["world_observation_std"], dtype=torch.float32, device=device
    ).reshape(1, -1).clamp_min(1e-6)
    policy_mean = torch.as_tensor(
        checkpoint["policy_state_mean"], dtype=torch.float32, device=device
    ).reshape(1, -1)
    policy_std = torch.as_tensor(
        checkpoint["policy_state_std"], dtype=torch.float32, device=device
    ).reshape(1, -1).clamp_min(1e-6)
    action_mean = torch.as_tensor(
        checkpoint["action_mean"], dtype=torch.float32, device=device
    ).reshape(1, -1)
    action_std = torch.as_tensor(
        checkpoint["action_std"], dtype=torch.float32, device=device
    ).reshape(1, -1).clamp_min(1e-6)
    continuation = torch.ones(len(samples), dtype=torch.float32, device=device)
    discount = 1.0
    returns = torch.zeros_like(continuation)
    previous_control = batch.base_actions
    first_value = None
    risk_values: List[torch.Tensor] = []
    progress_values: List[torch.Tensor] = []
    for _ in range(max(1, horizon)):
        raw_observation = world_observation * world_std + world_mean
        policy_observation = (raw_observation - policy_mean) / policy_std
        actor_input = torch.cat([policy_observation, model.feature(state)], dim=-1)
        mean, _, value = actor(actor_input)
        if first_value is None:
            first_value = value
        signed_longitudinal = torch.tanh(mean[:, 1:2] - mean[:, 2:3])
        decoded = torch.cat([
            torch.tanh(mean[:, 0:1]),
            torch.relu(signed_longitudinal),
            torch.relu(-signed_longitudinal),
            torch.sigmoid(mean[:, 3:4]),
        ], dim=-1)
        base_action = raw_observation[:, 28:31].clamp(
            torch.tensor([-1.0, 0.0, 0.0], device=device),
            torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        control = blend_with_simlingo(base_action, decoded)
        normalized_action = (control - action_mean) / action_std
        state, heads = model.imagine_step(
            state, normalized_action, deterministic=True
        )
        risk = torch.sigmoid(heads["risk_logit"])
        progress = symexp(heads["progress_symlog"]).clamp(-2.0, 5.0)
        reward = symexp(heads["reward_symlog"]).clamp(-10.0, 10.0)
        events = torch.sigmoid(heads["event_logits"])
        action_change = (control[:, :3] - previous_control[:, :3]).abs().mean(-1)
        utility = (
            0.20 * reward
            + 0.85 * progress
            - 1.50 * risk
            - 3.00 * events[:, 0]
            - 2.00 * events[:, 1]
            - 0.50 * events[:, 2]
            - 0.08 * action_change
        )
        returns = returns + discount * continuation * utility
        continuation = continuation * torch.sigmoid(
            heads["continuation_logit"]
        )
        discount *= 0.95
        world_observation = world_observation + heads["observation_delta"]
        # Previous control is part of the policy observation schema. The RSSM
        # persists those slots, so update them explicitly for imagined steps.
        world_raw = world_observation * world_std + world_mean
        world_raw = torch.cat([world_raw[:, :-4], control], dim=-1)
        world_observation = (world_raw - world_mean) / world_std
        previous_control = control
        risk_values.append(risk)
        progress_values.append(progress)
    assert first_value is not None
    actor_loss = -returns.mean()
    value_loss = F.smooth_l1_loss(first_value, returns.detach())
    metrics = {
        "imagined_return": float(returns.detach().mean().cpu()),
        "imagined_risk": float(torch.stack(risk_values).mean().detach().cpu()),
        "imagined_progress": float(
            torch.stack(progress_values).mean().detach().cpu()
        ),
    }
    return actor_loss, value_loss, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-route", default="33")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--imagination-horizon", type=int, default=3)
    parser.add_argument("--imagination-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--trace-pattern", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    source_path = args.source_checkpoint.expanduser().resolve()
    checkpoint = torch.load(source_path, map_location="cpu")
    config = RSSMConfig.from_dict(checkpoint.get("world_model_config"))
    model = TemporalRSSMWorldModel(config).to(device)
    model.load_state_dict(checkpoint["world_model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    patterns = args.trace_pattern or [
        "logs/dreamer_online_rl/webapp_*/trace.jsonl",
        "logs/dreamer_online_rl/*/traces/*.jsonl",
        "logs/dreamer_rl_campaign/*/traces/*.jsonl",
        "logs/action_dreaming_collect/*.jsonl",
    ]
    paths = v2.discover_traces(patterns)
    episodes, trace_audit = v2.load_episodes(paths, sequence_length=8)
    teacher_routes = sorted({
        episode.route_id for episode in episodes if episode.teacher_mask.any()
    })
    if args.validation_route not in teacher_routes:
        raise RuntimeError(
            f"validation route {args.validation_route} is not a teacher route; "
            f"available={teacher_routes}"
        )
    training_episodes = [
        episode for episode in episodes
        if episode.route_id != args.validation_route
    ]
    validation_episodes = [
        episode for episode in episodes
        if episode.route_id == args.validation_route
    ]
    train_teacher = extract_samples(
        model, training_episodes, checkpoint, device, teacher_only=True
    )
    validation_teacher = extract_samples(
        model, validation_episodes, checkpoint, device, teacher_only=True
    )
    anchors = extract_samples(
        model, training_episodes, checkpoint, device,
        teacher_only=False, anchor_stride=8,
    )
    if len(train_teacher) == 0 or len(validation_teacher) == 0:
        raise RuntimeError("teacher train/validation samples are empty")

    actor = v2.actor_from_checkpoint(checkpoint, device)
    reference = copy.deepcopy(actor).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    before_validation = actor_metrics(actor, validation_teacher, reference)
    before_training = actor_metrics(actor, train_teacher, reference)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter is not actor.log_std],
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best_state = copy.deepcopy(actor.state_dict())
    best_score = math.inf
    history: List[Dict[str, float]] = []
    started = time.time()
    for epoch in range(max(1, args.epochs)):
        actor.train()
        permutation = torch.randperm(len(train_teacher), generator=generator)
        epoch_rows = []
        for start in range(0, len(train_teacher), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            teacher_batch = train_teacher.index(indices).to(device)
            prediction = decoded_actor_output(actor, teacher_batch.inputs)
            active = teacher_batch.targets[:, 3] >= 0.5
            active_weights = torch.where(active, 4.0, 1.0).unsqueeze(-1)
            control_loss = (
                F.smooth_l1_loss(
                    prediction[:, :3], teacher_batch.targets[:, :3],
                    reduction="none",
                ) * active_weights
            ).mean()
            gate_loss = F.binary_cross_entropy(
                prediction[:, 3], teacher_batch.targets[:, 3]
            )

            anchor_count = min(len(anchors), len(teacher_batch))
            anchor_indices = torch.randint(
                len(anchors), (anchor_count,), generator=generator
            )
            anchor_batch = anchors.index(anchor_indices).to(device)
            with torch.no_grad():
                reference_prediction = decoded_actor_output(
                    reference, anchor_batch.inputs
                )
            anchor_prediction = decoded_actor_output(actor, anchor_batch.inputs)
            trust_loss = F.smooth_l1_loss(
                anchor_prediction, reference_prediction
            )
            imagination_loss, value_loss, imagined = imagined_actor_loss(
                actor,
                model,
                teacher_batch,
                checkpoint,
                horizon=args.imagination_horizon,
            )
            loss = (
                control_loss
                + 0.75 * gate_loss
                + 0.15 * trust_loss
                + args.imagination_weight * imagination_loss
                + 0.02 * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            optimizer.step()
            epoch_rows.append({
                "loss": float(loss.detach().cpu()),
                "control": float(control_loss.detach().cpu()),
                "gate": float(gate_loss.detach().cpu()),
                "trust": float(trust_loss.detach().cpu()),
                "value": float(value_loss.detach().cpu()),
                **imagined,
            })
        actor.eval()
        validation = actor_metrics(actor, validation_teacher, reference)
        anchor_validation = actor_metrics(actor, anchors, reference)
        score = (
            validation["control_mae_active"]
            + 0.25 * (1.0 - validation["gate_accuracy"])
            + 0.50 * anchor_validation["anchor_deviation"]
        )
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(actor.state_dict())
        row = {
            "epoch": epoch + 1,
            **{
                key: float(np.mean([item[key] for item in epoch_rows]))
                for key in epoch_rows[0]
            },
            "validation_control_mae": validation["control_mae_active"],
            "validation_gate_accuracy": validation["gate_accuracy"],
            "anchor_deviation": anchor_validation["anchor_deviation"],
            "latent_sensitivity": validation["latent_sensitivity"],
        }
        history.append(row)
        print(
            f"[rssm-actor-v3] epoch={epoch + 1}/{args.epochs} "
            f"loss={row['loss']:.4f} val_control={row['validation_control_mae']:.4f} "
            f"val_gate={row['validation_gate_accuracy']:.3f} "
            f"anchor={row['anchor_deviation']:.4f} "
            f"imagined_return={row['imagined_return']:.3f}",
            flush=True,
        )

    actor.load_state_dict(best_state)
    actor.eval()
    after_validation = actor_metrics(actor, validation_teacher, reference)
    after_training = actor_metrics(actor, train_teacher, reference)
    anchor_metrics = actor_metrics(actor, anchors, reference)
    control_improved = bool(
        math.isfinite(after_validation["control_mae_active"])
        and after_validation["control_mae_active"]
        <= before_validation["control_mae_active"] * 0.97
    )
    gate_valid = bool(
        after_validation["gate_accuracy"]
        >= max(0.80, before_validation["gate_accuracy"] - 0.01)
    )
    trust_valid = bool(anchor_metrics["anchor_deviation"] <= 0.12)
    latent_used = bool(after_validation["latent_sensitivity"] >= 0.005)
    accepted = control_improved and gate_valid and trust_valid and latent_used

    candidate = copy.deepcopy(checkpoint)
    candidate["policy"] = {
        key: value.detach().cpu() for key, value in actor.state_dict().items()
    }
    candidate["policy_model_type"] = "rssm_conditioned_actor_critic_v3"
    candidate["rssm_actor_v3"] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_checkpoint": str(source_path),
        "source_sha256": v2.sha256(source_path),
        "runtime_guard": False,
        "complementary_to_simlingo": True,
        "training": "validated_teacher_distillation_plus_short_rssm_imagination",
        "validation_route": args.validation_route,
        "teacher_routes": teacher_routes,
        "training_samples": len(train_teacher),
        "validation_samples": len(validation_teacher),
        "anchor_samples": len(anchors),
        "imagination_horizon": args.imagination_horizon,
        "imagination_weight": args.imagination_weight,
        "accepted_offline": accepted,
        "closed_loop_promoted": False,
        "before_training": before_training,
        "after_training": after_training,
        "before_validation": before_validation,
        "after_validation": after_validation,
        "anchor_metrics": anchor_metrics,
        "quality_gate": {
            "control_improved_3_percent": control_improved,
            "gate_valid": gate_valid,
            "normal_driving_trust_valid": trust_valid,
            "latent_feature_used": latent_used,
        },
        "history_tail": history[-10:],
        "training_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = args.output_dir / "last_attempt.pt"
    v2.atomic_torch_save(attempt_path, candidate)
    candidate_path = args.output_dir / "candidate_model.pt"
    if accepted:
        v2.atomic_torch_save(candidate_path, candidate)
    report = {
        "status": "candidate_saved" if accepted else "quality_gate_rejected",
        "accepted": accepted,
        "source_checkpoint": str(source_path),
        "source_sha256": v2.sha256(source_path),
        "teacher_routes": teacher_routes,
        "validation_route": args.validation_route,
        "training_samples": len(train_teacher),
        "validation_samples": len(validation_teacher),
        "anchor_samples": len(anchors),
        "before_training": before_training,
        "after_training": after_training,
        "before_validation": before_validation,
        "after_validation": after_validation,
        "anchor_metrics": anchor_metrics,
        "quality_gate": candidate["rssm_actor_v3"]["quality_gate"],
        "history": history,
        "trace_audit": trace_audit,
        "last_attempt": str(attempt_path),
        "candidate": str(candidate_path) if accepted else "",
    }
    v2.atomic_json(args.output_dir / "actor_training_report.json", report)
    print(
        f"[rssm-actor-v3] quality_gate={'PASS' if accepted else 'REJECT'} "
        f"control={before_validation['control_mae_active']:.4f}->"
        f"{after_validation['control_mae_active']:.4f} "
        f"gate={after_validation['gate_accuracy']:.3f} "
        f"anchor={anchor_metrics['anchor_deviation']:.4f} "
        f"latent={after_validation['latent_sensitivity']:.4f}",
        flush=True,
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
