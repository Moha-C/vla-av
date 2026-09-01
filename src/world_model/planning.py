"""Candidate generation, latent imagination and report utility scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import CandidateConfig, EvaluatorConfig
from .pairwise import PairwiseCalibrator
from .policy import ResidualActor, blend_control, signed_longitudinal, split_longitudinal
from .rssm import CompactRSSM, RSSMState


@dataclass
class Candidate:
    kind: str
    proposal: torch.Tensor
    assumed_alpha: float


@dataclass
class ImaginedCandidates:
    candidates: Tuple[Candidate, ...]
    world_actions: torch.Tensor
    predictions: Dict[str, torch.Tensor]
    features: torch.Tensor
    utilities: torch.Tensor
    selected_index: int


class CandidateGenerator:
    """Generate bounded alternatives around one native SimLingo command.

    There is intentionally no geometric veto in this class. Availability and
    traffic geometry are model inputs, so unsafe alternatives must be rejected
    by predicted risk/value rather than a hidden hand-written guard.
    """

    def __init__(self, config: Optional[CandidateConfig] = None):
        self.config = config or CandidateConfig()

    @staticmethod
    def _control(steer: torch.Tensor, longitudinal: torch.Tensor) -> torch.Tensor:
        throttle, brake = split_longitudinal(longitudinal)
        return torch.stack((steer.clamp(-1.0, 1.0), throttle, brake), dim=-1)

    def generate(
        self,
        native: torch.Tensor,
        observation: torch.Tensor,
        actor_proposal: Optional[torch.Tensor] = None,
        authority_alpha: Optional[float] = None,
    ) -> Tuple[Candidate, ...]:
        if native.shape != (3,):
            raise ValueError("candidate generation expects one [steer, throttle, brake] action")
        cfg = self.config
        non_native_alpha = float(
            cfg.assumed_alpha if authority_alpha is None else authority_alpha
        )
        candidates: List[Candidate] = [Candidate("native", native.clone(), 0.0)]
        native_long = signed_longitudinal(native)
        for factor in cfg.slow_factors:
            proposal = self._control(native[0], native_long * float(factor))
            candidates.append(Candidate("slow_%.2f" % float(factor), proposal, non_native_alpha))
        for level in cfg.emergency_brake_levels:
            proposal = torch.stack(
                (
                    native[0].clamp(-1.0, 1.0),
                    native.new_tensor(0.0),
                    native.new_tensor(float(level)).clamp(0.0, 1.0),
                )
            )
            candidates.append(
                Candidate(
                    "emergency_brake_%.2f" % float(level),
                    proposal,
                    non_native_alpha,
                )
            )
        for offset in cfg.steer_offsets:
            delta = max(-cfg.max_steer_delta, min(cfg.max_steer_delta, float(offset)))
            proposal = self._control(native[0] + delta, native_long)
            candidates.append(Candidate("steer_%+.2f" % delta, proposal, non_native_alpha))
        candidates.append(
            Candidate(
                "prepare_overtake_left",
                self._control(native[0] - cfg.overtake_steer, native_long),
                non_native_alpha,
            )
        )
        candidates.append(
            Candidate(
                "prepare_overtake_right",
                self._control(native[0] + cfg.overtake_steer, native_long),
                non_native_alpha,
            )
        )
        lane_offset = float(observation[7].detach().cpu())
        return_delta = cfg.return_steer if lane_offset < 0.0 else -cfg.return_steer
        candidates.append(
            Candidate(
                "return_to_lane",
                self._control(native[0] + return_delta, native_long),
                non_native_alpha,
            )
        )
        if cfg.include_actor_candidate and actor_proposal is not None:
            candidates.append(
                Candidate(
                    "actor_residual",
                    actor_proposal.clone(),
                    non_native_alpha,
                )
            )
        return tuple(candidates)


class CandidateEvaluator:
    def __init__(
        self,
        world_model: CompactRSSM,
        actor: ResidualActor,
        candidate_config: Optional[CandidateConfig] = None,
        evaluator_config: Optional[EvaluatorConfig] = None,
        pairwise: Optional[PairwiseCalibrator] = None,
    ):
        self.world_model = world_model
        self.actor = actor
        self.candidate_config = candidate_config or CandidateConfig()
        self.config = evaluator_config or EvaluatorConfig()
        self.pairwise = pairwise

    def imagine(
        self,
        latent: RSSMState,
        native: torch.Tensor,
        candidates: Sequence[Candidate],
        horizon: int,
        deterministic: bool = True,
    ) -> ImaginedCandidates:
        if latent.deterministic.shape[0] != 1:
            raise ValueError("runtime candidate evaluation expects batch size 1")
        count = len(candidates)
        state = latent.repeat_interleave(count)
        native_batch = native.unsqueeze(0).expand(count, -1)
        proposals = torch.stack([item.proposal for item in candidates], dim=0)
        alphas = torch.as_tensor(
            [item.assumed_alpha for item in candidates],
            dtype=native.dtype,
            device=native.device,
        ).clamp(0.0, 1.0)
        final_controls = blend_control(native_batch, proposals, alphas)
        world_action = final_controls
        world_actions = []
        predicted: Dict[str, List[torch.Tensor]] = {
            "progress": [],
            "risk": [],
            "continuation": [],
            "value": [],
            "collision": [],
            "offroad": [],
        }
        for step in range(max(1, int(horizon))):
            if step > 0:
                actor_output = self.actor(
                    self.world_model.feature(state), native_batch, deterministic
                )
                future_alpha = actor_output.alpha.clamp(0.0, 1.0)
                future_control = blend_control(
                    native_batch, actor_output.proposal, future_alpha
                )
                world_action = future_control
            state = self.world_model.imagine_step(state, world_action, deterministic)
            output = self.world_model.heads(state)
            world_actions.append(world_action)
            predicted["progress"].append(output.progress.squeeze(-1))
            predicted["risk"].append(output.risk.squeeze(-1))
            predicted["continuation"].append(output.continuation.squeeze(-1))
            predicted["value"].append(output.value.squeeze(-1))
            predicted["collision"].append(output.collision.squeeze(-1))
            predicted["offroad"].append(output.offroad.squeeze(-1))
        stacked = {key: torch.stack(value, dim=1) for key, value in predicted.items()}
        actions = torch.stack(world_actions, dim=1)
        continuation = stacked["continuation"].clamp(0.0, 1.0)
        discount = float(self.config.continuation_discount)
        weights = torch.ones_like(continuation)
        if continuation.shape[1] > 1:
            weights[:, 1:] = torch.cumprod(
                discount * continuation[:, :-1], dim=1
            )
        denominator = weights.sum(dim=1).clamp_min(1.0e-6)
        progress = (weights * stacked["progress"]).sum(dim=1) / denominator
        risk = (weights * stacked["risk"]).sum(dim=1) / denominator
        cont = (weights * continuation).sum(dim=1) / denominator
        value = stacked["value"][:, -1]
        change = torch.linalg.vector_norm(proposals - native_batch, dim=-1)
        features = torch.stack((progress, risk, cont, value, change), dim=-1)
        utilities = (
            self.config.lambda_progress * progress
            - self.config.lambda_risk * risk
            - self.config.lambda_change * change
        )
        if self.pairwise is not None:
            utilities = utilities + self.config.pairwise_scale * self.pairwise.tournament_bonus(features)
        selected = int(torch.argmax(utilities).item())
        return ImaginedCandidates(
            candidates=tuple(candidates),
            world_actions=actions,
            predictions=stacked,
            features=features,
            utilities=utilities,
            selected_index=selected,
        )
