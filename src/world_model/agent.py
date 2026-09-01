"""Clean SimLingo reference plus Dreamer residual orchestration."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .authority import LearnedAuthorityController
from .config import DreamerConfig, load_config
from .observation import DreamerObservation
from .pairwise import PairwiseCalibrator
from .planning import CandidateEvaluator, CandidateGenerator
from .policy import LatentCritic, ResidualActor
from .rssm import CompactRSSM, RSSMState


@dataclass
class DreamerStep:
    native_action: np.ndarray
    dreamer_action: np.ndarray
    final_action: np.ndarray
    alpha: float
    candidate_index: int
    candidate_kind: str
    information: Dict[str, Any]


class SimLingoDreamerAgent:
    """Dreamer complement operating around a native SimLingo action."""

    CHECKPOINT_VERSION = "report_aligned_dreamer_v2"

    @staticmethod
    def _architecture_signature(config: DreamerConfig) -> Dict[str, Any]:
        """Fields whose semantics/shapes are learned into the checkpoint."""

        return {
            "observation": asdict(config.observation),
            "rssm": asdict(config.rssm),
            "policy": asdict(config.policy),
            "pairwise_hidden_dim": config.pairwise.hidden_dim,
        }

    def __init__(
        self,
        config: Optional[DreamerConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or DreamerConfig()
        self.config.validate()
        if self.config.runtime.ablation == "B":
            raise ValueError(
                "Ablation B is the preserved legacy DreamerGuard runtime, not "
                "the report-aligned RSSM wrapper. Launch it through the legacy "
                "dashboard preset."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.world_model = CompactRSSM(self.config.rssm).to(self.device)
        self.actor = ResidualActor(
            self.world_model.feature_dim, self.config.policy
        ).to(self.device)
        self.critic = LatentCritic(
            self.world_model.feature_dim, self.config.policy
        ).to(self.device)
        self.pairwise: Optional[PairwiseCalibrator] = None
        if self.config.runtime.ablation == "E" or self.config.pairwise.enabled:
            self.pairwise = PairwiseCalibrator(
                feature_dim=5,
                hidden_dim=self.config.pairwise.hidden_dim,
            ).to(self.device)
        # Runtime ablations must not mutate the versioned config embedded in a
        # checkpoint (or another agent sharing the same config instance).
        candidate_config = copy.deepcopy(self.config.candidates)
        if self.config.runtime.ablation == "C":
            candidate_config.include_actor_candidate = False
        self.generator = CandidateGenerator(candidate_config)
        self.evaluator = CandidateEvaluator(
            self.world_model,
            self.actor,
            candidate_config,
            self.config.evaluator,
            self.pairwise,
        )
        authority_config = copy.deepcopy(self.config.authority)
        authority_config.learned = self.config.runtime.ablation in ("D", "E")
        self.authority = LearnedAuthorityController(authority_config)
        self.reset()

    def reset(self) -> None:
        self.latent: Optional[RSSMState] = None
        self.previous_world_action: Optional[torch.Tensor] = None
        self.authority.reset()

    @staticmethod
    def _native_array(action: Sequence[float]) -> np.ndarray:
        native = np.asarray(action, dtype=np.float32)[:3]
        if native.shape != (3,):
            raise ValueError("native action must contain steer, throttle, brake")
        return native

    def step(self, observation: DreamerObservation, native_action: Sequence[float]) -> DreamerStep:
        native = self._native_array(native_action)
        if self.config.runtime.ablation == "A":
            return DreamerStep(native, native.copy(), native.copy(), 0.0, 0, "native", {"ablation": "A"})
        obs = torch.as_tensor(
            observation.as_array(copy=False), device=self.device
        ).unsqueeze(0)
        native_tensor = torch.as_tensor(native, device=self.device)
        with torch.no_grad():
            if self.latent is None:
                self.latent = self.world_model.observe_initial(
                    obs, self.config.runtime.deterministic_latent
                )
            else:
                if self.previous_world_action is None:
                    raise RuntimeError("missing previous world action")
                self.latent, _ = self.world_model.observe_step(
                    self.latent,
                    self.previous_world_action,
                    obs,
                    self.config.runtime.deterministic_latent,
                )
            actor_output = self.actor(
                self.world_model.feature(self.latent),
                native_tensor.unsqueeze(0),
                self.config.runtime.deterministic_policy,
            )
            candidates = self.generator.generate(
                native_tensor,
                obs.squeeze(0),
                actor_output.proposal.squeeze(0),
                (
                    self.config.authority.fixed_alpha
                    if self.config.runtime.ablation == "C"
                    else float(actor_output.alpha.item())
                ),
            )
            imagination = self.evaluator.imagine(
                self.latent,
                native_tensor,
                candidates,
                self.config.rssm.imagination_horizon,
                deterministic=self.config.runtime.deterministic_latent,
            )
        index = imagination.selected_index
        candidate = imagination.candidates[index]
        learned_alpha = float(actor_output.alpha.item())
        force_alpha = None
        if self.config.runtime.shadow:
            force_alpha = 0.0
        elif self.config.runtime.ablation == "C":
            force_alpha = self.config.authority.fixed_alpha
        decision = self.authority.decide(
            learned_alpha=learned_alpha,
            candidate_is_native=index == 0,
            force_alpha=force_alpha,
        )
        dreamer_action = candidate.proposal.detach().cpu().numpy().astype(np.float32)
        final_action = self.authority.blend(native, dreamer_action, decision.alpha)
        self.previous_world_action = torch.as_tensor(
            final_action, device=self.device
        ).unsqueeze(0)
        features = imagination.features.detach().cpu().numpy()
        utilities = imagination.utilities.detach().cpu().numpy()
        information: Dict[str, Any] = {
            "ablation": self.config.runtime.ablation,
            "candidate_kinds": [item.kind for item in imagination.candidates],
            "candidate_utilities": utilities.tolist(),
            "candidate_features": features.tolist(),
            "selected_index": index,
            "selected_kind": candidate.kind,
            "native_predicted_progress": float(features[0, 0]),
            "native_predicted_risk": float(features[0, 1]),
            "selected_predicted_progress": float(features[index, 0]),
            "selected_predicted_risk": float(features[index, 1]),
            "selected_predicted_continuation": float(features[index, 2]),
            "selected_predicted_value": float(features[index, 3]),
            "selected_change_cost": float(features[index, 4]),
            "alpha_raw": decision.alpha_raw,
            "alpha": decision.alpha,
            "simlingo_weight": decision.native_weight,
            "dreamer_weight": decision.dreamer_weight,
            "native_action": native.tolist(),
            "dreamer_action": dreamer_action.tolist(),
            "final_action": final_action.tolist(),
        }
        return DreamerStep(
            native_action=native,
            dreamer_action=dreamer_action,
            final_action=final_action,
            alpha=decision.alpha,
            candidate_index=index,
            candidate_kind=candidate.kind,
            information=information,
        )

    def checkpoint_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "config": self.config.to_dict(),
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        if self.pairwise is not None:
            payload["pairwise"] = self.pairwise.state_dict()
        return payload

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = self.checkpoint_payload()
        payload["metadata"] = dict(metadata or {})
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(destination))

    @classmethod
    def load(
        cls,
        path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        runtime_overrides: Optional[Dict[str, Any]] = None,
    ) -> "SimLingoDreamerAgent":
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=target_device)
        if payload.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise ValueError("incompatible report-aligned Dreamer checkpoint")
        checkpoint_config = load_config(overrides=payload.get("config") or {})
        if config_path:
            requested_config = load_config(config_path)
            if cls._architecture_signature(requested_config) != cls._architecture_signature(
                checkpoint_config
            ):
                raise ValueError(
                    "runtime config changes checkpoint-trained observation/RSSM/"
                    "policy architecture; use the checkpoint config or a compatible YAML"
                )
        # Candidate generation, loss semantics and authority limits are part of
        # the learned checkpoint protocol.  A runtime YAML may verify tensor
        # compatibility, but it must not silently replace those semantics.
        config = checkpoint_config
        for key, value in (runtime_overrides or {}).items():
            if not hasattr(config.runtime, key):
                raise ValueError("unknown runtime override: %s" % key)
            setattr(config.runtime, key, value)
        config.validate()
        agent = cls(config=config, device=target_device)
        agent.world_model.load_state_dict(payload["world_model"])
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        if agent.pairwise is not None:
            if "pairwise" not in payload:
                raise ValueError(
                    "ablation E requires a separately trained pairwise "
                    "calibrator; this checkpoint does not contain one"
                )
            agent.pairwise.load_state_dict(payload["pairwise"])
        agent.world_model.eval()
        agent.actor.eval()
        agent.critic.eval()
        if agent.pairwise is not None:
            agent.pairwise.eval()
        return agent
