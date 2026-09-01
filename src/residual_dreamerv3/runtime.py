"""Stateful runtime with an explicit promotion lock.

Candidate checkpoints can be inspected in shadow mode, but cannot influence a
vehicle unless a fixed closed-loop evaluation has promoted them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from .config import ResidualDreamerConfig, load_config
from .model import RSSMState, ResidualDreamerV3


class CheckpointNotPromotedError(RuntimeError):
    pass


class ResidualDreamerRuntime:
    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: str = "cpu",
        allow_candidate_shadow: bool = False,
        allow_candidate_evaluation: bool = False,
    ):
        if allow_candidate_shadow and allow_candidate_evaluation:
            raise ValueError("candidate runtime must be either shadow or evaluation, not both")
        self.path = Path(checkpoint)
        payload = torch.load(str(self.path), map_location=device)
        if not isinstance(payload, Mapping):
            raise TypeError("invalid residual Dreamer checkpoint")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        status = str(metadata.get("status", "candidate"))
        if status != "promoted" and not (allow_candidate_shadow or allow_candidate_evaluation):
            raise CheckpointNotPromotedError(
                "checkpoint status is %s; only promoted checkpoints may control CARLA" % status
            )
        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError("checkpoint is missing its versioned configuration")
        self.config = load_config(overrides=config_payload)
        self.device = torch.device(device)
        self.model = ResidualDreamerV3(self.config).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.metadata = dict(metadata)
        self.shadow_only = status != "promoted" and allow_candidate_shadow
        self.evaluation_only = status != "promoted" and allow_candidate_evaluation
        self.state: Optional[RSSMState] = None
        self.previous_action: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.state = None
        self.previous_action = None

    @torch.no_grad()
    def step(self, observation: np.ndarray, applied_action: Optional[np.ndarray] = None) -> Dict[str, Any]:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        if observation_tensor.shape[1] != self.config.model.observation_dim:
            raise ValueError("unexpected observation dimension")
        previous = applied_action
        if previous is None and self.previous_action is not None:
            previous_tensor = self.previous_action
        elif previous is not None:
            previous_tensor = torch.as_tensor(previous, dtype=torch.float32, device=self.device).reshape(1, -1)
        else:
            previous_tensor = None
        if self.state is None or previous_tensor is None:
            self.state = self.model.world_model.observe_initial(observation_tensor, deterministic=True)
        else:
            self.state, _ = self.model.world_model.observe_step(
                self.state, previous_tensor, observation_tensor, deterministic=True
            )
        feature = self.model.world_model.feature(self.state)
        actor = self.model.actor(feature, observation_tensor, deterministic=True)
        dreamer_action = actor.final_action
        native = observation_tensor[:, 2:5]
        # A candidate loaded for diagnosis is physically incapable of taking
        # control: the public action remains SimLingo's native action.
        chosen = native if self.shadow_only else dreamer_action
        self.previous_action = chosen.detach()
        return {
            "action": chosen[0].cpu().numpy(),
            "dreamer_action": dreamer_action[0].cpu().numpy(),
            "native_action": native[0].cpu().numpy(),
            "proposal": actor.proposal[0].cpu().numpy(),
            "residual": actor.residual[0].cpu().numpy(),
            "authority": float(actor.authority[0].cpu()),
            "shadow_only": self.shadow_only,
            "evaluation_only": self.evaluation_only,
            "checkpoint_status": self.metadata.get("status", "candidate"),
            "guards_active": False,
        }
