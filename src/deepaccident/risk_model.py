"""Ego-camera temporal risk encoder for DeepAccident pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@dataclass(frozen=True)
class RiskEncoderConfig:
    embedding_dim: int = 128
    temporal_dim: int = 192
    dropout: float = 0.15
    pretrained_backbone: bool = True
    freeze_backbone_epochs: int = 2
    prediction_horizon_s: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RiskEncoderConfig":
        return cls(**dict(values))


class DeepAccidentRiskEncoder(nn.Module):
    """MobileNet frame encoder plus GRU temporal aggregation.

    The exported embedding is deliberately independent from the active RSSM.
    A later, validated adapter may append it to the compact observation, but
    this model never emits steering, throttle, or brake commands.
    """

    def __init__(self, config: Optional[RiskEncoderConfig] = None):
        super().__init__()
        self.config = config or RiskEncoderConfig()
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if self.config.pretrained_backbone
            else None
        )
        backbone = mobilenet_v3_small(weights=weights)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.frame_projection = nn.Sequential(
            nn.Linear(576, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
        )
        self.temporal = nn.GRU(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.temporal_dim,
            batch_first=True,
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(self.config.temporal_dim),
            nn.Linear(self.config.temporal_dim, self.config.embedding_dim),
            nn.SiLU(),
        )
        self.risk_head = nn.Linear(self.config.embedding_dim, 1)
        self.ttc_head = nn.Linear(self.config.embedding_dim, 1)

    def freeze_backbone(self, frozen: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must be [batch, time, 3, height, width]")
        batch, time = frames.shape[:2]
        flattened = frames.reshape(batch * time, *frames.shape[2:])
        features = self.pool(self.backbone(flattened)).flatten(1)
        frame_embeddings = self.frame_projection(features).reshape(batch, time, -1)
        temporal, _ = self.temporal(frame_embeddings)
        embedding = self.embedding_head(temporal[:, -1])
        risk_logit = self.risk_head(embedding).squeeze(-1)
        return {
            "embedding": embedding,
            "risk_logit": risk_logit,
            "risk": torch.sigmoid(risk_logit),
            "ttc_s": F.softplus(self.ttc_head(embedding).squeeze(-1)),
        }


def load_risk_encoder(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> DeepAccidentRiskEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "deepaccident_temporal_risk_encoder":
        raise ValueError("unsupported DeepAccident checkpoint kind")
    config = RiskEncoderConfig.from_dict(checkpoint["model_config"])
    # The checkpoint contains the complete backbone, so loading must never
    # trigger an external ImageNet download.
    config = RiskEncoderConfig(
        embedding_dim=config.embedding_dim,
        temporal_dim=config.temporal_dim,
        dropout=config.dropout,
        pretrained_backbone=False,
        freeze_backbone_epochs=config.freeze_backbone_epochs,
        prediction_horizon_s=config.prediction_horizon_s,
    )
    model = DeepAccidentRiskEncoder(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device or torch.device("cpu"))
    model.eval()
    return model
