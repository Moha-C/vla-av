"""Map-agnostic feature vector (Tier-2, opt-in).

Computes 7 features so the policy can generalize to unknown towns:
in_lane_center (1) + road-type one-hot (4) + visibility (1) + oncoming-traffic
flag (1). With Tier-2 enabled the env appends this vector directly to the
state; ``MapAgnosticStateWrapper`` is a thin helper kept for standalone use.
"""
import numpy as np

from utils.map_agnostic_features import (
    map_agnostic_vector, N_EXTRA_FEATURES, _ROAD_TYPE_ONEHOT,
)


class MapAgnosticStateWrapper:
    """Appends the map-agnostic feature vector to a base (non-Tier-2) state."""

    def __init__(self, config):
        self.config = config
        self.base_state_dim = config.state_dim
        self.augmented_state_dim = self.base_state_dim + N_EXTRA_FEATURES

    def augment_state(self, state, info):
        """Return the base state with map-agnostic features appended."""
        base = np.asarray(state, dtype=np.float32).reshape(-1)[:self.base_state_dim]
        extra = map_agnostic_vector(base, info, self.config)
        return np.concatenate([base, extra])
