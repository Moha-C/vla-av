"""All hyperparameters for the Dreamer-PPO CARLA project."""
from dataclasses import dataclass

from rewards.vru_reward import BASE_STATE_DIM, TIER1_EXTRA, TIER2_EXTRA


@dataclass
class Config:
    # Environment
    # The DEFAULT state is the original v1 28-dim layout — the checkpoint proven
    # to work in the downstream SimLingo/dreamer_guard.py integration:
    #   ego(6) + lane(4) + traffic(3) + vehicle_ahead(5) + 2 VRUs x 5 = 28.
    # The expanded safety state is OPT-IN via the two flags below; __post_init__
    # recomputes state_dim to the full trainable/exportable dimension:
    #   28 + 20 (Tier-1 rear/side vehicles) + 7 (Tier-2 map-agnostic features).
    state_dim: int = BASE_STATE_DIM
    enable_tier1_state: bool = False   # 28 -> 48: rear/side vehicle awareness
    enable_tier2_state: bool = False   # +7: map-agnostic generalization features
    action_dim: int = 4
    host: str = "localhost"
    port: int = 2000
    town: str = "Town01"
    fps: int = 10
    max_episode_steps: int = 1000

    # World model
    wm_hidden: int = 256
    lr_wm: float = 3e-4
    wm_batch_size: int = 256
    wm_warmup_steps: int = 1000   # transitions trained before dreaming starts

    # PPO
    hidden: int = 256
    lr_policy: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_size: int = 2048
    update_epochs: int = 10
    batch_size: int = 64
    num_episodes: int = 1000

    # Dreaming
    dream_k: int = 5
    w_progress: float = 1.0
    w_risk: float = 2.0
    w_value: float = 0.5

    # Reward weights
    w_prog: float = 1.0
    w_vru: float = 2.0
    w_safe: float = 1.0
    w_comfort: float = 0.1
    w_rules: float = 0.5

    # General vehicle safety (Tier 1: rear/side traffic awareness)
    w_vehicle: float = 1.0              # vehicle-safety weight (VRU stays 2.0)
    rear_risk_threshold: float = 3.0    # seconds TTC for a rear-collision penalty
    vehicle_proximity_sigma: float = 5.0    # distance scale for proximity penalty
    min_lane_change_clearance: float = 2.0  # min safe side gap for a lane change

    # Defensive driving mode (for unknown maps)
    defensive_mode: bool = False             # start in defensive mode
    unknown_map_detection: str = "manual"    # 'manual' or 'gps_comparison'
    defensive_w_vru_mult: float = 1.5        # increase VRU weight
    defensive_w_vehicle_mult: float = 1.5    # increase vehicle-safety weight
    defensive_dream_k: int = 8               # more candidate actions
    defensive_horizon_max: int = 5           # deeper lookahead
    defensive_disable_risky_maneuvers: bool = True
    sigma_d: float = 5.0
    lambda_ttc: float = 0.5
    tau_ttc: float = 2.0
    lambda_cross: float = 1.0
    eta1: float = 0.5  # lane departure weight in safety term
    eta2: float = 0.3  # general risk weight in safety term

    # S-DBS incremental-testing overrides.
    # When sdbs_force_fixed_params is True, SDBSPlanner.get_search_params
    # ignores difficulty-based auto-scaling and returns the fixed values below.
    # Start any new integration at horizon=1, groups=1 (equivalent to greedy
    # one-step dreaming), confirm it matches the plain dreamer variant, then
    # increase the horizon gradually to isolate where behaviour degrades.
    sdbs_force_fixed_params: bool = False
    sdbs_fixed_horizon: int = 1
    sdbs_fixed_groups: int = 1
    sdbs_fixed_beam_width: int = 4

    # SAFE-DREAM safety evaluation.
    # A beam-search candidate whose final composite score falls below this
    # threshold is counted as "rejected as unsafe" for the Unsafe Maneuver
    # Rejection Rate (UMRR) metric. Tune later once real score scales settle.
    unsafe_score_threshold: float = -5.0

    def __post_init__(self):
        # state_dim is derived, not free: it is always the base v1 layout plus
        # whichever opt-in tiers are enabled. This keeps a single source of
        # truth so models, env, and validators agree on the dimension.
        self.state_dim = (
            BASE_STATE_DIM
            + (TIER1_EXTRA if self.enable_tier1_state else 0)
            + (TIER2_EXTRA if self.enable_tier2_state else 0)
        )
