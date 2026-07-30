"""Hyperparameters for the S-DBS extension.

``SDBSConfig`` inherits every field of the base ``Config`` (so the existing
trainers keep working unchanged) and adds the S-DBS planner, diversity /
serendipity, difficulty-estimation, curriculum, PER, and auxiliary-loss knobs.
"""
from dataclasses import dataclass

from configs.config import Config


@dataclass
class SDBSConfig(Config):
    # -- S-DBS planner ------------------------------------------------- #
    beam_width_min: int = 4
    beam_width_max: int = 16
    horizon_min: int = 1
    horizon_max: int = 5
    num_groups_min: int = 1
    num_groups_max: int = 5
    compute_budget: int = 1000          # world-model forward passes per step

    # -- diversity and serendipity ------------------------------------- #
    lambda_g: float = 1.0               # diversity penalty weight
    eta_serendipity: float = 0.5        # serendipity bonus weight in planning
    eta_s: float = 0.1                  # intrinsic-reward weight during training
    alpha_novelty: float = 1.0
    beta_surprise: float = 1.0
    gamma_gain: float = 1.0

    # -- difficulty estimation ----------------------------------------- #
    w_vru_count: float = 0.2
    w_ttc_deficit: float = 0.3
    w_occlusion: float = 0.2
    w_risk_density: float = 0.15
    w_uncertainty: float = 0.15
    tau_safe: float = 1.5               # seconds, hard-stop threshold
    # tau_ttc is inherited from Config (2.0 s).

    # -- curriculum ---------------------------------------------------- #
    curriculum_stages: int = 3
    stage_unlock_threshold: float = 0.85
    scenarios_per_stage: int = 6
    goal_distance: float = 1.0          # route_progress is in [0, 1]
    max_scenarios_per_episode: int = 4  # cap scenarios visited per outer episode

    # -- prioritized experience replay --------------------------------- #
    per_alpha: float = 0.6
    per_epsilon: float = 1e-6
    scenario_bank_capacity: int = 1000

    # -- auxiliary losses ---------------------------------------------- #
    lambda_recon: float = 0.1
    lambda_risk_density: float = 0.1
    world_model_ensemble_size: int = 3
