"""VRU (Vulnerable Road User) safety reward.

Full implementation of the reward terms from the project spec: progress,
VRU risk (proximity + TTC + crosswalk), safety (collision / lane departure /
general risk), comfort (action smoothness), and traffic-rule violations.

State vector layout (dim=28), matching env/carla_env.py:
  ego (6):     [0] x, [1] y, [2] speed, [3] heading, [4] acc_x, [5] acc_y
  lane (4):    [6] lane_offset, [7] lane_width, [8] road_curvature, [9] is_junction
  traffic (3): [10] traffic_light_state, [11] dist_to_light, [12] route_progress
  vehicles (5):[13] dist, [14] speed, [15] heading, [16] rel_x, [17] rel_y
  VRU (10):    2 VRUs x (dist, speed, heading, rel_x, rel_y)
               VRU0: [18..22], VRU1: [23..27]

NOTE on the goal index: the spec lists ``DIST_TO_GOAL = 8``, but index 8 is
``road_curvature`` in this state layout. The actual goal signal is
``route_progress`` at index 12, which *increases* toward the goal — so the
progress reward is ``next - current`` (positive = closer to the goal).
"""
import numpy as np

# ---- State index constants ------------------------------------------------ #
EGO_X, EGO_Y = 0, 1
EGO_SPEED = 2
EGO_HEADING = 3
LANE_OFFSET = 6
ROUTE_PROGRESS = 12          # route completion in [0,1] (the goal signal)

VRU1_DIST = 18               # first VRU block starts here
VRU2_DIST = 23               # second VRU block starts here
VRU_BLOCK_SIZE = 5           # dist, speed, heading, rel_x, rel_y
VRU_DIST_INDICES = (VRU1_DIST, VRU2_DIST)


def compute_reward(state, next_state, action, prev_action, info, config):
    """Compute the total reward and its components for one transition.

    Returns (total_reward, reward_components_dict).
    """
    state = np.asarray(state, dtype=np.float32)
    next_state = np.asarray(next_state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    prev_action = np.asarray(prev_action, dtype=np.float32).reshape(-1)

    # ---- 1. PROGRESS -------------------------------------------------- #
    dist_before = float(state[ROUTE_PROGRESS])
    dist_after = float(next_state[ROUTE_PROGRESS])
    r_progress = dist_after - dist_before          # positive = closer to goal

    # ---- 2. VRU RISK (most important) --------------------------------- #
    ego_speed = float(next_state[EGO_SPEED])
    crosswalk = bool(info.get("crosswalk_conflict", False))
    r_vru = 0.0
    for vru_dist_idx in VRU_DIST_INDICES:
        dist_i = float(next_state[vru_dist_idx])

        # Proximity penalty — exponential, large when the VRU is close.
        r_proximity_i = -np.exp(-dist_i / config.sigma_d)

        # Time-to-collision penalty.
        ttc_i = dist_i / (ego_speed + 1e-6)
        r_ttc_i = -config.lambda_ttc if ttc_i < config.tau_ttc else 0.0

        # Crosswalk conflict penalty.
        r_cross_i = -config.lambda_cross if crosswalk else 0.0

        r_vru += float(r_proximity_i + r_ttc_i + r_cross_i)

    # ---- 3. SAFETY ---------------------------------------------------- #
    r_collision = -10.0 if info.get("collision", False) else 0.0
    r_lane_depart = -1.0 if info.get("lane_departure", False) else 0.0
    r_general_risk = -float(info.get("general_risk", 0.0))

    # ---- 4. COMFORT --------------------------------------------------- #
    delta_steer = abs(float(action[0]) - float(prev_action[0]))
    delta_throttle = abs(float(action[1]) - float(prev_action[1]))
    delta_brake = abs(float(action[2]) - float(prev_action[2]))
    r_comfort = -(delta_steer + delta_throttle + delta_brake)

    # ---- 5. RULES ----------------------------------------------------- #
    r_red_light = -2.0 if info.get("red_light_violation", False) else 0.0
    r_stop_sign = -1.0 if info.get("stop_sign_violation", False) else 0.0
    r_rules = r_red_light + r_stop_sign

    # ---- TOTAL -------------------------------------------------------- #
    r_total = (
        config.w_prog * r_progress
        + config.w_vru * r_vru
        + config.w_safe * (r_collision + r_lane_depart + r_general_risk)
        + config.w_comfort * r_comfort
        + config.w_rules * r_rules
    )

    components = {
        "progress": r_progress,
        "vru_risk": r_vru,
        "collision": r_collision,
        "lane_depart": r_lane_depart,
        "comfort": r_comfort,
        "rules": r_rules,
        "total": float(r_total),
    }
    return float(r_total), components


def compute_vru_risk_target(state, info, config):
    """Normalized VRU risk target in [0,1] for world-model supervision."""
    if "vru_risk" in info:
        return float(np.clip(info["vru_risk"], 0.0, 1.0))
    state = np.asarray(state, dtype=np.float32)
    risk = 0.0
    for idx in VRU_DIST_INDICES:
        dist = float(state[idx])
        if dist > 0:
            risk += float(np.exp(-dist / config.sigma_d))
    return float(np.clip(risk / len(VRU_DIST_INDICES), 0.0, 1.0))
