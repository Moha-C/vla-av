"""Core S-DBS (Serendipitous Diverse Beam Search) data structures.

This module holds the beam-search infrastructure used by ``sdbs_planner``:
the ``Plan`` container (a partial/complete action sequence with its imagined
rollout), the ``BeamState`` that manages groups of plans at a given lookahead
depth, and the spacetime-conflict-cell / diversity helpers used to keep the
search groups exploring genuinely different trajectories.

State-vector layout matches ``env/carla_env.py`` (dim=28); ego (x, y) live at
indices 0 and 1, which is what the conflict-cell discretization uses.
"""
from dataclasses import dataclass, field

import numpy as np

from rewards.vru_reward import EGO_X, EGO_Y


# ---------------------------------------------------------------------- #
# Plan: one partial or complete plan
# ---------------------------------------------------------------------- #
@dataclass(eq=False)            # eq=False keeps identity hash -> usable as dict key
class Plan:
    """A partial or complete plan (a sequence of actions + its imagined roll-out).

    ``imagined_states`` holds ``H + 1`` states (the root state followed by one
    imagined state per action); ``imagined_rewards`` holds the ``H`` imagined
    rewards. ``first_*`` fields cache the policy proposal that produced the
    plan's first action so PPO can train on a consistent (raw action, log-prob,
    value) triple even when execution is later overridden by a safety mandate.
    """
    actions: list = field(default_factory=list)
    imagined_states: list = field(default_factory=list)
    imagined_rewards: list = field(default_factory=list)
    maneuver: object = None        # semantic label (str) if hierarchical, else None
    score: float = 0.0             # total imagined return + serendipity bonuses
    g_value: float = 0.0           # discounted imagined return (+ terminal value)
    group_id: int = 0              # which beam group this plan belongs to

    # bookkeeping for PPO consistency and serendipity reporting
    first_raw_action: object = None
    first_log_prob: float = 0.0
    first_value: float = 0.0
    serendipity_value: float = 0.0
    collision_risk: float = 0.0        # predicted multi-agent collision risk


# ---------------------------------------------------------------------- #
# BeamState: groups of plans at a fixed depth
# ---------------------------------------------------------------------- #
class BeamState:
    """Manages G groups of partial plans plus the running incumbent."""

    def __init__(self, depth=0, num_groups=1):
        self.depth = depth
        self.groups = [[] for _ in range(max(1, num_groups))]
        self.incumbent = None
        self.conflict_cells = {}    # Plan -> set of (cell_x, cell_y, time) tuples

    def add_plan(self, plan, group_id):
        while len(self.groups) <= group_id:
            self.groups.append([])
        self.groups[group_id].append(plan)

    def get_top_per_group(self, b_per_group):
        """Trim each group in place to its top ``b_per_group`` plans by score."""
        for g in range(len(self.groups)):
            self.groups[g] = sorted(
                self.groups[g], key=lambda p: p.score, reverse=True
            )[:b_per_group]

    def get_incumbent(self):
        return self.incumbent

    def set_incumbent(self, plan):
        self.incumbent = plan


# ---------------------------------------------------------------------- #
# Conflict cells and diversity measures
# ---------------------------------------------------------------------- #
def compute_conflict_cells(imagined_states, discretization=0.5):
    """Discretize an imagined trajectory into spacetime conflict cells.

    Each imagined state contributes one ``(cell_x, cell_y, time)`` cell, where
    the spatial extent is divided into ``discretization`` (default 0.5 m) cells
    and ``time`` is the step index. With the project's 10 Hz stepping each step
    is 0.1 s, so the step index *is* the 0.1 s time cell.

    Returns a ``set`` of ``(cell_x, cell_y, time)`` integer tuples.
    """
    cells = set()
    for t, s in enumerate(imagined_states):
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        cx = int(np.floor(float(s[EGO_X]) / discretization))
        cy = int(np.floor(float(s[EGO_Y]) / discretization))
        cells.add((cx, cy, t))
    return cells


def jaccard_diversity(cells_a, cells_b):
    """Jaccard overlap of two conflict-cell sets: |A n B| / |A u B| in [0, 1].

    Higher = more overlap = *less* diverse. Empty union returns 0.0.
    """
    if not cells_a and not cells_b:
        return 0.0
    union = len(cells_a | cells_b)
    if union == 0:
        return 0.0
    return len(cells_a & cells_b) / union


def trajectory_distance_diversity(waypoints_a, waypoints_b, sigma_p=1.0):
    """RBF similarity of two imagined position sequences.

    Returns ``exp(-(1/H) * sum_i ||p_a[i] - p_b[i]||^2 / sigma_p)`` over the
    overlapping horizon ``H``. Close to 1 when the trajectories are similar,
    toward 0 as they diverge. (This is the trajectory-space alternative to the
    cell-based ``jaccard_diversity``.)
    """
    a = np.asarray(waypoints_a, dtype=np.float32).reshape(len(waypoints_a), -1)
    b = np.asarray(waypoints_b, dtype=np.float32).reshape(len(waypoints_b), -1)
    h = min(a.shape[0], b.shape[0])
    if h == 0:
        return 0.0
    sq = np.sum((a[:h] - b[:h]) ** 2, axis=-1)
    return float(np.exp(-(sq.mean()) / max(sigma_p, 1e-6)))
