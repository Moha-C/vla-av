"""Risk-aware curriculum learning with prioritized scenario replay.

Three coupled pieces:

  * ``ScenarioBank``  - the catalogue of scenarios with difficulty labels and
    rolling success rates.
  * ``SumTree`` / ``PrioritizedScenarioReplayer`` - prioritized experience
    replay over *scenarios*: scenarios the agent keeps failing (collisions,
    near-misses, TTC violations, progress deficit) get sampled more often, in
    O(log N) via a binary sum tree.
  * ``RiskAwareCurriculum`` - orchestrates three stages (empty roads ->
    crossings/yielding -> adversarial greedy traps), unlocking the next stage
    once the current one is reliably solved.
"""
from collections import deque

import numpy as np


# ---------------------------------------------------------------------- #
# ScenarioBank
# ---------------------------------------------------------------------- #
class ScenarioBank:
    """A collection of scenarios with difficulty labels and success records."""

    def __init__(self, success_window=20):
        self.scenarios = []
        self.difficulty = {}              # scenario_id -> float in [0, 1]
        self.success_record = {}          # scenario_id -> rolling success rate
        self._success_window = success_window
        self._windows = {}                # scenario_id -> deque of 0/1

    def add(self, scenario_id, difficulty):
        if scenario_id not in self.difficulty:
            self.scenarios.append(scenario_id)
            self._windows[scenario_id] = deque(maxlen=self._success_window)
            self.success_record[scenario_id] = 0.0
        self.difficulty[scenario_id] = float(difficulty)

    def record_success(self, scenario_id, success):
        if scenario_id not in self._windows:
            self.add(scenario_id, 0.5)
        self._windows[scenario_id].append(1.0 if success else 0.0)
        w = self._windows[scenario_id]
        self.success_record[scenario_id] = float(np.mean(w)) if w else 0.0


# ---------------------------------------------------------------------- #
# SumTree
# ---------------------------------------------------------------------- #
class SumTree:
    """Binary tree where each leaf holds a scenario priority and each internal
    node holds the sum of its children. Enables O(log N) priority sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity     # leaf index -> scenario_id
        self.id_to_leaf = {}              # scenario_id -> leaf index
        self.write = 0
        self.n = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _update_leaf(self, leaf, priority):
        idx = leaf + self.capacity - 1
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, scenario_id, priority):
        if scenario_id in self.id_to_leaf:
            self.update_priority(scenario_id, priority)
            return
        leaf = self.write
        old = self.data[leaf]
        if old is not None and old in self.id_to_leaf:
            del self.id_to_leaf[old]      # evict the overwritten scenario
        self.data[leaf] = scenario_id
        self.id_to_leaf[scenario_id] = leaf
        self._update_leaf(leaf, float(priority))
        self.write = (self.write + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def update_priority(self, scenario_id, new_priority):
        if scenario_id not in self.id_to_leaf:
            self.add(scenario_id, new_priority)
            return
        self._update_leaf(self.id_to_leaf[scenario_id], float(new_priority))

    def total_priority(self):
        return float(self.tree[0])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):        # reached a leaf
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        return self.data[idx - (self.capacity - 1)]

    def sample(self, batch_size):
        ids = [d for d in self.data if d is not None]
        total = self.total_priority()
        if total <= 0 or not ids:         # uniform fallback before any priorities
            if not ids:
                return []
            return list(np.random.choice(ids, size=min(batch_size, len(ids))))
        out = []
        for _ in range(batch_size):
            s = np.random.uniform(0, total)
            sid = self.get(s)
            out.append(sid if sid is not None else ids[0])
        return out


# ---------------------------------------------------------------------- #
# PrioritizedScenarioReplayer
# ---------------------------------------------------------------------- #
class PrioritizedScenarioReplayer:
    """Samples scenarios from a bank based on how badly the agent failed them."""

    def __init__(self, scenario_bank, capacity=1000, alpha=0.6, epsilon=1e-6,
                 w_collision=5.0, w_near=1.0, w_ttc=0.5, w_progress=1.0):
        self.bank = scenario_bank
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.w_c, self.w_n, self.w_ttc, self.w_p = (
            w_collision, w_near, w_ttc, w_progress
        )
        self.last_error = {}
        # Seed priorities from difficulty so harder scenarios start hotter.
        for sid in scenario_bank.scenarios:
            base = (scenario_bank.difficulty.get(sid, 0.5) + epsilon) ** alpha
            self.tree.add(sid, base)

    def record_episode(self, scenario_id, n_collisions, n_near_misses,
                       n_ttc_violations, progress_deficit):
        """Update a scenario's priority from its failure signal."""
        e = (self.w_c * float(n_collisions)
             + self.w_n * float(n_near_misses)
             + self.w_ttc * float(n_ttc_violations)
             + self.w_p * float(progress_deficit))
        self.last_error[scenario_id] = e
        priority = (abs(e) + self.epsilon) ** self.alpha
        self.tree.update_priority(scenario_id, priority)

    def sample_scenarios(self, batch_size):
        return self.tree.sample(batch_size)

    def unlock_next_stage(self, stage_scenarios, threshold):
        """True if the mean success over ``stage_scenarios`` clears ``threshold``."""
        if not stage_scenarios:
            return False
        rates = [self.bank.success_record.get(s, 0.0) for s in stage_scenarios]
        return float(np.mean(rates)) >= threshold


# ---------------------------------------------------------------------- #
# RiskAwareCurriculum
# ---------------------------------------------------------------------- #
class RiskAwareCurriculum:
    """Orchestrates training across 3 stages of increasing risk."""

    # (difficulty low, high) per stage.
    STAGE_DIFFICULTY = [(0.10, 0.30), (0.40, 0.60), (0.75, 0.95)]

    def __init__(self, config):
        self.config = config
        self.n_stages = int(getattr(config, "curriculum_stages", 3))
        self.unlock_threshold = float(getattr(config, "stage_unlock_threshold", 0.85))
        self.scenarios_per_stage = int(getattr(config, "scenarios_per_stage", 6))

        self.scenario_bank = ScenarioBank()
        self.stage_scenarios = self._build_stages()
        self._stage = 1
        self._active = set(self.stage_scenarios[0])

        self.replayer = PrioritizedScenarioReplayer(
            self.scenario_bank,
            capacity=int(getattr(config, "scenario_bank_capacity", 1000)),
            alpha=float(getattr(config, "per_alpha", 0.6)),
            epsilon=float(getattr(config, "per_epsilon", 1e-6)),
        )

    def _build_stages(self):
        stages = []
        per = max(1, self.scenarios_per_stage)
        for st in range(self.n_stages):
            lo, hi = self.STAGE_DIFFICULTY[min(st, len(self.STAGE_DIFFICULTY) - 1)]
            ids = []
            for i in range(per):
                sid = f"stage{st + 1}_{i:03d}"
                d = lo + (hi - lo) * (i / max(1, per - 1))
                self.scenario_bank.add(sid, d)
                ids.append(sid)
            stages.append(ids)
        return stages

    # -- queries ------------------------------------------------------- #
    def current_stage(self):
        return self._stage

    def get_active_scenarios(self):
        return sorted(self._active)

    # -- updates ------------------------------------------------------- #
    def record_rollout(self, scenario_id, stats):
        """Record one episode's outcome for curriculum + prioritized replay."""
        collisions = int(stats.get("vru_collisions", 0))
        completion = float(stats.get("route_completion", stats.get("progress", 0.0)))
        success = (collisions == 0 and completion >= 0.5)
        self.scenario_bank.record_success(scenario_id, success)
        self.replayer.record_episode(
            scenario_id=scenario_id,
            n_collisions=collisions,
            n_near_misses=int(stats.get("near_misses", 0)),
            n_ttc_violations=int(stats.get("ttc_violations", 0)),
            progress_deficit=max(
                0.0, float(getattr(self.config, "goal_distance", 1.0)) - completion
            ),
        )

    def _stage_success(self):
        ids = self.stage_scenarios[self._stage - 1]
        rates = [self.scenario_bank.success_record.get(i, 0.0) for i in ids]
        return float(np.mean(rates)) if rates else 0.0

    def should_advance_stage(self):
        if self._stage >= self.n_stages:
            return False
        return self._stage_success() >= self.unlock_threshold

    def advance_to_next_stage(self):
        if self._stage >= self.n_stages:
            return
        self._stage += 1
        new = self.stage_scenarios[self._stage - 1]
        self._active.update(new)
        for sid in new:
            base = (self.scenario_bank.difficulty[sid]
                    + self.replayer.epsilon) ** self.replayer.alpha
            self.replayer.tree.add(sid, base)
