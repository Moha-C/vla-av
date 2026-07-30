"""Per-episode SAFE-DREAM safety accumulator (additive telemetry only).

Collects the raw per-step signals the SAFE-DREAM metrics need — TTC margins,
safety distances, safety-envelope intrusions, ego jerk, disengagement-equivalent
events, and (for beam-search variants) per-decision candidate diagnostics — and
condenses them into the flat per-episode dict written by ``training.logger``.

Shared by the PPO baseline, the plain dreamer, and the S-DBS trainer so the CSV
schema stays consistent across variants. Beam-search fields stay at 0 for the
baseline (it never calls :meth:`record_plan_meta`), matching the documented
"0/NaN for baseline" behaviour of the SAFE-DREAM evaluator.
"""
import numpy as np

# Ego block layout (carla_env: ego = x, y, speed, heading, acc_x, acc_y).
EGO_ACC_X = 4
EGO_ACC_Y = 5

# Sentinel used when an episode observed no VRU/vehicle at all, so an absent
# conflict is never mistaken for a dangerously low TTC / distance.
NO_CONFLICT = 999.0


class EpisodeSafetyAccumulator:
    """Accumulate SAFE-DREAM safety telemetry for a single episode."""

    def __init__(self, config):
        self.dt = 1.0 / max(1, int(getattr(config, "fps", 10)))
        self.clearance = float(getattr(config, "min_lane_change_clearance", 2.0))
        self.reset()

    def reset(self):
        self.ttc_values = []            # every VRU + vehicle TTC this episode
        self.min_distance = float("inf")
        self.envelope_violations = 0    # timesteps within `clearance` of any agent
        self.prev_acc = None
        self.jerks = []
        self.disengagements = 0
        # Beam-search per-decision diagnostics (dreamer / sdbs only).
        self._n_eval = []
        self._n_rej = []
        self._score_means = []
        self._score_stds = []
        self._score_entropies = []
        self._risk_means = []
        self._chosen_risks = []

    # ------------------------------------------------------------------ #
    def record_step(self, info, state):
        """Fold one environment step's TTC / distance / jerk signals in."""
        info = info or {}
        for ttc in (list(info.get("vru_ttc_list", []))
                    + list(info.get("vehicle_ttc_list", []))):
            if ttc is not None and float(ttc) > 0:
                self.ttc_values.append(float(ttc))

        dists = (list(info.get("vru_distance_list", []))
                 + list(info.get("vehicle_distance_list", [])))
        dists = [float(d) for d in dists if d is not None and float(d) > 0]
        if dists:
            step_min = min(dists)
            self.min_distance = min(self.min_distance, step_min)
            if step_min < self.clearance:      # one intrusion per timestep
                self.envelope_violations += 1

        if state is not None:
            s = np.asarray(state, dtype=np.float32).reshape(-1)
            if s.shape[0] > EGO_ACC_Y:
                acc = np.array([s[EGO_ACC_X], s[EGO_ACC_Y]], dtype=np.float32)
                if self.prev_acc is not None:
                    jerk = float(np.linalg.norm(acc - self.prev_acc) / self.dt)
                    self.jerks.append(jerk)
                self.prev_acc = acc

    def record_disengagement(self):
        """One blocked/overridden unsafe action = one disengagement-equivalent."""
        self.disengagements += 1

    def record_plan_meta(self, meta):
        """Fold one S-DBS ``plan()`` metadata dict's beam diagnostics in."""
        meta = meta or {}
        self._n_eval.append(int(meta.get("n_candidates_evaluated", 0)))
        self._n_rej.append(int(meta.get("n_candidates_rejected_unsafe", 0)))
        scores = list(meta.get("candidate_scores", []) or [])
        risks = list(meta.get("candidate_risk_predictions", []) or [])
        if scores:
            self._score_means.append(float(np.mean(scores)))
            self._score_stds.append(float(np.std(scores)))
            self._score_entropies.append(self._softmax_entropy(scores))
        if risks:
            self._risk_means.append(float(np.mean(risks)))
        chosen = meta.get("chosen_risk_prediction")
        if chosen is not None:
            self._chosen_risks.append(float(chosen))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax_entropy(scores):
        """Shannon entropy of the softmax over one decision's candidate scores."""
        s = np.asarray(scores, dtype=np.float64)
        if s.size <= 1:
            return 0.0
        s = s - s.max()                        # numerical stability
        p = np.exp(s)
        p = p / p.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    def summarize(self):
        """Flat per-episode dict of SAFE-DREAM columns for ``training.logger``."""
        min_ttc = min(self.ttc_values) if self.ttc_values else NO_CONFLICT
        p5_ttc = (float(np.percentile(self.ttc_values, 5))
                  if self.ttc_values else NO_CONFLICT)
        min_dist = (self.min_distance if self.min_distance < float("inf")
                    else NO_CONFLICT)
        return {
            # Beam-search diagnostics (0 for the baseline variant).
            "n_candidates_evaluated": self._mean(self._n_eval),
            "n_candidates_rejected_unsafe": self._mean(self._n_rej),
            "candidate_score_mean": self._mean(self._score_means),
            "candidate_score_std": self._mean(self._score_stds),
            "candidate_score_entropy": self._mean(self._score_entropies),
            "mean_risk_prediction": self._mean(self._risk_means),
            "chosen_risk_prediction": self._mean(self._chosen_risks),
            # Applicable to every variant.
            "min_ttc_this_episode": min_ttc,
            "p5_ttc_this_episode": p5_ttc,
            "min_safety_distance_this_episode": min_dist,
            "safety_envelope_violations": self.envelope_violations,
            "jerk_mean": self._mean(self.jerks),
            "jerk_max": float(max(self.jerks)) if self.jerks else 0.0,
            "disengagement_count": self.disengagements,
        }
