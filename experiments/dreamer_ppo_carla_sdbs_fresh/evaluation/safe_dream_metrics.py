"""SAFE-DREAM aligned post-hoc safety metrics.

Reads a per-variant training-log CSV (baseline.csv / dreamer.csv / sdbs.csv)
produced by :class:`training.logger.Logger` and computes the SAFE-DREAM metrics
that are genuinely applicable to this world-model + RL system. Pure post-hoc
analysis: no training loop, no CARLA, no pandas (plain ``csv`` module, matching
the rest of the project).

Framework
---------
Based on Prof. Ben Yahia's SAFE-DREAM framework (July 2026). This module covers:

  * Section 3.2 — Decision Quality / Safety KPIs (collision & near-collision
    rates, TTC margins, safety distance, safety-envelope violations,
    disengagement rate, traffic-rule compliance, driving comfort);
  * Section 4 — novel Dreaming-specific KPIs (counterfactual coverage, future
    diversity, unsafe-maneuver rejection rate, risk-prediction accuracy,
    hazard-anticipation horizon), applicable only to the dreaming variants.

SAFE-DREAM dimensions that require capabilities this repo does not have
(perception stack, semantic/generative world model, LLM reasoning, formal
scenario DB) are **documented, not silently skipped** — see the
``not_applicable`` list in :meth:`generate_report`.

Distance normalization caveat
-----------------------------
Collision Rate (CR), Near-Collision Rate (NCR), and Disengagement Rate (DR) are
normalized "per million km". Exact traveled distance is **not tracked** yet, so
``distance_per_episode_km`` (default 0.15 km/episode) is an ESTIMATE used purely
for normalization. These three rates are therefore approximate until real
odometry is logged; every other metric is exact.
"""
import csv
import math

import numpy as np

MILLION_KM = 1_000_000.0

# SAFE-DREAM dimensions/metrics deliberately out of scope for this system.
NOT_APPLICABLE = [
    "Perception Quality — no perception stack, CARLA ground truth used",
    "Semantic World Model — world model is not semantic (no scene graph)",
    "Reasoning Quality — no LLM/causal reasoning module",
    "Hallucination Rate — no generative/visual world model",
    "Physical Realism vs real world — no real-world comparison data",
    "EU-CEM Traceability/Scenario Coverage — no formal scenario DB",
]


class SafeDreamEvaluator:
    """Compute SAFE-DREAM metrics from one variant's training-log CSV.

    Parameters
    ----------
    csv_path : str
        Path to a training-log CSV written by ``training.logger.Logger``.
    variant_name : str
        Human label ("baseline" / "dreamer" / "sdbs"); "baseline" additionally
        forces the dreaming-specific metrics to be reported N/A.
    distance_per_episode_km : float
        Estimated distance travelled per episode, used ONLY to normalize CR /
        NCR / DR to a per-million-km basis (see module docstring).
    """

    def __init__(self, csv_path, variant_name, distance_per_episode_km=0.15):
        self.csv_path = csv_path
        self.variant_name = variant_name
        self.distance_per_episode_km = float(distance_per_episode_km)
        self.rows, self.fieldnames = self._read_csv(csv_path)
        self.n_episodes = len(self.rows)

    # ------------------------------------------------------------------ #
    # CSV plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_csv(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        return rows, fieldnames

    def _has_column(self, name):
        return name in self.fieldnames

    def _col(self, name):
        """Float values for a column, skipping missing / blank / non-numeric."""
        out = []
        for r in self.rows:
            v = r.get(name)
            if v is None or v == "":
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            out.append(fv)
        return out

    def _sum(self, name):
        return float(sum(self._col(name)))

    def _distance_km(self):
        return self.distance_per_episode_km * self.n_episodes

    def _per_million_km(self, count):
        dist_km = self._distance_km()
        if dist_km <= 0:
            return 0.0
        return float(count) / dist_km * MILLION_KM

    def is_baseline(self):
        """True when this variant has no beam-search counterfactuals to report."""
        if str(self.variant_name).lower() == "baseline":
            return True
        vals = self._col("n_candidates_evaluated")
        return (not vals) or sum(vals) == 0

    def _has_risk_predictions(self):
        vals = self._col("chosen_risk_prediction")
        return bool(vals) and any(v != 0.0 for v in vals)

    # ------------------------------------------------------------------ #
    # Section 3.2 — Decision Quality / Safety KPIs
    # ------------------------------------------------------------------ #
    def collision_rate(self):
        """CR: (VRU + vehicle) collisions per million km (distance estimated)."""
        total = self._sum("vru_collisions") + self._sum("vehicle_collisions")
        return self._per_million_km(total)

    def near_collision_rate(self, ttc_threshold=1.5):
        """NCR: episodes with min TTC below ``ttc_threshold`` (s), per million km."""
        vals = self._col("min_ttc_this_episode")
        count = sum(1 for v in vals if v < ttc_threshold)
        return self._per_million_km(count)

    def min_ttc_stats(self):
        """Overall min and 5th-percentile of per-episode minimum TTC."""
        vals = self._col("min_ttc_this_episode")
        if not vals:
            return {"min": None, "p5": None}
        return {"min": float(min(vals)),
                "p5": float(np.percentile(vals, 5))}

    def min_safety_distance(self):
        """Overall minimum of the per-episode minimum safety distance."""
        vals = self._col("min_safety_distance_this_episode")
        return float(min(vals)) if vals else None

    def safety_envelope_violations(self):
        """Total safety-envelope intrusions across all episodes."""
        return int(round(self._sum("safety_envelope_violations")))

    def disengagement_rate(self):
        """DR: disengagement-equivalent events per million km (distance estimated)."""
        return self._per_million_km(self._sum("disengagement_count"))

    def traffic_rule_compliance(self):
        """TRC (%): 100 * (1 - rule_violations / total_rule_events).

        Requires explicit ``red_light_violations`` / ``stop_sign_violations`` /
        ``total_rule_events`` columns, which the current logger does not emit
        (only a summed ``r_rules`` reward exists, not per-event counts). Returns
        ``None`` with a warning when those columns are absent.
        """
        needed = ("red_light_violations", "stop_sign_violations",
                  "total_rule_events")
        if not all(self._has_column(c) for c in needed):
            print(f"[SAFE-DREAM] traffic_rule_compliance N/A for "
                  f"'{self.variant_name}': per-event rule-violation counts are "
                  f"not logged (only summed r_rules reward).")
            return None
        violations = (self._sum("red_light_violations")
                      + self._sum("stop_sign_violations"))
        total = self._sum("total_rule_events")
        if total <= 0:
            return 100.0
        return float((1.0 - violations / total) * 100.0)

    def driving_comfort(self):
        """Mean and max ego jerk across episodes (lower = more comfortable)."""
        jm = self._col("jerk_mean")
        jx = self._col("jerk_max")
        return {"mean_jerk": float(np.mean(jm)) if jm else None,
                "max_jerk": float(max(jx)) if jx else None}

    # ------------------------------------------------------------------ #
    # Section 4 — Novel Dreaming-specific KPIs (dreamer / sdbs only)
    # ------------------------------------------------------------------ #
    def counterfactual_coverage(self):
        """CC: mean number of imagined candidates evaluated per decision."""
        if self.is_baseline():
            print(f"[SAFE-DREAM] counterfactual_coverage N/A for "
                  f"'{self.variant_name}': no beam search / imagined candidates.")
            return None
        vals = self._col("n_candidates_evaluated")
        return float(np.mean(vals)) if vals else None

    def future_diversity(self):
        """FD: mean softmax entropy of per-decision candidate scores."""
        if self.is_baseline():
            print(f"[SAFE-DREAM] future_diversity N/A for "
                  f"'{self.variant_name}': no beam search / imagined candidates.")
            return None
        vals = self._col("candidate_score_entropy")
        return float(np.mean(vals)) if vals else None

    def unsafe_maneuver_rejection_rate(self):
        """UMRR (%): rejected-unsafe candidates / all evaluated candidates."""
        if self.is_baseline():
            print(f"[SAFE-DREAM] unsafe_maneuver_rejection_rate N/A for "
                  f"'{self.variant_name}': no beam search / imagined candidates.")
            return None
        evaluated = self._sum("n_candidates_evaluated")
        rejected = self._sum("n_candidates_rejected_unsafe")
        if evaluated <= 0:
            return None
        return float(rejected / evaluated * 100.0)

    def risk_prediction_accuracy(self):
        """RPA: 1 - mean|chosen_risk_prediction - observed_risk|.

        ``observed_risk`` is a PROXY, not ground truth: 1.0 if a collision or
        near-miss occurred that episode; else the (clamped-to-[0,1]) ``r_vru``
        reward-component magnitude if present; else 0.0. Meaningful only for the
        dreaming variants that actually log ``chosen_risk_prediction``.
        """
        if not self._has_risk_predictions():
            print(f"[SAFE-DREAM] risk_prediction_accuracy N/A for "
                  f"'{self.variant_name}': no chosen_risk_prediction logged.")
            return None
        errors = []
        has_rvru = self._has_column("r_vru")
        for r in self.rows:
            chosen = self._row_float(r, "chosen_risk_prediction")
            if chosen is None:
                continue
            observed = self._observed_risk(r, has_rvru)
            errors.append(abs(chosen - observed))
        if not errors:
            return None
        return float(1.0 - np.mean(errors))

    def hazard_anticipation_horizon(self, risk_rise_threshold=0.3):
        """HAH: mean anticipation lead for hazard episodes (coarse proxy).

        The true metric is the per-timestep gap between the predicted risk first
        crossing ``risk_rise_threshold`` and the actual near-miss/collision. The
        per-episode CSV holds no per-timestep risk trace, so this is a documented
        PROXY: for hazard episodes (collision or near-miss) whose episode-mean
        ``chosen_risk_prediction`` crossed the threshold, the lead is proxied by
        that episode's min TTC (seconds-to-conflict at closest approach).
        Returns ``None`` (with a warning) when fewer than 3 hazard episodes exist
        or when this variant logs no risk predictions.
        """
        if not self._has_risk_predictions():
            print(f"[SAFE-DREAM] hazard_anticipation_horizon N/A for "
                  f"'{self.variant_name}': no chosen_risk_prediction logged.")
            return None
        hazard_leads = []
        n_hazard = 0
        for r in self.rows:
            if not self._is_hazard_episode(r):
                continue
            n_hazard += 1
            chosen = self._row_float(r, "chosen_risk_prediction")
            min_ttc = self._row_float(r, "min_ttc_this_episode")
            if chosen is not None and chosen >= risk_rise_threshold \
                    and min_ttc is not None:
                hazard_leads.append(min_ttc)
        if n_hazard < 3:
            print(f"[SAFE-DREAM] hazard_anticipation_horizon N/A for "
                  f"'{self.variant_name}': only {n_hazard} hazard episode(s) "
                  f"(need >= 3).")
            return None
        if not hazard_leads:
            return 0.0
        return float(np.mean(hazard_leads))

    # ------------------------------------------------------------------ #
    # Risk-proxy helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_float(row, name):
        v = row.get(name)
        if v is None or v == "":
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(fv) else fv

    def _is_hazard_episode(self, row):
        vru_c = self._row_float(row, "vru_collisions") or 0.0
        veh_c = self._row_float(row, "vehicle_collisions") or 0.0
        min_ttc = self._row_float(row, "min_ttc_this_episode")
        near_miss = (min_ttc is not None and min_ttc < 1.5)
        return (vru_c > 0) or (veh_c > 0) or near_miss

    def _observed_risk(self, row, has_rvru):
        if self._is_hazard_episode(row):
            return 1.0
        if has_rvru:
            rvru = self._row_float(row, "r_vru")
            if rvru is not None:
                return float(min(1.0, max(0.0, abs(rvru))))
        return 0.0

    # ------------------------------------------------------------------ #
    # Safety gain (paper sign convention: SG = Risk_reactive - Risk_dreaming)
    # ------------------------------------------------------------------ #
    def _aggregate_risk_proxy(self):
        min_ttc = self.min_ttc_stats()["min"]
        min_ttc = min_ttc if min_ttc is not None else 0.1
        return (0.5 * self.collision_rate()
                + 0.3 * self.near_collision_rate()
                + 0.2 * (1.0 / max(min_ttc, 0.1)))

    def safety_gain(self, other_evaluator):
        """SG = other's aggregate risk proxy - self's aggregate risk proxy.

        Positive => *self* is safer than ``other_evaluator``. Intended use:
        ``SafeDreamEvaluator('sdbs.csv','sdbs').safety_gain(
             SafeDreamEvaluator('baseline.csv','baseline'))`` yields
        ``Risk_reactive - Risk_dreaming`` (paper convention).
        """
        return float(other_evaluator._aggregate_risk_proxy()
                     - self._aggregate_risk_proxy())

    # ------------------------------------------------------------------ #
    # Full report
    # ------------------------------------------------------------------ #
    def generate_report(self):
        """Every computed metric (None where not applicable) + skipped dims."""
        return {
            "variant": self.variant_name,
            "n_episodes": self.n_episodes,
            "distance_per_episode_km": self.distance_per_episode_km,
            "distance_assumption": (
                f"CR/NCR/DR normalized using an ESTIMATED "
                f"{self.distance_per_episode_km} km/episode "
                f"(exact odometry not tracked yet)."),
            # Section 3.2
            "collision_rate": self.collision_rate(),
            "near_collision_rate": self.near_collision_rate(),
            "min_ttc": self.min_ttc_stats(),
            "min_safety_distance": self.min_safety_distance(),
            "safety_envelope_violations": self.safety_envelope_violations(),
            "disengagement_rate": self.disengagement_rate(),
            "traffic_rule_compliance": self.traffic_rule_compliance(),
            "driving_comfort": self.driving_comfort(),
            # Section 4 (dreaming variants only)
            "counterfactual_coverage": self.counterfactual_coverage(),
            "future_diversity": self.future_diversity(),
            "unsafe_maneuver_rejection_rate": self.unsafe_maneuver_rejection_rate(),
            "risk_prediction_accuracy": self.risk_prediction_accuracy(),
            "hazard_anticipation_horizon": self.hazard_anticipation_horizon(),
            # Documented out-of-scope SAFE-DREAM dimensions.
            "not_applicable": list(NOT_APPLICABLE),
        }
