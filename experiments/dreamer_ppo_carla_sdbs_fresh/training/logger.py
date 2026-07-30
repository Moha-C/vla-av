"""CSV logger shared by the PPO baseline and Dreamer-PPO trainers.

Tier-3: the schema separates VRU safety (primary) from vehicle safety
(secondary) and tracks lane-change decisions, so a reviewer can read exactly
what happened each episode. Unknown stats default to 0, so older callers keep
working unchanged.
"""
import csv
import os


class Logger:
    """Append-style CSV logger (one row per episode)."""

    FIELDS = [
        "episode", "return",
        # --- VRU safety (PRIMARY) ---
        "vru_collisions", "vru_near_misses", "min_ttc_vru", "avg_distance_to_vru",
        # --- vehicle safety (SECONDARY) ---
        "vehicle_collisions", "vehicle_near_misses", "min_ttc_vehicle",
        "avg_distance_to_vehicle", "rear_incidents",
        # --- lane-change safety ---
        "lane_changes_attempted", "lane_changes_safe",
        "lane_changes_unsafe_prevented", "lane_change_success_rate",
        # --- reward components ---
        "r_progress", "r_vru", "r_collision",
        "r_vehicle_collision", "r_vehicle_proximity", "r_rear_risk",
        "r_comfort", "r_rules",
        # --- planning / losses ---
        "ppo_loss", "vf_loss", "entropy",
        "loss_wm", "wm_state_err", "wm_risk_err",
        "dreaming_active", "dreaming_steps", "defensive_mode_active",
        "avg_planning_latency_ms", "state_dim",
        # --- general ---
        "lane_departures", "route_completion", "episode_duration",
        # --- periodic eval ---
        "eval_return", "eval_vru_collisions", "eval_near_misses",
        "eval_route_completion", "eval_lane_departures",
        # --- SAFE-DREAM safety evaluation (additive; beam fields 0/NaN for
        #     the baseline variant, which has no beam search) ---
        "n_candidates_evaluated", "n_candidates_rejected_unsafe",
        "candidate_score_mean", "candidate_score_std", "candidate_score_entropy",
        "mean_risk_prediction", "chosen_risk_prediction",
        "min_ttc_this_episode", "p5_ttc_this_episode",
        "min_safety_distance_this_episode", "safety_envelope_violations",
        "jerk_mean", "jerk_max", "disengagement_count",
    ]

    # Compact view for the console summary table.
    SUMMARY_COLUMNS = [
        "episode", "vru_collisions", "vru_near_misses",
        "vehicle_collisions", "vehicle_near_misses",
        "lane_changes_safe", "lane_changes_unsafe_prevented",
        "route_completion", "return",
    ]

    def __init__(self, log_dir="logs", filename="training_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.csv_path = self.path        # alias used by tooling/tests
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, episode, stats):
        """Write one episode row. Missing fields default to 0."""
        row = {"episode": int(episode)}
        for key in self.FIELDS[1:]:
            value = stats.get(key, 0)
            row[key] = int(value) if isinstance(value, bool) else value
        self._writer.writerow(row)
        self._file.flush()

    def create_summary_table(self, episodes_range=(0, -1)):
        """Print a compact safety-progression table from the CSV written so far."""
        if not os.path.exists(self.path):
            return
        with open(self.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        lo, hi = episodes_range
        rows = rows[lo:None if hi == -1 else hi]
        if not rows:
            return

        headers = ["Episode", "VRU-Coll", "VRU-NM", "Veh-Coll", "Veh-NM",
                   "LC-Safe", "LC-Blocked", "Route%", "Return"]
        print("\n" + "=" * 84)
        print("TRAINING SUMMARY (VRU vs vehicle safety, lane changes)")
        print("=" * 84)
        print("  ".join(f"{h:>9}" for h in headers))
        print("-" * 84)
        for r in rows:
            vals = [
                r.get("episode", 0),
                r.get("vru_collisions", 0), r.get("vru_near_misses", 0),
                r.get("vehicle_collisions", 0), r.get("vehicle_near_misses", 0),
                r.get("lane_changes_safe", 0), r.get("lane_changes_unsafe_prevented", 0),
                f"{float(r.get('route_completion', 0)) * 100:.0f}",
                f"{float(r.get('return', 0)):.0f}",
            ]
            print("  ".join(f"{str(v):>9}" for v in vals))
        print("=" * 84 + "\n")

    def close(self):
        if not self._file.closed:
            self._file.close()
