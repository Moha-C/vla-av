"""Simple CSV logger shared by the PPO baseline and Dreamer-PPO trainers."""
import csv
import os


class Logger:
    """Append-style CSV logger.

    Writes one row per episode to ``<log_dir>/<filename>`` with the columns:
        episode, return, r_progress, r_vru, r_collision, r_comfort, r_rules,
        ppo_loss, vf_loss, entropy, loss_wm, wm_state_err, wm_risk_err,
        dreaming_active, dreaming_steps,
        vru_collisions, lane_departures, route_completion,
        eval_return, eval_vru_collisions, eval_near_misses,
        eval_route_completion, eval_lane_departures
    """

    FIELDS = [
        "episode", "return",
        "r_progress", "r_vru", "r_collision", "r_comfort", "r_rules",
        "ppo_loss", "vf_loss", "entropy",
        "loss_wm", "wm_state_err", "wm_risk_err",
        "dreaming_active", "dreaming_steps",
        "vru_collisions", "lane_departures", "route_completion",
        "eval_return", "eval_vru_collisions", "eval_near_misses",
        "eval_route_completion", "eval_lane_departures",
    ]

    def __init__(self, log_dir="logs", filename="training_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, episode, stats):
        """Write one episode row. Missing fields default to 0."""
        row = {"episode": int(episode)}
        for key in self.FIELDS[1:]:
            row[key] = stats.get(key, 0)
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()
