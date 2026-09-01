import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import dreamer_rssm_data_v4 as data_v4
from scripts import train_dreamer_rssm_v2 as v2
from scripts import train_dreamer_rssm_v4 as train_v4


def status(index: int) -> dict:
    state = [0.0] * 28
    state[0] = float(index) * 0.1
    state[2] = 2.0
    return {
        "timestamp": 100.0 + index,
        "mode": "rl_noguard",
        "state_vector": state,
        "base_action": {
            "steer": 0.0,
            "throttle": 0.2,
            "brake": 0.0,
            "intervention": 0.0,
        },
        "chosen_action": {
            "steer": 0.0,
            "throttle": 0.2,
            "brake": 0.0,
            "intervention": 0.0,
        },
        "front_vehicle_m": 30.0,
        "nearest_walker_m": 80.0,
        "nearest_bike_m": 80.0,
        "nearby_vehicles": [],
        "traffic_light": "none",
        "blocked_ticks": 0,
    }


class RSSMDataV4Test(unittest.TestCase):
    def test_exact_collision_cuts_post_impact_rows_and_labels_impact(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            trace = run / "trace.jsonl"
            rows = [
                {
                    "collector_time": 100.0 + index,
                    "route_id": "148",
                    "seed": "7",
                    "status": status(index),
                }
                for index in range(40)
            ]
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (run / "episode.json").write_text(
                json.dumps({
                    "metrics": {
                        "collisions": 1,
                        "vehicle_collisions": 1,
                        "offroad": 0,
                        "blocked": 0,
                    }
                }),
                encoding="utf-8",
            )
            (run / "collision_events.jsonl").write_text(
                json.dumps({
                    "event": "collision",
                    "wall_time": 125.2,
                    "collision_kind": "vehicle",
                    "source": "unit_test",
                }) + "\n",
                encoding="utf-8",
            )

            episodes, audit = data_v4.load_episodes([trace], sequence_length=8)

            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            self.assertEqual(episode.transitions, 26)
            self.assertEqual(float(episode.events[-1, 0]), 1.0)
            self.assertEqual(float(episode.continuation[-1]), 0.0)
            self.assertLessEqual(float(episode.rewards[-1]), -15.0)
            self.assertGreater(float(episode.risks[-10]), 0.2)
            self.assertEqual(audit[0]["collision_source"], "unit_test")
            self.assertEqual(audit[0]["impact_transition_index"], 25)

    def test_route_seed_split_never_leaks_segments(self):
        def episode(route: str, seed: str, suffix: str) -> v2.Episode:
            transitions = 8
            return v2.Episode(
                key=f"{route}:{seed}:{suffix}",
                route_id=route,
                seed=seed,
                source="test",
                observations=np.zeros((transitions + 1, 49), dtype=np.float32),
                actions=np.zeros((transitions, 4), dtype=np.float32),
                rewards=np.zeros(transitions, dtype=np.float32),
                continuation=np.ones(transitions, dtype=np.float32),
                risks=np.zeros(transitions, dtype=np.float32),
                progress=np.zeros(transitions, dtype=np.float32),
                events=np.zeros((transitions, 5), dtype=np.float32),
                teacher_targets=np.zeros((transitions, 4), dtype=np.float32),
                teacher_mask=np.zeros(transitions, dtype=np.float32),
            )

        episodes = [
            episode("148", "1", "a"),
            episode("148", "1", "b"),
            episode("148", "2", "a"),
            episode("54", "3", "a"),
            episode("93", "4", "a"),
        ]
        training, validation, _ = data_v4.split_route_seed_stratified(
            episodes, seed=11
        )
        training_groups = {(row.route_id, row.seed) for row in training}
        validation_groups = {(row.route_id, row.seed) for row in validation}
        self.assertFalse(training_groups & validation_groups)
        self.assertTrue(training_groups)
        self.assertTrue(validation_groups)

    def test_v4_gate_rejects_world_model_worse_than_persistence(self):
        validation = {
            "1": {"families": {"ego": {"persistence_ratio": 1.20}}},
            "5": {
                "families": {"decision": {"persistence_ratio": 1.80}},
                "changed_decision_persistence_ratio": 2.0,
                "risk_mae": 0.10,
                "event_brier": 0.05,
            },
        }
        collision = {
            "positive_windows": 10,
            "average_precision": 0.8,
            "recall": 0.9,
            "positive_risk_mean": 0.8,
            "negative_risk_mean": 0.1,
        }
        accepted, details = train_v4.quality_gate(validation, collision)
        self.assertFalse(accepted)
        self.assertFalse(details["checks"]["h1_ego_beats_persistence"])
        self.assertFalse(
            details["checks"]["h5_decision_near_or_better_than_persistence"]
        )

    def test_average_precision_rewards_collision_ranking(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.float32)
        good = train_v4.average_precision(
            labels, np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
        )
        bad = train_v4.average_precision(
            labels, np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
        )
        self.assertGreater(good, bad)


if __name__ == "__main__":
    unittest.main()
