import json
import tempfile
import unittest
from pathlib import Path

from scripts.curate_dreamer_rl_run import parse_result, validate_clean_overtake


def write_result(path: Path, *, collisions=0, offroad=0, route=100, min_speed=None) -> None:
    path.write_text(
        json.dumps({
            "_checkpoint": {
                "records": [{
                    "status": "Completed",
                    "scenario_name": "Accident_1",
                    "scores": {
                        "score_route": route,
                        "score_composed": route,
                        "score_penalty": 1.0,
                    },
                    "infractions": {
                        "collisions_layout": [],
                        "collisions_pedestrian": [],
                        "collisions_vehicle": ["collision"] * collisions,
                        "outside_route_lanes": ["offroad"] * offroad,
                        "red_light": [],
                        "vehicle_blocked": [],
                        "min_speed_infractions": min_speed or [],
                    },
                    "meta": {"duration_game": 10, "duration_system": 20},
                }]
            }
        }),
        encoding="utf-8",
    )


class CurateDreamerRunTest(unittest.TestCase):
    def test_min_speed_only_failure_is_still_clean_overtake(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            write_result(result, min_speed=["slow"] * 4)
            _, metrics = parse_result(result)
            validate_clean_overtake(metrics)
            self.assertEqual(metrics["route_completion"], 100)
            self.assertEqual(metrics["min_speed_infractions"], 4)

    def test_collision_is_rejected(self):
        self._assert_unsafe_rejected(collisions=1)

    def test_offroad_is_rejected(self):
        self._assert_unsafe_rejected(offroad=1)

    def _assert_unsafe_rejected(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            write_result(result, **kwargs)
            _, metrics = parse_result(result)
            with self.assertRaisesRegex(ValueError, "not a clean positive overtake"):
                validate_clean_overtake(metrics)


if __name__ == "__main__":
    unittest.main()
