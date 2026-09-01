import unittest

import numpy as np

from scripts.dreamer_online_rl_batch_update import bounded_rewards
from scripts.run_dreamer_curriculum_training import comparison_gate


class BoundedRewardTests(unittest.TestCase):
    def test_long_episode_has_fixed_clean_terminal_scale(self):
        raw = np.linspace(-200.0, 40.0, 4000, dtype=np.float32)
        result = bounded_rewards(raw, "clean_success", {"route_score": 100.0})
        self.assertAlmostEqual(float(result.sum()), 8.0, delta=1e-3)
        self.assertLessEqual(float(np.max(np.abs(result[:-1]))), 2.0)

    def test_collision_does_not_numerically_destroy_batch(self):
        raw = np.full(6000, -500.0, dtype=np.float32)
        result = bounded_rewards(raw, "collision", {"route_score": 20.0})
        self.assertAlmostEqual(float(result.sum()), -10.0, delta=1e-3)
        self.assertEqual(float(result[-1]), -10.0)

    def test_justified_wait_is_positive_but_below_success(self):
        raw = np.sin(np.arange(1000, dtype=np.float32)) * 80.0
        result = bounded_rewards(raw, "justified_wait", {"route_score": 10.0})
        self.assertAlmostEqual(float(result.sum()), 2.0, delta=1e-3)


class FrozenPromotionGateTests(unittest.TestCase):
    def baseline(self):
        return {
            "runs": 8,
            "unsafe_events": 1.0,
            "collisions": 1.0,
            "blocked": 1.0,
            "mean_route": 80.0,
            "mean_driving": 70.0,
            "success": 5.0,
        }

    def test_rejects_a_single_collision_regression(self):
        candidate = {**self.baseline(), "collisions": 2.0, "unsafe_events": 2.0}
        gate = comparison_gate(self.baseline(), candidate, final=True)
        self.assertFalse(gate["approved"])
        self.assertTrue(any("collision" in reason for reason in gate["reasons"]))

    def test_accepts_non_regressing_candidate(self):
        candidate = {
            **self.baseline(),
            "unsafe_events": 0.0,
            "collisions": 0.0,
            "blocked": 0.0,
            "mean_route": 85.0,
            "mean_driving": 76.0,
            "success": 6.0,
        }
        gate = comparison_gate(self.baseline(), candidate, final=True)
        self.assertTrue(gate["approved"], gate["reasons"])

    def test_rejects_an_equal_all_blocked_candidate(self):
        native = {
            "runs": 8,
            "unsafe_events": 0.0,
            "collisions": 0.0,
            "blocked": 8.0,
            "mean_route": 0.0,
            "mean_driving": 0.0,
            "success": 0.0,
        }
        gate = comparison_gate(native, dict(native), final=True)
        self.assertFalse(gate["approved"])
        self.assertTrue(any("strict" in reason for reason in gate["reasons"]))

    def test_accepts_a_safe_targeted_improvement_over_native(self):
        native = self.baseline()
        candidate = {
            **native,
            "blocked": 0.0,
            "mean_route": 84.0,
            "mean_driving": 74.0,
            "success": 6.0,
        }
        gate = comparison_gate(native, candidate, final=True)
        self.assertTrue(gate["approved"], gate["reasons"])
        self.assertEqual(
            gate["comparison"], "simlingo_plus_dreamer_vs_native_simlingo"
        )


if __name__ == "__main__":
    unittest.main()
