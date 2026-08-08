import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "simlingo" / "team_code"))

from dreamer_guard import DreamerGuard  # noqa: E402


class DreamerComplementMappingTest(unittest.TestCase):
    def setUp(self):
        self.guard = DreamerGuard.__new__(DreamerGuard)
        self.guard.config = SimpleNamespace(
            rl_action_space="residual",
            rl_steer_scale=0.18,
            rl_throttle_delta=0.35,
            rl_brake_delta=1.0,
        )
        self.guard.policy_action_semantics = "simlingo_target_control_with_learned_gate_v2"
        self.guard.previous_rl_policy_action = None
        self.base = np.asarray([0.20, 0.40, 1.00, 0.0], dtype=np.float32)

    def map(self, policy_action):
        return self.guard._policy_to_control_action(
            self.base,
            np.asarray(policy_action, dtype=np.float32),
        )

    def test_zero_gate_is_exactly_simlingo(self):
        actual = self.map([-1.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(actual[:3], self.base[:3], atol=1e-7)

    def test_neutral_residual_is_exactly_simlingo_for_any_gate(self):
        actual = self.map([0.0, 0.5, 0.5, 0.85])
        np.testing.assert_allclose(actual[:3], self.base[:3], atol=1e-7)
        self.assertAlmostEqual(float(actual[3]), 0.85, places=6)

    def test_full_gate_can_release_a_full_simlingo_brake(self):
        actual = self.map([0.0, 0.5, 0.0, 1.0])
        self.assertAlmostEqual(float(actual[2]), 0.0, places=6)

    def test_partial_gate_blends_instead_of_replacing_simlingo(self):
        actual = self.map([1.0, 1.0, 0.0, 0.25])
        expected_steer = 0.20 + 0.25 * 0.18
        expected_throttle = 0.40 + 0.25 * 0.35
        expected_brake = 1.00 + 0.25 * (0.00 - 1.00)
        np.testing.assert_allclose(
            actual[:3],
            np.asarray([expected_steer, expected_throttle, expected_brake]),
            atol=1e-6,
        )

    def test_absolute_target_is_still_a_continuous_simlingo_complement(self):
        self.guard.config.rl_action_space = "absolute"
        actual = self.map([-0.40, 0.80, 0.00, 0.25])
        expected = self.base[:3] + 0.25 * (
            np.asarray([-0.40, 0.80, 0.00], dtype=np.float32) - self.base[:3]
        )
        np.testing.assert_allclose(actual[:3], expected, atol=1e-6)

    def test_signed_longitudinal_blend_never_commands_both_pedals(self):
        self.guard.config.rl_action_space = "absolute"
        self.guard.policy_action_semantics = (
            "simlingo_signed_longitudinal_target_with_learned_gate_v3"
        )
        actual = self.map([-0.40, 0.00, 0.80, 0.75])
        self.assertEqual(min(float(actual[1]), float(actual[2])), 0.0)
        self.assertGreater(float(actual[2]), 0.0)

    def test_v2_policy_observation_contains_simlingo_and_lane_context(self):
        self.guard.state_dim = 28
        self.guard.policy_state_dim = 44
        self.guard.policy_input_semantics = "world_state_plus_simlingo_context_v2"
        state = np.arange(28, dtype=np.float32)
        context = {
            "blocked_ticks": 17,
            "left_front_m": 12,
            "left_rear_m": 34,
            "right_front_m": 56,
            "right_rear_m": 78,
            "left_ttc_s": 4,
            "right_ttc_s": 5,
            "left_oncoming_m": 21,
            "right_oncoming_m": 22,
            "left_oncoming_ttc_s": 6,
            "right_oncoming_ttc_s": 7,
            "left_lane_available": 1,
            "right_lane_available": 0,
        }
        observation = self.guard._policy_observation(state, self.base, context)
        self.assertEqual(observation.shape, (44,))
        np.testing.assert_allclose(observation[28:31], self.base[:3], atol=1e-7)
        np.testing.assert_allclose(
            observation[31:],
            np.asarray([17, 12, 34, 56, 78, 4, 5, 21, 22, 6, 7, 1, 0], dtype=np.float32),
        )

    def test_v3_policy_observation_uses_compact_clearances(self):
        self.guard.state_dim = 28
        self.guard.policy_state_dim = 42
        self.guard.policy_input_semantics = "world_state_plus_simlingo_compact_context_v3"
        state = np.arange(28, dtype=np.float32)
        context = {
            "blocked_ticks": 17,
            "left_front_m": 12,
            "left_rear_m": 34,
            "right_front_m": 56,
            "right_rear_m": 23,
            "left_ttc_s": 4,
            "right_ttc_s": 5,
            "left_oncoming_m": 21,
            "right_oncoming_m": 22,
            "left_oncoming_ttc_s": 6,
            "right_oncoming_ttc_s": 7,
            "left_lane_available": 1,
            "right_lane_available": 0,
        }
        observation = self.guard._policy_observation(state, self.base, context)
        self.assertEqual(observation.shape, (42,))
        np.testing.assert_allclose(
            observation[31:],
            np.asarray([17, 12, 23, 4, 5, 21, 22, 6, 7, 1, 0], dtype=np.float32),
        )

    def test_v4_policy_observation_carries_previous_learned_action(self):
        self.guard.state_dim = 28
        self.guard.policy_state_dim = 46
        self.guard.policy_input_semantics = "world_state_plus_simlingo_temporal_context_v4"
        self.guard.previous_rl_policy_action = np.asarray(
            [-0.24, 0.92, 0.0, 0.995], dtype=np.float32
        )
        observation = self.guard._policy_observation(
            np.arange(28, dtype=np.float32),
            self.base,
            {"blocked_ticks": 304, "left_lane_available": 1, "right_lane_available": 0},
        )
        self.assertEqual(observation.shape, (46,))
        np.testing.assert_allclose(
            observation[-4:], self.guard.previous_rl_policy_action, atol=1e-7
        )

    def test_v5_policy_observation_masks_global_pose_only(self):
        self.guard.state_dim = 28
        self.guard.policy_state_dim = 46
        self.guard.policy_input_semantics = (
            "world_state_plus_simlingo_map_invariant_temporal_context_v5"
        )
        self.guard.previous_rl_policy_action = np.asarray(
            [-0.24, 0.92, 0.0, 0.995], dtype=np.float32
        )
        state = np.arange(28, dtype=np.float32)
        observation = self.guard._policy_observation(
            state,
            self.base,
            {"blocked_ticks": 304, "left_lane_available": 1, "right_lane_available": 0},
        )
        self.assertEqual(observation.shape, (46,))
        np.testing.assert_allclose(observation[[0, 1, 3]], 0.0, atol=1e-7)
        self.assertEqual(float(observation[2]), float(state[2]))
        self.assertEqual(float(observation[8]), float(state[8]))
        np.testing.assert_allclose(
            observation[-4:], self.guard.previous_rl_policy_action, atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
