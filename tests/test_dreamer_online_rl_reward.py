import unittest

import numpy as np

from scripts.dreamer_online_rl_update import (
    MAP_INVARIANT_POLICY_INPUT_SEMANTICS,
    policy_state_from_status,
    step_reward,
    terminal_reward,
)


def status(*, x=0.0, speed=0.0, front=80.0, gate=0.0, brake=0.0, throttle=0.4):
    state = [0.0] * 28
    state[0] = x
    state[2] = speed
    state[18] = 80.0
    state[23] = 80.0
    return {
        "state_vector": state,
        "front_vehicle_m": front,
        "nearest_walker_m": 80.0,
        "nearest_bike_m": 80.0,
        "traffic_light": "green",
        "base_risk": 0.1,
        "chosen_risk": 0.1,
        "blocked_ticks": 0,
        "rl_intervention_strength": gate,
        "left_lane_available": True,
        "left_front_m": 80.0,
        "left_rear_m": 80.0,
        "left_clear_m": 80.0,
        "left_ttc_s": 99.0,
        "left_oncoming_m": 80.0,
        "left_oncoming_ttc_s": 99.0,
        "right_lane_available": False,
        "base_action": {"steer": 0.0, "throttle": 0.4, "brake": 0.0, "intervention": 0.0},
        "chosen_action": {
            "steer": 0.0,
            "throttle": throttle,
            "brake": brake,
            "intervention": gate,
        },
    }


class OnlineRewardTest(unittest.TestCase):
    def test_policy_state_reconstructs_v2_context_from_legacy_clearances(self):
        current = status(front=8.0)
        current.pop("left_front_m")
        current.pop("left_rear_m")
        current["left_clear_m"] = 13.0
        current["right_clear_m"] = 27.0
        observation = policy_state_from_status(current, 44, 28)
        self.assertEqual(observation.shape, (44,))
        self.assertEqual(float(observation[32]), 13.0)
        self.assertEqual(float(observation[33]), 13.0)
        self.assertEqual(float(observation[34]), 27.0)
        self.assertEqual(float(observation[35]), 27.0)

    def test_policy_state_reconstructs_v3_compact_context(self):
        current = status(front=8.0)
        current["left_front_m"] = 22.0
        current["left_rear_m"] = 13.0
        current["right_clear_m"] = 27.0
        observation = policy_state_from_status(current, 42, 28)
        self.assertEqual(observation.shape, (42,))
        self.assertEqual(float(observation[32]), 13.0)
        self.assertEqual(float(observation[33]), 27.0)

    def test_policy_state_reconstructs_v4_temporal_context(self):
        current = status(front=8.0)
        current["rl_previous_policy_action"] = [-0.24, 0.92, 0.0, 0.995]
        observation = policy_state_from_status(current, 46, 28)
        self.assertEqual(observation.shape, (46,))
        np.testing.assert_allclose(
            observation[-4:], current["rl_previous_policy_action"], atol=1e-7
        )

    def test_map_invariant_policy_state_ignores_global_carla_pose(self):
        first = status(front=8.0)
        second = status(front=8.0)
        first["state_vector"][0:4] = [100.0, 200.0, 0.0, -1.2]
        second["state_vector"][0:4] = [4306.0, 575.0, 0.0, 2.4]
        first_observation = policy_state_from_status(
            first,
            46,
            28,
            policy_input_semantics=MAP_INVARIANT_POLICY_INPUT_SEMANTICS,
        )
        second_observation = policy_state_from_status(
            second,
            46,
            28,
            policy_input_semantics=MAP_INVARIANT_POLICY_INPUT_SEMANTICS,
        )
        np.testing.assert_allclose(first_observation, second_observation, atol=1e-7)

    def test_unnecessary_intervention_is_penalized_on_clear_road(self):
        current = status(gate=1.0)
        nxt = status(x=0.5, speed=4.0, gate=1.0)
        _, parts = step_reward(current, nxt, np.zeros(4, dtype=np.float32))
        self.assertLess(parts["intervention_cost"], 0.0)
        self.assertLess(parts["unnecessary_intervention"], 0.0)

    def test_staying_stopped_behind_obstacle_with_open_lane_is_penalized(self):
        current = status(front=8.0, gate=0.8, brake=0.9, throttle=0.0)
        nxt = status(front=8.0, gate=0.8, brake=0.9, throttle=0.0)
        _, parts = step_reward(current, nxt, np.zeros(4, dtype=np.float32), stagnant_steps=50)
        self.assertLess(parts["stuck"], -0.1)

    def test_safe_overtake_does_not_receive_generic_override_penalty(self):
        current = status(front=8.0, gate=1.0, brake=0.0, throttle=0.5)
        current["base_action"] = {
            "steer": 0.0,
            "throttle": 0.0,
            "brake": 1.0,
            "intervention": 0.0,
        }
        current["chosen_action"]["steer"] = -0.15
        current["chosen_risk"] = 1.0
        nxt = status(x=0.5, speed=2.0, front=8.0, gate=1.0, brake=0.0, throttle=0.5)
        _, parts = step_reward(current, nxt, np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        self.assertEqual(parts["high_risk_override"], 0.0)
        self.assertGreater(parts["overtake_attempt"], 0.0)

    def test_unsafe_oncoming_overtake_keeps_strong_penalties(self):
        current = status(front=8.0, gate=1.0, brake=0.0, throttle=0.5)
        current["base_action"] = {
            "steer": 0.0,
            "throttle": 0.0,
            "brake": 1.0,
            "intervention": 0.0,
        }
        current["chosen_action"]["steer"] = -0.15
        current["chosen_risk"] = 1.0
        current["left_oncoming_m"] = 10.0
        current["left_oncoming_ttc_s"] = 1.0
        nxt = status(x=0.2, speed=1.0, front=8.0, gate=1.0, brake=0.0, throttle=0.5)
        _, parts = step_reward(current, nxt, np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        self.assertLess(parts["high_risk_override"], 0.0)
        self.assertLess(parts["unsafe_side"], 0.0)

    def test_collision_terminal_signal_is_strongly_negative(self):
        reward, parts = terminal_reward({
            "route_score": 10.0,
            "driving_score": 0.0,
            "collisions": 1.0,
            "vehicle_collisions": 1.0,
            "incomplete": 1.0,
        })
        self.assertLess(reward, -100.0)
        self.assertLess(parts["terminal_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
