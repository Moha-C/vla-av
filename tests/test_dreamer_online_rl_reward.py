import unittest

import numpy as np
import torch

from scripts.dreamer_online_rl_update import (
    ActorCritic,
    CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
    MAP_INVARIANT_POLICY_INPUT_SEMANTICS,
    OvertakeRewardTracker,
    WorldModel,
    enrich_current_oncoming,
    policy_state_from_status,
    progressive_oncoming_engagement_penalty,
    step_reward,
    terminal_reward,
    truncate_rows_at_first_collision,
    upgrade_policy_observation_checkpoint,
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


def overtaking_status(
    *,
    x=0.0,
    speed=0.0,
    front=80.0,
    lane_id=1,
    obstacle_forward=None,
    obstacle_clearance=0.0,
    steer=-0.2,
):
    current = status(x=x, speed=speed, front=front, gate=1.0, brake=0.0, throttle=0.5)
    current["blocked_ticks"] = 30
    current["base_action"] = {
        "steer": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
        "intervention": 0.0,
    }
    current["chosen_action"]["steer"] = steer
    current["ego_road_id"] = 10
    current["ego_lane_id"] = lane_id
    current["ego_lane_width_m"] = 3.5
    current["ego_lane_center_offset_m"] = 0.0
    current["front_vehicle_id"] = 42 if front < 80.0 else -1
    current["front_vehicle_clearance_m"] = max(0.0, front - 4.8)
    current["nearby_vehicles"] = []
    if obstacle_forward is not None:
        current["nearby_vehicles"].append({
            "id": 42,
            "forward_m": obstacle_forward,
            "lateral_m": 0.0,
            "distance_m": abs(obstacle_forward),
            "longitudinal_clearance_m": obstacle_clearance,
            "road_id": 10,
            "lane_id": 1,
        })
    return current


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

    def test_policy_state_v6_contains_current_oncoming_before_previous_action(self):
        current = status(front=8.0)
        current["current_oncoming_distance_m"] = 24.0
        current["current_oncoming_closing_speed_mps"] = 8.0
        current["current_oncoming_ttc_s"] = 3.0
        current["rl_previous_policy_action"] = [-0.2, 0.7, 0.1, 0.8]
        observation = policy_state_from_status(
            current,
            49,
            28,
            policy_input_semantics=CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
        )
        np.testing.assert_allclose(observation[42:45], [24.0, 8.0, 3.0], atol=1e-7)
        np.testing.assert_allclose(observation[45:49], current["rl_previous_policy_action"], atol=1e-7)

    def test_v4_to_v6_checkpoint_migration_preserves_old_network_output(self):
        torch.manual_seed(7)
        policy = ActorCritic(46, 4, 32)
        world_model = WorldModel(28, 4, 32)
        old_input = torch.randn(3, 46)
        old_mean, _, _ = policy.forward(old_input)
        checkpoint = {
            "policy": policy.state_dict(),
            "world_model": world_model.state_dict(),
            "policy_state_mean": np.zeros(46, dtype=np.float32),
            "policy_state_std": np.ones(46, dtype=np.float32),
            "policy_input_semantics": "world_state_plus_simlingo_temporal_context_v4",
        }
        checkpoint, migration = upgrade_policy_observation_checkpoint(checkpoint)
        migrated = ActorCritic(49, 4, 32)
        migrated.load_state_dict(checkpoint["policy"])
        new_input = torch.zeros(3, 49)
        new_input[:, :42] = old_input[:, :42]
        new_input[:, 45:49] = old_input[:, 42:46]
        new_mean, _, _ = migrated.forward(new_input)
        self.assertTrue(migration["applied"])
        torch.testing.assert_close(old_mean, new_mean)

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
        current["blocked_ticks"] = 120
        nxt = status(front=8.0, gate=0.8, brake=0.9, throttle=0.0)
        _, parts = step_reward(current, nxt, np.zeros(4, dtype=np.float32), stagnant_steps=50)
        self.assertLess(parts["stuck"], -0.1)

    def test_waiting_for_oncoming_traffic_is_not_penalized_as_stuck(self):
        current = status(front=8.0, gate=0.8, brake=0.9, throttle=0.0)
        current["blocked_ticks"] = 300
        current["left_oncoming_m"] = 12.0
        current["left_oncoming_ttc_s"] = 1.2
        current["right_lane_available"] = False
        nxt = status(front=8.0, gate=0.8, brake=0.9, throttle=0.0)
        _, parts = step_reward(current, nxt, np.zeros(4, dtype=np.float32), stagnant_steps=300)
        self.assertEqual(parts["stuck"], 0.0)

    def test_waiting_at_red_light_is_not_penalized_as_stuck(self):
        current = status(front=8.0, gate=0.8, brake=0.9, throttle=0.0)
        current["blocked_ticks"] = 300
        current["traffic_light"] = "red"
        nxt = status(front=8.0, gate=0.8, brake=0.9, throttle=0.0)
        _, parts = step_reward(current, nxt, np.zeros(4, dtype=np.float32), stagnant_steps=300)
        self.assertEqual(parts["stuck"], 0.0)

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
        self.assertLess(parts["oncoming_engagement"], 0.0)

    def test_oncoming_penalty_grows_continuously_as_ttc_shrinks(self):
        safe = progressive_oncoming_engagement_penalty(40.0, 6.0, 1.0)
        warning = progressive_oncoming_engagement_penalty(24.0, 3.5, 1.0)
        critical = progressive_oncoming_engagement_penalty(8.0, 1.0, 1.0)
        self.assertGreater(safe, warning)
        self.assertGreater(warning, critical)

    def test_braking_reduces_current_oncoming_engagement_penalty(self):
        accelerating = status(x=0.0, speed=7.0, gate=1.0, brake=0.0, throttle=0.8)
        braking = status(x=0.0, speed=7.0, gate=1.0, brake=1.0, throttle=0.0)
        for current in (accelerating, braking):
            current["current_oncoming_distance_m"] = 10.0
            current["current_oncoming_closing_speed_mps"] = 12.0
            current["current_oncoming_ttc_s"] = 0.83
        _, accelerating_parts = step_reward(accelerating, status(x=0.2, speed=7.0), np.zeros(4, dtype=np.float32))
        _, braking_parts = step_reward(braking, status(x=0.1, speed=5.0), np.zeros(4, dtype=np.float32))
        self.assertLess(accelerating_parts["oncoming_engagement"], braking_parts["oncoming_engagement"])

    def test_legacy_trace_enrichment_recovers_current_lane_oncoming_ttc(self):
        rows = []
        for index, distance in enumerate((20.0, 19.5, 19.0)):
            current = status(x=0.0, speed=6.0)
            current.update({
                "timestamp": 100.0 + index,
                "ego_road_id": 10,
                "ego_lane_id": -1,
                "ego_lane_width_m": 3.5,
                "nearby_vehicles": [{
                    "id": 99,
                    "forward_m": distance,
                    "lateral_m": 0.1,
                    "distance_m": distance,
                    "longitudinal_clearance_m": distance - 5.0,
                    "heading_dot": -0.99,
                    "road_id": 10,
                    "lane_id": -1,
                }],
            })
            rows.append({"collector_time": 100.0 + index, "status": current})
        enrich_current_oncoming(rows)
        self.assertEqual(rows[-1]["status"]["current_oncoming_actor_id"], 99)
        self.assertAlmostEqual(rows[-1]["status"]["current_oncoming_closing_speed_mps"], 10.0, delta=1.5)
        self.assertLess(rows[-1]["status"]["current_oncoming_ttc_s"], 2.5)

    def test_stationary_opposite_vehicle_repairs_explicit_false_legacy_label(self):
        current = status(x=0.0, speed=0.0)
        current.update({
            "ego_road_id": 10,
            "ego_lane_id": -1,
            "ego_lane_width_m": 3.5,
            "current_oncoming_distance_m": 80.0,
            "current_oncoming_ttc_s": 99.0,
            "nearby_vehicles": [{
                "id": 4875,
                "forward_m": 9.6,
                "lateral_m": -2.6,
                "heading_dot": -0.985,
                "relative_longitudinal_speed_mps": 0.0,
                "closing_speed_mps": 0.0,
                "is_oncoming": False,
                "road_id": 10,
                "lane_id": -1,
            }],
        })
        rows = [{"status": current}]
        enrich_current_oncoming(rows)
        repaired = rows[0]["status"]
        self.assertTrue(repaired["nearby_vehicles"][0]["is_oncoming"])
        self.assertEqual(repaired["current_oncoming_actor_id"], 4875)
        self.assertAlmostEqual(repaired["current_oncoming_distance_m"], 9.6)
        self.assertAlmostEqual(repaired["left_oncoming_m"], 9.6)

    def test_enrichment_overwrites_stale_oncoming_when_no_actor_exists(self):
        current = status(x=0.0, speed=0.0)
        current.update({
            "nearby_vehicles": [],
            "current_oncoming_distance_m": 7.0,
            "current_oncoming_ttc_s": 1.0,
            "current_oncoming_actor_id": 123,
            "left_oncoming_m": 5.0,
        })
        rows = [{"status": current}]
        enrich_current_oncoming(rows)
        repaired = rows[0]["status"]
        self.assertEqual(repaired["current_oncoming_distance_m"], 80.0)
        self.assertEqual(repaired["current_oncoming_ttc_s"], 99.0)
        self.assertEqual(repaired["current_oncoming_actor_id"], -1)
        self.assertEqual(repaired["left_oncoming_m"], 80.0)

    def test_collision_trace_is_cut_at_first_impact(self):
        rows = []
        for index in range(6):
            current = status(x=float(index), speed=5.0)
            current["timestamp"] = 100.0 + index
            rows.append({"collector_time": 100.0 + index, "status": current})
        truncated, collision = truncate_rows_at_first_collision(
            rows,
            [{"event": "collision", "wall_time": 103.2, "collision_kind": "vehicle"}],
            {"collisions": 1.0},
        )
        self.assertEqual(len(truncated), 5)
        self.assertEqual(collision["impact_status_index"], 4)
        self.assertEqual(collision["impact_transition_index"], 3)

    def test_legacy_collision_uses_bench2drive_actor_not_earlier_near_contact(self):
        rows = []
        for index in range(5):
            current = status(x=float(index), speed=5.0)
            current.update({
                "timestamp": 100.0 + index,
                "ego_road_id": 640,
                "ego_lane_id": -1,
                "nearby_vehicles": [],
            })
            rows.append({"collector_time": 100.0 + index, "status": current})
        rows[1]["status"]["nearby_vehicles"] = [{
            "id": 4713,
            "forward_m": 4.0,
            "lateral_m": 3.2,
            "longitudinal_clearance_m": 0.02,
            "heading_dot": 0.9,
            "road_id": 640,
            "lane_id": -2,
        }]
        rows[3]["status"]["nearby_vehicles"] = [{
            "id": 4888,
            "forward_m": 5.2,
            "lateral_m": -1.2,
            "longitudinal_clearance_m": 0.15,
            "heading_dot": -0.98,
            "road_id": 640,
            "lane_id": -1,
        }]
        truncated, collision = truncate_rows_at_first_collision(
            rows,
            [],
            {
                "collisions": 1.0,
                "vehicle_collisions": 1.0,
                "first_collision_actor_id": 4888,
            },
        )
        self.assertEqual(len(truncated), 4)
        self.assertEqual(collision["other_actor_id"], 4888)
        self.assertEqual(collision["impact_status_index"], 3)

    def test_front_distance_jump_alone_never_earns_clean_pass(self):
        tracker = OvertakeRewardTracker()
        current = overtaking_status(front=8.0, lane_id=1, obstacle_forward=8.0, obstacle_clearance=3.2)
        nxt = overtaking_status(x=0.5, speed=2.0, front=80.0, lane_id=1)
        _, parts = step_reward(
            current,
            nxt,
            np.zeros(4, dtype=np.float32),
            overtake_tracker=tracker,
        )
        self.assertEqual(parts["clean_pass"], 0.0)
        self.assertFalse(tracker.departed)

    def test_early_reentry_is_penalized_before_safe_rear_gap(self):
        tracker = OvertakeRewardTracker()
        approach = overtaking_status(front=8.0, lane_id=1, obstacle_forward=8.0, obstacle_clearance=3.2)
        adjacent = overtaking_status(x=0.5, speed=2.0, front=80.0, lane_id=2, obstacle_forward=4.0, obstacle_clearance=0.0)
        step_reward(approach, adjacent, np.zeros(4, dtype=np.float32), overtake_tracker=tracker)

        passed = overtaking_status(x=1.0, speed=4.0, front=80.0, lane_id=2, obstacle_forward=-1.0, obstacle_clearance=0.2)
        step_reward(adjacent, passed, np.zeros(4, dtype=np.float32), overtake_tracker=tracker)

        early_return = overtaking_status(x=1.5, speed=4.0, front=80.0, lane_id=1, obstacle_forward=-1.5, obstacle_clearance=0.5)
        _, parts = step_reward(passed, early_return, np.zeros(4, dtype=np.float32), overtake_tracker=tracker)
        self.assertLess(parts["early_reentry"], -4.0)
        self.assertLess(parts["unsafe_reentry"], 0.0)
        self.assertEqual(parts["clean_pass"], 0.0)

    def test_clean_pass_requires_safe_gap_and_twenty_stable_steps(self):
        tracker = OvertakeRewardTracker()
        approach = overtaking_status(front=8.0, lane_id=1, obstacle_forward=8.0, obstacle_clearance=3.2)
        adjacent = overtaking_status(x=0.5, speed=2.0, front=80.0, lane_id=2, obstacle_forward=4.0, obstacle_clearance=0.0)
        step_reward(approach, adjacent, np.zeros(4, dtype=np.float32), overtake_tracker=tracker)
        passed = overtaking_status(x=1.0, speed=4.0, front=80.0, lane_id=2, obstacle_forward=-6.0, obstacle_clearance=4.5)
        step_reward(adjacent, passed, np.zeros(4, dtype=np.float32), overtake_tracker=tracker)

        clean_bonus = 0.0
        previous = passed
        for index in range(20):
            returned = overtaking_status(
                x=1.5 + 0.2 * index,
                speed=4.0,
                front=80.0,
                lane_id=1,
                obstacle_forward=-6.0 - 0.2 * index,
                obstacle_clearance=4.5 + 0.2 * index,
            )
            _, parts = step_reward(
                previous,
                returned,
                np.zeros(4, dtype=np.float32),
                overtake_tracker=tracker,
            )
            clean_bonus += parts["clean_pass"]
            previous = returned
        self.assertEqual(clean_bonus, 8.0)
        self.assertEqual(tracker.phase, "idle")

    def test_incomplete_overtake_receives_terminal_maneuver_penalty(self):
        tracker = OvertakeRewardTracker()
        tracker.phase = "passing"
        tracker.departed = True
        self.assertEqual(tracker.finalize()["incomplete_overtake"], -8.0)

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

    def test_exact_impact_excludes_duplicate_terminal_collision_penalty(self):
        reward, parts = terminal_reward({
            "route_score": 10.0,
            "driving_score": 0.0,
            "collisions": 1.0,
            "vehicle_collisions": 1.0,
        }, exclude_collisions=True)
        self.assertEqual(parts["terminal_collision_penalty"], 0.0)
        self.assertGreater(reward, 0.0)

    def test_min_speed_observation_is_diagnostic_not_blind_wait_penalty(self):
        reward, parts = terminal_reward({
            "route_score": 100.0,
            "driving_score": 100.0,
            "collisions": 0.0,
            "offroad": 0.0,
            "red_lights": 0.0,
            "blocked": 0.0,
            "min_speed_infractions": 4.0,
        })
        self.assertEqual(parts["terminal_clean"], 8.0)
        self.assertEqual(parts["terminal_min_speed"], 0.0)
        self.assertEqual(parts["terminal_min_speed_observed"], 4.0)
        self.assertGreater(reward, 15.0)


if __name__ == "__main__":
    unittest.main()
