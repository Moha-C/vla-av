import ast
import importlib.util
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SIDECAR_PATH = ROOT / "scripts" / "cardreamer_shadow_sidecar.py"


def load_sidecar():
    spec = importlib.util.spec_from_file_location("cardreamer_shadow_sidecar", SIDECAR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def one_hot(index, count=15):
    action = np.zeros(count, dtype=np.float32)
    action[index] = 1.0
    return action


def test_official_discrete_action_decoding_matches_carla_sign_convention():
    sidecar = load_sidecar()

    hard_left = sidecar.decode_action(one_hot(14))
    hard_right_brake = sidecar.decode_action(one_hot(0))

    assert hard_left["acceleration"] == 2.0
    assert hard_left["proposed_control"] == {"throttle": 2.0 / 3.0, "brake": 0.0, "steer": -0.6}
    assert hard_left["maneuver"] == "left"
    assert hard_right_brake["acceleration"] == -2.0
    assert hard_right_brake["proposed_control"] == {"throttle": 0.0, "brake": 2.0 / 3.0, "steer": 0.6}
    assert hard_right_brake["maneuver"] == "right"


def test_mirror_adapter_changes_only_world_lateral_convention():
    sidecar = load_sidecar()

    native = sidecar.decode_action(one_hot(10), "native")
    mirrored = sidecar.decode_action(one_hot(10), "mirror")

    assert native["action_index"] == mirrored["action_index"] == 10
    assert native["acceleration"] == mirrored["acceleration"] == 2.0
    assert native["policy_frame_maneuver"] == mirrored["policy_frame_maneuver"] == "right"
    assert native["maneuver"] == "right"
    assert mirrored["maneuver"] == "left"
    assert native["proposed_control"]["steer"] == 0.6
    assert mirrored["proposed_control"]["steer"] == -0.6


def test_safe_left_overtake_is_coherent():
    sidecar = load_sidecar()
    geometry = sidecar.LaneGeometry(
        front_vehicle_m=8.0,
        left_lane_available=True,
        left_clear_m=20.0,
        left_oncoming_ttc_s=12.0,
        left_rear_ttc_s=10.0,
    )
    action = sidecar.decode_action(one_hot(14))

    result = sidecar.assess_proposal(action, geometry, 8, 18.0, 5.0, 7.0, 5.0)

    assert result["safe_overtake_opportunity"] is True
    assert result["coherent"] is True
    assert result["unsafe"] is False
    assert result["coherence_label"] == "coherent_overtake_proposal"


def test_left_overtake_with_low_oncoming_ttc_is_unsafe():
    sidecar = load_sidecar()
    geometry = sidecar.LaneGeometry(
        front_vehicle_m=8.0,
        left_lane_available=True,
        left_clear_m=20.0,
        left_oncoming_ttc_s=2.5,
        left_rear_ttc_s=10.0,
    )
    action = sidecar.decode_action(one_hot(14))

    result = sidecar.assess_proposal(action, geometry, 8, 18.0, 5.0, 7.0, 5.0)

    assert result["safe_overtake_opportunity"] is False
    assert result["coherent"] is False
    assert result["unsafe"] is True
    assert result["coherence_label"] == "unsafe_left_proposal"


def test_lateral_proposal_remains_unsafe_after_blocked_context_clears():
    sidecar = load_sidecar()
    geometry = sidecar.LaneGeometry(
        front_vehicle_m=40.0,
        front_vehicle_speed_mps=8.0,
        left_lane_available=True,
        left_clear_m=20.0,
        left_oncoming_ttc_s=2.5,
        left_rear_ttc_s=10.0,
    )
    action = sidecar.decode_action(one_hot(14))

    result = sidecar.assess_proposal(action, geometry, 0, 18.0, 5.0, 7.0, 5.0)

    assert result["blocked"] is False
    assert result["unsafe"] is True
    assert result["coherence_label"] == "unsafe_left_proposal"


def test_shadow_source_contains_no_carla_mutation_calls():
    tree = ast.parse(SIDECAR_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "apply_batch",
        "apply_batch_sync",
        "apply_control",
        "apply_settings",
        "destroy",
        "load_world",
        "set_autopilot",
        "set_simulate_physics",
        "set_target_velocity",
        "set_transform",
        "set_weather",
        "spawn_actor",
        "tick",
        "try_spawn_actor",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not (calls & forbidden), sorted(calls & forbidden)
