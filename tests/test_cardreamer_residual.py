import importlib.util
import json
import pathlib
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "external" / "simlingo" / "team_code" / "cardreamer_residual.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cardreamer_residual", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Control:
    def __init__(self, steer=0.0, throttle=0.0, brake=0.0):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake


def proposal(module, **updates):
    payload = {
        "timestamp": time.time(),
        "state": "observing",
        "mode": "residual",
        "lateral_adapter": "mirror",
        "checkpoint_sha256": module.OFFICIAL_OVERTAKE_SHA256,
        "decision_index": 12,
        "action_index": 14,
        "maneuver": "left",
        "unsafe": True,
        "coherence_label": "unsafe_left_proposal",
        "left_lane_available": True,
        "left_clear_m": 20.0,
        "left_rear_m": 20.0,
        "left_rear_ttc_s": 99.0,
        "left_oncoming_m": 80.0,
        "left_oncoming_ttc_s": 99.0,
        "right_lane_available": True,
        "right_clear_m": 20.0,
        "right_rear_m": 20.0,
        "right_rear_ttc_s": 99.0,
        "right_oncoming_m": 80.0,
        "right_oncoming_ttc_s": 99.0,
        "proposed_control": {"steer": -0.6, "throttle": 2.0 / 3.0, "brake": 0.0},
    }
    payload.update(updates)
    return payload


def test_unsafe_mirrored_proposal_can_be_applied_when_gate_is_disabled(tmp_path):
    module = load_module()
    status_path = tmp_path / "proposal.json"
    output_path = tmp_path / "applied.json"
    status_path.write_text(json.dumps(proposal(module)), encoding="utf-8")
    adapter = module.CarDreamerResidualAdapter(
        status_path,
        output_path,
        alpha=0.5,
        max_steer_delta=0.22,
        task_scoped_authority=False,
        traffic_gate_enabled=False,
    )
    control = Control(steer=0.0, throttle=0.4, brake=0.0)

    result, info = adapter.maybe_apply(control)

    assert result is control
    assert info["applied"] is True
    assert info["unsafe_proposal_accepted"] is True
    assert info["no_guard"] is True
    assert info["guard_enabled"] is False
    assert control.steer == -0.22
    assert round(control.throttle, 6) == round(0.4 + 0.5 * ((2.0 / 3.0) - 0.4), 6)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["experimental_unvalidated"] is True


def test_stale_proposal_does_not_change_simlingo_control(tmp_path):
    module = load_module()
    status_path = tmp_path / "proposal.json"
    output_path = tmp_path / "applied.json"
    status_path.write_text(
        json.dumps(proposal(module, timestamp=time.time() - 30.0)), encoding="utf-8"
    )
    adapter = module.CarDreamerResidualAdapter(status_path, output_path, max_status_age=2.0)
    control = Control(steer=0.15, throttle=0.3, brake=0.0)

    _, info = adapter.maybe_apply(control)

    assert info["applied"] is False
    assert info["reason"] == "stale_proposal"
    assert (control.steer, control.throttle, control.brake) == (0.15, 0.3, 0.0)


def test_non_mirrored_proposal_is_rejected_as_transport_mismatch(tmp_path):
    module = load_module()
    status_path = tmp_path / "proposal.json"
    output_path = tmp_path / "applied.json"
    status_path.write_text(
        json.dumps(proposal(module, lateral_adapter="native")), encoding="utf-8"
    )
    adapter = module.CarDreamerResidualAdapter(status_path, output_path)
    control = Control()

    _, info = adapter.maybe_apply(control)

    assert info["applied"] is False
    assert info["reason"] == "mirror_adapter_required"


def test_stable_rssm_overtake_releases_full_simlingo_brake(tmp_path):
    module = load_module()
    status_path = tmp_path / "proposal.json"
    output_path = tmp_path / "applied.json"
    adapter = module.CarDreamerResidualAdapter(
        status_path,
        output_path,
        alpha=0.5,
        engage_decisions=2,
        min_engagement_decisions=20,
    )

    status_path.write_text(
        json.dumps(proposal(module, decision_index=20, blocked=True)), encoding="utf-8"
    )
    first_control = Control(steer=0.0, throttle=0.0, brake=1.0)
    _, first = adapter.maybe_apply(first_control)

    assert first["engagement_active"] is False
    assert first["reason"] == "rssm_engagement_pending"
    assert first["longitudinal_mode"] == "simlingo_native"
    assert first_control.throttle == 0.0
    assert first_control.brake == 1.0

    status_path.write_text(
        json.dumps(proposal(module, decision_index=21, blocked=True)), encoding="utf-8"
    )
    second_control = Control(steer=0.0, throttle=0.0, brake=1.0)
    _, second = adapter.maybe_apply(second_control)

    assert second["engagement_active"] is True
    assert second["engagement_transition"] == "engaged"
    assert second["longitudinal_mode"] == "rssm_engaged"
    assert second_control.brake == 0.0
    assert round(second_control.throttle, 6) == round(1.0 / 3.0, 6)


def test_engagement_persists_through_a_straight_acceleration_proposal(tmp_path):
    module = load_module()
    status_path = tmp_path / "proposal.json"
    output_path = tmp_path / "applied.json"
    adapter = module.CarDreamerResidualAdapter(
        status_path,
        output_path,
        alpha=0.5,
        engage_decisions=2,
        min_engagement_decisions=20,
    )
    for decision_index in (30, 31):
        status_path.write_text(
            json.dumps(proposal(module, decision_index=decision_index, blocked=True)),
            encoding="utf-8",
        )
        adapter.maybe_apply(Control(brake=1.0))

    straight = proposal(
        module,
        decision_index=32,
        action_index=12,
        maneuver="straight_accelerate",
        blocked=True,
        proposed_control={"steer": 0.0, "throttle": 2.0 / 3.0, "brake": 0.0},
    )
    status_path.write_text(json.dumps(straight), encoding="utf-8")
    control = Control(brake=1.0)
    _, info = adapter.maybe_apply(control)

    assert info["engagement_active"] is True
    assert info["engagement_age_decisions"] == 1
    assert control.brake == 0.0
    assert round(control.throttle, 6) == round(1.0 / 3.0, 6)


def test_lateral_predictions_do_not_engage_without_a_blocking_vehicle(tmp_path):
    module = load_module()
    status_path = tmp_path / "proposal.json"
    output_path = tmp_path / "applied.json"
    adapter = module.CarDreamerResidualAdapter(
        status_path, output_path, alpha=0.5, engage_decisions=2
    )

    for decision_index in range(40, 46):
        status_path.write_text(
            json.dumps(
                proposal(module, decision_index=decision_index, blocked=False)
            ),
            encoding="utf-8",
        )
        control = Control(brake=1.0)
        _, info = adapter.maybe_apply(control)
        assert info["engagement_active"] is False
        assert info["reason"] == "simlingo_native_outside_overtake"
        assert info["longitudinal_mode"] == "simlingo_native"
        assert control.throttle == 0.0
        assert control.brake == 1.0


def test_rear_left_closing_vehicle_prevents_engagement(tmp_path):
    module = load_module()
    status_path = tmp_path / "proposal.json"
    output_path = tmp_path / "applied.json"
    adapter = module.CarDreamerResidualAdapter(
        status_path, output_path, alpha=0.5, engage_decisions=2
    )

    for decision_index in (50, 51):
        status_path.write_text(
            json.dumps(
                proposal(
                    module,
                    decision_index=decision_index,
                    blocked=True,
                    left_rear_m=8.0,
                    left_rear_ttc_s=2.0,
                )
            ),
            encoding="utf-8",
        )
        control = Control(steer=0.1, throttle=0.0, brake=1.0)
        _, info = adapter.maybe_apply(control)
        assert info["applied"] is False
        assert info["reason"] == "traffic_gate_wait"
        assert "rear_ttc" in info["traffic_gate_block_reason"]
        assert (control.steer, control.throttle, control.brake) == (0.1, 0.0, 1.0)


def test_oncoming_vehicle_hands_control_back_during_engagement(tmp_path):
    module = load_module()
    status_path = tmp_path / "proposal.json"
    output_path = tmp_path / "applied.json"
    adapter = module.CarDreamerResidualAdapter(
        status_path, output_path, alpha=0.5, engage_decisions=2
    )

    for decision_index in (60, 61):
        status_path.write_text(
            json.dumps(proposal(module, decision_index=decision_index, blocked=True)),
            encoding="utf-8",
        )
        _, info = adapter.maybe_apply(Control(brake=1.0))
    assert info["engagement_active"] is True

    status_path.write_text(
        json.dumps(
            proposal(
                module,
                decision_index=62,
                blocked=True,
                left_oncoming_m=12.0,
                left_oncoming_ttc_s=2.0,
            )
        ),
        encoding="utf-8",
    )
    control = Control(steer=0.05, throttle=0.2, brake=0.0)
    _, info = adapter.maybe_apply(control)

    assert info["applied"] is False
    assert info["reason"] == "traffic_gate_handoff"
    assert info["engagement_active"] is False
    assert "oncoming_ttc" in info["traffic_gate_block_reason"]
    assert (control.steer, control.throttle, control.brake) == (0.05, 0.2, 0.0)
