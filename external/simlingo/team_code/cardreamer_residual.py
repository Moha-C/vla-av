"""Consume official CarDreamer proposals as a residual over SimLingo control.

The official policy runs in a separate Python 3.10 process.  This module is
kept Python 3.8 compatible so the SimLingo agent can consume its latest JSON
proposal without importing JAX or CarDreamer.
"""

import json
import math
import os
import time
from pathlib import Path


OFFICIAL_OVERTAKE_SHA256 = (
    "123525828488d596e80dad0fad0681767cec937adcc04bf0d5aa8ee972aa8058"
)


def _enabled(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _as_float(value, default):
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return converted if math.isfinite(converted) else float(default)


class CarDreamerResidualAdapter:
    """Use a mirrored CarDreamer proposal as an overtaking complement.

    The official checkpoint was trained on a single slow lead vehicle.  Its
    residual authority is therefore scoped to an explicit overtaking episode.
    An optional traffic gate keeps it from entering an occupied/opposing lane;
    this gate is deterministic arbitration, not learned checkpoint behavior.
    """

    def __init__(
        self,
        status_path,
        control_status_path,
        alpha=0.35,
        max_steer_delta=0.22,
        max_status_age=6.0,
        expected_sha256=OFFICIAL_OVERTAKE_SHA256,
        engage_decisions=2,
        min_engagement_decisions=20,
        release_decisions=6,
        min_engage_throttle=0.15,
        min_lateral_steer=0.10,
        task_scoped_authority=True,
        traffic_gate_enabled=True,
        minimum_clearance=5.0,
        minimum_oncoming_ttc=7.0,
        minimum_rear_ttc=5.0,
        emergency_rear_clearance=3.0,
    ):
        self.status_path = Path(status_path)
        self.control_status_path = Path(control_status_path)
        self.alpha = _clip(float(alpha), 0.0, 1.0)
        self.max_steer_delta = max(0.0, float(max_steer_delta))
        self.max_status_age = max(0.1, float(max_status_age))
        self.expected_sha256 = str(expected_sha256 or "").strip().lower()
        self.engage_decisions = max(1, int(engage_decisions))
        self.min_engagement_decisions = max(0, int(min_engagement_decisions))
        self.release_decisions = max(1, int(release_decisions))
        self.min_engage_throttle = max(0.0, float(min_engage_throttle))
        self.min_lateral_steer = max(0.0, float(min_lateral_steer))
        self.task_scoped_authority = bool(task_scoped_authority)
        self.traffic_gate_enabled = bool(traffic_gate_enabled)
        self.minimum_clearance = max(0.0, float(minimum_clearance))
        self.minimum_oncoming_ttc = max(0.0, float(minimum_oncoming_ttc))
        self.minimum_rear_ttc = max(0.0, float(minimum_rear_ttc))
        self.emergency_rear_clearance = max(
            0.0, float(emergency_rear_clearance)
        )
        self._last_decision_index = None
        self._candidate_direction = None
        self._candidate_streak = 0
        self._engagement_active = False
        self._engagement_direction = None
        self._engagement_age = 0
        self._release_streak = 0
        self.last_info = {}

    @classmethod
    def from_env(cls):
        return cls(
            status_path=os.environ.get(
                "SIMLINGO_CARDREAMER_STATUS_PATH",
                "logs/simlingo_eval/cardreamer_runtime_status.json",
            ),
            control_status_path=os.environ.get(
                "SIMLINGO_CARDREAMER_CONTROL_STATUS_PATH",
                "logs/simlingo_eval/cardreamer_residual_control.json",
            ),
            alpha=os.environ.get("SIMLINGO_CARDREAMER_RESIDUAL_ALPHA", "0.35"),
            max_steer_delta=os.environ.get(
                "SIMLINGO_CARDREAMER_MAX_STEER_DELTA", "0.22"
            ),
            max_status_age=os.environ.get(
                "SIMLINGO_CARDREAMER_MAX_STATUS_AGE", "6.0"
            ),
            expected_sha256=os.environ.get(
                "SIMLINGO_CARDREAMER_EXPECTED_SHA256", OFFICIAL_OVERTAKE_SHA256
            ),
            engage_decisions=os.environ.get(
                "SIMLINGO_CARDREAMER_ENGAGE_DECISIONS", "2"
            ),
            min_engagement_decisions=os.environ.get(
                "SIMLINGO_CARDREAMER_MIN_ENGAGEMENT_DECISIONS", "20"
            ),
            release_decisions=os.environ.get(
                "SIMLINGO_CARDREAMER_RELEASE_DECISIONS", "6"
            ),
            min_engage_throttle=os.environ.get(
                "SIMLINGO_CARDREAMER_MIN_ENGAGE_THROTTLE", "0.15"
            ),
            min_lateral_steer=os.environ.get(
                "SIMLINGO_CARDREAMER_MIN_LATERAL_STEER", "0.10"
            ),
            task_scoped_authority=_enabled(
                os.environ.get("SIMLINGO_CARDREAMER_TASK_SCOPED_AUTHORITY", "1")
            ),
            traffic_gate_enabled=_enabled(
                os.environ.get("SIMLINGO_CARDREAMER_TRAFFIC_GATE", "1")
            ),
            minimum_clearance=os.environ.get(
                "SIMLINGO_CARDREAMER_MINIMUM_CLEARANCE", "5.0"
            ),
            minimum_oncoming_ttc=os.environ.get(
                "SIMLINGO_CARDREAMER_MINIMUM_ONCOMING_TTC", "7.0"
            ),
            minimum_rear_ttc=os.environ.get(
                "SIMLINGO_CARDREAMER_MINIMUM_REAR_TTC", "5.0"
            ),
            emergency_rear_clearance=os.environ.get(
                "SIMLINGO_CARDREAMER_EMERGENCY_REAR_CLEARANCE", "3.0"
            ),
        )

    def _lane_gate(self, status, direction, entry):
        if direction not in ("left", "right"):
            return {
                "open": False,
                "direction": direction,
                "phase": "entry" if entry else "committed",
                "reasons": ["no_lateral_direction"],
            }

        prefix = direction
        available = bool(status.get(prefix + "_lane_available", False))
        clearance = _as_float(status.get(prefix + "_clear_m"), 0.0)
        rear_m = _as_float(status.get(prefix + "_rear_m"), 0.0)
        rear_ttc = _as_float(status.get(prefix + "_rear_ttc_s"), 0.0)
        oncoming_m = _as_float(status.get(prefix + "_oncoming_m"), 0.0)
        oncoming_ttc = _as_float(
            status.get(prefix + "_oncoming_ttc_s"), 0.0
        )
        reasons = []
        if entry and not available:
            reasons.append("lane_unavailable")
        if entry and clearance < self.minimum_clearance:
            reasons.append("clearance")
        if rear_ttc < self.minimum_rear_ttc:
            reasons.append("rear_ttc")
        if oncoming_ttc < self.minimum_oncoming_ttc:
            reasons.append("oncoming_ttc")
        if not entry and rear_m < self.emergency_rear_clearance:
            reasons.append("rear_overlap")
        return {
            "open": not reasons,
            "direction": direction,
            "phase": "entry" if entry else "committed",
            "reasons": reasons,
            "clearance_m": clearance,
            "rear_m": rear_m,
            "rear_ttc_s": rear_ttc,
            "oncoming_m": oncoming_m,
            "oncoming_ttc_s": oncoming_ttc,
        }

    def _release_engagement(self, transition):
        self._engagement_active = False
        self._engagement_direction = None
        self._candidate_direction = None
        self._candidate_streak = 0
        self._release_streak = 0
        return transition

    def _update_engagement(self, status, proposed, entry_gate_open=True):
        """Track a temporal policy commitment from the RSSM task output.

        A stationary-obstacle flag activates the complementary policy. Two
        consecutive positive lateral proposals give CarDreamer enough
        longitudinal authority to release an otherwise permanent SimLingo
        brake command.  When enabled, the traffic gate must also be open.
        """
        decision_index = status.get("decision_index")
        if decision_index is None or decision_index == self._last_decision_index:
            return ""
        self._last_decision_index = decision_index

        maneuver = str(status.get("maneuver") or "")
        blocked_context = bool(status.get("blocked", False))
        proposal_drive = proposed["throttle"] - proposed["brake"]
        lateral_positive = (
            self.alpha > 0.0
            and blocked_context
            and entry_gate_open
            and maneuver in ("left", "right")
            and proposed["throttle"] >= self.min_engage_throttle
            and proposed["brake"] <= 0.05
            and abs(proposed["steer"]) >= self.min_lateral_steer
        )
        transition = ""

        if not self._engagement_active:
            if lateral_positive:
                if maneuver == self._candidate_direction:
                    self._candidate_streak += 1
                else:
                    self._candidate_direction = maneuver
                    self._candidate_streak = 1
                if self._candidate_streak >= self.engage_decisions:
                    self._engagement_active = True
                    self._engagement_direction = maneuver
                    self._engagement_age = 0
                    self._release_streak = 0
                    transition = "engaged"
            else:
                self._candidate_direction = None
                self._candidate_streak = 0
            return transition

        self._engagement_age += 1
        if proposed["brake"] > 0.05 and proposal_drive < 0.0:
            transition = self._release_engagement("released_by_rssm_brake")
        elif (
            self._engagement_age >= self.min_engagement_decisions
            and not blocked_context
        ):
            self._release_streak += 1
            if self._release_streak >= self.release_decisions:
                transition = self._release_engagement("released_to_simlingo")
        else:
            self._release_streak = 0
        return transition

    def _read_proposal(self):
        with self.status_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_control_status(self, payload):
        self.control_status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.control_status_path.with_suffix(
            self.control_status_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.control_status_path)

    def _finish(self, info):
        info.update(
            {
                "timestamp": time.time(),
                "mode": "cardreamer_rssm_mirror_residual",
                "complement_to_simlingo": True,
                "no_guard": not self.traffic_gate_enabled,
                "guard_enabled": self.traffic_gate_enabled,
                "traffic_gate_enabled": self.traffic_gate_enabled,
                "task_scoped_authority": self.task_scoped_authority,
                "privileged_information": True,
                "experimental_unvalidated": True,
            }
        )
        self.last_info = info
        try:
            self._write_control_status(info)
        except OSError:
            pass
        return info

    def maybe_apply(self, control):
        base = {
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        info = {
            "applied": False,
            "reason": "waiting_for_proposal",
            "alpha": self.alpha,
            "base_control": dict(base),
            "applied_control": dict(base),
        }
        try:
            status = self._read_proposal()
        except (OSError, ValueError, TypeError):
            return control, self._finish(info)

        now = time.time()
        try:
            age = max(0.0, now - float(status.get("timestamp", 0.0)))
        except (TypeError, ValueError):
            age = float("inf")
        info["proposal_age_seconds"] = age
        info["sidecar_state"] = status.get("state")
        info["sidecar_decision_index"] = status.get("decision_index")
        info["checkpoint_sha256"] = status.get("checkpoint_sha256")
        info["proposal_assessment"] = status.get("coherence_label")
        info["proposal_unsafe"] = bool(status.get("unsafe", False))
        info["proposal_maneuver"] = status.get("maneuver")
        info["proposal_action_index"] = status.get("action_index")
        info["front_vehicle_m"] = status.get("front_vehicle_m")
        info["left_clear_m"] = status.get("left_clear_m")
        info["left_rear_m"] = status.get("left_rear_m")
        info["left_rear_ttc_s"] = status.get("left_rear_ttc_s")
        info["left_oncoming_ttc_s"] = status.get("left_oncoming_ttc_s")
        info["right_clear_m"] = status.get("right_clear_m")
        info["right_rear_m"] = status.get("right_rear_m")
        info["right_rear_ttc_s"] = status.get("right_rear_ttc_s")
        info["right_oncoming_ttc_s"] = status.get("right_oncoming_ttc_s")

        rejection = None
        if status.get("state") != "observing":
            rejection = "sidecar_not_observing"
        elif status.get("mode") != "residual":
            rejection = "wrong_runtime_mode"
        elif status.get("lateral_adapter") != "mirror":
            rejection = "mirror_adapter_required"
        elif age > self.max_status_age:
            rejection = "stale_proposal"
        elif (
            self.expected_sha256
            and str(status.get("checkpoint_sha256", "")).lower()
            != self.expected_sha256
        ):
            rejection = "unexpected_checkpoint"

        proposal = status.get("proposed_control")
        if rejection is None and not isinstance(proposal, dict):
            rejection = "missing_proposed_control"
        if rejection is None:
            try:
                proposed = {
                    "steer": float(proposal["steer"]),
                    "throttle": float(proposal["throttle"]),
                    "brake": float(proposal["brake"]),
                }
            except (KeyError, TypeError, ValueError):
                rejection = "invalid_proposed_control"
                proposed = None
            if proposed is not None and not all(
                math.isfinite(value) for value in proposed.values()
            ):
                rejection = "non_finite_proposed_control"
        else:
            proposed = None

        if rejection is not None:
            info["reason"] = rejection
            return control, self._finish(info)

        info["proposed_control"] = dict(proposed)
        maneuver = str(status.get("maneuver") or "")
        proposal_direction = maneuver if maneuver in ("left", "right") else None
        entry_gate = self._lane_gate(status, proposal_direction, entry=True)
        info["traffic_gate"] = entry_gate
        gate_open_for_entry = (
            entry_gate["open"] if self.traffic_gate_enabled else True
        )
        transition = self._update_engagement(
            status, proposed, entry_gate_open=gate_open_for_entry
        )

        if self.traffic_gate_enabled and self._engagement_active:
            committed_gate = self._lane_gate(
                status, self._engagement_direction, entry=False
            )
            info["traffic_gate"] = committed_gate
            if not committed_gate["open"]:
                transition = self._release_engagement("traffic_gate_handoff")
                info.update(
                    {
                        "reason": "traffic_gate_handoff",
                        "engagement_transition": transition,
                        "engagement_active": False,
                        "engagement_direction": None,
                        "engagement_age_decisions": self._engagement_age,
                        "engagement_candidate_streak": 0,
                        "engagement_release_streak": 0,
                        "engagement_context_blocked": bool(
                            status.get("blocked", False)
                        ),
                        "longitudinal_mode": "simlingo_native",
                        "traffic_gate_block_reason": ",".join(
                            committed_gate["reasons"]
                        ),
                    }
                )
                return control, self._finish(info)

        if self.task_scoped_authority and not self._engagement_active:
            blocked_context = bool(status.get("blocked", False))
            if (
                blocked_context
                and proposal_direction
                and self.traffic_gate_enabled
                and not entry_gate["open"]
            ):
                reason = "traffic_gate_wait"
                info["traffic_gate_block_reason"] = ",".join(
                    entry_gate["reasons"]
                )
            elif blocked_context and proposal_direction:
                reason = "rssm_engagement_pending"
            elif blocked_context:
                reason = "waiting_for_rssm_overtake_intent"
            else:
                reason = "simlingo_native_outside_overtake"
            info.update(
                {
                    "reason": reason,
                    "engagement_active": False,
                    "engagement_direction": self._engagement_direction,
                    "engagement_age_decisions": self._engagement_age,
                    "engagement_candidate_streak": self._candidate_streak,
                    "engagement_release_streak": self._release_streak,
                    "engagement_transition": transition,
                    "engagement_context_blocked": blocked_context,
                    "longitudinal_mode": "simlingo_native",
                }
            )
            return control, self._finish(info)

        steer_delta = self.alpha * (proposed["steer"] - base["steer"])
        applied_steer = _clip(
            base["steer"]
            + _clip(steer_delta, -self.max_steer_delta, self.max_steer_delta),
            -1.0,
            1.0,
        )

        base_drive = _clip(base["throttle"] - base["brake"], -1.0, 1.0)
        proposal_drive = _clip(
            proposed["throttle"] - proposed["brake"], -1.0, 1.0
        )
        if self._engagement_active and proposal_drive >= 0.0:
            # Once the RSSM has expressed a stable overtaking intent, a native
            # SimLingo brake is no longer allowed to cancel its accelerator.
            # SimLingo still contributes any positive longitudinal command.
            applied_drive = (
                (1.0 - self.alpha) * max(0.0, base_drive)
                + self.alpha * proposal_drive
            )
            longitudinal_mode = "rssm_engaged"
        else:
            applied_drive = (
                (1.0 - self.alpha) * base_drive
                + self.alpha * proposal_drive
            )
            longitudinal_mode = "signed_blend"
        applied_drive = _clip(applied_drive, -1.0, 1.0)
        applied = {
            "steer": applied_steer,
            "throttle": max(0.0, applied_drive),
            "brake": max(0.0, -applied_drive),
        }

        control.steer = float(applied["steer"])
        control.throttle = float(applied["throttle"])
        control.brake = float(applied["brake"])
        info.update(
            {
                "applied": True,
                "reason": "residual_applied",
                "unsafe_proposal_accepted": bool(status.get("unsafe", False)),
                "applied_control": dict(applied),
                "base_longitudinal": base_drive,
                "proposal_longitudinal": proposal_drive,
                "applied_longitudinal": applied_drive,
                "longitudinal_mode": longitudinal_mode,
                "engagement_active": self._engagement_active,
                "engagement_direction": self._engagement_direction,
                "engagement_age_decisions": self._engagement_age,
                "engagement_candidate_streak": self._candidate_streak,
                "engagement_release_streak": self._release_streak,
                "engagement_transition": transition,
                "engagement_context_blocked": bool(status.get("blocked", False)),
                "control_delta": {
                    key: applied[key] - base[key]
                    for key in ("steer", "throttle", "brake")
                },
            }
        )
        return control, self._finish(info)
