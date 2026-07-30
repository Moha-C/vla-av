"""Records and explains every lane-change decision the planner makes.

For each lane change (or blocked attempt) it captures when, which direction,
the surrounding-vehicle context, whether a safety mandate blocked it, and a
short reason — so a reviewer can audit exactly why each maneuver happened.

The rear/left/right vehicle context only exists with Tier-1 state; in the base
28-dim layout those indices are read as 0.0 (no side-vehicle awareness).
"""
import json

from rewards.vru_reward import (
    EGO_SPEED, VEHICLE_BEHIND_DIST, VEHICLE_LEFT_DIST, VEHICLE_RIGHT_DIST,
)

LANE_CHANGE_STEER_THRESHOLD = 0.3


class LaneChangeExplainer:
    def __init__(self):
        self.lane_change_log = []

    def record_decision(self, timestep, action, state, info, mandate,
                        is_safe, reason, direction=None):
        """Append a lane-change decision record.

        Skips non-lane-change actions unless a mandate blocked an attempt (in
        which case the executed steering may have been clamped below threshold).
        """
        steering = float(action[0])
        blocked = mandate is not None
        if abs(steering) < LANE_CHANGE_STEER_THRESHOLD and not blocked:
            return

        if direction is None:
            if blocked and isinstance(mandate, dict) and mandate.get("direction"):
                direction = mandate["direction"]
            else:
                direction = "left" if steering < 0 else "right"

        mandate_reason = mandate.get("reason") if isinstance(mandate, dict) else None
        info = info or {}

        # Side/rear vehicle context only exists with Tier-1 state; fall back to
        # 0.0 for the base 28-dim layout instead of indexing out of bounds.
        def _get(idx):
            return float(state[idx]) if idx < len(state) else 0.0

        self.lane_change_log.append({
            "timestep": int(timestep),
            "steering": steering,
            "direction": direction,
            # Safety context
            "vehicle_rear_dist": _get(VEHICLE_BEHIND_DIST),
            "vehicle_rear_speed": _get(VEHICLE_BEHIND_DIST + 1),
            "vehicle_left_dist": _get(VEHICLE_LEFT_DIST),
            "vehicle_right_dist": _get(VEHICLE_RIGHT_DIST),
            "ego_speed": _get(EGO_SPEED),
            # Mandate
            "blocked_by_mandate": blocked,
            "mandate_reason": mandate_reason,
            # Outcome
            "is_safe": bool(is_safe),
            "reason": reason,
            # Planning context
            "dreaming_active": bool(info.get("dreaming_active", False)),
            "defensive_mode": bool(info.get("defensive_mode", False)),
            "scene_difficulty": float(info.get("scene_difficulty", 0.0)),
        })

    def print_summary(self):
        if not self.lane_change_log:
            print("No lane changes recorded.")
            return

        safe = sum(1 for d in self.lane_change_log if d["is_safe"])
        unsafe = sum(1 for d in self.lane_change_log if not d["is_safe"])
        blocked = sum(1 for d in self.lane_change_log if d["blocked_by_mandate"])

        print("\n" + "=" * 100)
        print("LANE CHANGE DECISIONS")
        print("=" * 100)
        print(f"Total: {len(self.lane_change_log)} | Safe: {safe} | "
              f"Unsafe: {unsafe} | Blocked: {blocked}")
        print("-" * 100)
        for d in self.lane_change_log[-10:]:
            status = "OK  SAFE" if d["is_safe"] else "XX UNSAFE"
            mandate_str = (f"[MANDATE: {d['mandate_reason']}]"
                           if d["blocked_by_mandate"] else "")
            print(f"t={d['timestep']:03d} | {d['direction']:5s} | "
                  f"rear={d['vehicle_rear_dist']:5.1f}m "
                  f"ego={d['ego_speed']:5.1f}m/s | "
                  f"{status} | {d['reason']:22s} {mandate_str}")
        print("=" * 100 + "\n")

    def export_decisions(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.lane_change_log, f, indent=2)
