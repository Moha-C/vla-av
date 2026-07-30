"""Defensive-driving mode controller (for unknown maps).

When the ego enters an unknown town it switches to a conservative regime:
heavier VRU/vehicle/collision weighting, deeper planning, and a block on risky
maneuvers (aggressive turns in intersections, speed in low visibility, etc.).

Scaling is applied *in place* on the shared config (and restored in place on
deactivate) so the env/planner that hold the same config object see the change.
"""
from rewards.vru_reward import EGO_SPEED, VEHICLE_AHEAD_DIST


class DefensiveDrivingController:
    def __init__(self, config):
        self.config = config
        self.defensive_mode_active = False
        self.last_action = None
        self._saved = {}

    # ------------------------------------------------------------------ #
    def _scale(self, attr, mult):
        if hasattr(self.config, attr):
            self._saved[attr] = getattr(self.config, attr)
            setattr(self.config, attr, getattr(self.config, attr) * mult)

    def _set(self, attr, value):
        self._saved[attr] = getattr(self.config, attr, None)
        setattr(self.config, attr, value)

    def activate_defensive_mode(self):
        """Scale weights/params for maximum safety. Idempotent."""
        if self.defensive_mode_active:
            return
        print("DEFENSIVE MODE ACTIVATED")
        self.defensive_mode_active = True
        self._saved = {}

        self._scale("w_vru", self.config.defensive_w_vru_mult)
        self._scale("w_vehicle", self.config.defensive_w_vehicle_mult)

        self._set("dream_k", self.config.defensive_dream_k)
        if hasattr(self.config, "horizon_max"):
            self._set("horizon_max", self.config.defensive_horizon_max)
        self._set("disable_risky_maneuvers",
                  self.config.defensive_disable_risky_maneuvers)

    def deactivate_defensive_mode(self):
        """Restore the pre-defensive config values (in place)."""
        if not self.defensive_mode_active:
            return
        print("DEFENSIVE MODE DEACTIVATED")
        self.defensive_mode_active = False
        for attr, value in self._saved.items():
            setattr(self.config, attr, value)
        self._saved = {}

    # ------------------------------------------------------------------ #
    def is_risky_action(self, action, state, info):
        """True if the action is unsafe enough to block in defensive mode."""
        if not self.defensive_mode_active:
            return False

        action = list(action)
        steering, throttle = float(action[0]), float(action[1])
        is_junction = bool(state[9])
        visibility = float((info or {}).get("visibility", 0.5))
        ego_speed = float(state[EGO_SPEED])

        # 1. Aggressive lane change / turn in an intersection.
        if is_junction and abs(steering) > 0.5:
            return True
        # 2. High speed in low visibility.
        if visibility < 0.5 and ego_speed > 8.0:
            return True
        # 3. Rapid steering change vs the previous action.
        if self.last_action is not None:
            if abs(steering - float(self.last_action[0])) > 0.8:
                return True
        # 4. Accelerating into a close vehicle ahead.
        if float(state[VEHICLE_AHEAD_DIST]) < 10.0 and throttle > 0.5:
            return True
        return False
