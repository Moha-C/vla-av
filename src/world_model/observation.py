"""Compact, named and normalized observation used by the RSSM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import ObservationConfig


DREAMER_OBSERVATION_FEATURES: Tuple[str, ...] = (
    "ego_speed",
    "ego_acceleration",
    "native_steer",
    "native_throttle",
    "native_brake",
    "progress_delta",
    "lane_edge_distance",
    "lane_center_offset",
    "left_clearance",
    "right_clearance",
    "left_front_distance",
    "left_rear_distance",
    "right_front_distance",
    "right_rear_distance",
    "front_obstacle_distance",
    "front_clearance",
    "front_relative_speed",
    "current_oncoming_distance",
    "current_oncoming_closing_speed",
    "current_oncoming_ttc",
    "left_oncoming_distance",
    "left_oncoming_ttc",
    "right_oncoming_distance",
    "right_oncoming_ttc",
    "left_lane_available",
    "right_lane_available",
    "nearest_vru_distance",
    "blocked_fraction",
    "overtake_phase",
    "return_distance",
    "traffic_light_state",
    "route_curvature",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _clip_scale(value: float, scale: float, signed: bool = False) -> float:
    if scale <= 0.0:
        raise ValueError("normalization scale must be positive")
    low = -1.0 if signed else 0.0
    return float(np.clip(value / scale, low, 1.0))


def _action_values(action: Any) -> Tuple[float, float, float]:
    if isinstance(action, Mapping):
        return (
            _number(action.get("steer")),
            _number(action.get("throttle")),
            _number(action.get("brake")),
        )
    if hasattr(action, "steer"):
        return (
            _number(getattr(action, "steer", 0.0)),
            _number(getattr(action, "throttle", 0.0)),
            _number(getattr(action, "brake", 0.0)),
        )
    values = list(()) if action is None else list(action)
    values += [0.0] * (3 - len(values))
    return tuple(_number(value) for value in values[:3])


def _traffic_light(value: Any) -> float:
    label = str(value or "").strip().lower()
    if "red" in label:
        return -1.0
    if "green" in label:
        return 1.0
    if "yellow" in label:
        return -0.5
    return 0.0


def _phase(context: Mapping[str, Any]) -> float:
    raw = context.get("overtake_phase")
    if isinstance(raw, str):
        phases = {
            "idle": 0.0,
            "prepare": 0.25,
            "depart": 0.50,
            "pass": 0.75,
            "return": 1.0,
        }
        return phases.get(raw.lower(), 0.0)
    if raw is not None:
        return float(np.clip(_number(raw), 0.0, 1.0))
    if _number(context.get("recovery_finish_active_ticks")) > 0:
        return 1.0
    if _number(context.get("recovery_active_ticks")) > 0:
        return 0.75
    if _number(context.get("recovery_commit_ticks")) > 0:
        return 0.50
    return 0.0


@dataclass(frozen=True)
class DreamerObservation:
    """A single normalized RSSM observation with stable feature ordering."""

    values: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float32)
        if array.shape != (len(DREAMER_OBSERVATION_FEATURES),):
            raise ValueError(
                "DreamerObservation must have shape (%d,), got %r"
                % (len(DREAMER_OBSERVATION_FEATURES), array.shape)
            )
        if not np.isfinite(array).all():
            raise ValueError("DreamerObservation contains non-finite values")
        object.__setattr__(self, "values", array)

    def as_array(self, copy: bool = True) -> np.ndarray:
        return self.values.copy() if copy else self.values

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(DREAMER_OBSERVATION_FEATURES, map(float, self.values)))


class DreamerObservationBuilder:
    """Build the report observation from map-invariant CARLA context.

    The builder is stateful only for acceleration and progress deltas. Calling
    :meth:`reset` at every route boundary prevents temporal leakage.
    """

    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()
        self.reset()

    def reset(self) -> None:
        self._previous_speed: Optional[float] = None
        self._previous_progress: Optional[float] = None

    def build(
        self,
        context: Mapping[str, Any],
        native_action: Any,
        dt: float = 0.05,
        update_state: bool = True,
    ) -> DreamerObservation:
        cfg = self.config
        steer, throttle, brake = _action_values(native_action)
        speed = max(0.0, _number(context.get("ego_speed_mps", context.get("speed_mps", context.get("speed", 0.0)))))
        acceleration = _number(context.get("ego_acceleration_mps2"), float("nan"))
        if not np.isfinite(acceleration):
            acceleration = 0.0 if self._previous_speed is None else (speed - self._previous_speed) / max(dt, 1.0e-3)

        progress = _number(context.get("route_progress_m", context.get("route_progress", context.get("route_completion", 0.0))))
        progress_delta = _number(context.get("progress_delta_m"), float("nan"))
        if not np.isfinite(progress_delta):
            progress_delta = 0.0 if self._previous_progress is None else progress - self._previous_progress

        lane_width = max(0.1, _number(context.get("ego_lane_width_m"), 3.5))
        lane_offset = _number(context.get("ego_lane_center_offset_m"))
        edge_distance = max(0.0, lane_width * 0.5 - abs(lane_offset))
        nearest_vru = min(
            _number(context.get("nearest_walker_m"), cfg.max_clearance_m),
            _number(context.get("nearest_bike_m"), cfg.max_clearance_m),
            _number(context.get("nearest_vru_m"), cfg.max_clearance_m),
        )

        values = np.asarray(
            [
                _clip_scale(speed, cfg.max_speed_mps),
                _clip_scale(acceleration, cfg.max_accel_mps2, signed=True),
                float(np.clip(steer, -1.0, 1.0)),
                float(np.clip(throttle, 0.0, 1.0)),
                float(np.clip(brake, 0.0, 1.0)),
                _clip_scale(progress_delta, cfg.max_progress_delta_m, signed=True),
                _clip_scale(edge_distance, cfg.max_lane_distance_m),
                _clip_scale(lane_offset, cfg.max_lane_distance_m, signed=True),
                _clip_scale(_number(context.get("left_clear_m"), cfg.max_clearance_m), cfg.max_clearance_m),
                _clip_scale(_number(context.get("right_clear_m"), cfg.max_clearance_m), cfg.max_clearance_m),
                _clip_scale(_number(context.get("left_front_m"), cfg.max_clearance_m), cfg.max_clearance_m),
                _clip_scale(_number(context.get("left_rear_m"), cfg.max_clearance_m), cfg.max_clearance_m),
                _clip_scale(_number(context.get("right_front_m"), cfg.max_clearance_m), cfg.max_clearance_m),
                _clip_scale(_number(context.get("right_rear_m"), cfg.max_clearance_m), cfg.max_clearance_m),
                _clip_scale(_number(context.get("front_vehicle_m", context.get("front_obstacle_m", cfg.max_clearance_m))), cfg.max_clearance_m),
                _clip_scale(_number(context.get("front_vehicle_clearance_m", cfg.max_clearance_m)), cfg.max_clearance_m),
                _clip_scale(_number(context.get("front_relative_speed_mps", context.get("front_closing_speed_mps", 0.0))), cfg.max_relative_speed_mps, signed=True),
                _clip_scale(_number(context.get("current_oncoming_distance_m", cfg.max_clearance_m)), cfg.max_clearance_m),
                _clip_scale(_number(context.get("current_oncoming_closing_speed_mps", 0.0)), cfg.max_relative_speed_mps),
                _clip_scale(_number(context.get("current_oncoming_ttc_s", cfg.max_ttc_s)), cfg.max_ttc_s),
                _clip_scale(_number(context.get("left_oncoming_m", cfg.max_clearance_m)), cfg.max_clearance_m),
                _clip_scale(_number(context.get("left_oncoming_ttc_s", cfg.max_ttc_s)), cfg.max_ttc_s),
                _clip_scale(_number(context.get("right_oncoming_m", cfg.max_clearance_m)), cfg.max_clearance_m),
                _clip_scale(_number(context.get("right_oncoming_ttc_s", cfg.max_ttc_s)), cfg.max_ttc_s),
                1.0 if bool(context.get("left_lane_available", False)) else 0.0,
                1.0 if bool(context.get("right_lane_available", False)) else 0.0,
                _clip_scale(nearest_vru, cfg.max_clearance_m),
                _clip_scale(_number(context.get("blocked_ticks", 0.0)), cfg.max_blocked_ticks),
                _phase(context),
                _clip_scale(_number(context.get("return_distance_m", 0.0)), cfg.max_return_distance_m),
                _traffic_light(context.get("traffic_light")),
                _clip_scale(_number(context.get("route_curvature", 0.0)), cfg.max_curvature, signed=True),
            ],
            dtype=np.float32,
        )
        if update_state:
            self._previous_speed = speed
            self._previous_progress = progress
        return DreamerObservation(values)
