"""Conservative Dreamer guard for SimLingo closed-loop evaluation.

The guard is intentionally not a replacement policy. It scores a small set of
actions around SimLingo's own control and may only apply an override when the
learned world model predicts a clear risk reduction under strict constraints.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


LIGHT_TO_FLOAT = {
    "red": 0.0,
    "yellow": 1.0,
    "green": 2.0,
    "none": 2.0,
    None: 2.0,
}


class WorldModel(nn.Module):
    def __init__(self, state_dim: int = 28, action_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head_state = nn.Linear(hidden, state_dim)
        self.head_risk = nn.Linear(hidden, 1)
        self.head_progress = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.head_state(x), self.sigmoid(self.head_risk(x)), self.head_progress(x)


@dataclass
class GuardConfig:
    checkpoint: str
    mode: str = "apply"
    variant: str = "dreamer_guard_v1"
    device: str = "cpu"
    status_path: str = ""
    risk_margin: float = 0.05
    max_progress_drop: float = 0.01
    max_steer_delta: float = 0.12
    max_brake_increase: float = 0.45
    hazard_front_m: float = 18.0
    w_progress: float = 1.0
    w_risk: float = 2.0
    action_penalty: float = 0.08
    log_every: int = 40
    recovery_enabled: bool = False
    recovery_min_ticks: int = 8
    recovery_front_m: float = 18.0
    recovery_clearance_m: float = 14.0
    recovery_oncoming_clearance_m: float = 48.0
    recovery_oncoming_min_ttc_s: float = 6.5
    recovery_min_ttc_s: float = 3.0
    recovery_throttle: float = 0.38
    recovery_steer: float = 0.34
    recovery_use_base_throttle: bool = True
    recovery_hold_ticks: int = 44
    recovery_exit_front_m: float = 24.0
    recovery_require_driving_lane: bool = True
    recovery_gap_enabled: bool = False
    recovery_gap_clearance_m: float = 8.0
    recovery_gap_min_ttc_s: float = 1.8
    recovery_gap_oncoming_clearance_m: float = 42.0
    recovery_gap_oncoming_min_ttc_s: float = 5.5
    recovery_gap_throttle: float = 0.46
    recovery_gap_initiative_ticks: int = 0
    recovery_gap_initiative_clearance_m: float = 0.0
    recovery_gap_initiative_min_ttc_s: float = 0.0
    recovery_gap_initiative_oncoming_clearance_m: float = 0.0
    recovery_max_risk: float = 1.01
    recovery_min_risk_drop: float = -1.0
    recovery_risk_weight: float = 0.0
    recovery_commit_lock_ticks: int = 0
    recovery_commit_entry_ticks: int = 18
    recovery_commit_cruise_ticks: int = 34
    recovery_commit_emergency_clearance_m: float = 3.2
    recovery_commit_emergency_ttc_s: float = 1.8
    recovery_commit_oncoming_min_ttc_s: float = 4.8
    recovery_finish_ticks: int = 34
    recovery_finish_steer_scale: float = 0.42
    recovery_finish_throttle: float = 0.42
    collision_shield_enabled: bool = False
    collision_shield_front_m: float = 12.0
    collision_shield_risk: float = 0.72
    collision_shield_min_speed: float = 0.25
    collision_shield_brake: float = 0.78


def _enabled_value(value: str) -> bool:
    return value.lower() not in ("", "0", "false", "no", "off")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return _enabled_value(str(value))


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _actor_speed(actor: Any) -> float:
    try:
        velocity = actor.get_velocity()
        return math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
    except Exception:
        return 0.0


def _actor_longitudinal_speed(actor: Any, forward: Any) -> float:
    try:
        velocity = actor.get_velocity()
        return float(velocity.x * forward.x + velocity.y * forward.y + velocity.z * forward.z)
    except Exception:
        return 0.0


def _traffic_light_state(hero: Any) -> str:
    try:
        state = str(hero.get_traffic_light_state()).lower()
    except Exception:
        return "none"
    if "red" in state:
        return "red"
    if "yellow" in state:
        return "yellow"
    if "green" in state:
        return "green"
    return "none"


class DreamerGuard:
    def __init__(self, config: GuardConfig):
        self.config = config
        self.device = torch.device(config.device)
        ckpt = torch.load(config.checkpoint, map_location=self.device)
        state_dict = ckpt.get("model")
        checkpoint_schema = "simlingo_guard"
        if state_dict is None and "world_model" in ckpt:
            state_dict = ckpt["world_model"]
            checkpoint_schema = "youma_dreamer_ppo"
        if state_dict is None:
            raise KeyError("Dreamer checkpoint must contain either 'model' or 'world_model'.")

        fc1_weight = state_dict.get("fc1.weight")
        if fc1_weight is None:
            raise KeyError("Dreamer checkpoint is missing world model layer 'fc1.weight'.")
        hidden = int(fc1_weight.shape[0])
        action_dim = 4
        state_dim = int(fc1_weight.shape[1]) - action_dim
        self.model = WorldModel(state_dim=state_dim, action_dim=action_dim, hidden=hidden).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.state_dim = state_dim
        self.checkpoint_schema = checkpoint_schema

        self.state_mean = torch.as_tensor(
            ckpt.get("state_mean", np.zeros(state_dim, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.state_std = torch.as_tensor(
            ckpt.get("state_std", np.ones(state_dim, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        ).clamp_min(1e-6)
        self.action_mean = torch.as_tensor(
            ckpt.get("action_mean", np.zeros(action_dim, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_std = torch.as_tensor(
            ckpt.get("action_std", np.ones(action_dim, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        ).clamp_min(1e-6)
        self.progress_mean = torch.as_tensor(
            ckpt.get("progress_mean", np.asarray([0.0], dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.progress_std = torch.as_tensor(
            ckpt.get("progress_std", np.asarray([1.0], dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        ).clamp_min(1e-6)
        self.last_log_time = 0.0
        self.blocked_ticks = 0
        self.recovery_active_ticks = 0
        self.recovery_side = 0
        self.recovery_commit_ticks = 0
        self.recovery_finish_active_ticks = 0
        self._candidate_meta: List[Dict[str, Any]] = []

    @classmethod
    def from_env(cls) -> Optional["DreamerGuard"]:
        enabled = os.environ.get("SIMLINGO_DREAMER_GUARD", "0")
        if not _enabled_value(enabled):
            return None

        checkpoint = os.environ.get("SIMLINGO_DREAMER_CHECKPOINT")
        if not checkpoint:
            checkpoint = os.path.join(
                os.environ.get("WORK_DIR", ""),
                "checkpoints",
                "dreamer_guard",
                "best_world_model.pt",
            )
        if not checkpoint or not os.path.exists(checkpoint):
            print(f"SIMLINGO_DREAMER_GUARD disabled: checkpoint not found: {checkpoint}", flush=True)
            return None

        mode = os.environ.get("SIMLINGO_DREAMER_GUARD_MODE", "apply").lower()
        if enabled.lower() == "shadow":
            mode = "shadow"
        variant = os.environ.get("SIMLINGO_DREAMER_VARIANT", "dreamer_guard_v1")
        recovery_default = "accident" in variant or "overtake" in variant
        config = GuardConfig(
            checkpoint=checkpoint,
            mode=mode,
            variant=variant,
            device=os.environ.get("SIMLINGO_DREAMER_DEVICE", "cpu"),
            status_path=os.environ.get("SIMLINGO_DREAMER_STATUS_PATH", ""),
            risk_margin=_as_float(os.environ.get("SIMLINGO_DREAMER_RISK_MARGIN"), 0.05),
            max_progress_drop=_as_float(os.environ.get("SIMLINGO_DREAMER_MAX_PROGRESS_DROP"), 0.01),
            max_steer_delta=_as_float(os.environ.get("SIMLINGO_DREAMER_MAX_STEER_DELTA"), 0.12),
            max_brake_increase=_as_float(os.environ.get("SIMLINGO_DREAMER_MAX_BRAKE_INCREASE"), 0.45),
            hazard_front_m=_as_float(os.environ.get("SIMLINGO_DREAMER_HAZARD_FRONT_M"), 18.0),
            w_progress=_as_float(os.environ.get("SIMLINGO_DREAMER_W_PROGRESS"), 1.0),
            w_risk=_as_float(os.environ.get("SIMLINGO_DREAMER_W_RISK"), 2.0),
            action_penalty=_as_float(os.environ.get("SIMLINGO_DREAMER_ACTION_PENALTY"), 0.08),
            log_every=max(1, int(_as_float(os.environ.get("SIMLINGO_DREAMER_LOG_EVERY"), 40))),
            recovery_enabled=_as_bool(os.environ.get("SIMLINGO_DREAMER_RECOVERY"), recovery_default),
            recovery_min_ticks=max(0, int(_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_MIN_TICKS"), 8))),
            recovery_front_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_FRONT_M"), 18.0),
            recovery_clearance_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_CLEARANCE_M"), 14.0),
            recovery_oncoming_clearance_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_ONCOMING_CLEARANCE_M"), 48.0),
            recovery_oncoming_min_ttc_s=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_ONCOMING_MIN_TTC"), 6.5),
            recovery_min_ttc_s=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_MIN_TTC"), 3.0),
            recovery_throttle=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_THROTTLE"), 0.38),
            recovery_steer=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_STEER"), 0.34),
            recovery_use_base_throttle=_as_bool(os.environ.get("SIMLINGO_DREAMER_RECOVERY_USE_BASE_THROTTLE"), True),
            recovery_hold_ticks=max(0, int(_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_HOLD_TICKS"), 44))),
            recovery_exit_front_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_EXIT_FRONT_M"), 24.0),
            recovery_require_driving_lane=_as_bool(os.environ.get("SIMLINGO_DREAMER_RECOVERY_REQUIRE_DRIVING_LANE"), True),
            recovery_gap_enabled=_as_bool(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP"), False),
            recovery_gap_clearance_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_CLEARANCE_M"), 8.0),
            recovery_gap_min_ttc_s=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_MIN_TTC"), 1.8),
            recovery_gap_oncoming_clearance_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_ONCOMING_CLEARANCE_M"), 42.0),
            recovery_gap_oncoming_min_ttc_s=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_ONCOMING_MIN_TTC"), 5.5),
            recovery_gap_throttle=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_THROTTLE"), 0.46),
            recovery_gap_initiative_ticks=max(0, int(_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_INITIATIVE_TICKS"), 0))),
            recovery_gap_initiative_clearance_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_INITIATIVE_CLEARANCE_M"), 0.0),
            recovery_gap_initiative_min_ttc_s=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_INITIATIVE_MIN_TTC"), 0.0),
            recovery_gap_initiative_oncoming_clearance_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_GAP_INITIATIVE_ONCOMING_CLEARANCE_M"), 0.0),
            recovery_max_risk=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_MAX_RISK"), 1.01),
            recovery_min_risk_drop=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_MIN_RISK_DROP"), -1.0),
            recovery_risk_weight=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_RISK_WEIGHT"), 0.0),
            recovery_commit_lock_ticks=max(0, int(_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_COMMIT_LOCK_TICKS"), 0))),
            recovery_commit_entry_ticks=max(1, int(_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_COMMIT_ENTRY_TICKS"), 18))),
            recovery_commit_cruise_ticks=max(1, int(_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_COMMIT_CRUISE_TICKS"), 34))),
            recovery_commit_emergency_clearance_m=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_COMMIT_EMERGENCY_CLEARANCE_M"), 3.2),
            recovery_commit_emergency_ttc_s=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_COMMIT_EMERGENCY_TTC"), 1.8),
            recovery_commit_oncoming_min_ttc_s=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_COMMIT_ONCOMING_MIN_TTC"), 4.8),
            recovery_finish_ticks=max(0, int(_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_FINISH_TICKS"), 34))),
            recovery_finish_steer_scale=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_FINISH_STEER_SCALE"), 0.42),
            recovery_finish_throttle=_as_float(os.environ.get("SIMLINGO_DREAMER_RECOVERY_FINISH_THROTTLE"), 0.42),
            collision_shield_enabled=_as_bool(os.environ.get("SIMLINGO_DREAMER_COLLISION_SHIELD"), False),
            collision_shield_front_m=_as_float(os.environ.get("SIMLINGO_DREAMER_COLLISION_SHIELD_FRONT_M"), 12.0),
            collision_shield_risk=_as_float(os.environ.get("SIMLINGO_DREAMER_COLLISION_SHIELD_RISK"), 0.72),
            collision_shield_min_speed=_as_float(os.environ.get("SIMLINGO_DREAMER_COLLISION_SHIELD_MIN_SPEED"), 0.25),
            collision_shield_brake=_as_float(os.environ.get("SIMLINGO_DREAMER_COLLISION_SHIELD_BRAKE"), 0.78),
        )
        guard = cls(config)
        print(
            "SIMLINGO_DREAMER_GUARD enabled: "
            f"variant={config.variant} mode={config.mode} checkpoint={config.checkpoint} "
            f"risk_margin={config.risk_margin} max_progress_drop={config.max_progress_drop}",
            flush=True,
        )
        print(
            f"SIMLINGO_DREAMER_CHECKPOINT schema={guard.checkpoint_schema} "
            f"state_dim={guard.state_dim}",
            flush=True,
        )
        return guard

    def build_state(self, agent: Any, tick_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        state = np.zeros(28, dtype=np.float32)
        speed = 0.0
        try:
            speed_value = tick_data.get("speed")
            if torch.is_tensor(speed_value):
                speed = float(speed_value.detach().float().cpu().reshape(-1)[0])
            else:
                speed = _as_float(speed_value)
        except Exception:
            speed = 0.0

        hero = getattr(agent, "hero_actor", None)
        if hero is None or not getattr(hero, "is_alive", False):
            try:
                agent.get_hero()
                hero = getattr(agent, "hero_actor", None)
            except Exception:
                hero = None

        yaw_rad = 0.0
        accel = 0.0
        loc_x = 0.0
        loc_y = 0.0
        light_state = "none"
        front_vehicle_m = 80.0
        nearest_vehicle_m = 80.0
        front_vehicle_rel_speed_mps = 0.0
        nearest_walker_m = 80.0
        nearest_bike_m = 80.0
        left_front_m = 80.0
        left_rear_m = 80.0
        right_front_m = 80.0
        right_rear_m = 80.0
        left_front_rel_speed_mps = 0.0
        left_rear_rel_speed_mps = 0.0
        right_front_rel_speed_mps = 0.0
        right_rear_rel_speed_mps = 0.0
        raw_left_front_m = 80.0
        raw_left_rear_m = 80.0
        raw_right_front_m = 80.0
        raw_right_rear_m = 80.0
        raw_left_front_rel_speed_mps = 0.0
        raw_left_rear_rel_speed_mps = 0.0
        raw_right_front_rel_speed_mps = 0.0
        raw_right_rear_rel_speed_mps = 0.0
        left_oncoming_m = 80.0
        right_oncoming_m = 80.0
        left_oncoming_rel_speed_mps = 0.0
        right_oncoming_rel_speed_mps = 0.0
        current_lane_ok = True
        left_lane_available = True
        right_lane_available = True
        left_lane_width = 3.5
        right_lane_width = 3.5
        road_map = None
        hero_lane_key = None
        left_lane_key = None
        right_lane_key = None

        if hero is not None and getattr(hero, "is_alive", False):
            transform = hero.get_transform()
            loc_x = float(transform.location.x)
            loc_y = float(transform.location.y)
            yaw_rad = math.radians(float(transform.rotation.yaw))
            try:
                acc = hero.get_acceleration()
                accel = math.sqrt(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z)
            except Exception:
                accel = 0.0
            light_state = _traffic_light_state(hero)

            try:
                world = hero.get_world()
                actors = world.get_actors()
                hero_loc = transform.location
                try:
                    road_map = world.get_map()
                    waypoint = road_map.get_waypoint(hero_loc, project_to_road=True)

                    def lane_key(lane: Any) -> Optional[Tuple[int, int]]:
                        if lane is None:
                            return None
                        return (int(getattr(lane, "road_id", 0)), int(getattr(lane, "lane_id", 0)))

                    def lane_ok(lane: Any) -> Tuple[bool, float]:
                        if lane is None:
                            return False, 0.0
                        lane_type = str(getattr(lane, "lane_type", "")).lower()
                        lane_width = float(getattr(lane, "lane_width", 0.0))
                        is_driving = "driving" in lane_type
                        return bool(is_driving and lane_width >= 2.4), lane_width

                    current_lane_ok, _ = lane_ok(waypoint)
                    hero_lane_key = lane_key(waypoint)
                    if waypoint is not None:
                        left_lane = waypoint.get_left_lane()
                        right_lane = waypoint.get_right_lane()
                        left_lane_available, left_lane_width = lane_ok(left_lane)
                        right_lane_available, right_lane_width = lane_ok(right_lane)
                        left_lane_key = lane_key(left_lane)
                        right_lane_key = lane_key(right_lane)
                    else:
                        current_lane_ok = False
                        left_lane_available = False
                        right_lane_available = False
                except Exception:
                    pass
                forward = transform.get_forward_vector()
                right = transform.get_right_vector()
                hero_longitudinal_speed = _actor_longitudinal_speed(hero, forward)
                hero_id = hero.id
                for actor in actors:
                    if not getattr(actor, "is_alive", False) or actor.id == hero_id:
                        continue
                    type_id = getattr(actor, "type_id", "")
                    if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
                        continue
                    loc = actor.get_location()
                    dx = loc.x - hero_loc.x
                    dy = loc.y - hero_loc.y
                    forward_m = dx * forward.x + dy * forward.y
                    lateral_m = dx * right.x + dy * right.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if type_id.startswith("vehicle."):
                        actor_long_speed = _actor_longitudinal_speed(actor, forward)
                        actor_rel_long_speed = actor_long_speed - hero_longitudinal_speed
                        actor_heading_dot = 1.0
                        try:
                            actor_forward = actor.get_transform().get_forward_vector()
                            actor_heading_dot = float(
                                actor_forward.x * forward.x
                                + actor_forward.y * forward.y
                                + actor_forward.z * forward.z
                            )
                        except Exception:
                            actor_heading_dot = 1.0
                        is_oncoming = actor_heading_dot < -0.25 and actor_long_speed < -0.5
                        nearest_vehicle_m = min(nearest_vehicle_m, dist)
                        if -6.2 <= lateral_m <= -0.8:
                            if forward_m >= 0.0 and is_oncoming and forward_m < left_oncoming_m:
                                left_oncoming_m = forward_m
                                left_oncoming_rel_speed_mps = actor_rel_long_speed
                            if forward_m >= 0.0 and forward_m < raw_left_front_m:
                                raw_left_front_m = forward_m
                                raw_left_front_rel_speed_mps = actor_rel_long_speed
                            elif forward_m < 0.0 and abs(forward_m) < raw_left_rear_m:
                                raw_left_rear_m = abs(forward_m)
                                raw_left_rear_rel_speed_mps = actor_rel_long_speed
                        elif 0.8 <= lateral_m <= 6.2:
                            if forward_m >= 0.0 and is_oncoming and forward_m < right_oncoming_m:
                                right_oncoming_m = forward_m
                                right_oncoming_rel_speed_mps = actor_rel_long_speed
                            if forward_m >= 0.0 and forward_m < raw_right_front_m:
                                raw_right_front_m = forward_m
                                raw_right_front_rel_speed_mps = actor_rel_long_speed
                            elif forward_m < 0.0 and abs(forward_m) < raw_right_rear_m:
                                raw_right_rear_m = abs(forward_m)
                                raw_right_rear_rel_speed_mps = actor_rel_long_speed
                        actor_lane_key = None
                        try:
                            if road_map is not None:
                                actor_wp = road_map.get_waypoint(loc, project_to_road=True)
                                if actor_wp is not None:
                                    actor_lane_key = (
                                        int(getattr(actor_wp, "road_id", 0)),
                                        int(getattr(actor_wp, "lane_id", 0)),
                                    )
                        except Exception:
                            actor_lane_key = None

                        if hero_lane_key is not None and actor_lane_key is not None:
                            same_lane = actor_lane_key == hero_lane_key
                            side_bucket = 0
                            if actor_lane_key == left_lane_key:
                                side_bucket = -1
                            elif actor_lane_key == right_lane_key:
                                side_bucket = 1
                        else:
                            same_lane = abs(lateral_m) < 2.2
                            side_bucket = 0
                            if -5.8 <= lateral_m <= -1.25:
                                side_bucket = -1
                            elif 1.25 <= lateral_m <= 5.8:
                                side_bucket = 1

                        if forward_m > 0.0 and same_lane and forward_m < front_vehicle_m:
                            front_vehicle_m = forward_m
                            front_vehicle_rel_speed_mps = actor_rel_long_speed
                        if side_bucket == -1:
                            if forward_m >= 0.0 and forward_m < left_front_m:
                                left_front_m = forward_m
                                left_front_rel_speed_mps = actor_rel_long_speed
                            elif forward_m < 0.0 and abs(forward_m) < left_rear_m:
                                left_rear_m = abs(forward_m)
                                left_rear_rel_speed_mps = actor_rel_long_speed
                        elif side_bucket == 1:
                            if forward_m >= 0.0 and forward_m < right_front_m:
                                right_front_m = forward_m
                                right_front_rel_speed_mps = actor_rel_long_speed
                            elif forward_m < 0.0 and abs(forward_m) < right_rear_m:
                                right_rear_m = abs(forward_m)
                                right_rear_rel_speed_mps = actor_rel_long_speed
                    elif type_id.startswith("walker."):
                        nearest_walker_m = min(nearest_walker_m, dist)
            except Exception:
                pass

        if raw_left_front_m < left_front_m:
            left_front_m = raw_left_front_m
            left_front_rel_speed_mps = raw_left_front_rel_speed_mps
        if raw_left_rear_m < left_rear_m:
            left_rear_m = raw_left_rear_m
            left_rear_rel_speed_mps = raw_left_rear_rel_speed_mps
        if raw_right_front_m < right_front_m:
            right_front_m = raw_right_front_m
            right_front_rel_speed_mps = raw_right_front_rel_speed_mps
        if raw_right_rear_m < right_rear_m:
            right_rear_m = raw_right_rear_m
            right_rear_rel_speed_mps = raw_right_rear_rel_speed_mps

        target = [0.0, 0.0]
        target_next = [0.0, 0.0]
        try:
            if getattr(agent, "target_points", None):
                target = list(np.asarray(agent.target_points[0], dtype=np.float32).reshape(-1)[:2])
                if len(agent.target_points) > 1:
                    target_next = list(np.asarray(agent.target_points[1], dtype=np.float32).reshape(-1)[:2])
                else:
                    target_next = target
        except Exception:
            pass

        state[0] = loc_x
        state[1] = loc_y
        state[2] = speed
        state[3] = yaw_rad
        state[4] = accel
        state[5] = 0.0
        state[6] = _as_float(target[1] if len(target) > 1 else 0.0)
        state[7] = 3.5
        dx1 = _as_float(target[0] if len(target) > 0 else 0.0)
        dy1 = _as_float(target[1] if len(target) > 1 else 0.0)
        dx2 = _as_float(target_next[0] if len(target_next) > 0 else dx1)
        dy2 = _as_float(target_next[1] if len(target_next) > 1 else dy1)
        state[8] = math.atan2(dy2 - dy1, max(abs(dx2 - dx1), 1e-3))
        state[9] = 0.0
        state[10] = LIGHT_TO_FLOAT.get(light_state, 2.0)
        state[11] = 50.0 if light_state == "none" else 20.0
        state[12] = 0.0
        state[13] = min(front_vehicle_m, nearest_vehicle_m)
        state[14] = max(0.0, speed + front_vehicle_rel_speed_mps)
        state[15] = 0.0
        state[16] = state[13]
        state[17] = 0.0
        for base, dist in ((18, nearest_walker_m), (23, nearest_bike_m)):
            state[base] = min(dist, 80.0)
            state[base + 1] = 0.0
            state[base + 2] = 0.0
            state[base + 3] = min(dist, 80.0)
            state[base + 4] = 0.0

        def oncoming_ttc(dist: float, rel_speed: float) -> float:
            closing = max(0.0, -float(rel_speed))
            if closing <= 0.5:
                return 99.0
            return float(dist) / max(0.1, closing)

        context = {
            "speed": speed,
            "front_vehicle_m": front_vehicle_m,
            "nearest_vehicle_m": nearest_vehicle_m,
            "left_front_m": left_front_m,
            "left_rear_m": left_rear_m,
            "right_front_m": right_front_m,
            "right_rear_m": right_rear_m,
            "left_front_rel_speed_mps": left_front_rel_speed_mps,
            "left_rear_rel_speed_mps": left_rear_rel_speed_mps,
            "right_front_rel_speed_mps": right_front_rel_speed_mps,
            "right_rear_rel_speed_mps": right_rear_rel_speed_mps,
            "left_oncoming_m": left_oncoming_m,
            "right_oncoming_m": right_oncoming_m,
            "left_oncoming_rel_speed_mps": left_oncoming_rel_speed_mps,
            "right_oncoming_rel_speed_mps": right_oncoming_rel_speed_mps,
            "left_oncoming_ttc_s": oncoming_ttc(left_oncoming_m, left_oncoming_rel_speed_mps),
            "right_oncoming_ttc_s": oncoming_ttc(right_oncoming_m, right_oncoming_rel_speed_mps),
            "left_clear_m": min(left_front_m, left_rear_m),
            "right_clear_m": min(right_front_m, right_rear_m),
            "left_ttc_s": self._side_ttc(-1, left_front_m, left_rear_m, left_front_rel_speed_mps, left_rear_rel_speed_mps),
            "right_ttc_s": self._side_ttc(1, right_front_m, right_rear_m, right_front_rel_speed_mps, right_rear_rel_speed_mps),
            "current_lane_ok": float(current_lane_ok),
            "left_lane_available": float(left_lane_available),
            "right_lane_available": float(right_lane_available),
            "left_lane_width": left_lane_width,
            "right_lane_width": right_lane_width,
            "target_lateral_m": dy1,
            "traffic_light": light_state,
        }
        return state, context

    def _recovery_context(self, base_action: np.ndarray, context: Dict[str, float]) -> bool:
        if not self.config.recovery_enabled:
            return False
        light = context.get("traffic_light")
        if light in ("red", "yellow"):
            return False
        speed = float(context.get("speed", 0.0))
        front_m = float(context.get("front_vehicle_m", 80.0))
        blocked_by_front = front_m <= self.config.recovery_front_m
        base_stopped = float(base_action[1]) <= 0.08 and float(base_action[2]) >= 0.45
        slow_gap_approach = (
            self.config.recovery_gap_enabled
            and front_m <= self.config.collision_shield_front_m
            and speed <= 3.5
            and float(base_action[1]) > 0.15
        )
        return blocked_by_front and (speed <= 0.9 or base_stopped or slow_gap_approach)

    def _update_blocked_ticks(self, base_action: np.ndarray, context: Dict[str, float]) -> None:
        if self._recovery_context(base_action, context):
            self.blocked_ticks += 1
        else:
            self.blocked_ticks = 0
        context["blocked_ticks"] = float(self.blocked_ticks)

    def _side_clearance(self, side: int, context: Dict[str, float]) -> float:
        if side < 0:
            return float(context.get("left_clear_m", 80.0))
        return float(context.get("right_clear_m", 80.0))

    def _side_ttc(self, side: int, front_m: float, rear_m: float, front_rel_speed: float, rear_rel_speed: float) -> float:
        ttc = 99.0
        if front_rel_speed < -0.5:
            ttc = min(ttc, front_m / max(0.1, -front_rel_speed))
        if rear_rel_speed > 0.5:
            ttc = min(ttc, rear_m / max(0.1, rear_rel_speed))
        return float(ttc)

    def _side_oncoming_hazard(
        self,
        side: int,
        context: Dict[str, float],
        clearance_required: float,
        ttc_required: float,
    ) -> bool:
        prefix = "left" if side < 0 else "right"
        oncoming_m = float(context.get(f"{prefix}_oncoming_m", 80.0))
        oncoming_rel = float(context.get(f"{prefix}_oncoming_rel_speed_mps", 0.0))
        closing_speed = max(0.0, -oncoming_rel)
        if closing_speed <= 0.5:
            return False
        ttc = oncoming_m / max(0.1, closing_speed)
        required_distance = max(clearance_required, ttc_required * closing_speed)
        return oncoming_m < required_distance or ttc < ttc_required

    def _commit_elapsed_ticks(self) -> int:
        if self.config.recovery_commit_lock_ticks <= 0:
            return 0
        return max(0, self.config.recovery_commit_lock_ticks - int(self.recovery_commit_ticks))

    def _side_oncoming_warning_during_commit(self, side: int, context: Dict[str, float]) -> bool:
        prefix = "left" if side < 0 else "right"
        oncoming_m = float(context.get(f"{prefix}_oncoming_m", 80.0))
        oncoming_ttc = float(context.get(f"{prefix}_oncoming_ttc_s", 99.0))
        warn_distance = max(
            self.config.recovery_commit_emergency_clearance_m,
            self.config.recovery_gap_oncoming_clearance_m * 0.62,
        )
        return oncoming_m < warn_distance or oncoming_ttc < self.config.recovery_commit_oncoming_min_ttc_s

    def _side_safe_for_recovery(self, side: int, context: Dict[str, float]) -> bool:
        prefix = "left" if side < 0 else "right"
        if self.config.recovery_require_driving_lane:
            if float(context.get("current_lane_ok", 1.0)) < 0.5:
                return False
            if float(context.get(f"{prefix}_lane_available", 1.0)) < 0.5:
                return False
        front_m = float(context.get(f"{prefix}_front_m", 80.0))
        rear_m = float(context.get(f"{prefix}_rear_m", 80.0))
        front_rel = float(context.get(f"{prefix}_front_rel_speed_mps", 0.0))
        rear_rel = float(context.get(f"{prefix}_rear_rel_speed_mps", 0.0))
        ttc = float(context.get(f"{prefix}_ttc_s", 99.0))

        if min(front_m, rear_m) < self.config.recovery_clearance_m:
            return False
        if self._side_oncoming_hazard(
            side,
            context,
            self.config.recovery_oncoming_clearance_m,
            self.config.recovery_oncoming_min_ttc_s,
        ):
            return False
        if front_rel < -0.5:
            required_front = max(
                self.config.recovery_oncoming_clearance_m,
                self.config.recovery_oncoming_min_ttc_s * -front_rel,
            )
            if front_m < required_front:
                return False
        if rear_rel > 0.5 and rear_m < self.config.recovery_min_ttc_s * rear_rel:
            return False
        return ttc >= self.config.recovery_min_ttc_s

    def _safe_recovery_sides(self, context: Dict[str, float]) -> List[int]:
        return [side for side in (-1, 1) if self._side_safe_for_recovery(side, context)]

    def _side_usable_for_gap_commit(self, side: int, context: Dict[str, float]) -> bool:
        if not self.config.recovery_gap_enabled:
            return False
        prefix = "left" if side < 0 else "right"
        if self.config.recovery_require_driving_lane:
            if float(context.get("current_lane_ok", 1.0)) < 0.5:
                return False
            if float(context.get(f"{prefix}_lane_available", 1.0)) < 0.5:
                return False

        front_m = float(context.get(f"{prefix}_front_m", 80.0))
        rear_m = float(context.get(f"{prefix}_rear_m", 80.0))
        front_rel = float(context.get(f"{prefix}_front_rel_speed_mps", 0.0))
        rear_rel = float(context.get(f"{prefix}_rear_rel_speed_mps", 0.0))
        ttc = float(context.get(f"{prefix}_ttc_s", 99.0))
        blocked_ticks = int(context.get("blocked_ticks", 0.0))
        initiative_ready = (
            self.config.recovery_gap_initiative_ticks > 0
            and blocked_ticks >= self.config.recovery_gap_initiative_ticks
        )
        clearance_required = self.config.recovery_gap_clearance_m
        ttc_required = self.config.recovery_gap_min_ttc_s
        oncoming_required = self.config.recovery_gap_oncoming_clearance_m
        oncoming_ttc_required = self.config.recovery_gap_oncoming_min_ttc_s
        if initiative_ready:
            if self.config.recovery_gap_initiative_clearance_m > 0.0:
                clearance_required = min(clearance_required, self.config.recovery_gap_initiative_clearance_m)
            if self.config.recovery_gap_initiative_min_ttc_s > 0.0:
                ttc_required = min(ttc_required, self.config.recovery_gap_initiative_min_ttc_s)
            if self.config.recovery_gap_initiative_oncoming_clearance_m > 0.0:
                oncoming_required = min(oncoming_required, self.config.recovery_gap_initiative_oncoming_clearance_m)
        # Oncoming traffic is never relaxed by the initiative path. A short gap
        # behind us can be usable; a short head-on gap is how the overtake fails.
        oncoming_required = max(oncoming_required, self.config.recovery_gap_oncoming_clearance_m)
        oncoming_ttc_required = max(oncoming_ttc_required, self.config.recovery_gap_oncoming_min_ttc_s)

        if min(front_m, rear_m) < clearance_required:
            return False
        if ttc < ttc_required:
            return False
        if self._side_oncoming_hazard(side, context, oncoming_required, oncoming_ttc_required):
            return False
        if front_rel < -0.5:
            required_front = max(
                oncoming_required,
                oncoming_ttc_required * -front_rel + clearance_required,
            )
            if front_m < required_front:
                return False
        if rear_rel > 0.5 and rear_m < ttc_required * rear_rel:
            return False
        return True

    def _gap_recovery_sides(self, context: Dict[str, float]) -> List[int]:
        return [
            side
            for side in (-1, 1)
            if not self._side_safe_for_recovery(side, context)
            and self._side_usable_for_gap_commit(side, context)
        ]

    def _side_emergency_during_commit(self, side: int, context: Dict[str, float]) -> bool:
        prefix = "left" if side < 0 else "right"
        if self.config.recovery_require_driving_lane:
            if float(context.get("current_lane_ok", 1.0)) < 0.5:
                return True
            if float(context.get(f"{prefix}_lane_available", 1.0)) < 0.5:
                return True
        clearance = self._side_clearance(side, context)
        ttc = float(context.get(f"{prefix}_ttc_s", 99.0))
        if clearance < self.config.recovery_commit_emergency_clearance_m:
            return True
        if ttc < self.config.recovery_commit_emergency_ttc_s:
            return True

        # The initiation path is intentionally strict about oncoming traffic.
        # Once the ego has already committed to the pass, stopping in the
        # opposite lane is often more dangerous than finishing the maneuver.
        # During commit we therefore only cancel for hard emergencies.
        oncoming_m = float(context.get(f"{prefix}_oncoming_m", 80.0))
        oncoming_ttc = float(context.get(f"{prefix}_oncoming_ttc_s", 99.0))
        if oncoming_m < self.config.recovery_commit_emergency_clearance_m:
            return True
        if oncoming_ttc < self.config.recovery_commit_emergency_ttc_s:
            return True
        return False

    def _recovery_commit_context(self, context: Dict[str, float]) -> bool:
        if (
            not self.config.recovery_enabled
            or self.recovery_commit_ticks <= 0
            or self.recovery_side == 0
        ):
            return False
        if context.get("traffic_light") in ("red", "yellow"):
            return False
        if self._side_emergency_during_commit(self.recovery_side, context):
            return False
        return float(context.get("front_vehicle_m", 80.0)) <= self.config.recovery_exit_front_m

    def _recovery_finish_context(self, context: Dict[str, float]) -> bool:
        if (
            not self.config.recovery_enabled
            or self.recovery_finish_active_ticks <= 0
            or self.recovery_side == 0
        ):
            return False
        if context.get("traffic_light") in ("red", "yellow"):
            return False
        if self._side_emergency_during_commit(self.recovery_side, context):
            return False
        return True

    def _collision_shield_row(
        self,
        base_action: np.ndarray,
        scored: List[Dict[str, Any]],
        context: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        if not self.config.collision_shield_enabled or not scored:
            return None

        base = scored[0]
        front_m = float(context.get("front_vehicle_m", 80.0))
        speed = float(context.get("speed", 0.0))
        base_risk = float(base.get("risk", 0.0))
        light = context.get("traffic_light", "none")
        safe_sides = self._safe_recovery_sides(context)
        gap_sides = self._gap_recovery_sides(context)
        if safe_sides or gap_sides:
            return None

        close_front = front_m <= self.config.collision_shield_front_m
        blocked_front = front_m <= self.config.recovery_front_m
        high_risk = base_risk >= self.config.collision_shield_risk
        red_or_yellow = light in ("red", "yellow")
        moving_or_pushing = speed >= self.config.collision_shield_min_speed or float(base_action[1]) > 0.20
        if not ((close_front and moving_or_pushing) or (blocked_front and high_risk) or red_or_yellow):
            return None

        brake = max(float(base_action[2]), self.config.collision_shield_brake)
        steer = _clip(float(base_action[0]) * 0.35, -0.18, 0.18)
        action = np.asarray([steer, 0.0, brake, 0.0], dtype=np.float32)
        reason = []
        if close_front:
            reason.append(f"front {front_m:.1f}m")
        if high_risk:
            reason.append(f"risk {base_risk:.2f}")
        if red_or_yellow:
            reason.append(str(light))
        reason.append("no safe side lane")
        return {
            "candidate_index": -1,
            "action": action,
            "risk": base_risk,
            "progress": float(base.get("progress", 0.0)),
            "score": float(base.get("score", 0.0)),
            "action_delta": float(np.abs(action[:3] - base_action[:3]).mean()),
            "kind": "collision_shield_hold",
            "side": 0,
            "clearance_m": min(float(context.get("left_clear_m", 80.0)), float(context.get("right_clear_m", 80.0))),
            "ttc_s": min(float(context.get("left_ttc_s", 99.0)), float(context.get("right_ttc_s", 99.0))),
            "shield_active": True,
            "shield_reason": ", ".join(reason),
            "safe_recovery_sides": safe_sides,
            "gap_recovery_sides": gap_sides,
        }

    def _recovery_active_context(self, context: Dict[str, float]) -> bool:
        if not self.config.recovery_enabled or self.recovery_active_ticks <= 0 or self.recovery_side == 0:
            return False
        if context.get("traffic_light") in ("red", "yellow"):
            return False
        if self._recovery_commit_context(context):
            return True
        if (
            not self._side_safe_for_recovery(self.recovery_side, context)
            and not self._side_usable_for_gap_commit(self.recovery_side, context)
            and not self._recovery_finish_context(context)
        ):
            return False
        return float(context.get("front_vehicle_m", 80.0)) <= self.config.recovery_exit_front_m

    def candidate_actions(self, base: np.ndarray, context: Dict[str, float]) -> List[np.ndarray]:
        steer, throttle, brake, _ = [float(x) for x in base]
        hazard = (
            context.get("front_vehicle_m", 80.0) <= self.config.hazard_front_m
            or context.get("traffic_light") in ("red", "yellow")
        )
        candidates = [base.copy()]
        meta: List[Dict[str, Any]] = [{"kind": "base", "side": 0}]
        candidates.append(np.asarray([steer, throttle * 0.70, brake, 1.0 if brake <= 0.5 else 0.0], dtype=np.float32))
        meta.append({"kind": "model_nearby", "side": 0})
        candidates.append(np.asarray([steer, throttle * 0.45, max(brake, 0.10), 1.0], dtype=np.float32))
        meta.append({"kind": "model_cautious", "side": 0})
        for delta in (-0.05, 0.05):
            candidates.append(np.asarray([_clip(steer + delta, -1.0, 1.0), throttle * 0.85, brake, 1.0], dtype=np.float32))
            meta.append({"kind": "model_steer_delta", "side": -1 if delta < 0 else 1})
        if hazard:
            candidates.append(np.asarray([steer, 0.0, max(brake, 0.45), 0.0], dtype=np.float32))
            meta.append({"kind": "hazard_hold", "side": 0})
            if context.get("front_vehicle_m", 80.0) <= 10.0 or context.get("traffic_light") == "red":
                candidates.append(np.asarray([steer, 0.0, max(brake, 0.75), 0.0], dtype=np.float32))
                meta.append({"kind": "hazard_strong_hold", "side": 0})

        recovery_active = self._recovery_active_context(context)
        recovery_finish = self._recovery_finish_context(context)
        if self._recovery_context(base, context) or recovery_active or recovery_finish:
            if self.config.recovery_use_base_throttle:
                recovery_throttle = max(throttle, self.config.recovery_throttle)
            else:
                recovery_throttle = self.config.recovery_throttle
            target_lateral = float(context.get("target_lateral_m", 0.0))
            sides = [-1, 1]
            if recovery_finish:
                sides = [self.recovery_side]
            elif recovery_active:
                sides = [self.recovery_side]
            elif abs(target_lateral) > 0.4:
                preferred = 1 if target_lateral > 0.0 else -1
                sides = [preferred, -preferred]
            for side in sides:
                clearance = self._side_clearance(side, context)
                side_safe = self._side_safe_for_recovery(side, context)
                gap_commit = (not side_safe) and self._side_usable_for_gap_commit(side, context)
                commit_continue = (
                    recovery_active
                    and self.recovery_commit_ticks > 0
                    and side == self.recovery_side
                    and not self._side_emergency_during_commit(side, context)
                )
                finish_continue = recovery_finish and side == self.recovery_side
                if not side_safe and not gap_commit and not commit_continue and not finish_continue:
                    continue
                ttc_key = "left_ttc_s" if side < 0 else "right_ttc_s"
                ttc_s = float(context.get(ttc_key, 99.0))
                side_throttle = recovery_throttle
                steer_plan = (
                    (self.config.recovery_steer * 0.65, 0.78),
                    (self.config.recovery_steer, 1.00),
                    (self.config.recovery_steer * 1.18, 0.86),
                )
                kind = "recovery_hold" if recovery_active else "recovery_overtake"
                if finish_continue:
                    side_throttle = max(recovery_throttle, self.config.recovery_finish_throttle)
                    steer_plan = (
                        (-self.config.recovery_steer * self.config.recovery_finish_steer_scale, 1.00),
                        (-self.config.recovery_steer * self.config.recovery_finish_steer_scale * 0.58, 0.95),
                    )
                    kind = "recovery_finish_pass"
                elif gap_commit or commit_continue:
                    side_throttle = max(recovery_throttle, self.config.recovery_gap_throttle)
                    kind = "recovery_commit_continue" if commit_continue else "recovery_gap_commit"
                    if commit_continue:
                        elapsed = self._commit_elapsed_ticks()
                        oncoming_warning = self._side_oncoming_warning_during_commit(side, context)
                        if oncoming_warning:
                            side_throttle = max(side_throttle, self.config.recovery_finish_throttle)
                            steer_plan = (
                                (-self.config.recovery_steer * 0.68, 1.00),
                                (-self.config.recovery_steer * 0.48, 0.94),
                            )
                            kind = "recovery_commit_recenter"
                        elif elapsed >= self.config.recovery_commit_cruise_ticks:
                            steer_plan = (
                                (0.0, 1.00),
                                (-self.config.recovery_steer * 0.22, 0.96),
                            )
                        elif elapsed >= self.config.recovery_commit_entry_ticks:
                            steer_plan = (
                                (self.config.recovery_steer * 0.28, 1.00),
                                (0.0, 0.98),
                                (-self.config.recovery_steer * 0.14, 0.94),
                            )
                        else:
                            steer_plan = (
                                (self.config.recovery_steer * 0.82, 0.92),
                                (self.config.recovery_steer * 1.04, 1.00),
                            )
                    else:
                        steer_plan = (
                            (self.config.recovery_steer * 0.82, 0.92),
                            (self.config.recovery_steer * 1.04, 1.00),
                        )
                for steer_mag, throttle_scale in steer_plan:
                    candidates.append(np.asarray([
                        _clip(side * steer_mag, -1.0, 1.0),
                        _clip(side_throttle * throttle_scale, 0.0, 1.0),
                        0.0,
                        1.0,
                    ], dtype=np.float32))
                    meta.append({"kind": kind, "side": side, "clearance_m": clearance, "ttc_s": ttc_s})
            if float(context.get("front_vehicle_m", 80.0)) > 4.0:
                creep_throttle = self.config.recovery_throttle * 0.55
                if self.config.recovery_use_base_throttle:
                    creep_throttle = max(throttle, creep_throttle)
                candidates.append(np.asarray([
                    _clip(steer, -1.0, 1.0),
                    _clip(creep_throttle, 0.0, 1.0),
                    0.0,
                    1.0,
                ], dtype=np.float32))
                meta.append({"kind": "recovery_creep", "side": 0})

        self._candidate_meta = meta
        return [np.asarray([_clip(a[0], -1.0, 1.0), _clip(a[1], 0.0, 1.0), _clip(a[2], 0.0, 1.0), _clip(a[3], 0.0, 1.0)], dtype=np.float32) for a in candidates]

    @torch.no_grad()
    def predict(self, state: np.ndarray, actions: Iterable[np.ndarray]) -> List[Dict[str, float]]:
        actions_np = np.asarray(list(actions), dtype=np.float32)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]), mode="constant")
        elif state.shape[0] > self.state_dim:
            state = state[:self.state_dim]
        state_np = np.repeat(state[None, :], actions_np.shape[0], axis=0).astype(np.float32)
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
        state_t = (state_t - self.state_mean) / self.state_std
        action_t = (action_t - self.action_mean) / self.action_std
        _, risk_hat, progress_hat = self.model(state_t, action_t)
        progress = progress_hat * self.progress_std.reshape(1) + self.progress_mean.reshape(1)
        risk_np = risk_hat.squeeze(-1).detach().float().cpu().numpy()
        progress_np = progress.squeeze(-1).detach().float().cpu().numpy()
        rows = []
        base = actions_np[0]
        meta = getattr(self, "_candidate_meta", [])
        for idx, action in enumerate(actions_np):
            row_meta = meta[idx] if idx < len(meta) else {}
            action_delta = float(np.abs(action[:3] - base[:3]).mean())
            score = (
                self.config.w_progress * float(progress_np[idx])
                - self.config.w_risk * float(risk_np[idx])
                - self.config.action_penalty * action_delta
            )
            rows.append({
                "candidate_index": idx,
                "action": action,
                "risk": float(risk_np[idx]),
                "progress": float(progress_np[idx]),
                "score": float(score),
                "action_delta": action_delta,
                "kind": row_meta.get("kind", "model"),
                "side": int(row_meta.get("side", 0)),
                "clearance_m": float(row_meta.get("clearance_m", 0.0)),
                "ttc_s": float(row_meta.get("ttc_s", 99.0)),
            })
        return rows

    def choose(self, scored: List[Dict[str, Any]], context: Dict[str, float]) -> Tuple[Dict[str, Any], bool]:
        base = scored[0]
        if self.config.mode == "full":
            chosen = max(scored, key=lambda row: row["score"])
            return chosen, int(chosen["candidate_index"]) != 0

        base_action = base["action"]
        hazard = (
            context.get("front_vehicle_m", 80.0) <= self.config.hazard_front_m
            or context.get("traffic_light") in ("red", "yellow")
        )

        recovery_active = self._recovery_active_context(context)
        recovery_finish = self._recovery_finish_context(context)
        recovery_ready = (
            int(context.get("blocked_ticks", 0.0)) >= self.config.recovery_min_ticks
            and self._recovery_context(base_action, context)
        )
        if self.config.recovery_enabled and (recovery_active or recovery_finish or recovery_ready):
            recovery_rows = [
                row
                for row in scored[1:]
                if row.get("kind") in (
                    "recovery_overtake",
                    "recovery_hold",
                    "recovery_gap_commit",
                    "recovery_commit_continue",
                    "recovery_commit_recenter",
                    "recovery_finish_pass",
                )
                and float(row.get("risk", 1.0)) <= self.config.recovery_max_risk
                and (float(base.get("risk", 0.0)) - float(row.get("risk", 1.0))) >= self.config.recovery_min_risk_drop
            ]
            if recovery_rows:
                target_lateral = float(context.get("target_lateral_m", 0.0))

                def recovery_score(row: Dict[str, Any]) -> float:
                    side = int(row.get("side", 0))
                    target_bonus = 0.0
                    if abs(target_lateral) > 0.4 and side == (1 if target_lateral > 0.0 else -1):
                        target_bonus = 4.0
                    gap_bonus = 1.8 if row.get("kind") == "recovery_gap_commit" else 0.0
                    commit_bonus = 2.6 if row.get("kind") == "recovery_commit_continue" else 0.0
                    recenter_bonus = 2.4 if row.get("kind") == "recovery_commit_recenter" else 0.0
                    finish_bonus = 2.2 if row.get("kind") == "recovery_finish_pass" else 0.0
                    return (
                        float(row.get("clearance_m", 0.0))
                        + target_bonus
                        + gap_bonus
                        + commit_bonus
                        + recenter_bonus
                        + finish_bonus
                        + min(float(row.get("ttc_s", 99.0)), 8.0)
                        + 0.02 * float(row.get("progress", 0.0))
                        - self.config.recovery_risk_weight * float(row.get("risk", 1.0))
                        - 0.35 * abs(float(row["action"][0]))
                    )

                chosen = max(recovery_rows, key=recovery_score)
                self.recovery_side = int(chosen.get("side", 0))
                chosen_kind = str(chosen.get("kind", ""))
                if chosen_kind != "recovery_finish_pass":
                    self.recovery_active_ticks = max(self.recovery_active_ticks, self.config.recovery_hold_ticks)
                if chosen_kind in ("recovery_overtake", "recovery_gap_commit"):
                    self.recovery_commit_ticks = max(
                        self.recovery_commit_ticks,
                        self.config.recovery_commit_lock_ticks,
                    )
                return chosen, True

        eligible = []
        for row in scored[1:]:
            action = row["action"]
            risk_drop = base["risk"] - row["risk"]
            progress_drop = base["progress"] - row["progress"]
            steer_delta = abs(float(action[0] - base_action[0]))
            brake_increase = max(0.0, float(action[2] - base_action[2]))
            if risk_drop < self.config.risk_margin:
                continue
            if progress_drop > self.config.max_progress_drop:
                continue
            if steer_delta > self.config.max_steer_delta:
                continue
            if brake_increase > self.config.max_brake_increase and not hazard:
                continue
            if brake_increase > 0.25 and not hazard:
                continue
            eligible.append((risk_drop, row["score"], row))
        if not eligible:
            if (
                self.recovery_active_ticks > 0
                and not self._recovery_active_context(context)
                and not self._recovery_finish_context(context)
            ):
                self.recovery_active_ticks = 0
                self.recovery_side = 0
                self.recovery_commit_ticks = 0
                self.recovery_finish_active_ticks = 0
            return base, False
        eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return eligible[0][2], True

    def maybe_override(self, agent: Any, tick_data: Dict[str, Any], control: Any) -> Tuple[Any, Dict[str, Any]]:
        base_action = np.asarray([
            float(control.steer),
            float(control.throttle),
            float(control.brake),
            0.0 if float(control.brake) > 0.5 else 1.0,
        ], dtype=np.float32)
        state, context = self.build_state(agent, tick_data)
        if self.recovery_commit_ticks > 0:
            self.recovery_commit_ticks -= 1
            if not self._recovery_commit_context(context):
                self.recovery_commit_ticks = 0
        if self.recovery_active_ticks > 0:
            self.recovery_active_ticks -= 1
            if not self._recovery_active_context(context):
                front_clear = float(context.get("front_vehicle_m", 80.0)) > self.config.recovery_exit_front_m
                can_finish = (
                    front_clear
                    and self.recovery_side != 0
                    and self.config.recovery_finish_ticks > 0
                    and not self._side_emergency_during_commit(self.recovery_side, context)
                )
                self.recovery_active_ticks = 0
                if can_finish:
                    self.recovery_finish_active_ticks = max(
                        self.recovery_finish_active_ticks,
                        self.config.recovery_finish_ticks,
                    )
                else:
                    self.recovery_side = 0
                    self.recovery_commit_ticks = 0
                    self.recovery_finish_active_ticks = 0
        if self.recovery_finish_active_ticks > 0:
            if not self._recovery_finish_context(context):
                self.recovery_finish_active_ticks = 0
                if self.recovery_active_ticks <= 0 and self.recovery_commit_ticks <= 0:
                    self.recovery_side = 0
            else:
                self.recovery_finish_active_ticks -= 1
                if (
                    self.recovery_finish_active_ticks <= 0
                    and self.recovery_active_ticks <= 0
                    and self.recovery_commit_ticks <= 0
                ):
                    self.recovery_side = 0
        self._update_blocked_ticks(base_action, context)
        candidates = self.candidate_actions(base_action, context)
        scored = self.predict(state, candidates)
        shield_row = (
            None
            if (self._recovery_commit_context(context) or self._recovery_finish_context(context))
            else self._collision_shield_row(base_action, scored, context)
        )
        if shield_row is not None:
            chosen, would_override = shield_row, True
        else:
            chosen, would_override = self.choose(scored, context)
        applied = bool(would_override and self.config.mode in ("apply", "full"))

        info = {
            "enabled": True,
            "mode": self.config.mode,
            "variant": self.config.variant,
            "would_override": bool(would_override),
            "applied": applied,
            "candidate_index": int(chosen["candidate_index"]),
            "base_risk": float(scored[0]["risk"]),
            "chosen_risk": float(chosen["risk"]),
            "base_progress": float(scored[0]["progress"]),
            "chosen_progress": float(chosen["progress"]),
            "chosen_kind": str(chosen.get("kind", "model")),
            "chosen_side": int(chosen.get("side", 0)),
            "blocked_ticks": int(context.get("blocked_ticks", 0.0)),
            "recovery_active_ticks": int(self.recovery_active_ticks),
            "recovery_commit_ticks": int(self.recovery_commit_ticks),
            "recovery_finish_active_ticks": int(self.recovery_finish_active_ticks),
            "recovery_side": int(self.recovery_side),
            "front_vehicle_m": float(context.get("front_vehicle_m", 80.0)),
            "left_clear_m": float(context.get("left_clear_m", 80.0)),
            "right_clear_m": float(context.get("right_clear_m", 80.0)),
            "left_ttc_s": float(context.get("left_ttc_s", 99.0)),
            "right_ttc_s": float(context.get("right_ttc_s", 99.0)),
            "left_oncoming_m": float(context.get("left_oncoming_m", 80.0)),
            "right_oncoming_m": float(context.get("right_oncoming_m", 80.0)),
            "left_oncoming_ttc_s": float(context.get("left_oncoming_ttc_s", 99.0)),
            "right_oncoming_ttc_s": float(context.get("right_oncoming_ttc_s", 99.0)),
            "left_lane_available": bool(float(context.get("left_lane_available", 1.0)) >= 0.5),
            "right_lane_available": bool(float(context.get("right_lane_available", 1.0)) >= 0.5),
            "traffic_light": context.get("traffic_light", "none"),
            "collision_shield_active": bool(chosen.get("shield_active", False)),
            "collision_shield_reason": str(chosen.get("shield_reason", "")),
            "safe_recovery_sides": list(chosen.get("safe_recovery_sides", self._safe_recovery_sides(context))),
            "gap_recovery_sides": list(chosen.get("gap_recovery_sides", self._gap_recovery_sides(context))),
            "state_vector": state.astype(np.float32).tolist(),
        }

        should_log = applied or (
            getattr(agent, "step", 0) > 0
            and getattr(agent, "step", 0) % self.config.log_every == 0
            and time.time() - self.last_log_time > 0.5
        )
        if should_log:
            self.last_log_time = time.time()
            print(
                "SIMLINGO_DREAMER_GUARD "
                f"step={getattr(agent, 'step', -1)} variant={self.config.variant} mode={self.config.mode} "
                f"candidate={info['candidate_index']} applied={int(applied)} "
                f"risk={info['base_risk']:.3f}->{info['chosen_risk']:.3f} "
                f"progress={info['base_progress']:.4f}->{info['chosen_progress']:.4f} "
                f"kind={info['chosen_kind']} blocked={info['blocked_ticks']} "
                f"hold={info['recovery_active_ticks']} commit={info['recovery_commit_ticks']} "
                f"finish={info['recovery_finish_active_ticks']} "
                f"side={info['recovery_side']} "
                f"front={info['front_vehicle_m']:.1f} left={info['left_clear_m']:.1f} "
                f"right={info['right_clear_m']:.1f} "
                f"laneL={int(info['left_lane_available'])} laneR={int(info['right_lane_available'])} "
                f"ttcL={info['left_ttc_s']:.1f} ttcR={info['right_ttc_s']:.1f} "
                f"onL={info['left_oncoming_m']:.1f}/{info['left_oncoming_ttc_s']:.1f} "
                f"onR={info['right_oncoming_m']:.1f}/{info['right_oncoming_ttc_s']:.1f} "
                f"safe={info['safe_recovery_sides']} gap={info['gap_recovery_sides']} "
                f"tl={info['traffic_light']} "
                f"shield={int(info['collision_shield_active'])} {info['collision_shield_reason']}",
                flush=True,
            )

        if self.config.status_path:
            self.write_status(info, base_action, chosen["action"])

        if not applied:
            return control, info

        action = chosen["action"]
        new_control = type(control)(
            steer=float(action[0]),
            throttle=float(action[1]),
            brake=float(action[2]),
        )
        for attr in ("hand_brake", "reverse", "manual_gear_shift", "gear"):
            if hasattr(control, attr) and hasattr(new_control, attr):
                try:
                    setattr(new_control, attr, getattr(control, attr))
                except Exception:
                    pass
        return new_control, info

    def write_status(self, info: Dict[str, Any], base_action: np.ndarray, chosen_action: np.ndarray) -> None:
        try:
            import json
            import pathlib
            import tempfile

            path = pathlib.Path(self.config.status_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": time.time(),
                "enabled": True,
                "mode": info.get("mode"),
                "variant": info.get("variant"),
                "would_override": bool(info.get("would_override")),
                "applied": bool(info.get("applied")),
                "candidate_index": int(info.get("candidate_index", 0)),
                "base_risk": float(info.get("base_risk", 0.0)),
                "chosen_risk": float(info.get("chosen_risk", 0.0)),
                "base_progress": float(info.get("base_progress", 0.0)),
                "chosen_progress": float(info.get("chosen_progress", 0.0)),
                "chosen_kind": str(info.get("chosen_kind", "model")),
                "chosen_side": int(info.get("chosen_side", 0)),
                "blocked_ticks": int(info.get("blocked_ticks", 0)),
                "recovery_active_ticks": int(info.get("recovery_active_ticks", 0)),
                "recovery_commit_ticks": int(info.get("recovery_commit_ticks", 0)),
                "recovery_finish_active_ticks": int(info.get("recovery_finish_active_ticks", 0)),
                "recovery_side": int(info.get("recovery_side", 0)),
                "front_vehicle_m": float(info.get("front_vehicle_m", 80.0)),
                "left_clear_m": float(info.get("left_clear_m", 80.0)),
                "right_clear_m": float(info.get("right_clear_m", 80.0)),
                "left_ttc_s": float(info.get("left_ttc_s", 99.0)),
                "right_ttc_s": float(info.get("right_ttc_s", 99.0)),
                "left_oncoming_m": float(info.get("left_oncoming_m", 80.0)),
                "right_oncoming_m": float(info.get("right_oncoming_m", 80.0)),
                "left_oncoming_ttc_s": float(info.get("left_oncoming_ttc_s", 99.0)),
                "right_oncoming_ttc_s": float(info.get("right_oncoming_ttc_s", 99.0)),
                "left_lane_available": bool(info.get("left_lane_available", True)),
                "right_lane_available": bool(info.get("right_lane_available", True)),
                "traffic_light": info.get("traffic_light", "none"),
                "collision_shield_active": bool(info.get("collision_shield_active", False)),
                "collision_shield_reason": str(info.get("collision_shield_reason", "")),
                "safe_recovery_sides": list(info.get("safe_recovery_sides", [])),
                "gap_recovery_sides": list(info.get("gap_recovery_sides", [])),
                "state_dim": int(self.state_dim),
                "state_vector": list(info.get("state_vector", [])),
                "base_action": {
                    "steer": float(base_action[0]),
                    "throttle": float(base_action[1]),
                    "brake": float(base_action[2]),
                },
                "chosen_action": {
                    "steer": float(chosen_action[0]),
                    "throttle": float(chosen_action[1]),
                    "brake": float(chosen_action[2]),
                },
            }
            with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, encoding="utf-8") as tmp:
                json.dump(payload, tmp, sort_keys=True)
                tmp.write("\n")
                tmp_path = pathlib.Path(tmp.name)
            tmp_path.replace(path)
        except Exception as exc:
            print(f"SIMLINGO_DREAMER_STATUS write failed: {exc}", flush=True)
