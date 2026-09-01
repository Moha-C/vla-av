"""CARLA context adapter for the independent report-aligned Dreamer branch."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.world_model.agent import SimLingoDreamerAgent
from src.world_model.config import load_config
from src.world_model.dataset import route_metadata
from src.world_model.observation import DreamerObservationBuilder


class _ContextOnlyExtractor:
    """Reuse the proven CARLA geometry scan without constructing a guard."""

    def __init__(self):
        from team_code.dreamer_guard import DreamerGuard

        self._implementation = object.__new__(DreamerGuard)

    def build(self, agent: Any, tick_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._implementation.build_state(agent, tick_data)


def _enabled(value: str) -> bool:
    return str(value).strip().lower() not in ("", "0", "false", "no", "off", "none")


def _action(control: Any) -> np.ndarray:
    return np.asarray(
        [
            float(getattr(control, "steer", 0.0)),
            float(getattr(control, "throttle", 0.0)),
            float(getattr(control, "brake", 0.0)),
        ],
        dtype=np.float32,
    )


def _control_dict(action: Any) -> Dict[str, float]:
    values = _action(action)
    return {
        "steer": float(values[0]),
        "throttle": float(values[1]),
        "brake": float(values[2]),
    }


class ReportNativeTraceCollector:
    """Record Phase-1 native SimLingo transitions without changing control.

    The collector runs inside the agent so every recorded action corresponds
    to the exact post-PID command sent by native SimLingo. Bench2Drive event
    labels are finalized after the route; no incident label is inferred here.
    """

    CONTEXT_KEYS = (
        "speed",
        "front_vehicle_m",
        "front_vehicle_clearance_m",
        "front_vehicle_id",
        "front_relative_speed_mps",
        "front_closing_speed_mps",
        "left_clear_m",
        "right_clear_m",
        "left_front_m",
        "left_rear_m",
        "right_front_m",
        "right_rear_m",
        "left_lane_available",
        "right_lane_available",
        "left_ttc_s",
        "right_ttc_s",
        "left_oncoming_m",
        "right_oncoming_m",
        "left_oncoming_ttc_s",
        "right_oncoming_ttc_s",
        "current_oncoming_distance_m",
        "current_oncoming_closing_speed_mps",
        "current_oncoming_ttc_s",
        "nearest_walker_m",
        "nearest_bike_m",
        "ego_lane_center_offset_m",
        "ego_lane_width_m",
        "ego_lane_id",
        "ego_road_id",
        "blocked_ticks",
        "traffic_light",
        "nearby_vehicles",
    )

    def __init__(self, trace_path: str):
        self.trace_path = Path(trace_path).resolve()
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse accidental append across route runs.
        if self.trace_path.exists() and self.trace_path.stat().st_size:
            raise FileExistsError(
                "native report trace already contains data: %s" % self.trace_path
            )
        self.context_extractor = _ContextOnlyExtractor()
        config = load_config(
            os.environ.get(
                "SIMLINGO_REPORT_DREAMER_CONFIG",
                str(ROOT / "configs/dreamer_report_aligned.yaml"),
            )
        )
        self.observation_builder = DreamerObservationBuilder(config.observation)
        self.progress_m = 0.0
        self.previous_position: Optional[np.ndarray] = None
        self.route_file = os.environ.get("ROUTE_FILE", os.environ.get("ROUTES", ""))
        self.metadata = route_metadata(self.route_file)
        self.metadata.update(
            {
                "route_id": os.environ.get(
                    "ROUTE_ID", self.metadata.get("route_xml_id", "unknown")
                ),
                "seed": os.environ.get("SEED", "unknown"),
                "town": os.environ.get(
                    "TOWN", self.metadata.get("town", "unknown")
                ),
                "result_path": os.environ.get("SIMLINGO_RESULT_JSON", ""),
            }
        )

    @classmethod
    def from_env(cls) -> Optional["ReportNativeTraceCollector"]:
        path = os.environ.get("SIMLINGO_REPORT_NATIVE_TRACE", "").strip()
        return cls(path) if path else None

    def reset(self) -> None:
        self.observation_builder.reset()
        self.progress_m = 0.0
        self.previous_position = None

    def _route_progress(self, state: np.ndarray) -> float:
        position = np.asarray(state[:2], dtype=np.float32)
        if self.previous_position is not None:
            distance = float(np.linalg.norm(position - self.previous_position))
            if np.isfinite(distance):
                self.progress_m += min(
                    distance,
                    self.observation_builder.config.max_progress_delta_m,
                )
        self.previous_position = position
        return self.progress_m

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): ReportNativeTraceCollector._json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ReportNativeTraceCollector._json_value(item) for item in value]
        return value

    def record(self, agent: Any, tick_data: Dict[str, Any], control: Any) -> None:
        state, context = self.context_extractor.build(agent, tick_data)
        context["route_progress_m"] = self._route_progress(state)
        context["ego_speed_mps"] = float(state[2])
        context["ego_acceleration_mps2"] = float(state[4])
        context["route_curvature"] = float(state[8])
        native = _action(control)
        observation = self.observation_builder.build(
            context,
            native,
            dt=float(getattr(agent, "carla_frame_rate", 0.05)),
        )
        native_dict = _control_dict(control)
        status = {
            key: self._json_value(context.get(key))
            for key in self.CONTEXT_KEYS
            if key in context
        }
        status.update(
            {
                "timestamp": time.time(),
                "mode": "simlingo_native",
                "variant": "simlingo_native_report_collect",
                "policy_source": "simlingo_native",
                "applied": False,
                "shadow": False,
                "alpha": 0.0,
                "dreamer_weight": 0.0,
                "simlingo_weight": 1.0,
                "base_action": native_dict,
                "native_action": native_dict,
                "final_action": native_dict,
                "state_vector": self._json_value(state),
                "route_progress_m": float(context["route_progress_m"]),
                "ego_speed_mps": float(context["ego_speed_mps"]),
                "ego_acceleration_mps2": float(
                    context["ego_acceleration_mps2"]
                ),
                "route_curvature": float(context["route_curvature"]),
            }
        )
        row = {
            "collector_time": time.time(),
            "route_file": self.route_file,
            "route_id": str(self.metadata.get("route_id", "unknown")),
            "town": str(self.metadata.get("town", "unknown")),
            "seed": str(self.metadata.get("seed", "unknown")),
            "weather": self._json_value(self.metadata.get("weather", "unknown")),
            "scenario": self.metadata.get("scenario", "unknown"),
            "result_path": self.metadata.get("result_path", ""),
            "observation": observation.as_dict(),
            "status": status,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class ReportDreamerAdapter:
    def __init__(
        self,
        checkpoint: str,
        config_path: str,
        ablation: str,
        shadow: bool,
        device: str,
        trace_path: str = "",
        status_path: str = "",
    ):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.config_path = str(Path(config_path).resolve())
        self.ablation = str(ablation).upper()
        self.shadow = bool(shadow)
        self.device = device
        self.model = SimLingoDreamerAgent.load(
            self.checkpoint,
            self.config_path,
            device,
            runtime_overrides={"ablation": self.ablation, "shadow": self.shadow},
        )
        self.observation_builder = DreamerObservationBuilder(self.model.config.observation)
        self.context_extractor = _ContextOnlyExtractor()
        self.trace_path = Path(trace_path).resolve() if trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path = Path(status_path).resolve() if status_path else None
        if self.status_path is not None:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_m = 0.0
        self.previous_position: Optional[np.ndarray] = None
        route_file = os.environ.get("ROUTE_FILE", os.environ.get("ROUTES", ""))
        self.run_metadata = route_metadata(route_file)
        self.run_metadata.update(
            {
                "route": route_file or "unknown",
                "seed": os.environ.get("SEED", "unknown"),
                "visual_weather": os.environ.get(
                    "SIMLINGO_VISUAL_WEATHER", "route"
                ),
            }
        )

    @classmethod
    def from_env(cls) -> Optional["ReportDreamerAdapter"]:
        mode = os.environ.get("SIMLINGO_REPORT_DREAMER_MODE", "off").lower()
        if not _enabled(mode):
            return None
        checkpoint = os.environ.get(
            "SIMLINGO_REPORT_DREAMER_CHECKPOINT",
            str(ROOT / "checkpoints/report_aligned_dreamer/production/report_dreamer.pt"),
        )
        config_path = os.environ.get(
            "SIMLINGO_REPORT_DREAMER_CONFIG",
            str(ROOT / "configs/dreamer_report_aligned.yaml"),
        )
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                "report-aligned Dreamer checkpoint not found: %s" % checkpoint
            )
        if not Path(config_path).exists():
            raise FileNotFoundError("report-aligned Dreamer config not found: %s" % config_path)
        ablation = os.environ.get("SIMLINGO_REPORT_DREAMER_ABLATION", "D").upper()
        if mode == "shadow":
            shadow = True
        else:
            shadow = _enabled(os.environ.get("SIMLINGO_REPORT_DREAMER_SHADOW", "0"))
        return cls(
            checkpoint=checkpoint,
            config_path=config_path,
            ablation=ablation,
            shadow=shadow,
            device=os.environ.get("SIMLINGO_REPORT_DREAMER_DEVICE", "cpu"),
            trace_path=os.environ.get("SIMLINGO_REPORT_DREAMER_TRACE", ""),
            status_path=os.environ.get(
                "SIMLINGO_REPORT_DREAMER_STATUS_PATH",
                os.environ.get("SIMLINGO_DREAMER_STATUS_PATH", ""),
            ),
        )

    def reset(self) -> None:
        self.model.reset()
        self.observation_builder.reset()
        self.progress_m = 0.0
        self.previous_position = None

    def _route_progress(self, state: np.ndarray) -> float:
        position = np.asarray(state[:2], dtype=np.float32)
        if self.previous_position is not None:
            distance = float(np.linalg.norm(position - self.previous_position))
            if np.isfinite(distance):
                self.progress_m += min(
                    distance, self.model.config.observation.max_progress_delta_m
                )
        self.previous_position = position
        return self.progress_m

    def _metadata(self) -> Dict[str, Any]:
        return {
            "map": self.run_metadata.get("town", "unknown"),
            "route": self.run_metadata.get("route", "unknown"),
            "route_xml_id": self.run_metadata.get("route_xml_id", "unknown"),
            "scenario": self.run_metadata.get("scenario", "unknown"),
            "scenario_name": self.run_metadata.get("scenario_name", "unknown"),
            "seed": self.run_metadata.get("seed", "unknown"),
            "weather": self.run_metadata.get("weather", "unknown"),
            "visual_weather": self.run_metadata.get("visual_weather", "route"),
        }

    def _write_trace(self, context: Dict[str, Any], observation: Any, info: Dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        record = {
            "timestamp": time.time(),
            **self._metadata(),
            "observation": observation.as_dict(),
            # Bench2Drive owns collision/off-road ground truth. These fields
            # are intentionally null online and are joined after the route.
            "reward": None,
            "collision": None,
            "offroad": None,
            "event_labels_available_online": False,
            "context": {
                key: ReportNativeTraceCollector._json_value(context.get(key))
                for key in (
                    "speed",
                    "ego_speed_mps",
                    "ego_acceleration_mps2",
                    "route_progress_m",
                    "route_curvature",
                    "front_vehicle_m",
                    "front_vehicle_clearance_m",
                    "front_vehicle_id",
                    "front_relative_speed_mps",
                    "front_closing_speed_mps",
                    "left_clear_m",
                    "right_clear_m",
                    "left_front_m",
                    "left_rear_m",
                    "right_front_m",
                    "right_rear_m",
                    "left_lane_available",
                    "right_lane_available",
                    "left_ttc_s",
                    "right_ttc_s",
                    "left_oncoming_m",
                    "right_oncoming_m",
                    "left_oncoming_ttc_s",
                    "right_oncoming_ttc_s",
                    "current_oncoming_distance_m",
                    "current_oncoming_closing_speed_mps",
                    "current_oncoming_ttc_s",
                    "nearest_walker_m",
                    "nearest_bike_m",
                    "ego_lane_center_offset_m",
                    "ego_lane_width_m",
                    "ego_lane_id",
                    "ego_road_id",
                    "blocked_ticks",
                    "overtake_phase",
                    "return_distance_m",
                    "traffic_light",
                )
            },
            **info,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_status(self, context: Dict[str, Any], info: Dict[str, Any]) -> None:
        if self.status_path is None:
            return
        native = info.get("native_action") or [0.0, 0.0, 0.0]
        chosen = info.get("dreamer_action") or native
        final = info.get("final_action") or native
        payload = {
            "timestamp": time.time(),
            "mode": "REPORT-%s" % self.ablation,
            "variant": "report_aligned_rssm_%s" % self.ablation.lower(),
            "applied": bool(info.get("applied")),
            "would_override": int(info.get("selected_index", 0)) != 0,
            "candidate_index": int(info.get("selected_index", 0)),
            "chosen_kind": info.get("selected_kind", "native"),
            "base_risk": float(info.get("native_predicted_risk", 0.0)),
            "chosen_risk": float(info.get("selected_predicted_risk", 0.0)),
            "base_progress": float(info.get("native_predicted_progress", 0.0)),
            "chosen_progress": float(info.get("selected_predicted_progress", 0.0)),
            "base_action": {
                "steer": float(native[0]),
                "throttle": float(native[1]),
                "brake": float(native[2]),
            },
            "chosen_action": {
                "steer": float(chosen[0]),
                "throttle": float(chosen[1]),
                "brake": float(chosen[2]),
            },
            "final_action": {
                "steer": float(final[0]),
                "throttle": float(final[1]),
                "brake": float(final[2]),
            },
            "alpha": float(info.get("alpha", 0.0)),
            "simlingo_weight": float(info.get("simlingo_weight", 1.0)),
            "dreamer_weight": float(info.get("dreamer_weight", 0.0)),
            "front_vehicle_m": float(context.get("front_vehicle_m", 80.0)),
            "left_clear_m": float(context.get("left_clear_m", 80.0)),
            "right_clear_m": float(context.get("right_clear_m", 80.0)),
            "left_lane_available": bool(context.get("left_lane_available", False)),
            "right_lane_available": bool(context.get("right_lane_available", False)),
            "left_ttc_s": float(context.get("left_ttc_s", 99.0)),
            "right_ttc_s": float(context.get("right_ttc_s", 99.0)),
            "left_oncoming_m": float(context.get("left_oncoming_m", 80.0)),
            "right_oncoming_m": float(context.get("right_oncoming_m", 80.0)),
            "left_oncoming_ttc_s": float(context.get("left_oncoming_ttc_s", 99.0)),
            "right_oncoming_ttc_s": float(context.get("right_oncoming_ttc_s", 99.0)),
            "traffic_light": context.get("traffic_light", "none"),
            "shadow": self.shadow,
            "report_aligned": True,
        }
        temporary = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.status_path)

    def maybe_apply(
        self,
        agent: Any,
        tick_data: Dict[str, Any],
        control: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        state, context = self.context_extractor.build(agent, tick_data)
        context["route_progress_m"] = self._route_progress(state)
        context["ego_speed_mps"] = float(state[2])
        context["ego_acceleration_mps2"] = float(state[4])
        context["route_curvature"] = float(state[8])
        native = _action(control)
        observation = self.observation_builder.build(
            context,
            native,
            dt=float(getattr(agent, "carla_frame_rate", 0.05)),
        )
        inference_started = time.perf_counter()
        decision = self.model.step(observation, native)
        inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
        info = dict(decision.information)
        info.update(
            {
                "mode": "report_aligned",
                "checkpoint": self.checkpoint,
                "shadow": self.shadow,
                "applied": bool(decision.alpha > self.model.config.authority.exact_native_epsilon),
                "front_vehicle_m": float(context.get("front_vehicle_m", 80.0)),
                "left_clear_m": float(context.get("left_clear_m", 80.0)),
                "right_clear_m": float(context.get("right_clear_m", 80.0)),
                "route_progress_m": float(context.get("route_progress_m", 0.0)),
                "ego_speed_mps": float(context.get("ego_speed_mps", context.get("speed", 0.0))),
                "front_relative_speed_mps": float(context.get("front_relative_speed_mps", context.get("front_closing_speed_mps", 0.0))),
                "current_oncoming_distance_m": float(context.get("current_oncoming_distance_m", 80.0)),
                "current_oncoming_closing_speed_mps": float(context.get("current_oncoming_closing_speed_mps", 0.0)),
                "current_oncoming_ttc_s": float(context.get("current_oncoming_ttc_s", 99.0)),
                "inference_latency_ms": inference_latency_ms,
            }
        )
        self._write_status(context, info)
        self._write_trace(context, observation, info)
        if not info["applied"]:
            return control, info
        try:
            import carla

            final = carla.VehicleControl(
                steer=float(decision.final_action[0]),
                throttle=float(decision.final_action[1]),
                brake=float(decision.final_action[2]),
            )
        except ImportError:
            final = control
            final.steer = float(decision.final_action[0])
            final.throttle = float(decision.final_action[1])
            final.brake = float(decision.final_action[2])
        return final, info
