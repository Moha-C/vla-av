"""Pygame visualization for CARLA camera frames and driving commands."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Your system is avx2 capable.*",
            category=RuntimeWarning,
        )
        import pygame
except ImportError as exc:  # pragma: no cover - exercised only without pygame installed.
    pygame = None
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualizerConfig:
    """Window and overlay settings for the live CARLA visualizer."""

    window_size: Tuple[int, int] = (896, 672)
    caption: str = "VLA-AV CARLA Camera"
    target_fps: int = 30
    background_color: Tuple[int, int, int] = (0, 0, 0)
    text_color: Tuple[int, int, int] = (235, 240, 245)
    accent_color: Tuple[int, int, int] = (90, 180, 255)
    success_color: Tuple[int, int, int] = (80, 220, 140)
    warning_color: Tuple[int, int, int] = (255, 120, 92)
    overlay_color: Tuple[int, int, int, int] = (0, 0, 0, 170)
    font_size: int = 20
    margin: int = 16


class PygameVisualizer:
    """Render camera observations with current vehicle commands overlaid."""

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        self.config = config or VisualizerConfig()
        self.screen: Optional[Any] = None
        self.clock: Optional[Any] = None
        self.font: Optional[Any] = None
        self.small_font: Optional[Any] = None
        self._is_open = False
        self._emergency_autopilot_requested = False

    def open(self) -> None:
        """Initialize pygame and create the display window."""

        self._ensure_pygame_available()
        if self._is_open:
            return

        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(self.config.window_size)
        pygame.display.set_caption(self.config.caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, self.config.font_size)
        self.small_font = pygame.font.Font(None, max(16, self.config.font_size - 4))
        self._is_open = True

    def process_events(self) -> bool:
        """Return False when the user closes the window or presses Escape/Q."""

        self.open()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self._emergency_autopilot_requested = True
        return True

    def consume_emergency_autopilot_request(self) -> bool:
        """Return True once after the user presses Space."""

        requested = self._emergency_autopilot_requested
        self._emergency_autopilot_requested = False
        return requested

    def render(
        self,
        image: Optional[np.ndarray],
        state: Optional[Any] = None,
        *,
        instruction: Optional[str] = None,
        frame_id: Optional[int] = None,
        vla_action: Optional[Any] = None,
        attack_status: Optional[str] = None,
        control_mode: Optional[str] = None,
        vla_applied: bool = False,
        safety_status: Optional[Any] = None,
        model_stats: Optional[Dict[str, Any]] = None,
        waiting_text: str = "Waiting for camera...",
    ) -> None:
        """Draw one RGB frame and overlay the latest driving state."""

        self.open()
        self.screen.fill(self.config.background_color)

        if image is None:
            self._draw_center_text(waiting_text)
        else:
            self._draw_image(image)

        overlay_data = self._extract_overlay_data(state)
        if frame_id is not None:
            overlay_data["frame"] = frame_id
        if instruction:
            overlay_data["instruction"] = instruction
        if vla_action is not None:
            overlay_data["vla_action"] = self._extract_vla_action(vla_action)
        if attack_status:
            overlay_data["attack_status"] = attack_status
        if control_mode:
            overlay_data["control_mode"] = control_mode
        overlay_data["vla_applied"] = bool(vla_applied)
        if safety_status is not None:
            overlay_data["safety_status"] = self._extract_safety_status(safety_status)
        if model_stats:
            overlay_data["model_stats"] = dict(model_stats)

        self._draw_overlay(overlay_data)
        pygame.display.flip()
        self.clock.tick(self.config.target_fps)

    def render_compare(
        self,
        image: Optional[np.ndarray],
        state: Optional[Any] = None,
        *,
        instruction: Optional[str] = None,
        frame_id: Optional[int] = None,
        vla_action: Optional[Any] = None,
        expert_action: Optional[Any] = None,
        steering_error: Optional[float] = None,
        attack_status: Optional[str] = None,
        safety_status: Optional[Any] = None,
        model_stats: Optional[Dict[str, Any]] = None,
        waiting_text: str = "Waiting for camera...",
    ) -> None:
        """Draw a side-by-side VLA versus autopilot expert comparison."""

        self.open()
        self.screen.fill(self.config.background_color)
        width, height = self.config.window_size
        overlay_height = 130
        panel_height = height - overlay_height

        if image is None:
            self._draw_center_text(waiting_text)
        else:
            left_bounds = pygame.Rect(0, 0, width // 2, panel_height)
            right_bounds = pygame.Rect(width // 2, 0, width - width // 2, panel_height)
            self._draw_image_in_bounds(image, left_bounds)
            self._draw_image_in_bounds(image, right_bounds)
            pygame.draw.line(
                self.screen,
                (30, 34, 42),
                (width // 2, 0),
                (width // 2, panel_height),
                2,
            )
            self._blit_text(
                self.screen,
                "VLA shadow (not applied)",
                (self.config.margin, self.config.margin),
                color=self.config.success_color,
            )
            self._blit_text(
                self.screen,
                "Autopilot expert (applied)",
                (width // 2 + self.config.margin, self.config.margin),
            )

        overlay_data = self._extract_overlay_data(state)
        if frame_id is not None:
            overlay_data["frame"] = frame_id
        if instruction:
            overlay_data["instruction"] = instruction
        if vla_action is not None:
            overlay_data["vla_action"] = self._extract_vla_action(vla_action)
        if expert_action is not None:
            overlay_data["expert_action"] = self._extract_vla_action(expert_action)
        if steering_error is not None:
            overlay_data["steering_error"] = float(steering_error)
        if attack_status:
            overlay_data["attack_status"] = attack_status
        if safety_status is not None:
            overlay_data["safety_status"] = self._extract_safety_status(safety_status)
        if model_stats:
            overlay_data["model_stats"] = dict(model_stats)

        self._draw_compare_overlay(overlay_data, overlay_height=overlay_height)
        pygame.display.flip()
        self.clock.tick(self.config.target_fps)

    def capture_frame_rgb(self) -> Optional[np.ndarray]:
        """Return the current pygame display surface as an RGB image."""

        if not self._is_open or self.screen is None:
            return None
        frame = pygame.surfarray.array3d(self.screen)
        return np.ascontiguousarray(np.swapaxes(frame, 0, 1))

    def close(self) -> None:
        """Close the pygame window and release display resources."""

        if self._is_open and pygame is not None:
            pygame.quit()
        self._is_open = False
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

    def __enter__(self) -> "PygameVisualizer":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _draw_image(self, image: np.ndarray) -> None:
        self._draw_image_in_bounds(
            image,
            pygame.Rect(0, 0, self.config.window_size[0], self.config.window_size[1]),
        )

    def _draw_image_in_bounds(self, image: np.ndarray, bounds: Any) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape H x W x 3, got {image.shape}."
            )

        image = np.ascontiguousarray(image, dtype=np.uint8)
        surface_array = np.ascontiguousarray(np.swapaxes(image, 0, 1))
        surface = pygame.surfarray.make_surface(surface_array)

        target_rect = self._fit_rect(image.shape[1], image.shape[0], bounds=bounds)
        if surface.get_size() != target_rect.size:
            surface = pygame.transform.smoothscale(surface, target_rect.size)

        self.screen.blit(surface, target_rect)

    def _draw_overlay(self, data: Dict[str, Any]) -> None:
        overlay_height = 250
        width, height = self.config.window_size
        overlay = pygame.Surface((width, overlay_height), pygame.SRCALPHA)
        overlay.fill(self.config.overlay_color)
        y = self.config.margin

        speed = data.get("speed_kmh", 0.0)
        autopilot = "ON" if data.get("autopilot", False) else "OFF"
        frame = data.get("frame")
        control_mode = data.get("control_mode")

        title_parts = [f"Speed: {speed:05.1f} km/h"]
        if frame is not None:
            title_parts.append(f"Frame: {frame}")
        self._blit_text(overlay, " | ".join(title_parts), (self.config.margin, y))
        y += 26

        if control_mode:
            color = self.config.success_color if "VLA" in str(control_mode) else self.config.text_color
            self._blit_text(overlay, f"Control: {control_mode}", (self.config.margin, y), color=color)
        else:
            self._blit_text(overlay, f"Autopilot: {autopilot}", (self.config.margin, y))
        y += 24

        model_stats = data.get("model_stats") or {}
        if model_stats:
            self._blit_text(overlay, self._format_model_stats(model_stats), (self.config.margin, y), small=True)
            y += 22

        attack_status = data.get("attack_status")
        if attack_status:
            self._blit_text(
                overlay,
                str(attack_status),
                (self.config.margin, y),
                color=self.config.warning_color,
            )
            y += 24

        safety_status = data.get("safety_status") or {}
        safety_message = safety_status.get("message")
        if safety_message:
            self._blit_text(
                overlay,
                str(safety_message),
                (self.config.margin, y),
                color=self.config.warning_color,
            )
            y += 24

        applied_label = "Applied by CARLA autopilot:"
        if data.get("vla_applied"):
            applied_label = "Applied by VLA:"
        elif not data.get("autopilot", False):
            applied_label = "Applied control:"
        self._blit_text(
            overlay,
            applied_label,
            (self.config.margin, y),
            small=True,
        )
        y += 20
        steering = float(data.get("steering", 0.0))
        throttle = float(data.get("throttle", 0.0))
        brake = float(data.get("brake", 0.0))
        self._draw_bar(overlay, "Steer", steering, -1.0, 1.0, y)
        y += 24
        self._draw_bar(overlay, "Throttle", throttle, 0.0, 1.0, y)
        y += 24
        self._draw_bar(overlay, "Brake", brake, 0.0, 1.0, y)
        y += 28

        vla_action = data.get("vla_action")
        if vla_action is not None:
            action_label = "VLA command applied" if data.get("vla_applied") else "VLA predicted - not applied"
            vla_text = (
                f"{action_label}: "
                f"steer {vla_action[0]:+.3f} | "
                f"throttle {vla_action[1]:.3f} | "
                f"brake {vla_action[2]:.3f}"
            )
            self._blit_text(overlay, vla_text, (self.config.margin, y), small=True)
            y += 24

        instruction = data.get("instruction")
        if instruction:
            self._blit_text(overlay, f"Instruction: {instruction}", (self.config.margin, y))

        self.screen.blit(overlay, (0, height - overlay_height))

    def _draw_compare_overlay(self, data: Dict[str, Any], *, overlay_height: int) -> None:
        width, height = self.config.window_size
        overlay = pygame.Surface((width, overlay_height), pygame.SRCALPHA)
        overlay.fill(self.config.overlay_color)
        y = self.config.margin

        speed = data.get("speed_kmh", 0.0)
        frame = data.get("frame")
        title = f"Compare mode: autopilot drives, VLA is shadow only | Speed: {speed:05.1f} km/h"
        if frame is not None:
            title += f" | Frame: {frame}"
        self._blit_text(overlay, title, (self.config.margin, y))
        y += 28

        vla_action = data.get("vla_action")
        expert_action = data.get("expert_action", (0.0, 0.0, 0.0))
        steering_error = float(data.get("steering_error", 0.0))
        if vla_action is None:
            comparison = (
                f"VLA steer: warming up | "
                f"Expert steer: {expert_action[0]:+.2f}"
            )
            color = self.config.warning_color
        else:
            comparison = (
                f"VLA steer: {vla_action[0]:+.2f} | "
                f"Expert steer: {expert_action[0]:+.2f} | "
                f"Error: {steering_error:.2f}"
            )
            color = self.config.success_color
        self._blit_text(overlay, comparison, (self.config.margin, y), color=color)
        y += 26

        model_stats = data.get("model_stats") or {}
        if model_stats:
            self._blit_text(overlay, self._format_model_stats(model_stats), (self.config.margin, y), small=True)
            y += 22

        safety_status = data.get("safety_status") or {}
        safety_message = safety_status.get("message")
        if safety_message:
            self._blit_text(
                overlay,
                str(safety_message),
                (self.config.margin, y),
                color=self.config.warning_color,
            )
        elif data.get("attack_status"):
            self._blit_text(
                overlay,
                str(data["attack_status"]),
                (self.config.margin, y),
                color=self.config.warning_color,
            )
        elif data.get("instruction"):
            self._blit_text(overlay, f"Instruction: {data['instruction']}", (self.config.margin, y))

        self.screen.blit(overlay, (0, height - overlay_height))

    def _draw_bar(
        self,
        surface: Any,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        y: int,
    ) -> None:
        x = self.config.margin
        label_width = 92
        bar_width = 260
        bar_height = 12
        bar_x = x + label_width
        bar_y = y + 4

        self._blit_text(surface, f"{label}: {value:+.2f}", (x, y), small=True)
        pygame.draw.rect(surface, (70, 76, 86), (bar_x, bar_y, bar_width, bar_height), 0)

        clamped = max(min_value, min(max_value, value))
        if min_value < 0.0:
            center_x = bar_x + bar_width // 2
            pygame.draw.line(
                surface,
                (160, 165, 174),
                (center_x, bar_y - 3),
                (center_x, bar_y + bar_height + 3),
                1,
            )
            filled_width = int((abs(clamped) / max_value) * (bar_width / 2))
            if clamped >= 0:
                rect = (center_x, bar_y, filled_width, bar_height)
            else:
                rect = (center_x - filled_width, bar_y, filled_width, bar_height)
        else:
            filled_width = int((clamped / max_value) * bar_width)
            rect = (bar_x, bar_y, filled_width, bar_height)

        color = self.config.warning_color if label == "Brake" and clamped > 0 else self.config.accent_color
        pygame.draw.rect(surface, color, rect, 0)

    def _draw_center_text(self, text: str) -> None:
        rendered = self.font.render(text, True, self.config.text_color)
        rect = rendered.get_rect(center=(self.config.window_size[0] // 2, self.config.window_size[1] // 2))
        self.screen.blit(rendered, rect)

    def _blit_text(
        self,
        surface: Any,
        text: str,
        position: Tuple[int, int],
        *,
        small: bool = False,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        font = self.small_font if small else self.font
        rendered = font.render(text, True, color or self.config.text_color)
        surface.blit(rendered, position)

    def _fit_rect(self, image_width: int, image_height: int, *, bounds: Optional[Any] = None) -> Any:
        if bounds is None:
            bounds = pygame.Rect(0, 0, self.config.window_size[0], self.config.window_size[1])

        scale = min(bounds.width / image_width, bounds.height / image_height)
        target_width = int(image_width * scale)
        target_height = int(image_height * scale)
        x = bounds.x + (bounds.width - target_width) // 2
        y = bounds.y + (bounds.height - target_height) // 2
        return pygame.Rect(x, y, target_width, target_height)

    @staticmethod
    def _extract_overlay_data(state: Optional[Any]) -> Dict[str, Any]:
        if state is None:
            return {}

        if hasattr(state, "to_dict"):
            state = state.to_dict()

        if not isinstance(state, dict):
            return {}

        control = state.get("control", {})
        return {
            "speed_kmh": float(state.get("speed_kmh", 0.0)),
            "steering": float(control.get("steering", control.get("steer", 0.0))),
            "throttle": float(control.get("throttle", 0.0)),
            "brake": float(control.get("brake", 0.0)),
            "autopilot": bool(control.get("autopilot", False)),
        }

    @staticmethod
    def _extract_vla_action(action: Any) -> Optional[Tuple[float, float, float]]:
        if action is None:
            return None

        if hasattr(action, "detach"):
            action = action.detach().cpu().flatten().tolist()
        elif isinstance(action, np.ndarray):
            action = action.reshape(-1).tolist()
        elif isinstance(action, dict):
            return (
                float(action.get("steering", action.get("steer", 0.0))),
                float(action.get("throttle", 0.0)),
                float(action.get("brake", 0.0)),
            )

        values = list(action)
        if len(values) < 3:
            raise ValueError("VLA action must contain steering, throttle, and brake.")
        return float(values[0]), float(values[1]), float(values[2])

    @staticmethod
    def _extract_safety_status(status: Any) -> Dict[str, Any]:
        if status is None:
            return {}
        if hasattr(status, "to_dict"):
            status = status.to_dict()
        if isinstance(status, dict):
            return dict(status)
        return {}

    @staticmethod
    def _format_model_stats(stats: Dict[str, Any]) -> str:
        policy = str(stats.get("policy", ""))
        if policy.startswith("Isaac-GR00T"):
            inference_ms = float(stats.get("inference_ms", stats.get("system1_ms", 0.0)))
            horizon_index = int(stats.get("horizon_index", 0))
            action_key = str(stats.get("action_key", "vehicle_control"))
            queue_remaining = int(stats.get("queue_remaining", 0))
            cached = "cached" if bool(stats.get("used_cached_action", False)) else "refreshed"
            return (
                f"{policy}: {inference_ms:.1f}ms | {action_key}[t={horizon_index}] "
                f"| {cached}, queue={queue_remaining}"
            )
        if policy.startswith("Alpamayo"):
            inference_ms = float(stats.get("inference_ms", stats.get("system2_ms", 0.0)))
            queue_remaining = int(stats.get("queue_remaining", 0))
            points = int(stats.get("trajectory_points", 0))
            cached = "cached" if bool(stats.get("used_cached_action", False)) else "refreshed"
            return (
                f"{policy}: trajectory {inference_ms:.1f}ms | {points} pts "
                f"| {cached}, queue={queue_remaining}"
            )

        system1_ms = float(stats.get("system1_ms", 0.0))
        system2_ms = float(stats.get("system2_ms", 0.0))
        if bool(stats.get("system2_refreshed", False)):
            system2_status = "refreshed"
        else:
            refresh_in = int(stats.get("refresh_in", 0))
            system2_status = f"cached, refresh in {refresh_in} frames"
        return f"System1: {system1_ms:.1f}ms | System2: {system2_ms:.1f}ms ({system2_status})"

    @staticmethod
    def _ensure_pygame_available() -> None:
        if _PYGAME_IMPORT_ERROR is not None:
            raise RuntimeError(
                "pygame is not installed. Activate the vla-av environment or install pygame."
            ) from _PYGAME_IMPORT_ERROR
