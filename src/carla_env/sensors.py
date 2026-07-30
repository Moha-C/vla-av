"""Sensor helpers for CARLA observations."""

from __future__ import annotations

import logging
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover - exercised only without CARLA installed.
    carla = None
    _CARLA_IMPORT_ERROR = exc
else:
    _CARLA_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RGBCameraConfig:
    """Configuration for the ego RGB camera sensor."""

    width: int = 224
    height: int = 224
    fov: float = 90.0
    sensor_tick: float = 0.05
    location: Tuple[float, float, float] = (1.5, 0.0, 2.4)
    rotation: Tuple[float, float, float] = (-15.0, 0.0, 0.0)
    blueprint_id: str = "sensor.camera.rgb"


@dataclass(frozen=True)
class CameraFrame:
    """One decoded RGB camera observation from CARLA."""

    frame_id: int
    timestamp: float
    image: np.ndarray


class RGBCameraSensor:
    """Attach an RGB camera to a vehicle and expose the latest NumPy frame."""

    camera_label = "RGB"

    def __init__(
        self,
        config: Optional[RGBCameraConfig] = None,
        *,
        client: Optional[Any] = None,
    ) -> None:
        self.config = config or RGBCameraConfig()
        self.client = client
        self.actor: Optional[Any] = None
        self._latest_frame: Optional[CameraFrame] = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def spawn(self, world: Any, parent_actor: Any) -> Any:
        """Create the CARLA camera actor and start listening for RGB frames."""

        self._ensure_carla_available()

        blueprint = world.get_blueprint_library().find(self.config.blueprint_id)
        blueprint.set_attribute("image_size_x", str(self.config.width))
        blueprint.set_attribute("image_size_y", str(self.config.height))
        blueprint.set_attribute("fov", str(self.config.fov))
        blueprint.set_attribute("sensor_tick", str(self.config.sensor_tick))

        transform = self._make_transform()
        with self._lock:
            self._latest_frame = None
            self._new_frame_event.clear()
        self.actor = world.spawn_actor(blueprint, transform, attach_to=parent_actor)

        if self.client is not None and hasattr(self.client, "register_actor"):
            self.client.register_actor(self.actor)

        weak_self = weakref.ref(self)
        self.actor.listen(lambda image: RGBCameraSensor._on_image(weak_self, image))
        LOGGER.info(
            "Spawned %s camera %sx%s on actor %s",
            self.camera_label,
            self.config.width,
            self.config.height,
            parent_actor.id,
        )
        return self.actor

    def get_latest_frame(self, *, copy: bool = True) -> Optional[CameraFrame]:
        """Return the newest decoded frame, or None if no frame has arrived yet."""

        with self._lock:
            if self._latest_frame is None:
                return None
            if not copy:
                return self._latest_frame

            return CameraFrame(
                frame_id=self._latest_frame.frame_id,
                timestamp=self._latest_frame.timestamp,
                image=self._latest_frame.image.copy(),
            )

    def get_latest_image(self, *, copy: bool = True) -> Optional[np.ndarray]:
        """Return only the latest RGB image array with shape H x W x 3."""

        frame = self.get_latest_frame(copy=copy)
        if frame is None:
            return None
        return frame.image

    def wait_for_frame(self, timeout_seconds: float = 2.0) -> CameraFrame:
        """Block until at least one frame is available or raise TimeoutError."""

        if not self._new_frame_event.wait(timeout_seconds):
            raise TimeoutError(
                f"No RGB camera frame received within {timeout_seconds:.1f}s."
            )

        frame = self.get_latest_frame(copy=True)
        if frame is None:
            raise TimeoutError("RGB camera event fired but no frame was stored.")
        return frame

    def stop(self) -> None:
        """Stop the CARLA sensor callback without destroying the actor."""

        if self.actor is None:
            return

        try:
            self.actor.stop()
        except RuntimeError as exc:
            LOGGER.debug("Ignoring CARLA camera stop failure: %s", exc)

    def destroy(self) -> None:
        """Stop and destroy the CARLA camera actor."""

        if self.actor is None:
            return

        actor = self.actor
        self.actor = None

        try:
            actor.stop()
            time.sleep(0.05)
        except RuntimeError as exc:
            LOGGER.debug("Ignoring CARLA camera stop failure: %s", exc)

        if self.client is not None and hasattr(self.client, "unregister_actor"):
            self.client.unregister_actor(actor)

        try:
            actor.destroy()
        except RuntimeError as exc:
            LOGGER.debug("Ignoring CARLA camera destroy failure: %s", exc)

    @staticmethod
    def carla_image_to_rgb(image: Any) -> np.ndarray:
        """Convert CARLA's raw BGRA image buffer into uint8 RGB NumPy format."""

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        return array[:, :, :3][:, :, ::-1].copy()

    def _decode_image(self, image: Any) -> np.ndarray:
        return self.carla_image_to_rgb(image)

    @staticmethod
    def _on_image(weak_self: weakref.ReferenceType["RGBCameraSensor"], image: Any) -> None:
        sensor = weak_self()
        if sensor is None:
            return

        rgb_image = sensor._decode_image(image)
        frame = CameraFrame(
            frame_id=int(image.frame),
            timestamp=float(image.timestamp),
            image=rgb_image,
        )

        with sensor._lock:
            sensor._latest_frame = frame
            sensor._new_frame_event.set()

    def _make_transform(self) -> Any:
        location = carla.Location(
            x=self.config.location[0],
            y=self.config.location[1],
            z=self.config.location[2],
        )
        rotation = carla.Rotation(
            pitch=self.config.rotation[0],
            yaw=self.config.rotation[1],
            roll=self.config.rotation[2],
        )
        return carla.Transform(location, rotation)

    @staticmethod
    def _ensure_carla_available() -> None:
        if _CARLA_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The CARLA Python package is not installed. Activate the vla-av "
                "environment or install carla==0.9.15."
            ) from _CARLA_IMPORT_ERROR


@dataclass(frozen=True)
class SemanticSegmentationCameraConfig(RGBCameraConfig):
    """Configuration for the ego semantic segmentation camera sensor."""

    blueprint_id: str = "sensor.camera.semantic_segmentation"


class SemanticSegmentationCameraSensor(RGBCameraSensor):
    """Attach a semantic segmentation camera and expose CityScapes-color frames."""

    camera_label = "semantic segmentation"

    def __init__(
        self,
        config: Optional[SemanticSegmentationCameraConfig] = None,
        *,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(config or SemanticSegmentationCameraConfig(), client=client)

    def _decode_image(self, image: Any) -> np.ndarray:
        if carla is not None and hasattr(carla, "ColorConverter"):
            image.convert(carla.ColorConverter.CityScapesPalette)
        return self.carla_image_to_rgb(image)


@dataclass(frozen=True)
class DepthCameraConfig(RGBCameraConfig):
    """Configuration for the ego depth camera sensor."""

    blueprint_id: str = "sensor.camera.depth"


class DepthCameraSensor(RGBCameraSensor):
    """Attach a depth camera and expose normalized grayscale RGB frames."""

    camera_label = "depth"

    def __init__(
        self,
        config: Optional[DepthCameraConfig] = None,
        *,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(config or DepthCameraConfig(), client=client)

    def _decode_image(self, image: Any) -> np.ndarray:
        depth = self.carla_image_to_depth(image)
        depth_uint8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(depth_uint8[:, :, None], 3, axis=2)

    @staticmethod
    def carla_image_to_depth(image: Any) -> np.ndarray:
        """Decode CARLA depth buffer into normalized float depth in [0, 1]."""

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4)).astype(np.float32)
        blue = array[:, :, 0]
        green = array[:, :, 1]
        red = array[:, :, 2]
        normalized = (red + green * 256.0 + blue * 65536.0) / (256.0**3 - 1.0)
        return normalized.astype(np.float32)


SemanticSegmentationCamera = SemanticSegmentationCameraSensor
DepthCamera = DepthCameraSensor


def _is_actor_alive(actor: Any) -> bool:
    try:
        return bool(actor.is_alive)
    except RuntimeError:
        return False
