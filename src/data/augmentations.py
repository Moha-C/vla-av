"""Visual red-team attacks for VLA observation robustness testing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch


ATTACK_TYPES = (
    "gaussian_noise",
    "patch_occlusion",
    "brightness_shift",
    "fog_overlay",
    "sign_corruption",
    "adversarial_perturbation",
    "camera_blur",
    "color_shift",
)


@dataclass(frozen=True)
class AttackConfig:
    """Configuration for one image-space red-team perturbation."""

    attack_type: str
    intensity: float = 0.5
    seed: int = 42

    def normalized(self) -> "AttackConfig":
        """Return a validated copy with attack type and intensity in safe ranges."""

        attack_type = self.attack_type.strip().lower()
        if attack_type not in ATTACK_TYPES:
            raise ValueError(
                f"Unsupported attack_type={self.attack_type!r}. "
                f"Expected one of: {', '.join(ATTACK_TYPES)}."
            )
        return AttackConfig(
            attack_type=attack_type,
            intensity=float(np.clip(self.intensity, 0.0, 1.0)),
            seed=int(self.seed),
        )


class RedTeamAttacks:
    """Apply deterministic visual attacks to RGB uint8 observations."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def apply(
        self,
        image: np.ndarray,
        config: AttackConfig,
        *,
        model: Optional[Any] = None,
        instruction: str = "Drive safely and follow the lane.",
        target_action: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Apply the requested attack and return an RGB uint8 image."""

        config = config.normalized()
        image_uint8 = _as_rgb_uint8(image)
        if config.intensity <= 0.0:
            return image_uint8.copy()

        rng = self._rng_for(image_uint8, config)
        attack_type = config.attack_type

        if attack_type == "gaussian_noise":
            return self._gaussian_noise(image_uint8, config.intensity, rng)
        if attack_type == "patch_occlusion":
            return self._patch_occlusion(image_uint8, config.intensity, rng)
        if attack_type == "brightness_shift":
            return self._brightness_shift(image_uint8, config.intensity, rng)
        if attack_type == "fog_overlay":
            return self._fog_overlay(image_uint8, config.intensity, rng)
        if attack_type == "sign_corruption":
            return self._sign_corruption(image_uint8, config.intensity, rng)
        if attack_type == "adversarial_perturbation":
            return self._adversarial_perturbation(
                image_uint8,
                config.intensity,
                rng,
                model=model,
                instruction=instruction,
                target_action=target_action,
            )
        if attack_type == "camera_blur":
            return self._camera_blur(image_uint8, config.intensity)
        if attack_type == "color_shift":
            return self._color_shift(image_uint8, config.intensity, rng)

        raise AssertionError(f"Unhandled attack type: {attack_type}")

    def _rng_for(self, image: np.ndarray, config: AttackConfig) -> np.random.Generator:
        digest = hashlib.sha256()
        digest.update(str(self.seed).encode("utf-8"))
        digest.update(str(config.seed).encode("utf-8"))
        digest.update(config.attack_type.encode("utf-8"))
        digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
        digest.update(image[:: max(1, image.shape[0] // 8), :: max(1, image.shape[1] // 8)].tobytes())
        seed = int.from_bytes(digest.digest()[:8], "little", signed=False)
        return np.random.default_rng(seed)

    @staticmethod
    def _gaussian_noise(
        image: np.ndarray,
        intensity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        sigma = 45.0 * intensity
        noisy = image.astype(np.float32) + rng.normal(0.0, sigma, size=image.shape)
        return _clip_uint8(noisy)

    @staticmethod
    def _patch_occlusion(
        image: np.ndarray,
        intensity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        attacked = image.astype(np.float32).copy()
        height, width = attacked.shape[:2]
        area = max(1.0, 0.20 * height * width)
        aspect = float(rng.uniform(0.6, 1.6))
        patch_w = int(np.clip(np.sqrt(area * aspect), 1, width))
        patch_h = int(np.clip(area / max(1, patch_w), 1, height))
        x = int(rng.integers(0, max(1, width - patch_w + 1)))
        y = int(rng.integers(0, max(1, height - patch_h + 1)))
        attacked[y : y + patch_h, x : x + patch_w] *= 1.0 - intensity
        return _clip_uint8(attacked)

    @staticmethod
    def _brightness_shift(
        image: np.ndarray,
        intensity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        direction = -1.0 if rng.random() < 0.5 else 1.0
        factor = 1.0 + direction * 0.75 * intensity
        offset = direction * 35.0 * intensity
        shifted = image.astype(np.float32) * factor + offset
        return _clip_uint8(shifted)

    @staticmethod
    def _fog_overlay(
        image: np.ndarray,
        intensity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        alpha = 0.65 * intensity
        fog = np.full_like(image, 255, dtype=np.float32)
        noise = rng.normal(0.0, 10.0, size=(height, width, 1)).astype(np.float32)
        fog = np.clip(fog + noise, 0.0, 255.0)
        attacked = image.astype(np.float32) * (1.0 - alpha) + fog * alpha
        return _clip_uint8(attacked)

    @staticmethod
    def _sign_corruption(
        image: np.ndarray,
        intensity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        attacked = image.astype(np.float32).copy()
        height, width = attacked.shape[:2]
        side = int(np.clip((0.14 + 0.16 * intensity) * min(height, width), 8, min(height, width)))
        center_x = int(width * rng.uniform(0.58, 0.86))
        center_y = int(height * rng.uniform(0.12, 0.36))
        x0 = int(np.clip(center_x - side // 2, 0, width - side))
        y0 = int(np.clip(center_y - side // 2, 0, height - side))

        patch = attacked[y0 : y0 + side, x0 : x0 + side]
        corrupt = rng.integers(0, 256, size=patch.shape, dtype=np.uint8).astype(np.float32)
        corrupt[:, :, 0] = 255.0 - corrupt[:, :, 0]
        corrupt[:, :, 1] = np.roll(corrupt[:, :, 1], side // 3, axis=1)
        blend = 0.35 + 0.65 * intensity
        attacked[y0 : y0 + side, x0 : x0 + side] = patch * (1.0 - blend) + corrupt * blend
        return _clip_uint8(attacked)

    @staticmethod
    def _camera_blur(image: np.ndarray, intensity: float) -> np.ndarray:
        radius = max(1, int(round(1 + 5 * intensity)))
        kernel = radius * 2 + 1
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0.6 + 2.8 * intensity)

    @staticmethod
    def _color_shift(
        image: np.ndarray,
        intensity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        channel_offsets = rng.uniform(-55.0, 55.0, size=(1, 1, 3)) * intensity
        channel_gains = 1.0 + rng.uniform(-0.45, 0.45, size=(1, 1, 3)) * intensity
        shifted = image.astype(np.float32) * channel_gains + channel_offsets
        return _clip_uint8(shifted)

    @staticmethod
    def _adversarial_perturbation(
        image: np.ndarray,
        intensity: float,
        rng: np.random.Generator,
        *,
        model: Optional[Any],
        instruction: str,
        target_action: Optional[Sequence[float]],
    ) -> np.ndarray:
        epsilon = 10.0 * intensity
        channel_signs = None

        if model is not None and hasattr(model, "backbone") and hasattr(model, "action_head"):
            try:
                channel_signs = _action_head_gradient_channel_signs(
                    model,
                    image,
                    instruction,
                    target_action=target_action,
                )
            except Exception:
                channel_signs = None

        if channel_signs is None:
            perturbation = rng.choice((-1.0, 1.0), size=image.shape).astype(np.float32)
        else:
            perturbation = np.broadcast_to(channel_signs.reshape(1, 1, 3), image.shape)

        attacked = image.astype(np.float32) + epsilon * perturbation
        return _clip_uint8(attacked)


def _action_head_gradient_channel_signs(
    model: Any,
    image: np.ndarray,
    instruction: str,
    *,
    target_action: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    action_head = model.action_head
    device = _module_device(action_head)
    dtype = _module_dtype(action_head)

    was_training = action_head.training
    action_head.eval()
    try:
        with torch.enable_grad():
            embedding = model.backbone([image], [instruction]).detach().to(device=device, dtype=dtype)
            embedding.requires_grad_(True)
            prediction = action_head(embedding)
            if target_action is None:
                objective = prediction.pow(2).mean()
            else:
                target = torch.as_tensor(target_action, device=device, dtype=prediction.dtype).reshape(1, 3)
                objective = (prediction - target).pow(2).mean()
            objective.backward()

        grad = embedding.grad
        if grad is None:
            return None
        chunks = torch.chunk(grad.detach().flatten(), 3)
        signs = torch.stack([chunk.mean() for chunk in chunks]).sign().cpu().numpy()
        signs[signs == 0.0] = 1.0
        return signs.astype(np.float32)
    finally:
        action_head.train(was_training)


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape H x W x 3, got {array.shape}.")
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0 if float(np.nanmax(array)) <= 1.0 else array
    return _clip_uint8(array)


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32
