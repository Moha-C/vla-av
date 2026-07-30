"""Cosmos-Transfer style CARLA semantic/depth to photorealistic RGB conversion."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


CityScapesColor = Tuple[int, int, int]


@dataclass(frozen=True)
class CosmosTransferConfig:
    """Runtime settings for CARLA-to-photo transfer."""

    backend: str = "stylized"
    output_size: Tuple[int, int] = (224, 224)
    cache_dir: str = "data/synthetic/transferred"
    seed: int = 42
    reference_blend: float = 0.78
    transfer_repo_dir: str = "external/cosmos-transfer2.5"
    transfer_python: Optional[str] = None
    disable_guardrails: bool = True


class CosmosTransfer:
    """Convert CARLA semantic/depth frames into photorealistic-looking RGB frames.

    The fast default backend is a local deterministic renderer that validates the
    complete CARLA semantic/depth pipeline. The class keeps the model identifiers
    for the real Cosmos-Transfer path so the backend can be swapped without
    changing callers once the Transfer checkpoints are available locally.
    """

    MODEL_ID = "nvidia/Cosmos-Transfer2.5-2B/general/seg"
    DEPTH_MODEL_ID = "nvidia/Cosmos-Transfer2.5-2B/general/depth"
    LEGACY_MODEL_ID = "nvidia/Cosmos-Transfer1-7B"

    _CITYSCAPES_TO_PHOTO: Dict[CityScapesColor, CityScapesColor] = {
        (128, 64, 128): (64, 64, 67),      # road
        (244, 35, 232): (154, 146, 140),   # sidewalk
        (70, 70, 70): (123, 118, 111),     # building
        (102, 102, 156): (118, 112, 108),  # wall
        (190, 153, 153): (126, 112, 102),  # fence
        (153, 153, 153): (86, 86, 84),     # pole
        (250, 170, 30): (224, 160, 54),    # traffic light
        (220, 220, 0): (190, 177, 75),     # traffic sign
        (107, 142, 35): (62, 106, 54),     # vegetation
        (152, 251, 152): (91, 122, 73),    # terrain
        (70, 130, 180): (158, 182, 197),   # sky
        (220, 20, 60): (88, 48, 50),       # person
        (255, 0, 0): (72, 45, 45),         # rider
        (0, 0, 142): (44, 48, 58),         # car
        (0, 0, 70): (42, 45, 52),          # truck
        (0, 60, 100): (44, 50, 58),        # bus
        (0, 80, 100): (50, 54, 58),        # train
        (0, 0, 230): (38, 42, 54),         # motorcycle
        (119, 11, 32): (36, 38, 42),       # bicycle
    }

    def __init__(self, config: CosmosTransferConfig | None = None) -> None:
        self.config = config or CosmosTransferConfig()
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def transfer(
        self,
        segmentation_image: np.ndarray,
        depth_image: np.ndarray,
        weather_prompt: str = "rainy night urban",
        reference_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one RGB photorealistic-looking frame with shape H x W x 3."""

        segmentation = self._ensure_rgb_uint8(segmentation_image, "segmentation_image")
        depth = self._ensure_depth(depth_image)
        reference = (
            self._ensure_rgb_uint8(reference_image, "reference_image")
            if reference_image is not None
            else None
        )

        backend = self.config.backend.lower().strip()
        if backend not in {"stylized", "mock", "local"}:
            LOGGER.warning(
                "CosmosTransfer backend=%s is not available locally yet; using stylized fallback.",
                self.config.backend,
            )

        return self._stylized_transfer(segmentation, depth, weather_prompt, reference)

    def write_transfer2_5_spec(
        self,
        *,
        name: str,
        prompt: str,
        video_path: str | Path,
        params_path: str | Path,
        negative_prompt: str | None = None,
        edge_control_path: str | Path | None = None,
        seg_control_path: str | Path | None = None,
        depth_control_path: str | Path | None = None,
        guidance: float = 7.0,
        seed: int | None = None,
        edge_weight: float = 0.0,
        edge_threshold: str = "medium",
        seg_weight: float = 0.9,
        depth_weight: float = 0.9,
        vis_weight: float = 0.2,
        resolution: str | None = None,
        max_frames: int | None = None,
        num_video_frames_per_chunk: int | None = None,
        num_steps: int | None = None,
        keep_input_resolution: bool | None = None,
    ) -> Path:
        """Write a Cosmos-Transfer2.5 params JSON for real offline inference."""

        params = {
            "name": name,
            "prompt": prompt,
            "video_path": str(Path(video_path).expanduser().resolve()),
            "guidance": int(guidance),
        }
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        if seed is not None:
            params["seed"] = int(seed)
        if edge_weight > 0:
            params["edge"] = {
                "control_weight": float(edge_weight),
                "preset_edge_threshold": edge_threshold,
            }
            if edge_control_path is not None:
                params["edge"]["control_path"] = str(Path(edge_control_path).expanduser().resolve())
        if seg_control_path is not None and seg_weight > 0:
            params["seg"] = {
                "control_path": str(Path(seg_control_path).expanduser().resolve()),
                "control_weight": float(seg_weight),
            }
        if depth_control_path is not None and depth_weight > 0:
            params["depth"] = {
                "control_path": str(Path(depth_control_path).expanduser().resolve()),
                "control_weight": float(depth_weight),
            }
        if vis_weight > 0:
            # Transfer2.5 can compute visual blur control from video_path on the fly.
            params["vis"] = {"control_weight": float(vis_weight)}
        if resolution is not None:
            params["resolution"] = str(resolution)
        if max_frames is not None:
            params["max_frames"] = int(max_frames)
        if num_video_frames_per_chunk is not None:
            params["num_video_frames_per_chunk"] = int(num_video_frames_per_chunk)
        if num_steps is not None:
            params["num_steps"] = int(num_steps)
        if keep_input_resolution is not None:
            params["keep_input_resolution"] = bool(keep_input_resolution)

        params_file = Path(params_path).expanduser().resolve()
        params_file.parent.mkdir(parents=True, exist_ok=True)
        params_file.write_text(json.dumps(params, indent=2), encoding="utf-8")
        return params_file

    def run_transfer2_5(
        self,
        params_file: str | Path,
        *,
        output_dir: str | Path,
        num_gpus: int = 1,
    ) -> subprocess.CompletedProcess[str]:
        """Run NVIDIA Cosmos-Transfer2.5 inference from an installed repo."""

        repo_dir = Path(self.config.transfer_repo_dir).expanduser().resolve()
        if not repo_dir.exists():
            raise RuntimeError(
                "Cosmos-Transfer2.5 repo is missing. Install it with:\n"
                "cd external\n"
                "git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git\n"
                "cd cosmos-transfer2.5\n"
                "uv python install 3.10\n"
                "printf '3.10\\n' > .python-version\n"
                "uv venv --python 3.10 --clear\n"
                "uv sync --python 3.10 --extra=cu128\n"
                "hf auth login"
            )

        python_path = (
            Path(self.config.transfer_python).expanduser().resolve()
            if self.config.transfer_python
            else repo_dir / ".venv" / "bin" / "python"
        )
        if not python_path.exists():
            raise RuntimeError(
                f"Cosmos-Transfer2.5 Python env not found at {python_path}. "
                "Run `uv sync --python 3.10 --extra=cu128` inside external/cosmos-transfer2.5 first."
            )

        params_path = Path(params_file).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        if num_gpus > 1:
            command = [
                str(python_path),
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(num_gpus),
                "--master_port",
                "12341",
                "-m",
                "examples.inference",
                "-i",
                str(params_path),
                "-o",
                str(output_path),
            ]
        else:
            command = [
                str(python_path),
                "examples/inference.py",
                "-i",
                str(params_path),
                "-o",
                str(output_path),
            ]
        if self.config.disable_guardrails:
            command.append("--disable-guardrails")

        LOGGER.info("Running Cosmos-Transfer2.5 inference: %s", " ".join(command))
        return subprocess.run(
            command,
            cwd=repo_dir,
            env=self._build_transfer_env(python_path),
            check=True,
            text=True,
        )

    def _build_transfer_env(self, python_path: Path) -> dict[str, str]:
        """Expose pip-packaged CUDA shared libraries to Transformer Engine."""

        env = os.environ.copy()
        try:
            site_packages = subprocess.check_output(
                [
                    str(python_path),
                    "-c",
                    "import site; print(site.getsitepackages()[0])",
                ],
                text=True,
            ).strip()
        except subprocess.CalledProcessError as exc:
            LOGGER.warning("Could not inspect Transfer2.5 Python site-packages: %s", exc)
            return env

        nvidia_root = Path(site_packages) / "nvidia"
        if (nvidia_root / "cuda_nvrtc").exists():
            env.setdefault("CUDA_HOME", str(nvidia_root))

        library_dirs: list[Path] = []
        if nvidia_root.exists():
            library_dirs = [
                child / "lib"
                for child in sorted(nvidia_root.iterdir())
                if (child / "lib").exists()
            ]
        if library_dirs:
            self._ensure_unversioned_cuda_symlinks(library_dirs)
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(
                [str(path) for path in library_dirs] + ([existing] if existing else [])
            )
        return env

    @staticmethod
    def _ensure_unversioned_cuda_symlinks(library_dirs: Sequence[Path]) -> None:
        """Some CUDA wheels expose libfoo.so.12, while TE dlopens libfoo.so."""

        for library_dir in library_dirs:
            for versioned_lib in library_dir.glob("*.so.*"):
                marker = ".so."
                if marker not in versioned_lib.name:
                    continue
                unversioned_name = versioned_lib.name.split(marker, maxsplit=1)[0] + ".so"
                unversioned_lib = library_dir / unversioned_name
                if unversioned_lib.exists():
                    continue
                try:
                    unversioned_lib.symlink_to(versioned_lib.name)
                    LOGGER.debug("Created CUDA compatibility symlink %s -> %s", unversioned_lib, versioned_lib.name)
                except OSError as exc:
                    LOGGER.debug("Could not create CUDA compatibility symlink %s: %s", unversioned_lib, exc)

    def _stylized_transfer(
        self,
        segmentation: np.ndarray,
        depth: np.ndarray,
        weather_prompt: str,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        h, w = self.config.output_size[1], self.config.output_size[0]
        segmentation = cv2.resize(segmentation, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        photo = self._semantic_palette_to_photo(segmentation)
        if reference is not None:
            reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            blend = float(np.clip(self.config.reference_blend, 0.0, 1.0))
            photo = reference * blend + photo * (1.0 - blend)
        photo = self._apply_depth_lighting(photo, depth)
        photo = self._add_lane_texture(photo, segmentation, depth)
        photo = self._apply_weather(photo, depth, weather_prompt)
        photo = self._finish_photo(photo, segmentation, weather_prompt)
        return np.clip(photo, 0, 255).astype(np.uint8)

    def _semantic_palette_to_photo(self, segmentation: np.ndarray) -> np.ndarray:
        pixels = segmentation.reshape(-1, 3).astype(np.int16)
        palette = np.asarray(list(self._CITYSCAPES_TO_PHOTO.keys()), dtype=np.int16)
        targets = np.asarray(list(self._CITYSCAPES_TO_PHOTO.values()), dtype=np.float32)
        distances = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        nearest = distances.argmin(axis=1)
        photo = targets[nearest].reshape(segmentation.shape).astype(np.float32)

        # Blend a little of the original segmentation color to preserve object boundaries.
        return photo * 0.88 + segmentation.astype(np.float32) * 0.12

    @staticmethod
    def _apply_depth_lighting(photo: np.ndarray, depth: np.ndarray) -> np.ndarray:
        depth = np.clip(depth.astype(np.float32), 0.0, 1.0)
        near = 1.0 - depth
        atmospheric = 0.72 + 0.38 * near
        shaded = photo * atmospheric[:, :, None]

        horizon_glow = np.linspace(1.08, 0.92, photo.shape[0], dtype=np.float32)[:, None, None]
        return shaded * horizon_glow

    @staticmethod
    def _add_lane_texture(photo: np.ndarray, segmentation: np.ndarray, depth: np.ndarray) -> np.ndarray:
        road_color = np.asarray((128, 64, 128), dtype=np.int16)
        road_mask = (((segmentation.astype(np.int16) - road_color) ** 2).sum(axis=2) < 1600).astype(np.float32)
        if road_mask.max() <= 0:
            return photo

        h, w = road_mask.shape
        grain_x = np.sin(np.linspace(0, 42, w, dtype=np.float32))[None, :]
        grain_y = np.cos(np.linspace(0, 28, h, dtype=np.float32))[:, None]
        asphalt_grain = (grain_x + grain_y) * 3.0
        photo += asphalt_grain[:, :, None] * road_mask[:, :, None]

        edges = cv2.Canny(segmentation, 80, 150).astype(np.float32) / 255.0
        photo += edges[:, :, None] * road_mask[:, :, None] * 24.0 * (1.0 - depth[:, :, None])
        return photo

    def _apply_weather(self, photo: np.ndarray, depth: np.ndarray, weather_prompt: str) -> np.ndarray:
        prompt = weather_prompt.lower()
        result = photo.astype(np.float32)
        far = np.clip(depth.astype(np.float32), 0.0, 1.0)

        if "night" in prompt:
            result *= np.asarray([0.48, 0.52, 0.66], dtype=np.float32)
            result += np.asarray([10, 14, 28], dtype=np.float32)
        elif "dusk" in prompt or "sunset" in prompt:
            result *= np.asarray([1.10, 0.92, 0.78], dtype=np.float32)
            result += np.asarray([18, 8, 0], dtype=np.float32)

        if "rain" in prompt:
            result *= 0.82
            streaks = self._rain_streaks(result.shape[:2], weather_prompt)
            result = np.maximum(result, streaks[:, :, None])
            result[:, :, 2] += 10.0

        if "fog" in prompt:
            fog_strength = 0.25 + 0.55 * far
            fog_color = np.asarray([188, 196, 198], dtype=np.float32)
            result = result * (1.0 - fog_strength[:, :, None]) + fog_color * fog_strength[:, :, None]

        if "snow" in prompt:
            result = result * 0.86 + np.asarray([35, 38, 42], dtype=np.float32)
            snow = self._snow_noise(result.shape[:2], weather_prompt)
            result = np.maximum(result, snow[:, :, None])

        return result

    def _finish_photo(
        self,
        photo: np.ndarray,
        segmentation: np.ndarray,
        weather_prompt: str,
    ) -> np.ndarray:
        seed = self._seed_for(segmentation, weather_prompt)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 3.0, size=photo.shape).astype(np.float32)
        photo = photo + noise
        photo = cv2.bilateralFilter(np.clip(photo, 0, 255).astype(np.uint8), 5, 32, 32).astype(np.float32)

        h, w = photo.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        radius = ((xx - w / 2.0) / max(w, 1)) ** 2 + ((yy - h / 2.0) / max(h, 1)) ** 2
        vignette = 1.0 - np.clip(radius * 0.55, 0.0, 0.22)
        return photo * vignette[:, :, None]

    def _rain_streaks(self, shape: Tuple[int, int], weather_prompt: str) -> np.ndarray:
        h, w = shape
        rng = np.random.default_rng(self._seed_for(np.zeros((1, 1, 3), dtype=np.uint8), weather_prompt))
        streaks = np.zeros((h, w), dtype=np.float32)
        count = max(12, (h * w) // 900)
        for _ in range(count):
            x = int(rng.integers(0, max(1, w)))
            y = int(rng.integers(0, max(1, h)))
            length = int(rng.integers(8, 22))
            cv2.line(streaks, (x, y), (min(w - 1, x + 4), min(h - 1, y + length)), 190, 1)
        return cv2.GaussianBlur(streaks, (3, 3), 0)

    def _snow_noise(self, shape: Tuple[int, int], weather_prompt: str) -> np.ndarray:
        rng = np.random.default_rng(self._seed_for(np.zeros((1, 1, 3), dtype=np.uint8), weather_prompt + "snow"))
        flakes = rng.random(shape, dtype=np.float32)
        snow = np.where(flakes > 0.985, 230.0, 0.0).astype(np.float32)
        return cv2.GaussianBlur(snow, (3, 3), 0)

    def _seed_for(self, image: np.ndarray, text: str) -> int:
        digest = hashlib.sha256()
        digest.update(str(self.config.seed).encode("utf-8"))
        digest.update(text.encode("utf-8"))
        digest.update(image[:8, :8].tobytes())
        return int.from_bytes(digest.digest()[:8], "big")

    @staticmethod
    def _ensure_rgb_uint8(image: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"{name} must have shape H x W x 3, got {array.shape}.")
        return np.clip(array, 0, 255).astype(np.uint8)

    @staticmethod
    def _ensure_depth(depth_image: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_image)
        if depth.ndim == 3:
            depth = depth.mean(axis=2)
        if depth.ndim != 2:
            raise ValueError(f"depth_image must have shape H x W or H x W x 3, got {depth.shape}.")
        depth = depth.astype(np.float32)
        max_value = float(depth.max()) if depth.size else 0.0
        if max_value > 1.0:
            depth = depth / 255.0
        return np.clip(depth, 0.0, 1.0)
