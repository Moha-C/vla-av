"""Synthetic driving data generation interface for NVIDIA Cosmos."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)

SCENARIOS = [
    {"weather": "clear", "time_of_day": "day", "road_type": "urban"},
    {"weather": "rainy", "time_of_day": "day", "road_type": "urban"},
    {"weather": "foggy", "time_of_day": "night", "road_type": "highway"},
    {"weather": "clear", "time_of_day": "night", "road_type": "urban"},
    {"weather": "clear", "time_of_day": "day", "road_type": "urban_corrupted"},
    {"weather": "clear", "time_of_day": "day", "road_type": "urban_occluded"},
]


@dataclass(frozen=True)
class CosmosConfig:
    """Configuration for mock or real Cosmos synthetic generation."""

    mode: str = "mock"
    cosmos_version: str = "predict2"
    model_variant: str = "Cosmos-Predict2-2B"
    output_dir: str = "data/synthetic"
    n_frames: int = 49
    prompt_template: str = (
        "Autonomous vehicle driving in {weather} weather, "
        "{time_of_day}, {road_type} road, camera view"
    )
    image_size: int = 224
    seed: int = 42
    hf_model_id: str = "nvidia/Cosmos-Predict2-2B-Video2World"
    text2image_model_id: str = "nvidia/Cosmos-Predict2-2B-Text2Image"
    sample_model_id: str = "nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned"
    predict25_model_id: str = "nvidia/Cosmos-Predict2.5-2B"
    predict25_revision: str = "diffusers/base/post-trained"
    predict25_num_steps: int = 36
    cosmos_repo_dir: str = "external/cosmos-predict2-main"
    cosmos_python: Optional[str] = None
    resolution: str = "480"
    fps: int = 10
    guidance: float = 7.0
    negative_prompt: Optional[str] = None
    disable_guardrail: bool = True
    disable_prompt_refiner: bool = True
    offload_guardrail: bool = True
    offload_prompt_refiner: bool = True
    offload_text_encoder: bool = True
    downcast_text_encoder: bool = True
    keep_videos: bool = True
    device: str = "cuda"
    dtype: str = "float16"
    conditional_input_path: Optional[str] = None
    num_conditional_frames: int = 1


class CosmosGenerator:
    """Generate RGB frame sequences with mock data or a real Cosmos backend."""

    MODEL_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
    TEXT2IMAGE_MODEL_ID = "nvidia/Cosmos-Predict2-2B-Text2Image"
    SAMPLE_ACTION_CONDITIONED_MODEL_ID = "nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned"
    ATTACK_PROMPTS = {
        "sign_corruption": "Urban autonomous driving scene with graffiti-covered road signs and confusing altered traffic signs, front dashcam view",
        "fog_attack": "Extremely dense fog reducing visibility to 5 meters during autonomous urban driving, front dashcam view",
        "glare_attack": "Blinding sunlight glare directly in the autonomous vehicle camera view on a city road",
        "occlusion": "Large truck blocking the road view ahead during autonomous city driving, front dashcam view",
    }

    def __init__(self, config: Optional[CosmosConfig] = None) -> None:
        self.config = config or CosmosConfig()
        self.mode = self.config.mode.lower()
        if self.mode not in {"mock", "real"}:
            raise ValueError(f"Unsupported Cosmos generation mode: {self.config.mode}")

        self._pipeline = None
        self._rng = np.random.default_rng(self.config.seed)

    def load_model(self, *, require_text2image: bool = True) -> None:
        """Validate the local Cosmos installation and checkpoints."""

        if self.mode != "real":
            return

        cosmos_version = self.config.cosmos_version.lower()
        cosmos_repo = self._cosmos_repo_dir()
        cosmos_python = self._cosmos_python()
        if not cosmos_repo.exists():
            raise RuntimeError(
                f"Cosmos repo not found: {cosmos_repo}. Run the Cosmos install step first."
            )
        if not cosmos_python.exists():
            raise RuntimeError(
                f"Cosmos Python env not found: {cosmos_python}. Expected the official "
                "Cosmos uv environment inside the configured repo."
            )

        if cosmos_version in {"predict2", "2", "cosmos-predict2"}:
            self._verify_local_checkpoints(require_text2image=require_text2image)
        elif cosmos_version in {"predict2.5", "2.5", "cosmos-predict2.5"}:
            self._verify_predict25_install()
        else:
            raise ValueError(f"Unsupported Cosmos version: {self.config.cosmos_version}")
        self._warn_if_huggingface_access_missing()
        LOGGER.info("Using Cosmos repo %s with Python %s", cosmos_repo, cosmos_python)

    def generate_scenario(
        self,
        weather_or_prompt: str,
        time_of_day: Optional[str] = None,
        road_type: Optional[str] = None,
        n_frames: Optional[int] = None,
        input_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Return RGB frames for one driving scenario."""

        frame_count = n_frames or self.config.n_frames
        if time_of_day is None and road_type is None:
            prompt = weather_or_prompt
            weather = "custom"
            time_of_day = "custom"
            road_type = "custom"
        else:
            weather = weather_or_prompt
            if time_of_day is None or road_type is None:
                raise ValueError("time_of_day and road_type must be provided together.")
            prompt = self._build_prompt(weather, time_of_day, road_type)

        if self.mode == "mock":
            return self._generate_mock_frames(prompt, weather, time_of_day, road_type, frame_count)
        return self._generate_real_frames(prompt, frame_count, input_path=input_path)

    def generate_attack_scenario(
        self,
        attack_type: str,
        n_frames: Optional[int] = None,
        *,
        input_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Generate a real or mock scenario for one red-team attack type."""

        key = attack_type.strip().lower()
        if key not in self.ATTACK_PROMPTS:
            raise ValueError(
                f"Unsupported attack_type={attack_type!r}. "
                f"Expected one of: {', '.join(self.ATTACK_PROMPTS)}."
            )
        return self.generate_scenario(
            self.ATTACK_PROMPTS[key],
            n_frames=n_frames or self.config.n_frames,
            input_path=input_path,
        )

    def _build_prompt(self, weather: str, time_of_day: str, road_type: str) -> str:
        return self.config.prompt_template.format(
            weather=weather,
            time_of_day=time_of_day,
            road_type=road_type,
        )

    def _generate_mock_frames(
        self,
        prompt: str,
        weather: str,
        time_of_day: str,
        road_type: str,
        n_frames: int,
    ) -> List[np.ndarray]:
        base_color = self._scenario_color(weather, time_of_day, road_type)
        prompt_seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(self.config.seed ^ prompt_seed)

        frames: List[np.ndarray] = []
        for frame_idx in range(n_frames):
            frame = self._mock_background(rng, base_color)
            self._draw_mock_road(frame, road_type, frame_idx)
            self._apply_mock_weather(frame, rng, weather, time_of_day)
            if road_type.endswith("corrupted"):
                self._apply_corruption(frame, rng, frame_idx)
            if road_type.endswith("occluded"):
                self._apply_occlusion(frame, rng, frame_idx)
            frames.append(frame.astype(np.uint8))
        return frames

    def _generate_real_frames(
        self,
        prompt: str,
        n_frames: int,
        *,
        input_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        conditioning_path = input_path or self.config.conditional_input_path
        self.load_model(require_text2image=not bool(conditioning_path))
        if self.config.cosmos_version.lower() in {"predict2.5", "2.5", "cosmos-predict2.5"}:
            video_path = self._run_predict25(prompt, n_frames, input_path=conditioning_path)
            return self._extract_video_frames(video_path, n_frames)

        if conditioning_path:
            video_path = self._run_video2world(prompt, Path(conditioning_path))
        else:
            video_path = self._run_text2world(prompt)
        return self._extract_video_frames(video_path, n_frames)

    def _run_predict25(
        self,
        prompt: str,
        n_frames: int,
        *,
        input_path: Optional[str] = None,
    ) -> Path:
        """Run Cosmos-Predict2.5 through its unified Diffusers inference script."""

        cosmos_repo = self._cosmos_repo_dir()
        cosmos_python = self._cosmos_python()
        script_path = cosmos_repo / "scripts" / "diffusers_inference.py"
        if not script_path.exists():
            raise RuntimeError(
                "Cosmos-Predict2.5 diffusers inference script not found: "
                f"{script_path}. Clone/setup nvidia-cosmos/cosmos-predict2.5 first."
            )

        inference_type = "text2world"
        resolved_input: Optional[Path] = None
        if input_path:
            resolved_input = Path(input_path).expanduser().resolve()
            if not resolved_input.exists():
                raise FileNotFoundError(f"Cosmos conditioning input not found: {resolved_input}")
            inference_type = self._predict25_inference_type(resolved_input)

        video_dir = Path(self.config.output_dir) / "_cosmos25_videos"
        request_dir = Path(self.config.output_dir) / "_cosmos25_requests"
        video_dir.mkdir(parents=True, exist_ok=True)
        request_dir.mkdir(parents=True, exist_ok=True)
        slug = self._prompt_slug(prompt + str(resolved_input or ""))
        request_path = request_dir / f"{slug}.json"
        save_path = video_dir / f"{slug}.mp4"

        request = {
            "inference_type": inference_type,
            "name": slug,
            "prompt": prompt,
            "negative_prompt": self.config.negative_prompt or self._default_predict25_negative_prompt(),
        }
        if resolved_input is not None:
            request["input_path"] = str(resolved_input)
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

        cmd = [
            str(cosmos_python),
            str(script_path),
            "--input_path",
            str(request_path.resolve()),
            "--output_path",
            str(save_path.resolve()),
            "--num_output_frames",
            str(max(1, int(n_frames))),
            "--model_id",
            self.config.predict25_model_id,
            "--revision",
            self.config.predict25_revision,
            "--num_steps",
            str(max(1, int(self.config.predict25_num_steps))),
            "--seed",
            str(self.config.seed),
            "--device",
            self.config.device,
            "--disable-safety-checker",
        ]

        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env = self._with_cosmos_cuda_libraries(env)

        LOGGER.info("Running Cosmos-Predict2.5 %s generation: %s", inference_type, " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                cwd=cosmos_repo,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Cosmos-Predict2.5 generation failed. Make sure the "
                "nvidia-cosmos/cosmos-predict2.5 repo is installed, Hugging Face "
                "access for nvidia/Cosmos-Predict2.5-2B is accepted, and no other "
                "large GPU job is using the A6000."
            ) from exc

        if not save_path.exists():
            raise RuntimeError(f"Cosmos-Predict2.5 did not create expected video: {save_path}")
        return save_path

    @staticmethod
    def _predict25_inference_type(input_path: Path) -> str:
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        suffix = input_path.suffix.lower()
        if suffix in image_suffixes:
            return "image2world"
        if suffix in video_suffixes:
            return "video2world"
        raise ValueError(
            f"Unsupported Cosmos-Predict2.5 conditioning input type: {input_path.suffix}"
        )

    @staticmethod
    def _default_predict25_negative_prompt() -> str:
        return (
            "cartoon, illustration, CGI-looking render, distorted vehicles, warped lane markings, "
            "melting buildings, flicker, temporal jitter, fisheye distortion, low resolution, "
            "compression artifacts, oversharpened texture, impossible geometry, duplicated objects"
        )

    def _run_video2world(self, prompt: str, input_path: Path) -> Path:
        cosmos_repo = self._cosmos_repo_dir()
        cosmos_python = self._cosmos_python()
        resolved_input = input_path.expanduser().resolve()
        if not resolved_input.exists():
            raise FileNotFoundError(f"Cosmos conditioning input not found: {resolved_input}")

        video_dir = Path(self.config.output_dir) / "_cosmos_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_name = self._prompt_slug(prompt + str(resolved_input)) + ".mp4"
        save_path = video_dir / video_name

        cmd = [
            str(cosmos_python),
            "-m",
            "examples.video2world",
            "--model_size",
            "2B",
            "--resolution",
            str(self.config.resolution),
            "--fps",
            str(self.config.fps),
            "--input_path",
            str(resolved_input),
            "--num_conditional_frames",
            str(self.config.num_conditional_frames),
            "--prompt",
            prompt,
            "--save_path",
            str(save_path.resolve()),
            "--seed",
            str(self.config.seed),
            "--guidance",
            str(self.config.guidance),
            "--num_gpus",
            "1",
        ]
        if self.config.disable_guardrail:
            cmd.append("--disable_guardrail")
        if self.config.disable_prompt_refiner:
            cmd.append("--disable_prompt_refiner")
        if self.config.offload_guardrail:
            cmd.append("--offload_guardrail")
        if self.config.offload_prompt_refiner:
            cmd.append("--offload_prompt_refiner")
        if self.config.offload_text_encoder:
            cmd.append("--offload_text_encoder")
        if self.config.downcast_text_encoder:
            cmd.append("--downcast_text_encoder")
        if self.config.negative_prompt:
            cmd.extend(["--negative_prompt", self.config.negative_prompt])

        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env = self._with_cosmos_cuda_libraries(env)

        LOGGER.info("Running Cosmos image/video-conditioned video2world: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                cwd=cosmos_repo,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Cosmos video2world generation failed. Use a clear dashcam-like "
                "input image for --input-path, keep guardrail/prompt-refiner disabled "
                "on a 48GB GPU, and ensure the 480p/10fps Video2World checkpoint exists."
            ) from exc

        if not save_path.exists():
            raise RuntimeError(f"Cosmos did not create expected video: {save_path}")
        return save_path

    def _run_text2world(self, prompt: str) -> Path:
        cosmos_repo = self._cosmos_repo_dir()
        cosmos_python = self._cosmos_python()
        video_dir = Path(self.config.output_dir) / "_cosmos_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_name = self._prompt_slug(prompt) + ".mp4"
        save_path = video_dir / video_name

        cmd = [
            str(cosmos_python),
            "-m",
            "examples.text2world",
            "--model_size",
            "2B",
            "--resolution",
            str(self.config.resolution),
            "--fps",
            str(self.config.fps),
            "--prompt",
            prompt,
            "--save_path",
            str(save_path.resolve()),
            "--seed",
            str(self.config.seed),
            "--guidance",
            str(self.config.guidance),
            "--num_gpus",
            "1",
        ]
        if self.config.disable_guardrail:
            cmd.append("--disable_guardrail")
        if self.config.disable_prompt_refiner:
            cmd.append("--disable_prompt_refiner")
        if self.config.offload_guardrail:
            cmd.append("--offload_guardrail")
        if self.config.offload_prompt_refiner:
            cmd.append("--offload_prompt_refiner")
        if self.config.offload_text_encoder:
            cmd.append("--offload_text_encoder")
        if self.config.downcast_text_encoder:
            cmd.append("--downcast_text_encoder")
        if self.config.negative_prompt:
            cmd.extend(["--negative_prompt", self.config.negative_prompt])

        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env = self._with_cosmos_cuda_libraries(env)

        LOGGER.info("Running Cosmos text2world generation: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                cwd=cosmos_repo,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Cosmos text2world generation failed. If the error mentions "
                "flash_attn_2 or libnvrtc, finish the official Cosmos GPU dependency "
                "setup in external/cosmos-predict2-main before retrying. If it "
                "mentions CUDA out of memory while loading Llama Guard or Cosmos "
                "Reason, rerun without --enable-guardrail and without "
                "--enable-prompt-refiner."
            ) from exc

        if not save_path.exists():
            raise RuntimeError(f"Cosmos did not create expected video: {save_path}")
        return save_path

    def _with_cosmos_cuda_libraries(self, env: Dict[str, str]) -> Dict[str, str]:
        """Expose pip-installed CUDA libs so Transformer Engine can find NVRTC."""

        nvidia_root = (
            self._cosmos_repo_dir()
            / ".venv"
            / "lib"
            / "python3.10"
            / "site-packages"
            / "nvidia"
        )
        if not nvidia_root.exists():
            return env

        lib_dirs = sorted(path for path in nvidia_root.glob("*/lib") if path.is_dir())
        existing = env.get("LD_LIBRARY_PATH", "")
        lib_path = ":".join(str(path) for path in lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}" if existing else lib_path
        env.setdefault("CUDA_HOME", str(nvidia_root))
        env.setdefault("CUDA_PATH", str(nvidia_root))
        return env

    def _extract_video_frames(self, video_path: Path, n_frames: int) -> List[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open generated Cosmos video: {video_path}")

        decoded: List[np.ndarray] = []
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                decoded.append(frame_rgb)
        finally:
            capture.release()

        if not decoded:
            raise RuntimeError(f"Cosmos video contained no readable frames: {video_path}")

        indices = np.linspace(0, len(decoded) - 1, num=n_frames).round().astype(int)
        frames: List[np.ndarray] = []
        for index in indices:
            frame = decoded[int(index)]
            resized = cv2.resize(
                frame,
                (self.config.image_size, self.config.image_size),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(resized.astype(np.uint8))
        return frames

    def _verify_local_checkpoints(self, *, require_text2image: bool = True) -> None:
        checkpoints = self._cosmos_repo_dir() / "checkpoints"
        text2image = checkpoints / "nvidia" / "Cosmos-Predict2-2B-Text2Image" / "model.pt"
        video2world = (
            checkpoints
            / "nvidia"
            / "Cosmos-Predict2-2B-Video2World"
            / f"model-{self.config.resolution}p-{self.config.fps}fps.pt"
        )
        required_paths = [video2world]
        if require_text2image:
            required_paths.append(text2image)
        missing = [path for path in required_paths if not path.exists()]
        if missing:
            commands = (
                "cd external/cosmos-predict2-main\n"
                f".venv/bin/python -m scripts.download_checkpoints --model_types text2image --model_sizes 2B\n"
                f".venv/bin/python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B "
                f"--resolution {self.config.resolution} --fps {self.config.fps}"
            )
            missing_text = "\n".join(str(path) for path in missing)
            raise RuntimeError(
                "Missing local Cosmos checkpoints:\n"
                f"{missing_text}\n\n"
                "Download them with:\n"
                f"{commands}"
            )

    def _verify_predict25_install(self) -> None:
        script_path = self._cosmos_repo_dir() / "scripts" / "diffusers_inference.py"
        if not script_path.exists():
            raise RuntimeError(
                "Cosmos-Predict2.5 is not installed yet. Expected:\n"
                f"{script_path}\n\n"
                "Install with:\n"
                "cd external\n"
                "git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git\n"
                "cd cosmos-predict2.5\n"
                "uv python install\n"
                "uv sync --extra=cu128"
            )

    def _warn_if_huggingface_access_missing(self) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required in vla-av for Cosmos access checks. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        api = HfApi()
        if self.config.cosmos_version.lower() in {"predict2.5", "2.5", "cosmos-predict2.5"}:
            repo_ids = (self.config.predict25_model_id,)
        else:
            repo_ids = (self.config.text2image_model_id, self.config.hf_model_id)
        for repo_id in repo_ids:
            try:
                api.model_info(repo_id)
            except Exception as exc:
                LOGGER.warning(
                    "Could not verify Hugging Face access for %s. Continuing because "
                    "local checkpoints exist. Original error: %s",
                    repo_id,
                    exc,
                )

    def _cosmos_repo_dir(self) -> Path:
        return Path(self.config.cosmos_repo_dir).expanduser().resolve()

    def _cosmos_python(self) -> Path:
        if self.config.cosmos_python:
            return Path(self.config.cosmos_python).expanduser().resolve()
        return self._cosmos_repo_dir() / ".venv" / "bin" / "python"

    def _prompt_slug(self, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt[:60])
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return f"{cleaned[:48]}_{digest}"

    def _mock_background(self, rng: np.random.Generator, base_color: np.ndarray) -> np.ndarray:
        size = self.config.image_size
        noise = rng.normal(loc=0.0, scale=32.0, size=(size, size, 3))
        gradient = np.linspace(0.85, 1.15, size, dtype=np.float32).reshape(size, 1, 1)
        frame = base_color.reshape(1, 1, 3) * gradient + noise
        return np.clip(frame, 0, 255)

    def _draw_mock_road(self, frame: np.ndarray, road_type: str, frame_idx: int) -> None:
        height, width, _ = frame.shape
        horizon = int(height * 0.43)
        road_color = np.asarray([48, 50, 54], dtype=np.float32)
        lane_color = np.asarray([235, 220, 110], dtype=np.float32)

        left_bottom = int(width * 0.12)
        right_bottom = int(width * 0.88)
        center = width // 2 + int(np.sin(frame_idx * 0.08) * 8)

        for y in range(horizon, height):
            t = (y - horizon) / max(1, height - horizon)
            left = int((1.0 - t) * (center - 18) + t * left_bottom)
            right = int((1.0 - t) * (center + 18) + t * right_bottom)
            frame[y, max(0, left) : min(width, right)] = road_color + t * 30.0

            if y % 22 < 12:
                lane_half_width = max(1, int(2 + t * 3))
                lane_center = int(center + np.sin(frame_idx * 0.03) * 5)
                frame[y, lane_center - lane_half_width : lane_center + lane_half_width] = lane_color

        if "highway" in road_type:
            frame[horizon:height:18, width // 4 : width // 4 + 2] = [245, 245, 245]
            frame[horizon:height:18, 3 * width // 4 : 3 * width // 4 + 2] = [245, 245, 245]

    def _apply_mock_weather(
        self,
        frame: np.ndarray,
        rng: np.random.Generator,
        weather: str,
        time_of_day: str,
    ) -> None:
        if time_of_day == "night":
            frame *= np.asarray([0.35, 0.40, 0.55], dtype=np.float32).reshape(1, 1, 3)

        if weather == "rainy":
            drops = rng.integers(0, self.config.image_size, size=(120, 2))
            for y, x in drops:
                y2 = min(self.config.image_size, y + 8)
                frame[y:y2, x : min(self.config.image_size, x + 1)] = [180, 200, 230]

        if weather == "foggy":
            fog = np.asarray([190, 195, 200], dtype=np.float32)
            frame[:] = frame * 0.55 + fog.reshape(1, 1, 3) * 0.45

    def _apply_corruption(self, frame: np.ndarray, rng: np.random.Generator, frame_idx: int) -> None:
        height, width, _ = frame.shape
        stripe_y = (frame_idx * 7) % height
        frame[stripe_y : min(height, stripe_y + 14), :] = rng.integers(0, 255, size=(1, width, 3))

    def _apply_occlusion(self, frame: np.ndarray, rng: np.random.Generator, frame_idx: int) -> None:
        height, width, _ = frame.shape
        box_w = width // 4
        box_h = height // 5
        x = int((width - box_w) * (0.5 + 0.45 * np.sin(frame_idx * 0.05)))
        y = int(height * 0.32 + rng.integers(-8, 9))
        frame[y : y + box_h, x : x + box_w] = [8, 8, 8]

    @staticmethod
    def _scenario_color(weather: str, time_of_day: str, road_type: str) -> np.ndarray:
        color = np.asarray([125, 150, 170], dtype=np.float32)
        if weather == "clear":
            color += [30, 30, 25]
        if weather == "rainy":
            color += [-25, -10, 20]
        if weather == "foggy":
            color += [50, 45, 40]
        if time_of_day == "night":
            color += [-60, -55, -35]
        if "urban" in road_type:
            color += [15, 5, -5]
        if "highway" in road_type:
            color += [-10, 15, 5]
        return np.clip(color, 0, 255)
