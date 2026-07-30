"""JSON-lines worker for running Alpamayo from its own Python 3.12 environment."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np
import torch
from PIL import Image


PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", default="external/alpamayo1.5")
    parser.add_argument("--model-path", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    return parser.parse_args()


def torch_dtype(dtype: str) -> torch.dtype:
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def decode_image(image_b64: str) -> torch.Tensor:
    data = base64.b64decode(image_b64.encode("ascii"))
    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def prepare_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {
            "location": (0.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 0.0),
            "speed_kmh": 0.0,
        }
    return {
        "location": tuple(state.get("location", (0.0, 0.0, 0.0))),
        "rotation": tuple(state.get("rotation", (0.0, 0.0, 0.0))),
        "speed_kmh": float(state.get("speed_kmh", 0.0)),
    }


def padded_states(history: Deque[Dict[str, Any]], n_steps: int) -> list[Dict[str, Any]]:
    if not history:
        zero = {
            "location": (0.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 0.0),
            "speed_kmh": 0.0,
        }
        return [zero for _ in range(n_steps)]
    states = list(history)
    while len(states) < n_steps:
        states.insert(0, states[0])
    return states[-n_steps:]


def ego_history_xyz(history: Deque[Dict[str, Any]], n_steps: int) -> torch.Tensor:
    states = padded_states(history, n_steps)
    current = states[-1]
    current_xy = np.asarray(current["location"][:2], dtype=np.float32)
    yaw = math.radians(float(current["rotation"][1]))
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    # CARLA yaw=0 points along +x and +y is vehicle-right. Alpamayo uses
    # AV-style local coordinates where positive lateral means left.
    left = np.asarray([math.sin(yaw), -math.cos(yaw)], dtype=np.float32)
    xyz = []
    for state in states:
        loc = np.asarray(state["location"][:2], dtype=np.float32)
        delta = loc - current_xy
        xyz.append([float(delta @ forward), float(delta @ left), 0.0])
    return torch.tensor(xyz, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def ego_history_rot(history: Deque[Dict[str, Any]], n_steps: int) -> torch.Tensor:
    states = padded_states(history, n_steps)
    current_yaw = math.radians(float(states[-1]["rotation"][1]))
    rotations = []
    for state in states:
        yaw = math.radians(float(state["rotation"][1])) - current_yaw
        rotations.append(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    return torch.tensor(rotations, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def stack_frames(history: Deque[torch.Tensor], n_frames: int) -> torch.Tensor:
    if not history:
        raise RuntimeError("No image history is available.")
    frames = list(history)
    while len(frames) < n_frames:
        frames.insert(0, frames[0])
    return torch.stack(frames[-n_frames:], dim=0)


def emit(payload: Dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload, separators=(",", ":")) + "\n")
    PROTOCOL_STDOUT.flush()


def main() -> None:
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()
    for candidate in (repo_path / "src", repo_path):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    dtype = torch_dtype(args.dtype)
    load_kwargs: Dict[str, Any] = {"dtype": dtype}
    if args.attn_implementation and args.attn_implementation != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = Alpamayo1_5.from_pretrained(args.model_path, **load_kwargs)
    except ValueError as exc:
        message = str(exc)
        if (
            "scaled_dot_product_attention" not in message
            and "attn_implementation=\"eager\"" not in message
            and "attn_implementation='eager'" not in message
        ):
            raise
        load_kwargs["attn_implementation"] = "eager"
        model = Alpamayo1_5.from_pretrained(args.model_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        model = Alpamayo1_5.from_pretrained(args.model_path, **load_kwargs)
    model = model.to(args.device).eval()
    processor = helper.get_processor(model.tokenizer)

    image_history: Deque[torch.Tensor] = deque(maxlen=args.num_frames)
    state_history: Deque[Dict[str, Any]] = deque(maxlen=args.history_steps)
    emit({"ok": True, "event": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("cmd") == "shutdown":
            emit({"ok": True, "event": "shutdown"})
            return

        started_at = time.perf_counter()
        image_history.append(decode_image(request["image_b64"]))
        state_history.append(prepare_state(request.get("state")))
        frames = stack_frames(image_history, args.num_frames)
        camera_indices = torch.tensor([args.camera_index], dtype=torch.int64)
        xyz = ego_history_xyz(state_history, args.history_steps)
        rot = ego_history_rot(state_history, args.history_steps)
        nav_text = request.get("nav_text")

        if hasattr(helper, "create_message"):
            messages = helper.create_message(
                frames=frames,
                camera_indices=camera_indices,
                num_frames_per_camera=args.num_frames,
                nav_text=nav_text,
                use_nav_prompt=True,
            )
        else:
            messages = helper.create_message_using_data(
                {
                    "image_frames": frames,
                    "camera_indices": camera_indices,
                    "ego_history_xyz": xyz,
                    "ego_history_rot": rot,
                    "nav_text": nav_text,
                },
                num_frames_per_camera=args.num_frames,
                use_hist_traj=True,
                use_nav_prompt=True,
            )

        tokenized = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": tokenized,
            "ego_history_xyz": xyz,
            "ego_history_rot": rot,
        }
        model_inputs = helper.to_device(model_inputs, args.device)
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=dtype,
            enabled=str(args.device).startswith("cuda"),
        ):
            pred_xyz, _pred_rot, _extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=float(args.top_p),
                temperature=float(args.temperature),
                num_traj_samples=int(args.num_traj_samples),
                max_generation_length=int(args.max_generation_length),
                return_extra=True,
            )

        pred_array = pred_xyz.detach().float().cpu().numpy()
        if pred_array.ndim == 5:
            trajectory = pred_array[0, 0, 0, :, :2]
        elif pred_array.ndim == 4:
            trajectory = pred_array[0, 0, :, :2]
        elif pred_array.ndim == 3:
            trajectory = pred_array[0, :, :2]
        else:
            raise RuntimeError(f"Unexpected Alpamayo trajectory shape: {pred_array.shape}")
        emit(
            {
                "ok": True,
                "trajectory": np.asarray(trajectory, dtype=np.float32).tolist(),
                "inference_ms": (time.perf_counter() - started_at) * 1000.0,
            }
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - worker errors are reported to parent.
        emit({"ok": False, "error": repr(exc)})
        raise
