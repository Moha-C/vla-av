"""JSON-lines worker for running official Alpamayo R1 from its own Python env."""

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
    parser.add_argument("--repo-path", default="external/alpamayo_official")
    parser.add_argument(
        "--model-path",
        default="vm_backups/official_sft/intermediate/stage2/checkpoint-10528",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--num-traj-sets", type=int, default=1)
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
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


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


def create_runtime_message(helper: Any, frames: torch.Tensor, nav_text: Optional[str]) -> list[dict[str, Any]]:
    messages = helper.create_message(frames=frames)
    nav_text = (nav_text or "").strip()
    if not nav_text:
        return messages

    runtime_policy = (
        "Driving policy for this exact frame: obey red and yellow lights by braking "
        "before the stop line; proceed on green only when the lane, junction and "
        "crosswalk are clear; make a complete stop at stop signs; yield to pedestrians, "
        "cyclists, scooters, motorcycles and vehicles with priority; follow the road "
        "geometry and lane markings instead of driving off road."
    )
    user_content = messages[1].get("content", [])
    policy_item = {
        "type": "text",
        "text": f"{nav_text}\n{runtime_policy}\n",
    }
    if user_content and user_content[-1].get("type") == "text":
        user_content.insert(len(user_content) - 1, policy_item)
    else:
        user_content.append(policy_item)
    messages[1]["content"] = user_content
    return messages


def emit(payload: Dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload, separators=(",", ":")) + "\n")
    PROTOCOL_STDOUT.flush()


def main() -> None:
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()
    for candidate in (repo_path / "src", repo_path):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    dtype = torch_dtype(args.dtype)
    model = AlpamayoR1.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model = model.to(args.device).eval()
    processor = helper.get_processor(model.tokenizer)

    image_history: Deque[torch.Tensor] = deque(maxlen=args.num_frames)
    state_history: Deque[Dict[str, Any]] = deque(maxlen=args.history_steps)
    emit({"ok": True, "event": "ready", "model_type": "alpamayo_r1"})

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
        xyz = ego_history_xyz(state_history, args.history_steps)
        rot = ego_history_rot(state_history, args.history_steps)

        messages = create_runtime_message(helper, frames, request.get("nav_text"))
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
            pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=float(args.top_p),
                temperature=float(args.temperature),
                num_traj_samples=int(args.num_traj_samples),
                num_traj_sets=int(args.num_traj_sets),
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
            raise RuntimeError(f"Unexpected Alpamayo R1 trajectory shape: {pred_array.shape}")

        cot = None
        if isinstance(extra, dict) and "cot" in extra:
            try:
                cot = json_safe(extra["cot"][0])
            except Exception:
                cot = str(extra["cot"])

        emit(
            {
                "ok": True,
                "trajectory": np.asarray(trajectory, dtype=np.float32).tolist(),
                "cot": cot,
                "inference_ms": (time.perf_counter() - started_at) * 1000.0,
            }
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - worker errors are reported to parent.
        emit({"ok": False, "error": repr(exc)})
        raise
