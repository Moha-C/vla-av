"""Temporal front-camera clips from a prepared DeepAccident manifest."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("%s:%d is not a JSON object" % (path, line_number))
            rows.append(item)
    return rows


def _image_tensor(
    path: Path,
    image_size: Tuple[int, int],
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> torch.Tensor:
    with Image.open(str(path)) as image:
        image = image.convert("RGB")
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class DeepAccidentClipDataset(Dataset):
    """Clips stay within a scenario and preserve the original frame order."""

    def __init__(
        self,
        dataset_root: Path,
        manifest: Path,
        split: str,
        clip_length: int = 3,
        frame_stride: int = 5,
        image_height: int = 224,
        image_width: int = 384,
        augment: bool = False,
        seed: int = 230401168,
        max_clips: int = 0,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if clip_length < 1 or frame_stride < 1:
            raise ValueError("clip_length and frame_stride must be positive")
        self.dataset_root = dataset_root.resolve()
        self.image_size = (int(image_height), int(image_width))
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(manifest):
            if row.get("split") == split:
                grouped[str(row["scenario_key"])].append(row)
        self.rows_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
        self.clips: List[Tuple[str, Tuple[int, ...]]] = []
        span = (clip_length - 1) * frame_stride
        for key, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: int(row["frame_position"]))
            self.rows_by_scenario[key] = rows
            for end in range(span, len(rows)):
                indices = tuple(end - offset * frame_stride for offset in reversed(range(clip_length)))
                self.clips.append((key, indices))
        if max_clips > 0 and len(self.clips) > max_clips:
            rng = random.Random(seed + {"train": 0, "validation": 1, "test": 2}[split])
            rng.shuffle(self.clips)
            self.clips = self.clips[:max_clips]
        if not self.clips:
            raise RuntimeError("no temporal clips for split %s" % split)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.clips)

    def targets(self) -> torch.Tensor:
        values = []
        for key, indices in self.clips:
            row = self.rows_by_scenario[key][indices[-1]]
            values.append(float(bool(row["event_within_horizon"])))
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key, indices = self.clips[index]
        rows = self.rows_by_scenario[key]
        brightness = 1.0
        contrast = 1.0
        if self.augment:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
            brightness = rng.uniform(0.85, 1.15)
            contrast = rng.uniform(0.90, 1.10)
        frames = [
            _image_tensor(
                self.dataset_root / str(rows[position]["image_path"]),
                self.image_size,
                brightness,
                contrast,
            )
            for position in indices
        ]
        target_row = rows[indices[-1]]
        risk = float(bool(target_row["event_within_horizon"]))
        seconds = float(target_row["seconds_to_event"])
        return {
            "frames": torch.stack(frames, dim=0),
            "risk": torch.tensor(risk, dtype=torch.float32),
            "ttc_s": torch.tensor(max(0.0, seconds) if risk else 0.0, dtype=torch.float32),
            "ttc_mask": torch.tensor(risk, dtype=torch.float32),
            "scenario_key": key,
            "frame_number": int(target_row["frame_number"]),
        }


def scenario_keys(dataset: DeepAccidentClipDataset) -> Sequence[str]:
    return tuple(sorted(dataset.rows_by_scenario))
