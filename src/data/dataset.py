"""PyTorch datasets for collected CARLA expert-driving episodes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "all"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class EpisodeRecord:
    """One frame-level training sample from an episode JSONL file."""

    frame_id: int
    timestamp: float
    steering: float
    throttle: float
    brake: float
    speed_kmh: float
    instruction: str
    image_path: Path
    source: str = "real"


def _is_synthetic_source(source: str) -> bool:
    normalized = source.lower()
    return normalized != "real" and (
        normalized == "synthetic"
        or normalized.startswith("cosmos")
        or "synthetic" in normalized
    )


def _load_episode_records(data_dir: Path, *, source: str) -> List[EpisodeRecord]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    records: List[EpisodeRecord] = []
    for episode_dir in sorted(data_dir.glob("episode_*")):
        jsonl_path = episode_dir / "episode.jsonl"
        if not jsonl_path.exists():
            continue

        with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_path = episode_dir / row["image_path"]
                records.append(
                    EpisodeRecord(
                        frame_id=int(row["frame_id"]),
                        timestamp=float(row["timestamp"]),
                        steering=float(row["steering"]),
                        throttle=float(row["throttle"]),
                        brake=float(row["brake"]),
                        speed_kmh=float(row["speed_kmh"]),
                        instruction=str(row["instruction"]),
                        image_path=image_path,
                        source=str(row.get("source", source)),
                    )
                )

    if not records:
        raise RuntimeError(f"No episode samples found in {data_dir}.")
    return records


class CARLAEpisodeDataset(Dataset):
    """Load CARLA RGB frames, language instructions, and expert actions."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        real_dir: str | Path = "data/raw",
        synthetic_dir: str | Path | None = None,
        synthetic_weight: float = 0.0,
        split: SplitName = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be in (0, 1).")
        if not 0.0 <= synthetic_weight < 1.0:
            raise ValueError("synthetic_weight must be in [0, 1).")

        self.real_dir = Path(data_dir) if data_dir is not None else Path(real_dir)
        self.synthetic_dir = Path(synthetic_dir) if synthetic_dir else None
        self.synthetic_weight = synthetic_weight
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.image_size = image_size

        real_records = self._apply_split(self._load_records(self.real_dir, source="real"), source_offset=0)
        synthetic_records: List[EpisodeRecord] = []
        if self.synthetic_dir is not None and self.synthetic_dir.exists() and self.synthetic_weight > 0.0:
            synthetic_records = self._apply_split(
                self._load_records(self.synthetic_dir, source="synthetic"),
                source_offset=1_000_000,
            )

        self.records = self._mix_records(real_records, synthetic_records)
        if not self.records:
            raise RuntimeError(f"No samples found for split={split} in {self.real_dir}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """Return image_tensor, instruction_str, action_tensor."""

        record = self.records[index]
        image_tensor = self._load_image(record.image_path)
        action_tensor = torch.tensor(
            [record.steering, record.throttle, record.brake],
            dtype=torch.float32,
        )
        return image_tensor, record.instruction, action_tensor

    def _load_records(self, data_dir: Path, *, source: str) -> List[EpisodeRecord]:
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

        records: List[EpisodeRecord] = []
        records.extend(_load_episode_records(data_dir, source=source))
        return records

    def get_stats(self) -> str:
        """Return a readable summary of real versus synthetic samples."""

        real_count = sum(1 for record in self.records if not _is_synthetic_source(record.source))
        synthetic_count = len(self.records) - real_count
        total = max(1, len(self.records))
        synthetic_pct = 100.0 * synthetic_count / total
        return (
            f"Dataset: {real_count} real frames + {synthetic_count} synthetic frames "
            f"({synthetic_pct:.1f}% synthetic)"
        )

    def _apply_split(self, records: List[EpisodeRecord], *, source_offset: int) -> List[EpisodeRecord]:
        if self.split == "all":
            return records

        indices = list(range(len(records)))
        random.Random(self.seed + source_offset).shuffle(indices)
        split_at = int(len(indices) * self.train_ratio)
        split_at = min(max(split_at, 1), len(indices) - 1)

        selected_indices = indices[:split_at] if self.split == "train" else indices[split_at:]
        selected_indices = sorted(selected_indices)
        return [records[index] for index in selected_indices]

    def _mix_records(
        self,
        real_records: List[EpisodeRecord],
        synthetic_records: List[EpisodeRecord],
    ) -> List[EpisodeRecord]:
        if not synthetic_records or self.synthetic_weight <= 0.0:
            return real_records
        if not real_records:
            return synthetic_records

        target_synthetic = int(round(len(real_records) * self.synthetic_weight / (1.0 - self.synthetic_weight)))
        rng = random.Random(self.seed + 2_000_000)
        sampled_synthetic = [
            synthetic_records[rng.randrange(len(synthetic_records))]
            for _ in range(max(1, target_synthetic))
        ]

        mixed = real_records + sampled_synthetic
        rng.shuffle(mixed)
        return mixed

    def _load_image(self, image_path: Path) -> torch.Tensor:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing frame image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class MixedDataset(Dataset):
    """Combine CARLA expert episodes with Cosmos synthetic episodes at a target ratio."""

    def __init__(
        self,
        real_episodes: str | Path = "data/raw",
        synthetic_episodes: str | Path = "data/synthetic",
        *,
        synthetic_ratio: float = 0.4,
        augment_real: bool = True,
        split: SplitName = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        if not 0.0 <= synthetic_ratio < 1.0:
            raise ValueError("synthetic_ratio must be in [0, 1).")

        self.real_dir = Path(real_episodes)
        self.synthetic_dir = Path(synthetic_episodes)
        self.synthetic_ratio = synthetic_ratio
        self.augment_real = augment_real
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.image_size = image_size
        self._pair_rng = random.Random(seed + 4_000_000)

        self.real_records = self._apply_split(
            _load_episode_records(self.real_dir, source="real"),
            source_offset=0,
        )
        self.synthetic_records = self._apply_split(
            _load_episode_records(self.synthetic_dir, source="synthetic"),
            source_offset=1_000_000,
        )
        self.records = self._mix_records(self.real_records, self.synthetic_records)
        if not self.records:
            raise RuntimeError("MixedDataset contains no samples.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """Return image_tensor, instruction_str, action_tensor."""

        record = self.records[index]
        image_tensor = self._load_image(
            record.image_path,
            augment=self.augment_real and not _is_synthetic_source(record.source),
            seed=self.seed + index,
        )
        action_tensor = torch.tensor(
            [record.steering, record.throttle, record.brake],
            dtype=torch.float32,
        )
        return image_tensor, record.instruction, action_tensor

    def get_stats(self) -> str:
        """Return the effective mixed-sample composition."""

        real_count = sum(1 for record in self.records if not _is_synthetic_source(record.source))
        synthetic_count = len(self.records) - real_count
        total = max(1, len(self.records))
        synthetic_pct = 100.0 * synthetic_count / total
        return (
            f"Dataset: {real_count} real frames + {synthetic_count} synthetic frames "
            f"({synthetic_pct:.1f}% synthetic)"
        )

    def sample_domain_pairs(
        self,
        batch_size: int,
    ) -> Optional[Tuple[torch.Tensor, List[str], torch.Tensor, List[str]]]:
        """Sample real/synthetic image pairs for latent-domain alignment."""

        if not self.real_records or not self.synthetic_records or batch_size <= 0:
            return None

        real_images: List[torch.Tensor] = []
        synthetic_images: List[torch.Tensor] = []
        real_instructions: List[str] = []
        synthetic_instructions: List[str] = []
        for pair_idx in range(batch_size):
            real_record = self._pair_rng.choice(self.real_records)
            synthetic_record = self._pair_rng.choice(self.synthetic_records)
            real_images.append(
                self._load_image(
                    real_record.image_path,
                    augment=self.augment_real and self.split == "train",
                    seed=self.seed + pair_idx,
                )
            )
            synthetic_images.append(
                self._load_image(
                    synthetic_record.image_path,
                    augment=False,
                    seed=self.seed + 10_000 + pair_idx,
                )
            )
            real_instructions.append(real_record.instruction)
            synthetic_instructions.append(synthetic_record.instruction)

        return (
            torch.stack(real_images),
            real_instructions,
            torch.stack(synthetic_images),
            synthetic_instructions,
        )

    def _apply_split(self, records: List[EpisodeRecord], *, source_offset: int) -> List[EpisodeRecord]:
        if self.split == "all":
            return records

        indices = list(range(len(records)))
        random.Random(self.seed + source_offset).shuffle(indices)
        split_at = int(len(indices) * self.train_ratio)
        split_at = min(max(split_at, 1), len(indices) - 1)
        selected_indices = indices[:split_at] if self.split == "train" else indices[split_at:]
        selected_indices = sorted(selected_indices)
        return [records[index] for index in selected_indices]

    def _mix_records(
        self,
        real_records: List[EpisodeRecord],
        synthetic_records: List[EpisodeRecord],
    ) -> List[EpisodeRecord]:
        if not synthetic_records or self.synthetic_ratio <= 0.0:
            return list(real_records)

        target_synthetic = int(round(len(real_records) * self.synthetic_ratio / (1.0 - self.synthetic_ratio)))
        rng = random.Random(self.seed + 2_000_000)
        sampled_synthetic = [
            synthetic_records[rng.randrange(len(synthetic_records))]
            for _ in range(max(1, target_synthetic))
        ]

        mixed = list(real_records) + sampled_synthetic
        rng.shuffle(mixed)
        return mixed

    def _load_image(self, image_path: Path, *, augment: bool, seed: int) -> torch.Tensor:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing frame image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        if augment:
            array = self._augment_real_image(array, seed=seed)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    @staticmethod
    def _augment_real_image(array: np.ndarray, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        brightness = float(rng.uniform(0.92, 1.08))
        contrast = float(rng.uniform(0.92, 1.08))
        noise_std = float(rng.uniform(0.0, 0.015))

        mean = array.mean(axis=(0, 1), keepdims=True)
        augmented = (array - mean) * contrast + mean
        augmented = augmented * brightness
        if noise_std > 0.0:
            augmented = augmented + rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        return np.clip(augmented, 0.0, 1.0).astype(np.float32)


def build_train_val_datasets(
    data_dir: str | Path = "data/raw",
    *,
    synthetic_dir: str | Path | None = None,
    synthetic_weight: float = 0.0,
    synthetic_ratio: float | None = None,
    augment_real: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Create deterministic 80/20 train and validation datasets."""

    effective_synthetic_ratio = synthetic_weight if synthetic_ratio is None else synthetic_ratio
    if synthetic_dir is not None and effective_synthetic_ratio > 0.0:
        train_dataset = MixedDataset(
            real_episodes=data_dir,
            synthetic_episodes=synthetic_dir,
            synthetic_ratio=effective_synthetic_ratio,
            augment_real=augment_real,
            split="train",
            train_ratio=train_ratio,
            seed=seed,
        )
        val_dataset = MixedDataset(
            real_episodes=data_dir,
            synthetic_episodes=synthetic_dir,
            synthetic_ratio=effective_synthetic_ratio,
            augment_real=False,
            split="val",
            train_ratio=train_ratio,
            seed=seed,
        )
        return train_dataset, val_dataset

    train_dataset = CARLAEpisodeDataset(
        data_dir,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
    )
    val_dataset = CARLAEpisodeDataset(
        data_dir,
        split="val",
        train_ratio=train_ratio,
        seed=seed,
    )
    return train_dataset, val_dataset
