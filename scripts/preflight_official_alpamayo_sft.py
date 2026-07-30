#!/usr/bin/env python3
"""Smoke-test the official Alpamayo SFT stack without training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def _inspect_mapping(name: str, value: Any) -> None:
    print(f"\n===== {name} =====")
    if not isinstance(value, dict):
        print(type(value).__name__, value)
        return
    for key, item in value.items():
        if hasattr(item, "shape"):
            print(key, tuple(item.shape), item.dtype)
        elif isinstance(item, dict):
            print(key, "dict", sorted(item.keys()))
        else:
            print(key, type(item).__name__, str(item)[:180])


def _to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    args = parser.parse_args()

    root = Path(args.official_root).expanduser().resolve()
    checkpoint = str(Path(args.checkpoint_path).expanduser().resolve())
    config_dir = root / "finetune/sft/configs"

    overrides = [
        "data.train_dataset.max_samples=2",
        "data.val_dataset.max_samples=2",
        "trainer.dataloader_num_workers=0",
        "trainer.per_device_train_batch_size=1",
        "trainer.per_device_eval_batch_size=1",
    ]
    if args.config_name == "sft_carla_stage1":
        overrides.append(f"model.checkpoint_path={checkpoint}")
    if args.config_name == "sft_carla_stage2":
        overrides.append(f"model.pretrained_model_name_or_path={checkpoint}")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    print("===== composed config =====")
    print(OmegaConf.to_yaml(cfg))

    print("\n===== instantiate model =====")
    model = instantiate(cfg.model, _convert_="partial")
    print("model class:", type(model).__name__)
    print("model config class:", type(model.config).__name__)

    print("\n===== instantiate datasets =====")
    train_dataset = instantiate(
        cfg.data.train_dataset,
        _convert_="partial",
        model_config=model.config,
    )
    val_dataset = instantiate(
        cfg.data.val_dataset,
        _convert_="partial",
        model_config=model.config,
    )
    print("train len:", len(train_dataset))
    print("val len:", len(val_dataset))

    print("\n===== instantiate collator =====")
    collate_fn = instantiate(
        cfg.data.collate_fn,
        _convert_="partial",
        model_config=model.config,
    )

    item = train_dataset[0]
    _inspect_mapping("item smoke", item)

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    _inspect_mapping("collate smoke", batch)

    print("\n===== forward smoke, no optimizer, no training =====")
    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    batch = _to_device(batch, device)
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_available(),
    ):
        output = model(**batch)
    loss = getattr(output, "loss", None)
    print("loss:", float(loss.detach().cpu()) if loss is not None else None)
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
