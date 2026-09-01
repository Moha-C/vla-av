from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.deepaccident.data import DeepAccidentClipDataset, scenario_keys
from src.deepaccident.index import DeepAccidentIndexConfig, build_index
from src.deepaccident.risk_model import DeepAccidentRiskEncoder, RiskEncoderConfig
from src.deepaccident.training import binary_metrics, promotion_gate, risk_ttc_loss


def _scenario(
    root: Path,
    category_name: str,
    scenario_id: str,
    frames: int = 8,
    vehicle_role: str = "ego_vehicle",
    colliding_agents: str = "ego none",
) -> None:
    category = root / category_name
    camera = category / vehicle_role / "Camera_Front" / scenario_id
    labels = category / vehicle_role / "label" / scenario_id
    meta = category / "meta"
    camera.mkdir(parents=True)
    labels.mkdir(parents=True)
    meta.mkdir(parents=True, exist_ok=True)
    agents = colliding_agents if "accident" in category_name else "none none"
    (meta / (scenario_id + ".txt")).write_text(
        "ClearNoon 1 car 2 car 0 front front 0\n colliding agents: %s\n" % agents,
        encoding="utf-8",
    )
    for frame in range(frames):
        name = "%s_%03d" % (scenario_id, frame)
        pixels = np.full((24, 32, 3), frame * 8, dtype=np.uint8)
        Image.fromarray(pixels).save(str(camera / (name + ".jpg")))
        (labels / (name + ".txt")).write_text("1.0 0.0\n", encoding="utf-8")


def _prepared_dataset(tmp_path: Path) -> tuple:
    root = tmp_path / "DeepAccident_data"
    for index in range(3):
        _scenario(
            root,
            "type1_subtype1_accident",
            "Town10_type001_subtype0001_scenario%05d" % index,
        )
        _scenario(
            root,
            "type1_subtype1_normal",
            "Town10_type001_subtype0001_scenario%05d" % index,
        )
    output = tmp_path / "processed"
    audit = build_index(
        root,
        output,
        DeepAccidentIndexConfig(fps=2.0, prediction_horizon_s=1.0, split_seed=7),
    )
    return root, output, audit


def test_index_preserves_scenarios_and_temporal_targets(tmp_path: Path) -> None:
    root, output, audit = _prepared_dataset(tmp_path)
    assert audit["scenarios"] == 6
    assert audit["frames"] == 48
    assert audit["accident_scenarios"] == 3
    assert audit["normal_scenarios"] == 3
    assert audit["positive_horizon_frames"] == 9
    assert audit["rejected"] == []
    assert audit["vehicle_roles"] == {"ego_vehicle": 6}

    rows = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    accident = [row for row in rows if row["is_accident_scenario"]]
    normal = [row for row in rows if not row["is_accident_scenario"]]
    assert all(not row["event_within_horizon"] for row in normal)
    by_scenario = {}
    for row in accident:
        by_scenario.setdefault(row["scenario_key"], []).append(row)
    assert all(sum(item["event_within_horizon"] for item in values) == 3 for values in by_scenario.values())

    scenarios = [json.loads(line) for line in (output / "scenarios.jsonl").read_text().splitlines()]
    keys_by_split = {
        split: {row["scenario_key"] for row in scenarios if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert all(keys_by_split.values())
    assert keys_by_split["train"].isdisjoint(keys_by_split["validation"])
    assert keys_by_split["train"].isdisjoint(keys_by_split["test"])
    split_by_pair = {}
    for row in scenarios:
        split_by_pair.setdefault(row["scenario_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_pair.values())
    assert audit["paired_split_groups"] == 3
    assert all(row["vehicle_role"] == "ego_vehicle" for row in scenarios)


def test_index_uses_the_actor_named_in_collision_metadata(tmp_path: Path) -> None:
    root = tmp_path / "DeepAccident_data"
    for index in range(3):
        scenario_id = "Town03_type001_subtype0001_scenario%05d" % index
        for category in ("type1_subtype1_accident", "type1_subtype1_normal"):
            _scenario(
                root,
                category,
                scenario_id,
                vehicle_role="ego_vehicle",
                colliding_agents="ego_behind none",
            )
            _scenario(
                root,
                category,
                scenario_id,
                vehicle_role="ego_vehicle_behind",
                colliding_agents="ego_behind none",
            )
    output = tmp_path / "processed"
    audit = build_index(root, output, DeepAccidentIndexConfig(split_seed=11))
    scenarios = [json.loads(line) for line in (output / "scenarios.jsonl").read_text().splitlines()]
    assert audit["scenarios"] == 6
    assert audit["vehicle_roles"] == {"ego_vehicle_behind": 6}
    assert all(row["vehicle_role"] == "ego_vehicle_behind" for row in scenarios)


def test_clip_dataset_never_crosses_scenarios(tmp_path: Path) -> None:
    root, output, _ = _prepared_dataset(tmp_path)
    dataset = DeepAccidentClipDataset(
        root,
        output / "frames.jsonl",
        split="train",
        clip_length=3,
        frame_stride=1,
        image_height=32,
        image_width=48,
    )
    sample = dataset[0]
    assert sample["frames"].shape == (3, 3, 32, 48)
    assert torch.isfinite(sample["frames"]).all()
    assert len(scenario_keys(dataset)) == 2
    assert set(dataset.targets().tolist()) == {0.0, 1.0}


def test_risk_encoder_shapes_and_loss() -> None:
    model = DeepAccidentRiskEncoder(
        RiskEncoderConfig(
            embedding_dim=32,
            temporal_dim=48,
            pretrained_backbone=False,
        )
    )
    output = model(torch.randn(2, 3, 3, 64, 96))
    assert output["embedding"].shape == (2, 32)
    assert output["risk"].shape == (2,)
    assert output["ttc_s"].shape == (2,)
    assert ((output["risk"] >= 0.0) & (output["risk"] <= 1.0)).all()
    loss, components = risk_ttc_loss(
        output,
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor(1.0),
        prediction_horizon_s=2.0,
    )
    assert torch.isfinite(loss)
    assert components["total"] > 0.0


def test_binary_metrics_uses_validation_threshold() -> None:
    metrics = binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.2, 0.7, 0.9]),
    )
    assert metrics["average_precision"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_promotion_gate_rejects_small_or_weak_evaluation() -> None:
    metrics = {
        "average_precision": 0.20,
        "positive_rate": 0.15,
        "roc_auc": 0.55,
        "true_positive": 3.0,
        "false_negative": 7.0,
        "false_positive": 20.0,
        "true_negative": 70.0,
    }
    decision = promotion_gate(
        {
            "validation": metrics,
            "test": metrics,
            "dataset_audit": {"source_scenario_groups": 8},
        }
    )
    assert decision["passed"] is False
    assert decision["decision"] == "hold_not_promoted"
    assert decision["checks"]["enough_independent_source_groups"] is False
