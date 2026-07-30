"""Export a CARLA dataset frame to use as Cosmos Video2World conditioning input."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="data/reference/carla_reference.png")
    parser.add_argument(
        "--prefer-recovery",
        action="store_true",
        help="Prefer frames tagged recovery_active=true.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index inside the filtered frame list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for jsonl_path in sorted(Path(args.data_dir).glob("episode_*/episode.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_episode_dir"] = jsonl_path.parent
                records.append(row)

    if not records:
        raise RuntimeError(f"No dataset frames found under {args.data_dir}.")

    if args.prefer_recovery:
        filtered = [record for record in records if record.get("recovery_active") is True]
        records = filtered or records
    else:
        non_recovery = [record for record in records if record.get("recovery_active") is not True]
        records = non_recovery or records

    record = records[args.index % len(records)]
    source_path = record["_episode_dir"] / record["image_path"]
    if not source_path.exists():
        raise FileNotFoundError(f"Referenced image does not exist: {source_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    print(f"Exported reference frame: {source_path} -> {output_path}")
    print(
        "Metadata: "
        f"frame_id={record.get('frame_id')} "
        f"speed_kmh={float(record.get('speed_kmh', 0.0)):.1f} "
        f"recovery_active={record.get('recovery_active', False)}"
    )


if __name__ == "__main__":
    main()
