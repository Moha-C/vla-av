#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


VIDEO_KEY = "observation.images.ego_view"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(path):
    tasks = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tasks[int(item["task_index"])] = item["task"]
    return tasks


def load_episodes(path):
    episodes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            episodes.append(json.loads(line))
    return episodes


def format_template(template, episode_index, chunks_size, video_key=None):
    episode_chunk = episode_index // chunks_size
    kwargs = {
        "episode_chunk": episode_chunk,
        "episode_index": episode_index,
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    return template.format(**kwargs)


def count_frames(frames_dir):
    return sum(1 for _ in frames_dir.glob("frame_*.jpg"))


def extract_video(video_path, frames_dir, jpeg_quality, overwrite):
    frames_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for frame in frames_dir.glob("frame_*.jpg"):
            frame.unlink()

    if count_frames(frames_dir) > 0 and not overwrite:
        return

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-start_number",
        "0",
        "-q:v",
        str(jpeg_quality),
        str(frames_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def array_value(value, index, default=0.0):
    if value is None:
        return default
    try:
        return float(value[index])
    except (TypeError, IndexError, ValueError):
        return default


def write_episode_metadata(df, episode_dir, output_root, tasks):
    actions_csv = episode_dir / "actions.csv"
    metadata_jsonl = episode_dir / "metadata.jsonl"

    fieldnames = [
        "episode_index",
        "frame_index",
        "frame_path",
        "timestamp",
        "steering",
        "throttle",
        "brake",
        "speed_ratio",
        "red_light",
        "near_stop",
        "task_index",
        "instruction",
        "done",
    ]

    with actions_csv.open("w", encoding="utf-8", newline="") as csv_file, metadata_jsonl.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in df.to_dict(orient="records"):
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
            task_index = int(row.get("task_index", 0))
            instruction = tasks.get(task_index, "")
            action = row.get("action")
            state = row.get("observation.state")
            frame_path = (
                Path("runs")
                / f"episode_{episode_index:06d}"
                / "front_camera_frames"
                / f"frame_{frame_index:06d}.jpg"
            )

            item = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "frame_path": str(frame_path),
                "timestamp": float(row.get("timestamp", 0.0)),
                "steering": array_value(action, 0),
                "throttle": array_value(action, 1),
                "brake": array_value(action, 2),
                "speed_ratio": array_value(state, 3),
                "red_light": int(round(array_value(state, 4))),
                "near_stop": int(round(array_value(state, 5))),
                "task_index": task_index,
                "instruction": instruction,
                "done": bool(row.get("next.done", False)),
            }
            writer.writerow(item)

            jsonl_item = {
                **item,
                "image_path": str(output_root / frame_path),
                "action": {
                    "steering": item["steering"],
                    "throttle": item["throttle"],
                    "brake": item["brake"],
                },
            }
            jsonl_file.write(json.dumps(jsonl_item, ensure_ascii=False) + "\n")


def append_manifest(df, manifest_file, output_root, tasks):
    for row in df.to_dict(orient="records"):
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        task_index = int(row.get("task_index", 0))
        action = row.get("action")
        state = row.get("observation.state")
        rel_frame_path = (
            Path("runs")
            / f"episode_{episode_index:06d}"
            / "front_camera_frames"
            / f"frame_{frame_index:06d}.jpg"
        )
        item = {
            "episode_index": episode_index,
            "source_run": f"episode_{episode_index:06d}",
            "frame_index": frame_index,
            "image_path": str(rel_frame_path),
            "absolute_image_path": str(output_root / rel_frame_path),
            "timestamp": float(row.get("timestamp", 0.0)),
            "steering": array_value(action, 0),
            "throttle": array_value(action, 1),
            "brake": array_value(action, 2),
            "speed_ratio": array_value(state, 3),
            "red_light": int(round(array_value(state, 4))),
            "near_stop": int(round(array_value(state, 5))),
            "task_index": task_index,
            "instruction": tasks.get(task_index, ""),
            "action": {
                "steering": array_value(action, 0),
                "throttle": array_value(action, 1),
                "brake": array_value(action, 2),
            },
            "done": bool(row.get("next.done", False)),
        }
        manifest_file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export GROOT CARLA episodes as ordered front-camera frames plus actions/instructions."
    )
    parser.add_argument(
        "--dataset",
        default="backup_1ere_version/data/groot_carla_v2_instr",
        help="Path to the GROOT CARLA dataset root.",
    )
    parser.add_argument(
        "--output",
        default="exports/maram_groot_carla_frames",
        help="Output directory for the exported frame dataset.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg JPEG quality, lower is better.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already extracted frames.")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    output_root = Path(args.output).resolve()
    info = load_json(dataset / "meta" / "info.json")
    tasks = load_tasks(dataset / "meta" / "tasks.jsonl")
    episodes = load_episodes(dataset / "meta" / "episodes.jsonl")
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]

    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset / "meta" / "info.json", output_root / "source_info.json")
    shutil.copy2(dataset / "meta" / "tasks.jsonl", output_root / "tasks.jsonl")
    shutil.copy2(dataset / "meta" / "episodes.jsonl", output_root / "episodes.jsonl")

    summary = {
        "source_dataset": str(dataset),
        "output_dataset": str(output_root),
        "fps": info.get("fps"),
        "video_key": VIDEO_KEY,
        "episodes_exported": len(episodes),
        "features": {
            "front_camera_frames": "runs/episode_xxxxxx/front_camera_frames/frame_000000.jpg",
            "per_episode_actions": "runs/episode_xxxxxx/actions.csv",
            "per_episode_metadata": "runs/episode_xxxxxx/metadata.jsonl",
            "global_manifest": "manifest.jsonl",
            "action": ["steering", "throttle", "brake"],
            "state": ["steering", "throttle", "brake", "speed_ratio", "red_light", "near_stop"],
        },
    }
    (output_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    chunks_size = int(info.get("chunks_size", 1000))
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for n, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            episode_name = f"episode_{episode_index:06d}"
            episode_dir = output_root / "runs" / episode_name
            frames_dir = episode_dir / "front_camera_frames"
            parquet_path = dataset / format_template(info["data_path"], episode_index, chunks_size)
            video_path = dataset / format_template(
                info["video_path"], episode_index, chunks_size, video_key=VIDEO_KEY
            )

            if not parquet_path.exists():
                raise FileNotFoundError(parquet_path)
            if not video_path.exists():
                raise FileNotFoundError(video_path)

            print(f"[{n}/{len(episodes)}] {episode_name}: extracting frames and metadata")
            df = pd.read_parquet(parquet_path)
            extract_video(video_path, frames_dir, args.jpeg_quality, args.overwrite)
            write_episode_metadata(df, episode_dir, output_root, tasks)
            append_manifest(df, manifest_file, output_root, tasks)

    print(f"\nDone. Export written to: {output_root}")
    print(f"Global manifest: {manifest_path}")


if __name__ == "__main__":
    main()
