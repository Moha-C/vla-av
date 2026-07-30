"""Build an Alpamayo-ready manifest from CARLA + Cosmos-Transfer2.5 runs.

The script expects run directories produced by ``scripts/cosmos_transfer_real.py``.
Each run should contain:

- ``episode.jsonl`` with CARLA autopilot labels and ego poses.
- ``transfer_output/<run-name>.mp4`` or another non-control Transfer2.5 video.

It extracts photorealistic frames and writes a JSONL manifest with the original
instruction, CARLA expert action, traffic-law metadata, and local-frame
history/future ego trajectories. The manifest is intentionally simple so it can
be converted either to an Alpamayo-style dataset or uploaded to a B200 training
machine without carrying the full CARLA simulator outputs.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the active environment.
    cv2 = None


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="data/synthetic/transferred_real")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help="Specific run directory to include. Can be passed multiple times.",
    )
    parser.add_argument("--run-glob", default="transfer25_*")
    parser.add_argument("--metadata-name", default="episode.jsonl")
    parser.add_argument("--output-dir", default="data/alpamayo_transfer_dataset")
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-run", type=int, default=None)
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--future-steps", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return records


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [Path(path).expanduser().resolve() for path in args.run_dir]
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    return sorted(path for path in runs_dir.glob(args.run_glob) if path.is_dir())


def find_transfer_video(run_dir: Path) -> Optional[Path]:
    output_dir = run_dir / "transfer_output"
    exact = output_dir / f"{run_dir.name}.mp4"
    if exact.exists():
        return exact

    candidates = sorted(output_dir.glob("*.mp4"))
    usable = [
        path
        for path in candidates
        if "control" not in path.stem.lower()
        and not path.stem.lower().startswith("carla_")
    ]
    if not usable:
        return None

    preferred = [
        path
        for path in usable
        if any(token in path.stem.lower() for token in ("natural", "hood", "dashcam"))
    ]
    return preferred[0] if preferred else usable[0]


def record_timestamp(record: dict[str, Any], fallback: float) -> float:
    value = record.get("timestamp")
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def ego_location(record: dict[str, Any]) -> tuple[float, float, float]:
    ego_state = record.get("ego_state") or {}
    location = ego_state.get("location") or (0.0, 0.0, 0.0)
    return (float(location[0]), float(location[1]), float(location[2]))


def ego_yaw_rad(record: dict[str, Any]) -> float:
    ego_state = record.get("ego_state") or {}
    rotation = ego_state.get("rotation") or (0.0, 0.0, 0.0)
    return math.radians(float(rotation[1]))


def nearest_record_index(timestamps: list[float], target_timestamp: float) -> int:
    insertion = bisect.bisect_left(timestamps, target_timestamp)
    if insertion <= 0:
        return 0
    if insertion >= len(timestamps):
        return len(timestamps) - 1
    before = insertion - 1
    after = insertion
    if abs(timestamps[after] - target_timestamp) < abs(timestamps[before] - target_timestamp):
        return after
    return before


def local_xyz_sequence(
    records: list[dict[str, Any]],
    timestamps: list[float],
    center_idx: int,
    offsets_seconds: Iterable[float],
) -> list[list[float]]:
    center = records[center_idx]
    center_time = timestamps[center_idx]
    cx, cy, cz = ego_location(center)
    yaw = ego_yaw_rad(center)
    forward = (math.cos(yaw), math.sin(yaw))
    left = (-math.sin(yaw), math.cos(yaw))

    sequence: list[list[float]] = []
    for offset in offsets_seconds:
        target_idx = nearest_record_index(timestamps, center_time + float(offset))
        tx, ty, tz = ego_location(records[target_idx])
        dx = tx - cx
        dy = ty - cy
        sequence.append(
            [
                float(dx * forward[0] + dy * forward[1]),
                float(dx * left[0] + dy * left[1]),
                float(tz - cz),
            ]
        )
    return sequence


def output_image_path(
    output_dir: Path,
    run_name: str,
    frame_idx: int,
    image_format: str,
) -> tuple[Path, str]:
    relative = Path("images") / run_name / f"frame_{frame_idx:06d}.{image_format}"
    return output_dir / relative, str(relative)


def selected_frame_indices(
    metadata_count: int,
    *,
    frame_stride: int,
    max_frames: Optional[int],
) -> list[int]:
    indices = list(range(0, metadata_count, max(1, frame_stride)))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def extract_frames_with_ffmpeg(
    *,
    transfer_video: Path,
    output_dir: Path,
    run_name: str,
    image_format: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    frame_dir = output_dir / "images" / run_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / f"frame_%06d.{image_format}"
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(transfer_video),
        "-start_number",
        "0",
    ]
    if image_format == "jpg":
        command.extend(["-q:v", "2"])
    command.append(str(pattern))
    subprocess.run(command, check=True, text=True)


def make_manifest_record(
    *,
    source_record: dict[str, Any],
    image_relpath: str,
    source_run: str,
    transfer_video: Path,
    transfer_frame_idx: int,
    history_xyz: list[list[float]],
    future_xyz: list[list[float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    steering = float(source_record.get("steering", 0.0))
    throttle = float(source_record.get("throttle", 0.0))
    brake = float(source_record.get("brake", 0.0))
    record = dict(source_record)
    record.update(
        {
            "format": "carla_cosmos_transfer_alpamayo_v1",
            "source_run": source_run,
            "source_transfer_video": str(transfer_video),
            "transfer_frame_index": int(transfer_frame_idx),
            "image_path": image_relpath,
            "photoreal_frame_path": image_relpath,
            "camera_indices": [int(args.camera_index)],
            "ego_history_xyz": history_xyz,
            "ego_future_xyz": future_xyz,
            "ego_history_seconds": [
                float(-(args.history_steps - 1 - idx) * args.dt)
                for idx in range(args.history_steps)
            ],
            "ego_future_seconds": [
                float((idx + 1) * args.dt)
                for idx in range(args.future_steps)
            ],
            "action": {
                "steering": steering,
                "throttle": throttle,
                "brake": brake,
            },
        }
    )
    return record


def process_run(
    run_dir: Path,
    *,
    output_dir: Path,
    manifest_file: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata_path = run_dir / args.metadata_name
    if not metadata_path.exists():
        LOGGER.warning("Skipping %s: missing %s", run_dir, metadata_path.name)
        return {"run": run_dir.name, "frames": 0, "skipped": "missing_metadata"}

    transfer_video = find_transfer_video(run_dir)
    if transfer_video is None:
        LOGGER.warning("Skipping %s: no Transfer2.5 output video found", run_dir)
        return {"run": run_dir.name, "frames": 0, "skipped": "missing_transfer_video"}

    metadata = load_jsonl(metadata_path)
    if not metadata:
        LOGGER.warning("Skipping %s: metadata is empty", run_dir)
        return {"run": run_dir.name, "frames": 0, "skipped": "empty_metadata"}

    timestamps = [
        record_timestamp(record, fallback=float(idx) * args.dt)
        for idx, record in enumerate(metadata)
    ]
    if cv2 is None:
        extract_frames_with_ffmpeg(
            transfer_video=transfer_video,
            output_dir=output_dir,
            run_name=run_dir.name,
            image_format=args.image_format,
            dry_run=args.dry_run,
        )
        written = 0
        for frame_idx in selected_frame_indices(
            len(metadata),
            frame_stride=args.frame_stride,
            max_frames=args.max_frames_per_run,
        ):
            image_path, image_relpath = output_image_path(
                output_dir,
                run_dir.name,
                frame_idx,
                args.image_format,
            )
            if not args.dry_run and not image_path.exists():
                LOGGER.debug("Stopping %s at missing extracted frame %s", run_dir.name, image_path)
                break

            history_offsets = [
                -(args.history_steps - 1 - idx) * args.dt
                for idx in range(args.history_steps)
            ]
            future_offsets = [
                (idx + 1) * args.dt
                for idx in range(args.future_steps)
            ]
            record = make_manifest_record(
                source_record=metadata[frame_idx],
                image_relpath=image_relpath,
                source_run=run_dir.name,
                transfer_video=transfer_video,
                transfer_frame_idx=frame_idx,
                history_xyz=local_xyz_sequence(metadata, timestamps, frame_idx, history_offsets),
                future_xyz=local_xyz_sequence(metadata, timestamps, frame_idx, future_offsets),
                args=args,
            )
            if not args.dry_run:
                manifest_file.write(json.dumps(record) + "\n")
            written += 1

        LOGGER.info("Prepared %s photoreal frames from %s", written, run_dir.name)
        return {
            "run": run_dir.name,
            "frames": written,
            "metadata": str(metadata_path),
            "transfer_video": str(transfer_video),
            "extractor": "ffmpeg",
        }

    capture = cv2.VideoCapture(str(transfer_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open Transfer2.5 video: {transfer_video}")

    written = 0
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded >= len(metadata):
                break
            frame_idx = decoded
            decoded += 1
            if frame_idx % max(1, args.frame_stride) != 0:
                continue
            if args.max_frames_per_run is not None and written >= args.max_frames_per_run:
                break

            image_path, image_relpath = output_image_path(
                output_dir,
                run_dir.name,
                frame_idx,
                args.image_format,
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.dry_run:
                if args.image_format == "jpg":
                    cv2.imwrite(
                        str(image_path),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                    )
                else:
                    cv2.imwrite(str(image_path), frame)

            history_offsets = [
                -(args.history_steps - 1 - idx) * args.dt
                for idx in range(args.history_steps)
            ]
            future_offsets = [
                (idx + 1) * args.dt
                for idx in range(args.future_steps)
            ]
            record = make_manifest_record(
                source_record=metadata[frame_idx],
                image_relpath=image_relpath,
                source_run=run_dir.name,
                transfer_video=transfer_video,
                transfer_frame_idx=frame_idx,
                history_xyz=local_xyz_sequence(metadata, timestamps, frame_idx, history_offsets),
                future_xyz=local_xyz_sequence(metadata, timestamps, frame_idx, future_offsets),
                args=args,
            )
            if not args.dry_run:
                manifest_file.write(json.dumps(record) + "\n")
            written += 1
    finally:
        capture.release()

    LOGGER.info("Prepared %s photoreal frames from %s", written, run_dir.name)
    return {
        "run": run_dir.name,
        "frames": written,
        "metadata": str(metadata_path),
        "transfer_video": str(transfer_video),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    run_dirs = discover_run_dirs(args)
    if not run_dirs:
        raise RuntimeError("No run directories matched the requested inputs.")

    summaries: list[dict[str, Any]] = []
    mode = "w"
    with manifest_path.open(mode, encoding="utf-8") as manifest_file:
        for run_dir in run_dirs:
            summaries.append(
                process_run(
                    run_dir,
                    output_dir=output_dir,
                    manifest_file=manifest_file,
                    args=args,
                )
            )

    summary = {
        "format": "carla_cosmos_transfer_alpamayo_v1",
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "total_frames": sum(int(item.get("frames", 0)) for item in summaries),
        "runs": summaries,
        "history_steps": int(args.history_steps),
        "future_steps": int(args.future_steps),
        "dt": float(args.dt),
        "frame_stride": int(args.frame_stride),
        "dry_run": bool(args.dry_run),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Wrote summary: %s", summary_path)


if __name__ == "__main__":
    main()
