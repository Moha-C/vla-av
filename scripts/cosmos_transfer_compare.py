"""Pygame viewer comparing CARLA controls with a Cosmos-Transfer2.5 output video."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pygame is required for cosmos_transfer_compare.py") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=560)
    parser.add_argument("--middle", default="seg", choices=("seg", "depth"))
    return parser.parse_args()


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

    def read_rgb(self) -> np.ndarray | None:
        ok, frame = self.capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def reset(self) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        self.capture.release()


def find_transfer_output(run_dir: Path) -> Path | None:
    output_dir = run_dir / "transfer_output"
    if not output_dir.exists():
        return None

    preferred = output_dir / f"{run_dir.name}.mp4"
    if preferred.exists():
        return preferred

    candidates = sorted(
        path
        for path in output_dir.rglob("*.mp4")
        if "_control_" not in path.name and "_mask_" not in path.name
    )
    return candidates[0] if candidates else None


def wait_for_output(run_dir: Path, explicit_output: Path | None) -> Path:
    while True:
        output_path = explicit_output if explicit_output is not None and explicit_output.exists() else find_transfer_output(run_dir)
        if output_path is not None and output_path.exists():
            return output_path
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                pygame.quit()
                sys.exit(0)
        time.sleep(1.0)


def draw_text(screen: Any, text: str, x: int, y: int, font: Any) -> None:
    surface = font.render(text, True, (235, 240, 245))
    screen.blit(surface, (x, y))


def draw_image(screen: Any, image: np.ndarray, bounds: Any) -> None:
    image = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    surface = pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))
    target = fit_rect(image.shape[1], image.shape[0], bounds)
    if surface.get_size() != target.size:
        surface = pygame.transform.smoothscale(surface, target.size)
    screen.blit(surface, target)


def fit_rect(src_w: int, src_h: int, bounds: Any) -> Any:
    scale = min(bounds.width / max(src_w, 1), bounds.height / max(src_h, 1))
    width = max(1, int(src_w * scale))
    height = max(1, int(src_h * scale))
    x = bounds.x + (bounds.width - width) // 2
    y = bounds.y + (bounds.height - height) // 2
    return pygame.Rect(x, y, width, height)


def draw_wait_screen(screen: Any, font: Any, run_dir: Path) -> None:
    screen.fill((8, 10, 12))
    draw_text(screen, "Waiting for Cosmos-Transfer2.5 output...", 28, 40, font)
    draw_text(screen, str(run_dir / "transfer_output"), 28, 76, font)
    pygame.display.flip()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    rgb_path = run_dir / "carla_rgb.mp4"
    middle_path = run_dir / ("carla_depth.mp4" if args.middle == "depth" else "carla_seg.mp4")
    for path in (rgb_path, middle_path):
        if not path.exists():
            raise FileNotFoundError(path)

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((args.window_width, args.window_height))
    pygame.display.set_caption("VLA-AV Transfer2.5 Compare")
    font = pygame.font.Font(None, 24)

    explicit_output = Path(args.output_video).expanduser().resolve() if args.output_video else None
    output_path = explicit_output if explicit_output and explicit_output.exists() else find_transfer_output(run_dir)
    if output_path is None:
        if not args.wait:
            raise FileNotFoundError(f"No Transfer2.5 output mp4 found under {run_dir / 'transfer_output'}")
        draw_wait_screen(screen, font, run_dir)
        output_path = wait_for_output(run_dir, explicit_output)

    readers = [
        VideoReader(rgb_path),
        VideoReader(middle_path),
        VideoReader(output_path),
    ]
    labels = ("CARLA RGB", f"CARLA {args.middle}", "Cosmos-Transfer2.5")
    clock = pygame.time.Clock()
    label_h = 36

    try:
        running = True
        paused = False
        frames: list[np.ndarray | None] = [None, None, None]
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if not paused:
                for idx, reader in enumerate(readers):
                    frame = reader.read_rgb()
                    if frame is None:
                        reader.reset()
                        frame = reader.read_rgb()
                    frames[idx] = frame

            screen.fill((8, 10, 12))
            width, height = screen.get_size()
            column_w = width // 3
            for idx, frame in enumerate(frames):
                x = idx * column_w
                pygame.draw.rect(screen, (24, 28, 34), pygame.Rect(x, 0, column_w, label_h))
                draw_text(screen, labels[idx], x + 12, 10, font)
                if frame is not None:
                    draw_image(screen, frame, pygame.Rect(x, label_h, column_w, height - label_h))
                if idx > 0:
                    pygame.draw.line(screen, (34, 38, 44), (x, 0), (x, height), 2)
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        for reader in readers:
            reader.close()
        pygame.quit()


if __name__ == "__main__":
    main()
