#!/usr/bin/env python3
import argparse
import os
import pathlib
import time

import cv2
import pygame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--title", default="SimLingo replay")
    parser.add_argument("--no-frame-skip", action="store_true")
    args = parser.parse_args()

    video_path = pathlib.Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    speed = max(0.1, args.speed)
    delay = 1.0 / max(1.0, fps * speed)
    use_frame_skip = not args.no_frame_skip

    print(
        f"[simlingo-replay] Opening {video_path} "
        f"{width}x{height} {fps:.2f}fps frames={frame_count} "
        f"speed=x{speed:g} frame_skip={int(use_frame_skip)} "
        f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}",
        flush=True,
    )
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption(f"{args.title} x{args.speed:g}")

    running = True
    start_time = time.monotonic()
    last_frame_index = -1
    displayed_frames = 0
    skipped_frames = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        if use_frame_skip:
            elapsed = time.monotonic() - start_time
            target_frame = int(elapsed * fps * speed)

            if frame_count and target_frame >= frame_count:
                break

            if target_frame <= last_frame_index:
                next_frame_time = (last_frame_index + 1) / max(1.0, fps * speed)
                sleep_time = next_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.05))
                continue

            if target_frame > last_frame_index + 1:
                skipped_frames += target_frame - last_frame_index - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ok, frame = cap.read()
        if not ok:
            break
        if use_frame_skip:
            last_frame_index = target_frame
        else:
            last_frame_index += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        displayed_frames += 1
        if not use_frame_skip:
            time.sleep(delay)

    cap.release()
    pygame.quit()
    wall_time = max(time.monotonic() - start_time, 0.001)
    played_video_seconds = max(last_frame_index, 0) / max(fps, 1.0)
    print(
        f"[simlingo-replay] closed displayed={displayed_frames} skipped={skipped_frames} "
        f"video_time={played_video_seconds:.2f}s wall_time={wall_time:.2f}s "
        f"effective_speed=x{played_video_seconds / wall_time:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
