#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import time
from collections import deque

import carla
import cv2
import numpy as np
import pygame


def find_ego(world):
    vehicles = list(world.get_actors().filter("vehicle.*"))
    preferred_roles = {"hero", "ego", "ego_vehicle", "hero0"}
    for actor in vehicles:
        if actor.attributes.get("role_name") in preferred_roles:
            return actor
    for actor in vehicles:
        if actor.attributes.get("role_name") not in {"scenario", "background"}:
            return actor
    return vehicles[0] if vehicles else None


def camera_transform(mode):
    if mode == "chase":
        return carla.Transform(carla.Location(x=-6.5, z=3.0), carla.Rotation(pitch=-14.0))
    if mode == "top":
        return carla.Transform(carla.Location(x=0.0, z=28.0), carla.Rotation(pitch=-90.0))
    if mode == "wheel":
        # A literal in-cabin camera clips through several CARLA vehicle meshes and can
        # destabilize the render. This keeps the driver's feel from just behind the hood.
        return carla.Transform(carla.Location(x=1.15, y=0.0, z=1.45), carla.Rotation(pitch=-5.0))
    return carla.Transform(carla.Location(x=1.3, z=1.75), carla.Rotation(pitch=-2.0))


def set_spectator(world, ego, mode):
    transform = ego.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    if mode == "top":
        loc = transform.location + carla.Location(z=42.0)
        rot = carla.Rotation(pitch=-90.0, yaw=transform.rotation.yaw)
    else:
        loc = transform.location + carla.Location(
            x=-8.0 * math.cos(yaw),
            y=-8.0 * math.sin(yaw),
            z=4.0,
        )
        rot = carla.Rotation(pitch=-16.0, yaw=transform.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(loc, rot))


def force_visual_weather(world, visual_weather):
    if visual_weather == "route":
        return
    presets = {
        "day": carla.WeatherParameters.ClearNoon,
        "soft": carla.WeatherParameters.CloudyNoon,
        "sunset": carla.WeatherParameters.ClearSunset,
    }
    world.set_weather(presets.get(visual_weather, carla.WeatherParameters.ClearNoon))


def draw_status(screen, font, width, height, message):
    screen.fill((15, 23, 42))
    title = font.render(message, True, (255, 250, 240))
    hint = font.render("SimLingo Ego POV | q/esc to close", True, (203, 213, 225))
    screen.blit(title, (24, 24))
    screen.blit(hint, (24, height - 42))
    pygame.display.flip()


def blit_frame(screen, frame):
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surface, (0, 0))


def draw_overlay(screen, font, width, height, message, alpha=170):
    legend = pygame.Surface((360, 92), pygame.SRCALPHA)
    legend.fill((245, 247, 250, 185))
    entries = (
        ((255, 0, 0), "Predicted path WPs"),
        ((0, 220, 0), "Predicted speed WPs"),
        ((0, 70, 255), "Target points"),
    )
    for idx, (color, label) in enumerate(entries):
        y = 18 + idx * 26
        pygame.draw.circle(legend, color, (22, y + 7), 6)
        legend_text = font.render(label, True, (15, 23, 42))
        legend.blit(legend_text, (40, y - 2))
    screen.blit(legend, (16, 16))

    overlay = pygame.Surface((width, 46), pygame.SRCALPHA)
    overlay.fill((31, 41, 55, alpha))
    screen.blit(overlay, (0, height - 46))
    text = font.render(message, True, (255, 250, 240))
    screen.blit(text, (16, height - 34))


def read_dreamer_status(path):
    if not path:
        return None
    status_path = pathlib.Path(path)
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        payload["stale"] = time.time() - float(payload.get("timestamp", 0.0)) > 3.0
        return payload
    except Exception:
        return None


def read_cot_status(path):
    if not path:
        return None
    status_path = pathlib.Path(path)
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        payload["stale"] = time.time() - float(payload.get("timestamp", 0.0)) > 8.0
        return payload
    except Exception:
        return None


def wrap_to_width(font, text, max_width, max_lines=2):
    words = str(text or "").split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(words) > 0:
        while lines and font.size(lines[-1] + "...")[0] > max_width and len(lines[-1]) > 4:
            lines[-1] = lines[-1][:-1]
        if lines:
            lines[-1] = lines[-1].rstrip(" .,;:") + "..."
    return lines or [""]


def draw_cot_overlay(screen, font, width, height, status, dreamer_status=None):
    if not status:
        return

    panel_w = min(680, max(420, width - 32))
    panel_h = 176
    x = 16
    y = 122
    surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    surface.fill((8, 13, 22, 205))

    risk = str(status.get("risk_level", "unknown")).lower()
    stale = bool(status.get("stale"))
    dreamer_status = dreamer_status or {}
    dreamer_risk = float(dreamer_status.get("base_risk", 0.0) or 0.0)
    front_m = float(dreamer_status.get("front_vehicle_m", 80.0) or 80.0)
    critical_disagreement = risk in ("low", "unknown") and (dreamer_risk >= 0.70 or front_m <= 8.0)

    if stale:
        color = (148, 163, 184)
        badge = "STALE"
    elif critical_disagreement:
        color = (248, 113, 113)
        badge = "CHECK GEOMETRY"
    elif risk in ("critical", "high"):
        color = (248, 113, 113)
        badge = risk.upper()
    elif risk == "medium":
        color = (251, 191, 36)
        badge = "MEDIUM"
    elif risk == "low":
        color = (126, 242, 162)
        badge = "LOW"
    else:
        color = (110, 231, 249)
        badge = "OBSERVE"

    pygame.draw.rect(surface, color, pygame.Rect(0, 0, 6, panel_h))
    mode = str(status.get("mode", "cot")).upper()
    model = pathlib.Path(str(status.get("model", "external"))).name
    title = font.render(f"External CoT {mode} | {badge} | {model}", True, color)
    surface.blit(title, (18, 12))

    scene = status.get("scene_type", "unknown")
    hazard = status.get("main_hazard", "unknown")
    hint = status.get("safe_hint", "observe_only")
    confidence = float(status.get("confidence", 0.0) or 0.0)
    lines = [
        f"scene {scene} | hazard {hazard} | conf {confidence:.2f}",
        f"hint {hint}",
    ]
    if status.get("error"):
        lines.append(f"error {status.get('reason', '')}")
    elif critical_disagreement:
        lines.append(
            f"warning CoT says {risk}, but Dreamer sees risk {dreamer_risk:.2f} / front {front_m:.1f}m"
        )
        lines.extend(wrap_to_width(font, f"why {status.get('reason', '')}", panel_w - 36, max_lines=1))
    else:
        lines.extend(wrap_to_width(font, f"why {status.get('reason', '')}", panel_w - 36, max_lines=2))

    for idx, line in enumerate(lines[:5]):
        text = font.render(line, True, (235, 241, 249))
        surface.blit(text, (18, 38 + idx * 24))
    screen.blit(surface, (x, y))


def draw_dreamer_overlay(screen, font, width, status):
    if not status:
        return

    panel_w = min(560, max(360, width - 410))
    gap_sides = status.get("gap_recovery_sides") or []
    panel_h = 172 if status.get("collision_shield_active") or gap_sides else 150
    x = width - panel_w - 16
    y = 16
    surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    surface.fill((8, 13, 22, 205))

    mode = str(status.get("mode", "off")).upper()
    applied = bool(status.get("applied"))
    would = bool(status.get("would_override"))
    stale = bool(status.get("stale"))
    if stale:
        badge = "STALE"
        color = (148, 163, 184)
    elif applied:
        badge = "APPLIED"
        color = (126, 242, 162)
    elif would:
        badge = "WOULD"
        color = (255, 209, 102)
    else:
        badge = "HOLD"
        color = (110, 231, 249)

    pygame.draw.rect(surface, color, pygame.Rect(0, 0, 6, panel_h))
    title = font.render(f"Dreamer {mode} | {badge}", True, color)
    surface.blit(title, (18, 12))

    risk_line = (
        f"risk {float(status.get('base_risk', 0.0)):.3f} -> "
        f"{float(status.get('chosen_risk', 0.0)):.3f}   "
        f"progress {float(status.get('base_progress', 0.0)):.4f} -> "
        f"{float(status.get('chosen_progress', 0.0)):.4f}"
    )
    scene_line = (
        f"candidate {int(status.get('candidate_index', 0))}   "
        f"{status.get('chosen_kind', 'model')}   "
        f"blocked {int(status.get('blocked_ticks', 0))}   "
        f"hold {int(status.get('recovery_active_ticks', 0))}   "
        f"commit {int(status.get('recovery_commit_ticks', 0))}   "
        f"finish {int(status.get('recovery_finish_active_ticks', 0))}   "
        f"side {int(status.get('recovery_side', 0))}   "
        f"front {float(status.get('front_vehicle_m', 80.0)):.1f}m   "
        f"light {status.get('traffic_light', 'none')}"
    )
    clearance_line = (
        f"left clear {float(status.get('left_clear_m', 80.0)):.1f}m   "
        f"right clear {float(status.get('right_clear_m', 80.0)):.1f}m   "
        f"lane L/R {int(bool(status.get('left_lane_available', True)))}/"
        f"{int(bool(status.get('right_lane_available', True)))}   "
        f"TTC L/R {float(status.get('left_ttc_s', 99.0)):.1f}/"
        f"{float(status.get('right_ttc_s', 99.0)):.1f}s"
    )
    oncoming_line = (
        f"oncoming L/R {float(status.get('left_oncoming_m', 80.0)):.1f}/"
        f"{float(status.get('right_oncoming_m', 80.0)):.1f}m   "
        f"TTC {float(status.get('left_oncoming_ttc_s', 99.0)):.1f}/"
        f"{float(status.get('right_oncoming_ttc_s', 99.0)):.1f}s"
    )
    base = status.get("base_action", {})
    chosen = status.get("chosen_action", {})
    action_line = (
        "S "
        f"{float(base.get('steer', 0.0)):+.2f}->{float(chosen.get('steer', 0.0)):+.2f}  "
        "T "
        f"{float(base.get('throttle', 0.0)):.2f}->{float(chosen.get('throttle', 0.0)):.2f}  "
        "B "
        f"{float(base.get('brake', 0.0)):.2f}->{float(chosen.get('brake', 0.0)):.2f}"
    )

    lines = [risk_line, scene_line, clearance_line, oncoming_line, action_line]
    if status.get("collision_shield_active"):
        reason = str(status.get("collision_shield_reason", ""))
        lines.append(f"shield active: {reason[:82]}")
    elif gap_sides:
        lines.append(f"gap commit available: sides {gap_sides}")

    for idx, line in enumerate(lines):
        text = font.render(line, True, (235, 241, 249))
        surface.blit(text, (18, 38 + idx * 22))
    screen.blit(surface, (x, y))


def draw_held_frame(screen, font, width, height, frame, message):
    blit_frame(screen, frame)
    draw_overlay(screen, font, width, height, message, alpha=190)
    pygame.display.flip()


def record_frame(writer, screen):
    if writer is None:
        return
    frame = pygame.surfarray.array3d(screen).swapaxes(0, 1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def publish_cot_frame(frame, args, stats):
    if not args.cot_frame_path:
        return
    now = time.time()
    if now - float(stats.get("last_cot_publish", 0.0)) < args.cot_frame_interval:
        return
    stats["last_cot_publish"] = now
    try:
        out = frame
        if args.cot_frame_width > 0 and frame.shape[1] > args.cot_frame_width:
            scale = args.cot_frame_width / float(frame.shape[1])
            out = cv2.resize(
                frame,
                (args.cot_frame_width, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        path = pathlib.Path(args.cot_frame_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.jpg")
        cv2.imwrite(str(tmp), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        tmp.replace(path)
    except Exception as exc:
        stats["cot_publish_error"] = str(exc)


def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    matrix = np.identity(3)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def project_world_location(location, camera_transform, calibration):
    world_point = np.array([location.x, location.y, location.z, 1.0])
    world_to_camera = np.array(camera_transform.get_inverse_matrix())
    camera_point = world_to_camera.dot(world_point)
    # CARLA uses x-forward/y-right/z-up; pinhole projection expects z-forward.
    point = np.array([camera_point[1], -camera_point[2], camera_point[0]])
    if point[2] <= 0.1:
        return None
    image_point = calibration.dot(point)
    x = image_point[0] / image_point[2]
    y = image_point[1] / image_point[2]
    return int(x), int(y), float(point[2])


def traffic_light_marker_location(light):
    try:
        if hasattr(light, "trigger_volume"):
            loc = light.get_transform().transform(light.trigger_volume.location)
            return carla.Location(loc.x, loc.y, loc.z + 4.0)
        loc = light.get_transform().location
        return carla.Location(loc.x, loc.y, loc.z + 3.0)
    except RuntimeError:
        return None


def traffic_light_state_style(light):
    try:
        state = str(light.get_state()).rsplit(".", 1)[-1]
    except RuntimeError:
        state = "Unknown"
    if state == "Red":
        return state, (255, 64, 64), "R"
    if state == "Yellow":
        return state, (255, 210, 64), "Y"
    if state == "Green":
        return state, (42, 230, 110), "G"
    return state, (170, 180, 190), "?"


def draw_traffic_light_overlay(screen, font, world, sensor, width, height, fov, max_distance, max_markers):
    if sensor is None or not sensor.is_alive:
        return
    try:
        camera_transform = sensor.get_transform()
        camera_location = camera_transform.location
        lights = list(world.get_actors().filter("traffic.traffic_light*"))
    except Exception:
        return

    calibration = build_projection_matrix(width, height, fov)
    markers = []
    for light in lights:
        location = traffic_light_marker_location(light)
        if location is None:
            continue
        try:
            distance = camera_location.distance(location)
        except RuntimeError:
            continue
        if distance > max_distance:
            continue
        projected = project_world_location(location, camera_transform, calibration)
        if projected is None:
            continue
        x, y, depth = projected
        if x < -40 or x > width + 40 or y < -40 or y > height + 40:
            continue
        state, color, label = traffic_light_state_style(light)
        markers.append((depth, x, y, color, label, state, light.id))

    markers.sort(reverse=True)
    if max_markers > 0:
        markers = markers[-max_markers:]
    for _depth, x, y, color, label, state, light_id in markers:
        radius = 10
        badge = pygame.Surface((68, 30), pygame.SRCALPHA)
        badge.fill((8, 13, 22, 175))
        pygame.draw.circle(badge, color, (15, 15), radius)
        pygame.draw.circle(badge, (255, 255, 255), (15, 15), radius, 2)
        text = font.render(label, True, (8, 13, 22))
        badge.blit(text, (10, 4))
        state_text = font.render(str(light_id)[-3:], True, (235, 241, 249))
        badge.blit(state_text, (31, 5))
        screen.blit(badge, (max(0, min(width - 68, x - 15)), max(0, min(height - 76, y - 30))))


def connect_client(host, port, timeout):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            client = carla.Client(host, port)
            client.set_timeout(4.0)
            client.get_world()
            return client
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Could not connect to CARLA on {host}:{port}: {last_error}")


def destroy_sensor(sensor):
    if sensor is None:
        return
    try:
        sensor.stop()
    except Exception:
        pass
    try:
        sensor.destroy()
    except Exception:
        pass


def enhance_frame(arr, args):
    frame = arr.astype(np.float32)
    frame = (frame - 127.5) * args.contrast + 127.5 + args.brightness
    if abs(args.saturation - 1.0) > 1e-3:
        gray = (
            frame[:, :, 0:1] * 0.299
            + frame[:, :, 1:2] * 0.587
            + frame[:, :, 2:3] * 0.114
        )
        frame = gray + (frame - gray) * args.saturation
    return np.clip(frame, 0, 255).astype(np.uint8)


def spawn_camera(world, ego, args, frame_queue, stats):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(args.width))
    blueprint.set_attribute("image_size_y", str(args.height))
    blueprint.set_attribute("fov", str(args.fov))
    sensor = world.spawn_actor(blueprint, camera_transform(args.mode), attach_to=ego)

    def on_image(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        raw_mean = float(arr.mean())
        raw_p95 = float(np.percentile(arr, 95))
        stats["last_raw_mean"] = raw_mean
        stats["last_raw_p95"] = raw_p95
        last_good_mean = float(stats.get("last_good_mean", 0.0))
        dropped_hard_black = raw_mean < args.min_valid_brightness or raw_p95 < args.min_valid_p95
        dropped_flash = (
            last_good_mean > args.min_valid_brightness * 2.0
            and raw_mean < last_good_mean * args.dark_drop_ratio
        )
        if not args.disable_black_frame_filter and (dropped_hard_black or dropped_flash):
            stats["dropped_dark_frames"] = stats.get("dropped_dark_frames", 0) + 1
            return
        stats["last_good_mean"] = raw_mean
        arr = enhance_frame(arr, args)
        frame_queue.append((arr, time.time()))

    sensor.listen(on_image)
    return sensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--mode", choices=["front", "chase", "top", "wheel"], default="chase")
    parser.add_argument("--visual-weather", choices=["day", "soft", "sunset", "route"], default="day")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--stale-frame-seconds", type=float, default=60.0)
    parser.add_argument("--min-valid-brightness", type=float, default=12.0)
    parser.add_argument("--min-valid-p95", type=float, default=24.0)
    parser.add_argument("--dark-drop-ratio", type=float, default=0.45)
    parser.add_argument("--brightness", type=float, default=8.0)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--saturation", type=float, default=1.10)
    parser.add_argument("--max-fps", type=int, default=45)
    parser.add_argument("--disable-black-frame-filter", action="store_true")
    parser.add_argument("--record-path")
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--dreamer-status-path", default="")
    parser.add_argument("--cot-status-path", default="")
    parser.add_argument("--cot-frame-path", default="")
    parser.add_argument("--cot-frame-interval", type=float, default=2.0)
    parser.add_argument("--cot-frame-width", type=int, default=768)
    parser.add_argument("--traffic-light-overlay", action="store_true")
    parser.add_argument("--traffic-light-overlay-distance", type=float, default=160.0)
    parser.add_argument("--traffic-light-overlay-max", type=int, default=80)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("SimLingo Ego POV")
    font = pygame.font.SysFont("IBM Plex Sans, Segoe UI, Arial", 20)
    small_font = pygame.font.SysFont("IBM Plex Sans, Segoe UI, Arial", 14, bold=True)
    writer = None
    if args.record_path:
        record_path = pathlib.Path(args.record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(record_path), fourcc, args.record_fps, (args.width, args.height))
        if not writer.isOpened():
            print(f"[simlingo-pov] Could not open video writer: {record_path}", flush=True)
            writer = None
        else:
            print(f"[simlingo-pov] Recording Pygame view to {record_path}", flush=True)

    draw_status(screen, font, args.width, args.height, f"Connecting to CARLA on {args.host}:{args.port}...")
    client = connect_client(args.host, args.port, args.timeout)
    frame_queue = deque(maxlen=1)
    camera_stats = {"last_raw_mean": 0.0, "dropped_dark_frames": 0}
    last_frame = None
    last_frame_at = 0.0
    sensor = None
    ego = None
    attached_ego_id = None
    start = time.time()

    try:
        while time.time() - start < args.timeout and ego is None:
            world = client.get_world()
            ego = find_ego(world)
            if ego is None:
                draw_status(screen, font, args.width, args.height, "Waiting for SimLingo ego vehicle...")
                time.sleep(1.0)

        if ego is None:
            raise RuntimeError("No ego vehicle found. Start SimLingo first.")

        world = client.get_world()
        sensor = spawn_camera(world, ego, args, frame_queue, camera_stats)
        attached_ego_id = ego.id

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            try:
                world = client.get_world()
            except Exception:
                if last_frame is not None:
                    draw_held_frame(screen, font, args.width, args.height, last_frame, "Waiting for CARLA world...")
                else:
                    draw_status(screen, font, args.width, args.height, "Waiting for CARLA world...")
                time.sleep(0.5)
                continue

            if ego is None or not ego.is_alive:
                destroy_sensor(sensor)
                sensor = None
                attached_ego_id = None
                ego = find_ego(world)
                if ego is None:
                    if last_frame is not None:
                        draw_held_frame(
                            screen,
                            font,
                            args.width,
                            args.height,
                            last_frame,
                            "Waiting for SimLingo ego vehicle...",
                        )
                    else:
                        draw_status(screen, font, args.width, args.height, "Waiting for SimLingo ego vehicle...")
                    time.sleep(0.5)
                    continue

            if sensor is None or not sensor.is_alive or attached_ego_id != ego.id:
                destroy_sensor(sensor)
                try:
                    sensor = spawn_camera(world, ego, args, frame_queue, camera_stats)
                    attached_ego_id = ego.id
                except Exception as exc:
                    message = f"Waiting for camera attach: {exc}"
                    if last_frame is not None:
                        draw_held_frame(screen, font, args.width, args.height, last_frame, message)
                    else:
                        draw_status(screen, font, args.width, args.height, message)
                    sensor = None
                    time.sleep(0.5)
                    continue

            try:
                force_visual_weather(world, args.visual_weather)
                set_spectator(world, ego, args.mode)
            except Exception:
                pass

            if frame_queue:
                frame, frame_at = frame_queue[-1]
                last_frame = frame
                last_frame_at = frame_at
                blit_frame(screen, frame)
                publish_cot_frame(frame, args, camera_stats)
            elif last_frame is not None and time.time() - last_frame_at <= args.stale_frame_seconds:
                draw_held_frame(screen, font, args.width, args.height, last_frame, "Waiting for next camera frame...")
                clock.tick(args.max_fps)
                continue
            else:
                draw_status(screen, font, args.width, args.height, "Waiting for camera frames...")
                clock.tick(min(args.max_fps, 10))
                continue

            vel = ego.get_velocity()
            speed_kmh = 3.6 * math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
            message = f"SimLingo VLA | ego POV={args.mode} | speed={speed_kmh:05.1f} km/h | q/esc to close"
            if camera_stats.get("dropped_dark_frames", 0):
                message += f" | dark frames held={camera_stats['dropped_dark_frames']}"
            draw_overlay(screen, font, args.width, args.height, message)
            dreamer_status = read_dreamer_status(args.dreamer_status_path)
            draw_cot_overlay(
                screen,
                font,
                args.width,
                args.height,
                read_cot_status(args.cot_status_path),
                dreamer_status,
            )
            draw_dreamer_overlay(screen, font, args.width, dreamer_status)
            if args.traffic_light_overlay:
                draw_traffic_light_overlay(
                    screen,
                    small_font,
                    world,
                    sensor,
                    args.width,
                    args.height,
                    args.fov,
                    args.traffic_light_overlay_distance,
                    args.traffic_light_overlay_max,
                )
            record_frame(writer, screen)
            pygame.display.flip()
            clock.tick(args.max_fps)
    finally:
        if writer is not None:
            writer.release()
            print(f"[simlingo-pov] Recording finalized: {args.record_path}", flush=True)
        destroy_sensor(sensor)
        pygame.quit()


if __name__ == "__main__":
    main()
