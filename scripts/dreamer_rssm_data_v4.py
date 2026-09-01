#!/usr/bin/env python3
"""Collision-aware ordered Bench2Drive data for the RSSM V4 candidate.

The original RSSM loader inferred terminal failures from a global result file
and attached them to the last retained transition.  Curriculum runs record an
exact collision wall time next to each trace, so V4 aligns the terminal target
to the first real impact and never trains on post-impact controls.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts import dreamer_online_rl_update as core
from scripts import train_dreamer_rssm_v2 as v2


def _json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def trace_metrics(path: Path, route_id: str, seed: str) -> Dict[str, float]:
    payload = _json(path.parent / "episode.json")
    raw = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    if raw:
        return {
            str(key): core.as_float(value)
            for key, value in raw.items()
            if isinstance(value, (int, float, str, bool))
        }
    return v2.result_metrics(route_id, seed)


def collision_events_for_trace(path: Path) -> List[Dict[str, Any]]:
    return core.read_collision_events(path.parent / "collision_events.jsonl")


def _terminal_risk_ramp(
    risks: List[float],
    lookback: int = 24,
) -> None:
    """Teach anticipation without pretending an impact happened early."""

    count = min(max(1, int(lookback)), len(risks))
    start = len(risks) - count
    for index in range(start, len(risks)):
        phase = float(index - start + 1) / float(count)
        risks[index] = max(risks[index], 0.20 + 0.80 * phase)


def build_episode(
    path: Path,
    rows: List[Dict[str, Any]],
    segment_index: int,
    teacher_paths: set,
    metrics: Dict[str, float],
    terminal_collision: Optional[Dict[str, Any]] = None,
    terminal_metrics: bool = False,
) -> Optional[v2.Episode]:
    route_id = str(rows[0].get("route_id", "unknown"))
    seed = str(rows[0].get("seed", "unknown"))
    is_teacher = str(path.resolve()) in teacher_paths
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    continuations: List[float] = []
    risks: List[float] = []
    progress: List[float] = []
    events: List[np.ndarray] = []
    teacher_targets: List[np.ndarray] = []
    teacher_mask: List[float] = []
    first_base = core.action_dict(rows[0].get("status") or {}, "base_action")
    previous_action = np.asarray(
        [first_base[0], first_base[1], first_base[2], 0.0], dtype=np.float32
    )
    previous_reward_action = previous_action.copy()
    tracker = core.OvertakeRewardTracker()
    stagnant = 0

    for row, next_row in zip(rows, rows[1:]):
        status = row.get("status") or {}
        next_status = next_row.get("status") or {}
        observation = v2.observation_from_status(status, previous_action)
        action = v2.actual_action(status)
        next_observation = v2.observation_from_status(next_status, action)
        if observation is None or next_observation is None:
            continue
        state = np.asarray(status.get("state_vector") or [], dtype=np.float32)
        nxt = np.asarray(next_status.get("state_vector") or [], dtype=np.float32)
        if state.shape[0] < v2.WORLD_STATE_DIM or nxt.shape[0] < v2.WORLD_STATE_DIM:
            continue
        if np.linalg.norm(nxt[:2] - state[:2]) < 0.03 and max(0.0, float(nxt[2])) < 0.2:
            stagnant += 1
        else:
            stagnant = 0
        try:
            reward, _ = core.step_reward(
                status,
                next_status,
                previous_reward_action,
                stagnant_steps=stagnant,
                overtake_tracker=tracker,
            )
        except Exception:
            reward = 0.0
        risk, step_progress, event = v2.geometric_targets(
            status, next_status, action
        )
        applied = bool(status.get("applied"))
        base = core.action_dict(status, "base_action")
        chosen = core.action_dict(status, "chosen_action")
        target_control = chosen[:3] if applied else base[:3]
        target = np.asarray(
            [*target_control, 0.995 if applied else 0.005], dtype=np.float32
        )

        if not observations:
            observations.append(observation)
        observations.append(next_observation)
        actions.append(action)
        rewards.append(float(np.clip(reward, -20.0, 20.0)))
        continuations.append(1.0)
        risks.append(risk)
        progress.append(step_progress)
        events.append(event)
        teacher_targets.append(target)
        teacher_mask.append(1.0 if is_teacher else 0.0)
        previous_action = action
        previous_reward_action = action

    if len(actions) < 8 or len(observations) != len(actions) + 1:
        return None
    continuations[-1] = 0.0

    if terminal_collision is not None:
        _terminal_risk_ramp(risks)
        events[-1][0] = 1.0
        risks[-1] = 1.0
        rewards[-1] = min(rewards[-1], -15.0)
    if terminal_metrics and metrics.get("offroad", 0.0) > 0.0:
        events[-1][1] = 1.0
        risks[-1] = max(risks[-1], 0.90)
        rewards[-1] = min(rewards[-1], -8.0)
    if terminal_metrics and metrics.get("blocked", 0.0) > 0.0:
        events[-1][4] = 1.0
        risks[-1] = max(risks[-1], 0.75)
        rewards[-1] = min(rewards[-1], -4.0)

    if is_teacher:
        source = "validated_guard_teacher"
    elif terminal_collision is not None:
        source = "curriculum_collision"
    elif terminal_metrics and metrics.get("blocked", 0.0) > 0.0:
        source = "curriculum_blocked"
    elif terminal_metrics and metrics.get("success", 0.0) > 0.0:
        source = "curriculum_success"
    elif "dreamer_online_rl" in str(path):
        source = "online_rl"
    else:
        source = "guard_trace"

    return v2.Episode(
        key=f"{route_id}:{seed}:{segment_index}:{path.name}",
        route_id=route_id,
        seed=seed,
        source=source,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        continuation=np.asarray(continuations, dtype=np.float32),
        risks=np.asarray(risks, dtype=np.float32),
        progress=np.asarray(progress, dtype=np.float32),
        events=np.asarray(events, dtype=np.float32),
        teacher_targets=np.asarray(teacher_targets, dtype=np.float32),
        teacher_mask=np.asarray(teacher_mask, dtype=np.float32),
    )


def load_episodes(
    paths: Sequence[Path],
    sequence_length: int,
) -> Tuple[List[v2.Episode], List[Dict[str, Any]]]:
    teacher_paths = v2.validated_teacher_paths()
    episodes: List[v2.Episode] = []
    audit: List[Dict[str, Any]] = []
    for path in paths:
        rows = v2.read_jsonl(path)
        rows.sort(
            key=lambda row: core.as_float(
                (row.get("status") or {}).get("timestamp"),
                core.as_float(row.get("collector_time")),
            )
        )
        if not rows:
            audit.append({"path": str(path), "rows": 0, "accepted_segments": 0})
            continue
        route_id = str(rows[0].get("route_id", "unknown"))
        seed = str(rows[0].get("seed", "unknown"))
        metrics = trace_metrics(path, route_id, seed)
        core.enrich_current_oncoming(rows)
        collision_events = collision_events_for_trace(path)
        rows, collision = core.truncate_rows_at_first_collision(
            rows, collision_events, metrics
        )
        segments = v2.split_contiguous(rows, sequence_length + 1)
        accepted = 0
        transitions = 0
        for segment_index, segment in enumerate(segments):
            terminal = segment_index == len(segments) - 1
            episode = build_episode(
                path,
                segment,
                segment_index,
                teacher_paths,
                metrics,
                terminal_collision=collision if terminal else None,
                terminal_metrics=terminal,
            )
            if episode is not None:
                episodes.append(episode)
                accepted += 1
                transitions += episode.transitions
        audit.append({
            "path": str(path),
            "rows": len(rows),
            "segments": len(segments),
            "accepted_segments": accepted,
            "transitions": transitions,
            "route_id": route_id,
            "seed": seed,
            "validated_teacher": str(path.resolve()) in teacher_paths,
            "metrics": metrics,
            "collision_aligned": collision is not None,
            "collision_source": str((collision or {}).get("source", "")),
            "impact_status_index": (collision or {}).get("impact_status_index"),
            "impact_transition_index": (collision or {}).get(
                "impact_transition_index"
            ),
        })
    return episodes, audit


def split_route_seed_stratified(
    episodes: Sequence[v2.Episode],
    seed: int,
) -> Tuple[List[v2.Episode], List[v2.Episode], List[str]]:
    """Hold out seeds within represented routes and whole singleton routes."""

    groups: Dict[Tuple[str, str], List[v2.Episode]] = {}
    for episode in episodes:
        groups.setdefault((episode.route_id, episode.seed), []).append(episode)
    by_route: Dict[str, List[Tuple[str, str]]] = {}
    for key in groups:
        by_route.setdefault(key[0], []).append(key)
    rng = random.Random(seed)
    validation_keys = set()
    singleton_routes: List[str] = []
    for route, keys in sorted(by_route.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        if len(keys) >= 2:
            collision_keys = [
                key for key in keys
                if any(float(row.events[:, 0].sum()) > 0.0 for row in groups[key])
            ]
            # Safety validation needs real impacts. Prefer one collision seed
            # whenever another seed on the same route remains available.
            validation_keys.add(collision_keys[0] if collision_keys else keys[0])
        else:
            singleton_routes.append(route)
    rng.shuffle(singleton_routes)
    singleton_validation_count = max(1, int(round(len(singleton_routes) * 0.25)))
    for route in singleton_routes[:singleton_validation_count]:
        validation_keys.add(by_route[route][0])

    training: List[v2.Episode] = []
    validation: List[v2.Episode] = []
    for key, rows in groups.items():
        (validation if key in validation_keys else training).extend(rows)
    if not training or not validation:
        raise RuntimeError("route/seed-stratified RSSM split is empty")
    labels = sorted(f"{route}:{seed_value}" for route, seed_value in validation_keys)
    return training, validation, labels


def dataset_summary(episodes: Sequence[v2.Episode]) -> Dict[str, Any]:
    sources: Dict[str, int] = {}
    for episode in episodes:
        sources[episode.source] = sources.get(episode.source, 0) + 1
    return {
        "episodes": len(episodes),
        "transitions": int(sum(episode.transitions for episode in episodes)),
        "routes": sorted({episode.route_id for episode in episodes}),
        "seeds": len({(episode.route_id, episode.seed) for episode in episodes}),
        "sources": sources,
        "collision_transitions": int(
            sum(float(episode.events[:, 0].sum()) for episode in episodes)
        ),
        "offroad_transitions": int(
            sum(float(episode.events[:, 1].sum()) for episode in episodes)
        ),
        "blocked_transitions": int(
            sum(float(episode.events[:, 4].sum()) for episode in episodes)
        ),
        "teacher_transitions": int(
            sum(float(episode.teacher_mask.sum()) for episode in episodes)
        ),
    }
