#!/usr/bin/env python3
"""Manage immutable, production, candidate, and promoted Dreamer checkpoints.

Training is never allowed to mutate the checkpoint used by the dashboard.  A
candidate can only replace production through the explicit ``promote`` command
and a frozen-evaluation report whose promotion gate passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "external" / "simlingo" / "checkpoints"
KIND_DIRS = {
    "ppo": CHECKPOINT_ROOT / "dreamer_ppo_rl_noguard",
    "sdbs": CHECKPOINT_ROOT / "dreamer_sdbs_rl_noguard",
}


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(destination.parent), delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        temporary = Path(tmp.name)
    temporary.replace(path)


def kind_dir(kind: str) -> Path:
    return KIND_DIRS[kind]


def role_paths(kind: str) -> Dict[str, Path]:
    directory = kind_dir(kind)
    return {
        "production": directory / "production_model.pt",
        "candidate": directory / "candidate_model.pt",
        "best": directory / "best_model.pt",
        "legacy": directory / "latest_rl_model.pt",
        "manifest": directory / "checkpoint_roles.json",
    }


def backup_if_present(path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    destination = path.parent / "role_backups" / f"{path.stem}_{reason}_{timestamp()}{path.suffix}"
    atomic_copy(path, destination)
    return destination


def resolve(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def init_roles(args: argparse.Namespace) -> Dict[str, Any]:
    source = resolve(args.source)
    paths = role_paths(args.kind)
    digest = sha256(source)
    protected_dir = kind_dir(args.kind) / "protected_snapshots"
    protected_name = f"{args.label}_{digest[:12]}.pt"
    protected = protected_dir / protected_name
    if protected.exists() and sha256(protected) != digest:
        raise RuntimeError(f"protected checkpoint hash mismatch: {protected}")
    if not protected.exists():
        atomic_copy(source, protected)
    os.chmod(protected, 0o444)

    backups: Dict[str, str] = {}
    for role in ("production", "candidate", "best"):
        backup = backup_if_present(paths[role], "before_role_init")
        if backup is not None:
            backups[role] = str(backup)
        atomic_copy(protected, paths[role])

    payload: Dict[str, Any] = {
        "schema": "dreamer_checkpoint_roles_v1",
        "kind": args.kind,
        "initialized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protected_source": str(source),
        "protected_checkpoint": str(protected),
        "protected_sha256": digest,
        "label": args.label,
        "roles": {
            role: {"path": str(paths[role]), "sha256": sha256(paths[role])}
            for role in ("production", "candidate", "best")
        },
        "legacy_checkpoint": str(paths["legacy"]),
        "backups": backups,
        "promotion_history": [],
    }
    atomic_json(paths["manifest"], payload)
    return payload


def load_manifest(kind: str) -> Dict[str, Any]:
    path = role_paths(kind)["manifest"]
    if not path.exists():
        raise RuntimeError(
            f"checkpoint roles are not initialized for {kind}; run the init command first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def reset_candidate(args: argparse.Namespace) -> Dict[str, Any]:
    paths = role_paths(args.kind)
    manifest = load_manifest(args.kind)
    source_role = args.from_role
    source = paths[source_role]
    if not source.exists():
        raise FileNotFoundError(source)
    backup = backup_if_present(paths["candidate"], "before_candidate_reset")
    atomic_copy(source, paths["candidate"])
    manifest["roles"]["candidate"] = {
        "path": str(paths["candidate"]),
        "sha256": sha256(paths["candidate"]),
        "reset_from": source_role,
        "reset_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if backup is not None:
        manifest.setdefault("candidate_backups", []).append(str(backup))
    atomic_json(paths["manifest"], manifest)
    return manifest


def promote(args: argparse.Namespace) -> Dict[str, Any]:
    paths = role_paths(args.kind)
    manifest = load_manifest(args.kind)
    evaluation_path = resolve(args.evaluation)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not bool(evaluation.get("promotion_approved")):
        raise RuntimeError("frozen evaluation did not approve promotion")

    candidate = paths["candidate"]
    expected_hash = str(evaluation.get("candidate_sha256", ""))
    candidate_hash = sha256(candidate)
    if expected_hash and expected_hash != candidate_hash:
        raise RuntimeError(
            f"candidate changed after evaluation: expected {expected_hash}, got {candidate_hash}"
        )

    production_backup = backup_if_present(paths["production"], "before_promotion")
    atomic_copy(candidate, paths["best"])
    atomic_copy(candidate, paths["production"])
    if args.sync_legacy:
        backup_if_present(paths["legacy"], "before_promoted_sync")
        atomic_copy(candidate, paths["legacy"])

    event = {
        "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_sha256": candidate_hash,
        "evaluation": str(evaluation_path),
        "production_backup": str(production_backup) if production_backup else None,
        "sync_legacy": bool(args.sync_legacy),
    }
    manifest.setdefault("promotion_history", []).append(event)
    for role in ("production", "candidate", "best"):
        manifest["roles"][role] = {
            "path": str(paths[role]),
            "sha256": sha256(paths[role]),
        }
    atomic_json(paths["manifest"], manifest)
    return {"status": "promoted", **event, "manifest": str(paths["manifest"])}


def status(args: argparse.Namespace) -> Dict[str, Any]:
    paths = role_paths(args.kind)
    manifest = load_manifest(args.kind)
    current: Dict[str, Any] = {}
    for role, path in paths.items():
        if role == "manifest":
            continue
        current[role] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else None,
            "mtime": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)
            ) if path.exists() else None,
        }
    return {"manifest": manifest, "current": current}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--kind", choices=tuple(KIND_DIRS), required=True)
    init_parser.add_argument("--source", required=True)
    init_parser.add_argument("--label", default="protected_performant_20260808")

    reset_parser = subparsers.add_parser("reset-candidate")
    reset_parser.add_argument("--kind", choices=tuple(KIND_DIRS), required=True)
    reset_parser.add_argument("--from-role", choices=("production", "best"), default="production")

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--kind", choices=tuple(KIND_DIRS), required=True)
    promote_parser.add_argument("--evaluation", required=True)
    promote_parser.add_argument("--sync-legacy", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--kind", choices=tuple(KIND_DIRS), required=True)

    args = parser.parse_args()
    if args.command == "init":
        result = init_roles(args)
    elif args.command == "reset-candidate":
        result = reset_candidate(args)
    elif args.command == "promote":
        result = promote(args)
    else:
        result = status(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
