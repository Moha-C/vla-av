"""UDP bridge for SUMO-driven red-team attacks in the live CARLA demo."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.data.augmentations import AttackConfig, RedTeamAttacks


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveAttack:
    """One currently active SUMO attack command."""

    config: AttackConfig
    received_at: float


class SUMORedTeamAttackServer:
    """Listen for SUMO UDP messages and apply attacks to RGB frames."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 5005,
        ttl_seconds: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.ttl_seconds = float(ttl_seconds)
        self.attacks = RedTeamAttacks(seed=seed)
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._active_attack: Optional[ActiveAttack] = None

    def start(self) -> None:
        """Start the background UDP listener."""

        if self._thread is not None and self._thread.is_alive():
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.2)
        self._socket = sock
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        LOGGER.info("Listening for SUMO red-team attacks on UDP %s:%s", self.host, self.port)

    def stop(self) -> None:
        """Stop the UDP listener and clear any active attack."""

        self._stop_event.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            self._active_attack = None

    def get_active_attack(self) -> Optional[AttackConfig]:
        """Return the current attack unless its TTL has expired."""

        with self._lock:
            active = self._active_attack
            if active is None:
                return None
            if self.ttl_seconds > 0.0 and time.monotonic() - active.received_at > self.ttl_seconds:
                self._active_attack = None
                return None
            return active.config

    def get_status_text(self) -> Optional[str]:
        """Return overlay text for the current active attack."""

        config = self.get_active_attack()
        if config is None:
            return None
        return f"ATTACK ACTIVE: {config.attack_type} ({config.intensity:.2f})"

    def apply_to_image(
        self,
        image: np.ndarray,
        *,
        model: Optional[Any] = None,
        instruction: str = "Drive safely and follow the lane.",
        target_action: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Apply the active attack to the frame, or return a copy unchanged."""

        config = self.get_active_attack()
        if config is None:
            return image.copy()
        return self.attacks.apply(
            image,
            config,
            model=model,
            instruction=instruction,
            target_action=target_action,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sock = self._socket
            if sock is None:
                return

            try:
                payload, address = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return

            try:
                self._handle_payload(payload)
            except Exception as exc:
                LOGGER.warning("Ignoring malformed SUMO attack message from %s: %s", address, exc)

    def _handle_payload(self, payload: bytes) -> None:
        message = json.loads(payload.decode("utf-8"))
        if not isinstance(message, dict):
            raise ValueError("Expected a JSON object.")

        attack_type = str(message.get("attack_type", "")).strip().lower()
        if attack_type in {"", "none", "clear", "off", "stop"}:
            with self._lock:
                self._active_attack = None
            LOGGER.info("Cleared SUMO red-team attack.")
            return

        intensity = float(message.get("intensity", 0.5))
        config = AttackConfig(attack_type=attack_type, intensity=intensity).normalized()
        with self._lock:
            self._active_attack = ActiveAttack(config=config, received_at=time.monotonic())
        LOGGER.info("Activated SUMO red-team attack: %s %.2f", config.attack_type, config.intensity)


def parse_attack_message(message: Dict[str, Any]) -> AttackConfig:
    """Validate one SUMO JSON attack command without starting a socket server."""

    return AttackConfig(
        attack_type=str(message["attack_type"]),
        intensity=float(message.get("intensity", 0.5)),
    ).normalized()
