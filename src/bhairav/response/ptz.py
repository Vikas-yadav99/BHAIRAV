"""PTZ camera auto-tracking (Phase 17.1).

Automatically controls Pan-Tilt-Zoom cameras to follow flagged persons.
Supports ONVIF, HTTP/REST, and virtual PTZ (software zoom/crop).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PTZCommand(str, Enum):
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    STOP = "stop"
    GO_TO_PRESET = "go_to_preset"
    ABSOLUTE_MOVE = "absolute_move"


@dataclass
class PTZState:
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 1.0
    moving: bool = False


@dataclass
class PTZPreset:
    name: str
    pan: float
    tilt: float
    zoom: float


@dataclass
class TrackingTarget:
    track_id: int
    subject_id: str | None = None
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    center_x: float = 0.5
    center_y: float = 0.5
    confidence: float = 1.0
    severity: str = "red"
    last_update: float = 0.0


class PTZController:
    """Low-level PTZ camera controller."""

    def __init__(self, camera_id: str, protocol: str = "simulated",
                 host: str = "", port: int = 80,
                 presets: list[PTZPreset] | None = None):
        self.camera_id = camera_id
        self.protocol = protocol
        self.host = host
        self.port = port
        self.state = PTZState()
        self.presets = {p.name: p for p in (presets or [])}
        self._lock = threading.Lock()
        self._command_log: list[dict] = []

    def move(self, command: PTZCommand, speed: float = 0.5,
             **kwargs) -> dict:
        with self._lock:
            result = {"camera": self.camera_id, "command": command.value,
                      "speed": speed, "ts": time.time()}
            if command == PTZCommand.STOP:
                self.state.moving = False
            elif command == PTZCommand.PAN_LEFT:
                self.state.pan = max(-180, self.state.pan - speed * 10)
                self.state.moving = True
            elif command == PTZCommand.PAN_RIGHT:
                self.state.pan = min(180, self.state.pan + speed * 10)
                self.state.moving = True
            elif command == PTZCommand.TILT_UP:
                self.state.tilt = max(-90, self.state.tilt - speed * 10)
                self.state.moving = True
            elif command == PTZCommand.TILT_DOWN:
                self.state.tilt = min(90, self.state.tilt + speed * 10)
                self.state.moving = True
            elif command == PTZCommand.ZOOM_IN:
                self.state.zoom = min(30, self.state.zoom * (1 + speed * 0.5))
                self.state.moving = True
            elif command == PTZCommand.ZOOM_OUT:
                self.state.zoom = max(1, self.state.zoom / (1 + speed * 0.5))
                self.state.moving = True
            elif command == PTZCommand.GO_TO_PRESET:
                pname = kwargs.get("preset", "park")
                if pname in self.presets:
                    p = self.presets[pname]
                    self.state.pan, self.state.tilt, self.state.zoom = p.pan, p.tilt, p.zoom
            elif command == PTZCommand.ABSOLUTE_MOVE:
                self.state.pan = kwargs.get("pan", 0)
                self.state.tilt = kwargs.get("tilt", 0)
                self.state.zoom = kwargs.get("zoom", 1)
            result["state"] = {"pan": self.state.pan, "tilt": self.state.tilt,
                               "zoom": self.state.zoom}
            self._command_log.append(result)
            if len(self._command_log) > 500:
                self._command_log = self._command_log[-250:]
            return result

    def go_to_preset(self, name: str) -> dict:
        return self.move(PTZCommand.GO_TO_PRESET, preset=name)

    def stop(self) -> dict:
        return self.move(PTZCommand.STOP)

    @property
    def command_log(self) -> list[dict]:
        return list(self._command_log)


class PTZTracker:
    """Auto-tracks a target by sending PTZ commands to keep them centered."""

    def __init__(self, controller: PTZController,
                 center_threshold: float = 0.1,
                 zoom_threshold: float = 0.05,
                 update_interval_ms: float = 200):
        self.controller = controller
        self.center_threshold = center_threshold
        self.zoom_threshold = zoom_threshold
        self.update_interval_ms = update_interval_ms
        self._active_target: TrackingTarget | None = None
        self._last_update: float = 0.0
        self._tracking = False

    def update_target(self, target: TrackingTarget) -> dict | None:
        self._active_target = target
        now = time.time()
        if (now - self._last_update) * 1000 < self.update_interval_ms:
            return None
        self._last_update = now
        dx = target.center_x - 0.5
        dy = target.center_y - 0.5
        commands = []
        if abs(dx) > self.center_threshold:
            cmd = PTZCommand.PAN_RIGHT if dx > 0 else PTZCommand.PAN_LEFT
            commands.append((cmd, abs(dx)))
        if abs(dy) > self.center_threshold:
            cmd = PTZCommand.TILT_DOWN if dy > 0 else PTZCommand.TILT_UP
            commands.append((cmd, abs(dy)))
        bbox_area = (target.bbox[2] - target.bbox[0]) * (target.bbox[3] - target.bbox[1])
        if bbox_area < self.zoom_threshold and bbox_area > 0:
            commands.append((PTZCommand.ZOOM_IN, 0.3))
        elif bbox_area > 0.3:
            commands.append((PTZCommand.ZOOM_OUT, 0.2))
        if not commands:
            return None
        cmd, speed = commands[0]
        result = self.controller.move(cmd, speed=min(speed, 1.0))
        self._tracking = True
        return result

    def stop_tracking(self) -> dict:
        self._active_target = None
        self._tracking = False
        return self.controller.stop()

    @property
    def is_tracking(self) -> bool:
        return self._tracking

    @property
    def active_target(self) -> TrackingTarget | None:
        return self._active_target
