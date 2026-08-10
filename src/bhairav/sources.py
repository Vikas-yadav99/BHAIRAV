"""Camera / video source layer.

Turns any supported input - synthetic scene, video file, webcam index, or a
network stream (RTSP/RTMP/HTTP) - into a uniform `(kind, monitor, opener)`
bundle so the pipeline can reconnect live cameras with backoff and report
connect/drop health through /api/status.

RTSP notes
----------
* Transport is forced to TCP (more reliable than UDP over lossy links) and
  FFmpeg buffering is minimized for lower latency.
* `cv2.VideoCapture` on an unreachable URL can block for a long time, so
  network opens use `CAP_PROP_OPEN_TIMEOUT_MSEC` and retry with backoff.
"""
from __future__ import annotations

import os
import time
from enum import Enum

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
RTSP_OPTS = "rtsp_transport=tcp;fflags=nobuffer;flags=low_delay"


class SourceKind(str, Enum):
    BLOB = "blob"          # synthetic scene
    FILE = "file"          # local video file
    CAMERA = "camera"      # webcam / capture card index
    NETWORK = "network"    # rtsp:// rtmp:// http(s):// live stream


def classify_source(source: str | int | None) -> tuple[SourceKind, str]:
    """Return (kind, human description) for a source spec."""
    if source is None or str(source) == "blob":
        return SourceKind.BLOB, "synthetic scene"
    s = str(source).strip()
    if s.isdigit():
        return SourceKind.CAMERA, f"camera index {s}"
    low = s.lower()
    if low.startswith(("rtsp://", "rtmp://", "rtspx://", "http://", "https://")):
        # live streams and remote files both arrive over the network
        return SourceKind.NETWORK, s
    if os.path.splitext(s)[1].lower() in VIDEO_EXTS:
        return SourceKind.FILE, s
    return SourceKind.FILE, s


class SourceMonitor:
    """Thread-safe connect/drop health, surfaced in /api/status."""

    def __init__(self, kind: SourceKind, description: str):
        self.kind = kind
        self.description = description
        self._lock = __import__("threading").Lock()
        self.attempts = 0
        self.connects = 0
        self.drops = 0
        self.last_error: str | None = None
        self.last_connect_ts: float | None = None

    def connecting(self) -> None:
        with self._lock:
            self.attempts += 1

    def connected(self) -> None:
        with self._lock:
            self.connects += 1
            self.last_connect_ts = time.time()

    def dropped(self, error: str) -> None:
        with self._lock:
            self.drops += 1
            self.last_error = error

    def snapshot(self) -> dict:
        with self._lock:
            if self.kind is SourceKind.BLOB:
                # synthetic scene is always available; nothing to connect
                connected = True
            else:
                connected = (self.last_connect_ts is not None
                             and (time.time() - self.last_connect_ts) < 60)
            return {
                "kind": self.kind.value,
                "description": self.description,
                "attempts": self.attempts,
                "connects": self.connects,
                "drops": self.drops,
                "last_error": self.last_error,
                "connected": connected,
            }


def open_capture(source: str | int, monitor: SourceMonitor | None = None,
                 retries: int = 5, base_delay: float = 1.0,
                 open_timeout_ms: int = 10_000) -> cv2.VideoCapture:
    """Open a capture for `source`, retrying network sources with backoff.

    Raises RuntimeError if it cannot open after `retries` attempts.
    """
    kind, _ = classify_source(source)
    if kind is SourceKind.NETWORK:
        # low latency + TCP transport for live streams (read per-open by OpenCV)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_OPTS

    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        if monitor is not None:
            monitor.connecting()
        src: str | int = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(src)
        if kind is SourceKind.NETWORK:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, open_timeout_ms)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            if monitor is not None:
                monitor.connected()
            return cap
        cap.release()
        last_exc = RuntimeError(f"cannot open video source: {source}")
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"cannot open video source: {source}") from last_exc
