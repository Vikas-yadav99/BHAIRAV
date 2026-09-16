"""Camera Manager — runtime RTSP camera management for BHAIRAV.

Provides:
- Add/remove RTSP cameras at runtime via API
- Get live snapshots (JPEG) from any camera
- List all active cameras with status
- Health monitoring per camera

Usage:
    manager = CameraManager()
    manager.add_camera("CAM-01", "rtsp://192.168.1.100:554/stream1")
    snapshot = manager.get_snapshot("CAM-01")  # returns JPEG bytes
"""
from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger("bhairav.camera_manager")


@dataclass
class CameraState:
    """Runtime state for a single camera."""
    id: str
    name: str
    rtsp_url: str
    status: str = "disconnected"  # connected | disconnected | error
    last_frame_time: float = 0.0
    frames_processed: int = 0
    errors: int = 0
    last_error: str = ""
    fps: float = 0.0
    _fps_counter: int = 0
    _fps_window_start: float = 0.0


class CameraManager:
    """Manages multiple RTSP camera connections at runtime.

    Thread-safe. Each camera runs in its own reader thread.
    """

    def __init__(self):
        self._cameras: dict[str, CameraState] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._snapshots: dict[str, bytes] = {}  # Latest JPEG per camera
        self._running = False
        self._lock = threading.Lock()

    def add_camera(self, camera_id: str, rtsp_url: str, name: str = "") -> dict:
        """Register and start processing an RTSP camera."""
        with self._lock:
            if camera_id in self._cameras:
                return {"ok": False, "error": f"Camera {camera_id} already exists"}

            state = CameraState(
                id=camera_id,
                name=name or camera_id,
                rtsp_url=rtsp_url,
            )
            self._cameras[camera_id] = state

        # Start reader thread
        t = threading.Thread(
            target=self._reader_loop,
            args=(camera_id,),
            daemon=True,
            name=f"cam-reader-{camera_id}",
        )
        self._threads[camera_id] = t
        t.start()

        log.info("Camera added: %s -> %s", camera_id, rtsp_url)
        return {"ok": True, "camera_id": camera_id, "status": "connecting"}

    def remove_camera(self, camera_id: str) -> dict:
        """Stop and remove a camera."""
        with self._lock:
            if camera_id not in self._cameras:
                return {"ok": False, "error": f"Camera {camera_id} not found"}
            del self._cameras[camera_id]
            self._snapshots.pop(camera_id, None)

        log.info("Camera removed: %s", camera_id)
        return {"ok": True}

    def list_cameras(self) -> list[dict]:
        """List all cameras with their current status."""
        with self._lock:
            return [
                {
                    "id": c.id,
                    "name": c.name,
                    "rtsp_url": c.rtsp_url,
                    "status": c.status,
                    "last_frame_time": c.last_frame_time,
                    "frames_processed": c.frames_processed,
                    "errors": c.errors,
                    "fps": round(c.fps, 1),
                }
                for c in self._cameras.values()
            ]

    def get_snapshot(self, camera_id: str) -> bytes | None:
        """Get the latest JPEG snapshot from a camera. Returns None if unavailable."""
        return self._snapshots.get(camera_id)

    def get_camera(self, camera_id: str) -> dict | None:
        """Get status of a single camera."""
        with self._lock:
            c = self._cameras.get(camera_id)
            if c is None:
                return None
            return {
                "id": c.id, "name": c.name, "rtsp_url": c.rtsp_url,
                "status": c.status, "last_frame_time": c.last_frame_time,
                "frames_processed": c.frames_processed, "errors": c.errors,
                "last_error": c.last_error, "fps": round(c.fps, 1),
            }

    def _reader_loop(self, camera_id: str):
        """Background thread: reads frames from RTSP, stores latest snapshot."""
        try:
            import cv2
        except ImportError:
            log.error("cv2 not available — cannot read RTSP for %s", camera_id)
            return

        state = self._cameras.get(camera_id)
        if state is None:
            return

        state._fps_window_start = time.time()
        backoff = 1.0

        while camera_id in self._cameras:
            try:
                cap = cv2.VideoCapture(state.rtsp_url)
                if not cap.isOpened():
                    state.status = "error"
                    state.last_error = "Cannot open RTSP stream"
                    time.sleep(min(backoff, 30))
                    backoff = min(backoff * 2, 30)
                    continue

                state.status = "connected"
                backoff = 1.0
                log.info("Connected to camera %s: %s", camera_id, state.rtsp_url)

                while camera_id in self._cameras:
                    ret, frame = cap.read()
                    if not ret:
                        state.errors += 1
                        state.last_error = "Frame read failed"
                        break

                    # Encode as JPEG
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self._snapshots[camera_id] = jpeg.tobytes()

                    state.frames_processed += 1
                    state.last_frame_time = time.time()

                    # FPS calculation
                    state._fps_counter += 1
                    elapsed = time.time() - state._fps_window_start
                    if elapsed >= 5.0:
                        state.fps = state._fps_counter / elapsed
                        state._fps_counter = 0
                        state._fps_window_start = time.time()

                cap.release()

            except Exception as exc:
                state.status = "error"
                state.last_error = str(exc)[:200]
                state.errors += 1
                log.warning("Camera %s error: %s", camera_id, exc)
                time.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)

        state.status = "disconnected"
        log.info("Camera %s reader stopped", camera_id)


# Singleton for server-wide use
_default_manager: CameraManager | None = None


def get_camera_manager() -> CameraManager:
    """Get or create the global camera manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CameraManager()
    return _default_manager
