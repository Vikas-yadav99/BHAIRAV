"""Face tracking, trajectory prediction, and IMM filtering.

Modules:
- _KalmanFilter2D: Base Kalman filter for 2D position + velocity
- _IMMFilter2D: Interacting Multiple Model (CV + CA + Stopped)
- CameraCalibration: Homography-based pixel-to-world mapping
- FaceInferencePool: Async ThreadPoolExecutor for YuNet/SFace
- TrajectoryPredictor: Multi-camera trajectory tracking with IMM
- LiveFaceMonitor: Real-time face detection + recognition
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence scoring via chi-squared (NIS-based)
# ---------------------------------------------------------------------------

def _chi2_confidence(nis: float, dof: int = 2) -> float:
    """Convert Normalized Innovation Squared to a confidence score [0, 1].

    NIS ~ chi2(dof). When NIS equals dof (expected), confidence = 0.5.
    Low NIS means excellent model fit (high confidence); high NIS means
    the measurement is an outlier (low confidence).
    """
    if nis < 0:
        return 0.5
    return math.exp(-nis * 0.5)


# ---------------------------------------------------------------------------
# Kalman Filter (base)
# ---------------------------------------------------------------------------

class _KalmanFilter2D:
    """Kalman filter for 2D position + velocity.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 5.0):
        self.state = np.zeros(4, dtype=np.float64)  # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float64) * 100.0  # covariance
        self.Q = np.eye(4, dtype=np.float64) * process_noise  # process noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise  # meas noise
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.initialized = False
        self.nis = 0.0

    def init_state(self, x: float, y: float):
        self.state[:] = [x, y, 0.0, 0.0]
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.initialized = True

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def predict(self, dt: float) -> np.ndarray:
        F = self._F(dt)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        return self.state[:2].copy()

    def update(self, x: float, y: float):
        z = np.array([x, y], dtype=np.float64)
        y_innov = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        # NIS
        try:
            S_inv = np.linalg.inv(S)
            self.nis = float(y_innov @ S_inv @ y_innov)
        except np.linalg.LinAlgError:
            self.nis = 0.0
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_innov
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    @property
    def position_uncertainty(self) -> float:
        return float(self.P[0, 0] + self.P[1, 1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])


class _ConstantAccelerationFilter(_KalmanFilter2D):
    """Kalman filter with constant acceleration model.

    State: [x, y, vx, vy, ax, ay]
    """

    def __init__(self, process_noise: float = 5.0, measurement_noise: float = 5.0):
        self.state = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.initialized = False
        self.nis = 0.0

    def init_state(self, x: float, y: float):
        self.state[:] = [x, y, 0.0, 0.0, 0.0, 0.0]
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.initialized = True

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt * dt
        F[1, 5] = 0.5 * dt * dt
        return F

    def predict(self, dt: float) -> np.ndarray:
        F = self._F(dt)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        return self.state[:2].copy()

    def update(self, x: float, y: float):
        z = np.array([x, y], dtype=np.float64)
        y_innov = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
            self.nis = float(y_innov @ S_inv @ y_innov)
        except np.linalg.LinAlgError:
            self.nis = 0.0
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_innov
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    @property
    def position_uncertainty(self) -> float:
        return float(self.P[0, 0] + self.P[1, 1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])


class _StoppedFilter(_KalmanFilter2D):
    """Kalman filter with very low process noise — assumes target is stationary."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 5.0):
        super().__init__(process_noise=process_noise, measurement_noise=measurement_noise)


# ---------------------------------------------------------------------------
# IMM Filter (Interacting Multiple Model)
# ---------------------------------------------------------------------------

class _IMMFilter2D:
    """Interacting Multiple Model filter running 3 sub-filters in parallel:
    - Constant Velocity (CV): for walking targets
    - Constant Acceleration (CA): for speeding up/slowing down
    - Stopped (S): for stationary targets

    Automatically blends model probabilities based on how well each
    model fits the observations.
    """

    def __init__(self):
        self.cv = _KalmanFilter2D(process_noise=1.0, measurement_noise=5.0)
        self.ca = _ConstantAccelerationFilter(process_noise=5.0, measurement_noise=5.0)
        self.stopped = _StoppedFilter(process_noise=0.01, measurement_noise=5.0)
        # Model probabilities (uniform prior)
        self.probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float64)
        self.filters = [self.cv, self.ca, self.stopped]
        self.model_names = ["constant_velocity", "constant_acceleration", "stopped"]
        self.initialized = False

    def init_state(self, x: float, y: float):
        for f in self.filters:
            f.init_state(x, y)
        self.probs[:] = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        self.initialized = True

    @property
    def active_model(self) -> str:
        idx = int(np.argmax(self.probs))
        return self.model_names[idx]

    @property
    def position_uncertainty(self) -> float:
        # Weighted average of uncertainties
        return float(sum(
            p * f.position_uncertainty for p, f in zip(self.probs, self.filters)
        ))

    @property
    def velocity(self) -> tuple[float, float]:
        # Weighted average
        vx = sum(p * f.velocity[0] for p, f in zip(self.probs, self.filters))
        vy = sum(p * f.velocity[1] for p, f in zip(self.probs, self.filters))
        return float(vx), float(vy)

    def predict(self, dt: float) -> np.ndarray:
        predictions = []
        for f in self.filters:
            predictions.append(f.predict(dt))
        # Weighted average
        result = np.zeros(2, dtype=np.float64)
        for p, pred in zip(self.probs, predictions):
            result += p * pred
        return result

    def update(self, x: float, y: float):
        if not self.initialized:
            self.init_state(x, y)
            return

        # Compute likelihoods
        likelihoods = np.zeros(3, dtype=np.float64)
        for i, f in enumerate(self.filters):
            f.update(x, y)
            # Likelihood from NIS
            nis = f.nis
            likelihoods[i] = max(1e-30, math.exp(-0.5 * nis))

        # Update model probabilities
        combined = self.probs * likelihoods
        total = combined.sum()
        if total > 0:
            self.probs = combined / total
        else:
            self.probs[:] = [1.0 / 3, 1.0 / 3, 1.0 / 3]


# ---------------------------------------------------------------------------
# Camera Calibration (homography-based)
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """Maps between pixel coordinates and world coordinates via homography.

    Uses 4+ corresponding point pairs (pixel -> world) to compute a 3x3
    homography matrix that transforms normalized (0..1) coordinates to
    real-world positions in meters.
    """
    camera_id: str = ""
    # 3x3 homography matrix (pixel -> world)
    H: Optional[np.ndarray] = field(default=None, repr=False)
    # 3x3 inverse (world -> pixel)
    H_inv: Optional[np.ndarray] = field(default=None, repr=False)
    # FOV in degrees (estimated from calibration)
    fov_degrees: float = 90.0
    # Calibration quality (0-1, based on reprojection error)
    quality: float = 0.0
    # Point pairs used for calibration
    _src_pts: list = field(default_factory=list, repr=False)
    _dst_pts: list = field(default_factory=list, repr=False)

    def set_from_correspondences(
        self, pixel_points: list[tuple[float, float]], world_points: list[tuple[float, float]]
    ) -> bool:
        """Compute homography from >= 4 point pairs.

        Args:
            pixel_points: List of (x, y) in pixel/normalized coordinates
            world_points: List of (x, y) in world coordinates (meters)

        Returns:
            True if calibration succeeded, False otherwise.
        """
        if len(pixel_points) < 4 or len(world_points) < 4:
            log.warning("Need >= 4 point pairs for calibration, got %d", len(pixel_points))
            return False
        if len(pixel_points) != len(world_points):
            log.warning("Point count mismatch: %d pixel vs %d world", len(pixel_points), len(world_points))
            return False

        src = np.array(pixel_points, dtype=np.float64)
        dst = np.array(world_points, dtype=np.float64)

        try:
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        except Exception:
            log.exception("Homography computation failed")
            return False

        if H is None:
            log.warning("Homography computation returned None")
            return False

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self._src_pts = pixel_points
        self._dst_pts = world_points

        # Compute reprojection error for quality metric
        errors = []
        for sp, wp in zip(pixel_points, world_points):
            projected = self.pixel_to_world(sp[0], sp[1])
            if projected:
                err = math.sqrt((projected[0] - wp[0]) ** 2 + (projected[1] - wp[1]) ** 2)
                errors.append(err)
        self.quality = max(0.0, 1.0 - (np.mean(errors) if errors else 1.0))

        log.info("Camera %s calibrated: quality=%.2f", self.camera_id, self.quality)
        return True

    def pixel_to_world(self, px: float, py: float) -> Optional[tuple[float, float]]:
        """Convert pixel coordinates to world coordinates (meters)."""
        if self.H is None:
            return None
        pt = np.array([px, py, 1.0], dtype=np.float64)
        world = self.H @ pt
        if abs(world[2]) < 1e-10:
            return None
        return float(world[0] / world[2]), float(world[1] / world[2])

    def world_to_pixel(self, wx: float, wy: float) -> Optional[tuple[float, float]]:
        """Convert world coordinates to pixel coordinates."""
        if self.H_inv is None:
            return None
        pt = np.array([wx, wy, 1.0], dtype=np.float64)
        pixel = self.H_inv @ pt
        if abs(pixel[2]) < 1e-10:
            return None
        return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])

    def fov_contains(self, wx: float, wy: float, range_m: float = 50.0) -> bool:
        """Check if a world point is within the camera's FOV."""
        if self.H_inv is None:
            return True  # uncalibrated = assume visible
        px, py = self.world_to_pixel(wx, wy)
        if px is None:
            return False
        # Rough FOV check: if pixel is within frame bounds (assuming 1920x1080)
        return 0 <= px <= 1920 and 0 <= py <= 1080

    def world_distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Distance between two world points in meters."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ---------------------------------------------------------------------------
# Face Inference Pool (async YuNet/SFace)
# ---------------------------------------------------------------------------

@dataclass
class _InferenceRequest:
    frame: np.ndarray
    camera_id: str
    future: object  # concurrent.futures.Future


class FaceInferencePool:
    """Dedicated thread pool for YuNet/SFace face inference.

    Submits work asynchronously so the main pipeline thread is never
    blocked by face detection. Drops frames when the queue is full
    instead of blocking.
    """

    def __init__(self, max_workers: int = 2, max_queue: int = 10):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="face-infer")
        self._max_queue = max_queue
        self._queue_size = 0
        self._lock = threading.Lock()
        self._total_submitted = 0
        self._total_dropped = 0
        self._detector = None
        self._recognizer = None

    def set_models(self, detector, recognizer):
        """Set the YuNet detector and SFace recognizer (must be set before use)."""
        self._detector = detector
        self._recognizer = recognizer

    def submit(self, frame: np.ndarray, camera_id: str = "") -> Optional[object]:
        """Submit a frame for face inference. Returns a Future, or None if dropped."""
        with self._lock:
            if self._queue_size >= self._max_queue:
                self._total_dropped += 1
                return None
            self._queue_size += 1
            self._total_submitted += 1

        future = self._pool.submit(self._run_inference, frame, camera_id)
        future.add_done_callback(lambda _: self._decrement_queue())
        return future

    def _decrement_queue(self):
        with self._lock:
            self._queue_size = max(0, self._queue_size - 1)

    def _run_inference(self, frame: np.ndarray, camera_id: str):
        """Run YuNet + SFace on a frame."""
        if self._detector is None:
            return {"faces": [], "camera_id": camera_id}
        try:
            h, w = frame.shape[:2]
            self._detector.setInputSize((w, h))
            faces = self._detector.detect(frame)
            results = []
            if faces[1] is not None:
                for i, face in enumerate(faces[1]):
                    embedding = None
                    if self._recognizer is not None:
                        # Crop face region
                        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                        x, y = max(0, x), max(0, y)
                        face_crop = frame[y:y + fh, x:x + fw]
                        if face_crop.size > 0:
                            embedding = self._recognizer.alignCrop(frame, face)
                    results.append({
                        "bbox": [float(face[0]), float(face[1]), float(face[2]), float(face[3])],
                        "confidence": float(face[-1]) if len(face) > 4 else 0.5,
                        "embedding": embedding,
                    })
            return {"faces": results, "camera_id": camera_id}
        except Exception as e:
            log.debug("Face inference error: %s", e)
            return {"faces": [], "camera_id": camera_id}

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "queue_size": self._queue_size,
                "total_submitted": self._total_submitted,
                "total_dropped": self._total_dropped,
            }


    def list_tracked_persons(self) -> list[dict]:
        """Return list of all tracked person IDs and their stats."""
        with self._lock:
            result = []
            for key, traj in self._trajectories.items():
                result.append({
                    "person_id": traj.person_id,
                    "camera_id": traj.camera_id,
                    "observations": len(traj.positions),
                    "last_seen": traj.last_seen,
                    "model": traj.imm.active_model if traj.imm.initialized else None,
                })
            return result

    def get_recent_positions(self, person_id: str, seconds: float = 5.0) -> list[dict]:
        """Get recent positions for a person within the last N seconds."""
        with self._lock:
            cutoff = time.time() - seconds
            positions = []
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id:
                    for x, y, ts in traj.positions:
                        if ts >= cutoff:
                            positions.append({"x": x, "y": y, "timestamp": ts, "camera_id": traj.camera_id})
            return positions

    def get_trajectory(self, person_id: str, camera_id: str | None = None) -> dict | None:
        """Get full trajectory for a person. If camera_id is None, returns first match."""
        with self._lock:
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id:
                    if camera_id is None or traj.camera_id == camera_id:
                        return {
                            "person_id": person_id,
                            "camera_id": traj.camera_id,
                            "positions": traj.positions,
                            "world_positions": traj.world_positions,
                            "first_seen": traj.first_seen,
                            "last_seen": traj.last_seen,
                            "observations": len(traj.positions),
                        }
            return None

    def predict(self, person_id: str, camera_id: str = None, horizon_sec: float = 2.0, steps: int = 5, seconds_ahead: float = None) -> dict | None:
        """Predict future position. Accepts either camera_id or auto-detects."""
        if seconds_ahead is not None:
            horizon_sec = seconds_ahead
        with self._lock:
            if camera_id:
                key = f"{camera_id}:{person_id}"
                traj = self._trajectories.get(key)
                if traj and len(traj.positions) >= self.min_positions:
                    dt_per_step = horizon_sec / steps
                    predictions = []
                    for _ in range(steps):
                        pred = traj.imm.predict(dt_per_step)
                        predictions.append((float(pred[0]), float(pred[1])))
                    return {
                        "person_id": person_id, "camera_id": camera_id,
                        "predictions": predictions,
                        "active_model": traj.imm.active_model,
                        "confidence": _chi2_confidence(traj.imm.cv.nis),
                    }
            # Auto-detect camera
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id and len(traj.positions) >= self.min_positions:
                    cam = traj.camera_id
                    dt_per_step = horizon_sec / steps
                    predictions = []
                    for _ in range(steps):
                        pred = traj.imm.predict(dt_per_step)
                        predictions.append((float(pred[0]), float(pred[1])))
                    return {
                        "person_id": person_id, "camera_id": cam,
                        "predictions": predictions,
                        "active_model": traj.imm.active_model,
                        "confidence": _chi2_confidence(traj.imm.cv.nis),
                    }
            return None

    def shutdown(self):
        self._pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Trajectory data structure
# ---------------------------------------------------------------------------

@dataclass
class _PersonTrajectory:
    """Trajectory for a single person across frames."""
    person_id: str
    camera_id: str
    positions: list = field(default_factory=list)  # [(x, y, timestamp), ...]
    last_seen: float = 0.0
    imm: _IMMFilter2D = field(default_factory=_IMMFilter2D)
    first_seen: float = 0.0
    world_positions: list = field(default_factory=list)  # [(wx, wy, timestamp), ...]


# ---------------------------------------------------------------------------
# Trajectory Predictor
# ---------------------------------------------------------------------------

class TrajectoryPredictor:
    """Multi-camera trajectory predictor with IMM filtering.

    Tracks person positions across frames, maintains per-person IMM filters,
    supports cross-camera linking via world coordinates, and persists
    trajectories to JSONL.
    """

    def __init__(
        self,
        zones: list = None,
        persist_path: str | Path | None = None,
        min_positions_for_prediction: int = 5,
        enable_cross_camera_linking: bool = True,
        vanished_timeout_sec: float = 5.0,
    ):
        self.zones = zones or []
        self._persist_path = Path(persist_path) if persist_path else None
        self.min_positions = min_positions_for_prediction
        self.enable_cross_camera_linking = enable_cross_camera_linking
        self.vanished_timeout = vanished_timeout_sec
        self._trajectories: dict[str, _PersonTrajectory] = {}
        self._vanished: dict[str, _PersonTrajectory] = {}
        self._cameras: dict[str, CameraCalibration] = {}
        self._lock = threading.Lock()
        self._persist_file = None
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_file = open(self._persist_path, "a", encoding="utf-8")

    def register_camera(self, calibration: CameraCalibration):
        """Register a camera calibration for cross-camera linking."""
        with self._lock:
            self._cameras[calibration.camera_id] = calibration

    def update(
        self,
        person_id: str,
        x: float,
        y: float,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ):
        """Update a person's position from a detection."""
        with self._lock:
            key = f"{camera_id}:{person_id}"
            traj = self._trajectories.get(key)
            if traj is None:
                traj = _PersonTrajectory(
                    person_id=person_id, camera_id=camera_id,
                    first_seen=timestamp,
                )
                self._trajectories[key] = traj

            traj.positions.append((x, y, timestamp))
            traj.last_seen = timestamp

            # Update IMM filter
            if not traj.imm.initialized:
                traj.imm.init_state(x, y)
            else:
                dt = timestamp - (traj.positions[-2][2] if len(traj.positions) > 1 else timestamp)
                if dt > 0:
                    traj.imm.predict(dt)
                traj.imm.update(x, y)

            # Store world coordinates if calibrated
            cal = self._cameras.get(camera_id)
            if cal:
                world = cal.pixel_to_world(x, y)
                if world:
                    traj.world_positions.append((world[0], world[1], timestamp))

            # Persist
            if self._persist_file:
                record = {
                    "person_id": person_id, "camera_id": camera_id,
                    "x": x, "y": y, "frame_id": frame_id, "timestamp": timestamp,
                    "model": traj.imm.active_model,
                    "uncertainty": traj.imm.position_uncertainty,
                }
                if traj.world_positions:
                    wx, wy, _ = traj.world_positions[-1]
                    record["world_x"] = wx
                    record["world_y"] = wy
                try:
                    self._persist_file.write(json.dumps(record, default=str) + "\n")
                    self._persist_file.flush()
                except Exception:
                    pass

    def predict(
        self, person_id: str, camera_id: str, horizon_sec: float = 2.0, steps: int = 5
    ) -> Optional[dict]:
        """Predict future position for a person.

        Returns None if insufficient data for prediction.
        """
        with self._lock:
            key = f"{camera_id}:{person_id}"
            traj = self._trajectories.get(key)
            if traj is None:
                return None
            if len(traj.positions) < self.min_positions:
                return None

            # Predict future
            dt_per_step = horizon_sec / steps
            predictions = []
            for _ in range(steps):
                pred = traj.imm.predict(dt_per_step)
                predictions.append((float(pred[0]), float(pred[1])))

            # World coordinates for predictions
            world_preds = []
            cal = self._cameras.get(camera_id)
            if cal:
                for px, py in predictions:
                    w = cal.pixel_to_world(px, py)
                    if w:
                        world_preds.append(w)

            vx, vy = traj.imm.velocity
            speed = math.sqrt(vx ** 2 + vy ** 2)

            result = {
                "person_id": person_id,
                "camera_id": camera_id,
                "current_position": traj.positions[-1][:2],
                "predictions": predictions,
                "active_model": traj.imm.active_model,
                "model_probabilities": dict(zip(
                    ["constant_velocity", "constant_acceleration", "stopped"],
                    [float(p) for p in traj.imm.probs]
                )),
                "confidence": _chi2_confidence(traj.imm.cv.nis),
                "speed": speed,
                "uncertainty": traj.imm.position_uncertainty,
                "observations": len(traj.positions),
            }
            if world_preds:
                result["world_predictions"] = world_preds
                # Last known world position
                if traj.world_positions:
                    wx, wy, _ = traj.world_positions[-1]
                    result["world_x"] = wx
                    result["world_y"] = wy

            return result

    def predict_multi(
        self, person_id: str, horizon_sec=None, steps: int = 5
    ) -> list[dict]:
        """Predict across all cameras for a person.

        Args:
            person_id: Person to predict for
            horizon_sec: Single horizon (float) or list of horizons (list[float])
            steps: Number of prediction steps per horizon

        Only returns predictions for cameras where the person has been
        observed with enough data (>= min_positions).
        """
        # Normalize horizons
        if horizon_sec is None:
            horizons = [2.0]
        elif isinstance(horizon_sec, (int, float)):
            horizons = [float(horizon_sec)]
        else:
            horizons = [float(h) for h in horizon_sec]

        results = []
        with self._lock:
            keys = [k for k in self._trajectories if k.endswith(f":{person_id}")]
            for key in keys:
                cam_id = key.split(":")[0]
                traj = self._trajectories.get(key)
                if traj is None or len(traj.positions) < self.min_positions:
                    continue
                # Predict for each horizon
                for h in horizons:
                    pred = self._predict_at_horizon(traj, person_id, cam_id, h, steps)
                    if pred is not None:
                        results.append(pred)
        return results

    def _predict_at_horizon(self, traj, person_id, camera_id, horizon_sec, steps):
        """Internal: predict at a specific horizon for a trajectory."""
        dt_per_step = horizon_sec / steps
        predictions = []
        for _ in range(steps):
            pred = traj.imm.predict(dt_per_step)
            predictions.append((float(pred[0]), float(pred[1])))
        world_preds = []
        cal = self._cameras.get(camera_id)
        if cal:
            for px, py in predictions:
                w = cal.pixel_to_world(px, py)
                if w:
                    world_preds.append(w)
        vx, vy = traj.imm.velocity
        result = {
            "person_id": person_id,
            "camera_id": camera_id,
            "horizon_sec": horizon_sec,
            "predictions": predictions,
            "active_model": traj.imm.active_model,
            "model_probabilities": dict(zip(
                ["constant_velocity", "constant_acceleration", "stopped"],
                [float(p) for p in traj.imm.probs]
            )),
            "confidence": _chi2_confidence(traj.imm.cv.nis),
            "speed": math.sqrt(vx ** 2 + vy ** 2),
            "uncertainty": traj.imm.position_uncertainty,
            "observations": len(traj.positions),
        }
        if world_preds:
            result["world_predictions"] = world_preds
        return result

    def get_trajectory(self, person_id: str, camera_id: str) -> Optional[dict]:
        """Get full trajectory for a person."""
        with self._lock:
            key = f"{camera_id}:{person_id}"
            traj = self._trajectories.get(key)
            if traj is None:
                return None
            return {
                "person_id": person_id,
                "camera_id": camera_id,
                "positions": traj.positions,
                "world_positions": traj.world_positions,
                "first_seen": traj.first_seen,
                "last_seen": traj.last_seen,
                "observations": len(traj.positions),
            }

    def stats(self) -> dict:
        """Return predictor statistics."""
        with self._lock:
            tracked = list(self._trajectories.values())
            now = time.time()
            return {
                "tracked_persons": len(tracked),
                "active_persons": sum(
                    1 for t in tracked if now - t.last_seen < 30
                ),
                "total_positions": sum(
                    len(t.positions) for t in tracked
                ),
                "vanished_pool": len(self._vanished),
                "registered_cameras": len(self._cameras),
                "avg_position_uncertainty": (
                    round(float(np.mean([
                        t.imm.position_uncertainty for t in tracked
                    ])), 6) if tracked else 0.0
                ),
                "persist_path": (
                    str(self._persist_path) if self._persist_path else None
                ),
            }


    def list_tracked_persons(self) -> list[dict]:
        """Return list of all tracked person IDs and their stats."""
        with self._lock:
            result = []
            for key, traj in self._trajectories.items():
                result.append({
                    "person_id": traj.person_id,
                    "camera_id": traj.camera_id,
                    "observations": len(traj.positions),
                    "last_seen": traj.last_seen,
                    "model": traj.imm.active_model if traj.imm.initialized else None,
                })
            return result

    def get_recent_positions(self, person_id: str, seconds: float = 5.0) -> list[dict]:
        """Get recent positions for a person within the last N seconds."""
        with self._lock:
            cutoff = time.time() - seconds
            positions = []
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id:
                    for x, y, ts in traj.positions:
                        if ts >= cutoff:
                            positions.append({"x": x, "y": y, "timestamp": ts, "camera_id": traj.camera_id})
            return positions

    def get_trajectory(self, person_id: str, camera_id: str | None = None) -> dict | None:
        """Get full trajectory for a person. If camera_id is None, returns first match."""
        with self._lock:
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id:
                    if camera_id is None or traj.camera_id == camera_id:
                        return {
                            "person_id": person_id,
                            "camera_id": traj.camera_id,
                            "positions": traj.positions,
                            "world_positions": traj.world_positions,
                            "first_seen": traj.first_seen,
                            "last_seen": traj.last_seen,
                            "observations": len(traj.positions),
                        }
            return None

    def predict(self, person_id: str, camera_id: str = None, horizon_sec: float = 2.0, steps: int = 5, seconds_ahead: float = None) -> dict | None:
        """Predict future position. Accepts either camera_id or auto-detects."""
        if seconds_ahead is not None:
            horizon_sec = seconds_ahead
        with self._lock:
            if camera_id:
                key = f"{camera_id}:{person_id}"
                traj = self._trajectories.get(key)
                if traj and len(traj.positions) >= self.min_positions:
                    dt_per_step = horizon_sec / steps
                    predictions = []
                    for _ in range(steps):
                        pred = traj.imm.predict(dt_per_step)
                        predictions.append((float(pred[0]), float(pred[1])))
                    return {
                        "person_id": person_id, "camera_id": camera_id,
                        "predictions": predictions,
                        "active_model": traj.imm.active_model,
                        "confidence": _chi2_confidence(traj.imm.cv.nis),
                    }
            # Auto-detect camera
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id and len(traj.positions) >= self.min_positions:
                    cam = traj.camera_id
                    dt_per_step = horizon_sec / steps
                    predictions = []
                    for _ in range(steps):
                        pred = traj.imm.predict(dt_per_step)
                        predictions.append((float(pred[0]), float(pred[1])))
                    return {
                        "person_id": person_id, "camera_id": cam,
                        "predictions": predictions,
                        "active_model": traj.imm.active_model,
                        "confidence": _chi2_confidence(traj.imm.cv.nis),
                    }
            return None

    def shutdown(self):
        """Flush and close persistence file."""
        with self._lock:
            if self._persist_file:
                try:
                    self._persist_file.flush()
                    self._persist_file.close()
                except Exception:
                    pass
                self._persist_file = None


# ---------------------------------------------------------------------------
# Live Face Monitor Config
# ---------------------------------------------------------------------------

@dataclass
class LiveFaceMonitorConfig:
    """Configuration for LiveFaceMonitor."""
    enabled: bool = False
    detector_model: str = "yunet"
    recognizer_model: str = "sface"
    confidence_threshold: float = 0.5
    max_queue: int = 10
    max_workers: int = 2
    gallery_path: str | None = None


# ---------------------------------------------------------------------------
# Live Face Monitor
# ---------------------------------------------------------------------------

class LiveFaceMonitor:
    """Real-time face detection + recognition using YuNet/SFace.

    Uses FaceInferencePool for non-blocking inference.
    """

    def __init__(self, config: LiveFaceMonitorConfig | None = None):
        self.config = config or LiveFaceMonitorConfig()
        self._pool = FaceInferencePool(
            max_workers=self.config.max_workers,
            max_queue=self.config.max_queue,
        )
        self._lock = threading.Lock()
        self._detections_count = 0
        self._running = False

    def start(self):
        """Start the face monitor (load models)."""
        if not self.config.enabled:
            log.info("Face monitor disabled")
            return
        try:
            import cv2
            # Try to load YuNet
            model_path = Path("models/face_detection_yunet.onnx")
            if model_path.exists():
                detector = cv2.FaceDetectorYN.create(
                    str(model_path), "", (320, 320)
                )
                # Try to load SFace
                sface_path = Path("models/face_recognition_sface.onnx")
                recognizer = None
                if sface_path.exists():
                    recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
                self._pool.set_models(detector, recognizer)
                self._running = True
                log.info("Face monitor started (YuNet + %s)",
                         "SFace" if recognizer else "no recognizer")
            else:
                log.warning("YuNet model not found at %s", model_path)
        except Exception as e:
            log.warning("Failed to start face monitor: %s", e)

    def process_frame(self, frame: np.ndarray, camera_id: str = "") -> Optional[object]:
        """Submit a frame for face detection (non-blocking).

        Returns a Future that resolves to face detection results,
        or None if the frame was dropped (queue full).
        """
        if not self._running:
            return None
        return self._pool.submit(frame, camera_id)


    def list_tracked_persons(self) -> list[dict]:
        """Return list of all tracked person IDs and their stats."""
        with self._lock:
            result = []
            for key, traj in self._trajectories.items():
                result.append({
                    "person_id": traj.person_id,
                    "camera_id": traj.camera_id,
                    "observations": len(traj.positions),
                    "last_seen": traj.last_seen,
                    "model": traj.imm.active_model if traj.imm.initialized else None,
                })
            return result

    def get_recent_positions(self, person_id: str, seconds: float = 5.0) -> list[dict]:
        """Get recent positions for a person within the last N seconds."""
        with self._lock:
            cutoff = time.time() - seconds
            positions = []
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id:
                    for x, y, ts in traj.positions:
                        if ts >= cutoff:
                            positions.append({"x": x, "y": y, "timestamp": ts, "camera_id": traj.camera_id})
            return positions

    def get_trajectory(self, person_id: str, camera_id: str | None = None) -> dict | None:
        """Get full trajectory for a person. If camera_id is None, returns first match."""
        with self._lock:
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id:
                    if camera_id is None or traj.camera_id == camera_id:
                        return {
                            "person_id": person_id,
                            "camera_id": traj.camera_id,
                            "positions": traj.positions,
                            "world_positions": traj.world_positions,
                            "first_seen": traj.first_seen,
                            "last_seen": traj.last_seen,
                            "observations": len(traj.positions),
                        }
            return None

    def predict(self, person_id: str, camera_id: str = None, horizon_sec: float = 2.0, steps: int = 5, seconds_ahead: float = None) -> dict | None:
        """Predict future position. Accepts either camera_id or auto-detects."""
        if seconds_ahead is not None:
            horizon_sec = seconds_ahead
        with self._lock:
            if camera_id:
                key = f"{camera_id}:{person_id}"
                traj = self._trajectories.get(key)
                if traj and len(traj.positions) >= self.min_positions:
                    dt_per_step = horizon_sec / steps
                    predictions = []
                    for _ in range(steps):
                        pred = traj.imm.predict(dt_per_step)
                        predictions.append((float(pred[0]), float(pred[1])))
                    return {
                        "person_id": person_id, "camera_id": camera_id,
                        "predictions": predictions,
                        "active_model": traj.imm.active_model,
                        "confidence": _chi2_confidence(traj.imm.cv.nis),
                    }
            # Auto-detect camera
            for key, traj in self._trajectories.items():
                if traj.person_id == person_id and len(traj.positions) >= self.min_positions:
                    cam = traj.camera_id
                    dt_per_step = horizon_sec / steps
                    predictions = []
                    for _ in range(steps):
                        pred = traj.imm.predict(dt_per_step)
                        predictions.append((float(pred[0]), float(pred[1])))
                    return {
                        "person_id": person_id, "camera_id": cam,
                        "predictions": predictions,
                        "active_model": traj.imm.active_model,
                        "confidence": _chi2_confidence(traj.imm.cv.nis),
                    }
            return None

    def shutdown(self):
        """Shutdown the face monitor."""
        self._running = False
        self._pool.shutdown()

    @property
    def status(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "pool_stats": self._pool.stats,
        }
