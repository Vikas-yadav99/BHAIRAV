"""Per-track motion history: smoothed velocity, speed, heading, wobble.

Used by the Phase 2 behavior rules (fall / fight / chase). All values are
computed in pixel space from centroid positions, so they work with any
detector (blob or YOLO).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class MotionBuffer:
    """Rolling (timestamp, x, y) samples per track_id.

    Velocity is the mean displacement over the most recent `window_sec` of
    history (min 2 samples), which smooths per-frame jitter.
    """

    def __init__(self, window_sec: float = 1.0, max_age_sec: float = 3.0):
        self.window_sec = window_sec
        self.max_age_sec = max_age_sec
        self._buf: dict[int, deque] = {}

    def push(self, track_id: int, t: float, x: float, y: float) -> None:
        dq = self._buf.setdefault(track_id, deque(maxlen=256))
        dq.append((t, x, y))
        cutoff = t - self.window_sec
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _samples(self, track_id: int, now: float) -> list[tuple] | None:
        dq = self._buf.get(track_id)
        if not dq:
            return None
        cutoff = now - self.window_sec
        samples = [(t, x, y) for (t, x, y) in dq if t >= cutoff]
        if len(samples) < 2:
            return None
        return samples

    def velocity(self, track_id: int, now: float) -> tuple[float, float] | None:
        samples = self._samples(track_id, now)
        if samples is None:
            return None
        (t0, x0, y0), (t1, x1, y1) = samples[0], samples[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return None
        return (x1 - x0) / dt, (y1 - y0) / dt

    def speed(self, track_id: int, now: float) -> float | None:
        v = self.velocity(track_id, now)
        return None if v is None else float(np.hypot(*v))

    def step_velocities(self, track_id: int, now: float) -> list[tuple[float, float]]:
        """Per-sample velocities between consecutive samples in the window.

        Unlike `velocity` (net displacement over the window), this preserves
        short spikes and oscillation - e.g. a 1 Hz jostle has ~zero net
        displacement but high per-step speed.
        """
        samples = self._samples(track_id, now)
        if not samples or len(samples) < 2:
            return []
        out: list[tuple[float, float]] = []
        for (t0, x0, y0), (t1, x1, y1) in zip(samples, samples[1:]):
            dt = t1 - t0
            if dt <= 1e-6:
                continue
            out.append(((x1 - x0) / dt, (y1 - y0) / dt))
        return out

    def mean_speed(self, track_id: int, now: float) -> float:
        """Mean of per-step speeds - robust to oscillatory motion."""
        vs = self.step_velocities(track_id, now)
        if not vs:
            return 0.0
        return float(np.mean([np.hypot(vx, vy) for vx, vy in vs]))

    def peak_downward_vy(self, track_id: int, now: float) -> float:
        """Largest per-step downward (image +y) velocity in the window."""
        vs = self.step_velocities(track_id, now)
        return max((vy for _, vy in vs), default=0.0)

    def heading_deg(self, track_id: int, now: float) -> float | None:
        v = self.velocity(track_id, now)
        if v is None:
            return None
        return float(np.degrees(np.arctan2(v[1], v[0])))

    def wobble_deg(self, track_id: int, now: float) -> float:
        """Std dev of heading over the window - high for erratic motion
        (fighting), near zero for smooth walking/running."""
        samples = self._samples(track_id, now)
        if samples is None or len(samples) < 3:
            return 0.0
        angles: list[float] = []
        for (t0, x0, y0), (t1, x1, y1) in zip(samples, samples[1:]):
            dt = t1 - t0
            if dt <= 1e-6:
                continue
            angles.append(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        if len(angles) < 2:
            return 0.0
        return float(np.std(np.unwrap(np.radians(angles))) / np.pi * 180.0)

    def prune(self, active_ids: set[int], now: float) -> None:
        stale = [
            tid for tid, dq in self._buf.items()
            if tid not in active_ids or (dq and now - dq[-1][0] > self.max_age_sec)
        ]
        for tid in stale:
            del self._buf[tid]
