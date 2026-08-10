"""Fight detection (Phase 2).

Signals:
  - two person tracks in close proximity
  - both moving with high speed (relative to a normal walk)
  - erratic motion (high heading wobble) sustained over a window

Red severity. Confidence combines proximity, speed, wobble and duration.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..types import Alert, FrameState, Severity
from .kinematics import MotionBuffer


class FightRule:
    name = "fight"
    enabled = True

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.severity = Severity(config.get("severity", "red"))
        self.proximity_norm = float(config.get("proximity_norm", 0.10))   # x frame diagonal
        self.speed_norm = float(config.get("speed_norm", 0.08))           # x frame height / s
        self.min_speed_norm = float(config.get("min_speed_norm", 0.03))   # both parties must move
        self.wobble_deg = float(config.get("wobble_deg", 25.0))
        self.duration_sec = float(config.get("duration_sec", 1.5))
        self.reset_gap = float(config.get("reset_gap", 1.0))
        self.buf = MotionBuffer(window_sec=1.0)
        self._pairs: dict[tuple[int, int], dict] = {}

    def evaluate(self, state: FrameState, zones: list) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        diag = float(np.hypot(state.frame_w, state.frame_h))
        h = float(state.frame_h)
        info: dict[int, dict] = {}
        for tr in state.tracks:
            if not tr.is_person:
                continue
            cx, cy = tr.centroid
            self.buf.push(tr.track_id, state.timestamp, cx, cy)
            info[tr.track_id] = {
                "x": cx, "y": cy,
                # mean per-step speed (window-averaged velocity is ~0 for
                # oscillatory scuffling, so it would miss real fights)
                "speed": self.buf.mean_speed(tr.track_id, state.timestamp),
                "wobble": self.buf.wobble_deg(tr.track_id, state.timestamp),
            }
        persons = [tr for tr in state.tracks if tr.is_person]
        t = state.timestamp
        for a, b in itertools.combinations(persons, 2):
            ia, ib = info[a.track_id], info[b.track_id]
            dist = np.hypot(ia["x"] - ib["x"], ia["y"] - ib["y"])
            if dist > self.proximity_norm * diag:
                continue
            max_speed = max(ia["speed"], ib["speed"])
            if max_speed < self.speed_norm * h:
                continue
            # Both people must be genuinely moving: a stationary bystander
            # parked near the scuffle must not trigger the fight (jitter alone
            # can look like heading wobble).
            if min(ia["speed"], ib["speed"]) < self.min_speed_norm * h:
                continue
            min_wobble = min(ia["wobble"], ib["wobble"])
            if min_wobble < self.wobble_deg:
                continue
            key = tuple(sorted((a.track_id, b.track_id)))
            st = self._pairs.setdefault(key, {"since": t, "last_ok": t})
            if t - st["last_ok"] > self.reset_gap:
                st["since"] = t
            st["last_ok"] = t
            dur = t - st["since"]
            if dur >= self.duration_sec:
                conf = min(0.95, 0.5
                           + 0.15 * (1 - dist / (self.proximity_norm * diag))
                           + 0.15 * min(max_speed / (2 * self.speed_norm * h), 1.0)
                           + 0.15 * min(min_wobble / (2 * self.wobble_deg), 1.0)
                           + 0.05 * min(dur / (3 * self.duration_sec), 1.0))
                alerts.append(Alert(
                    rule=self.name, zone=None, track_id=key[0],
                    severity=self.severity,
                    message=f"FIGHT between #{key[0]} and #{key[1]} "
                            f"(sep {dist:.0f}px, speed {max_speed:.0f}px/s)",
                    frame_id=state.frame_id, timestamp=t,
                    details={"tracks": list(key), "separation_px": round(dist, 1),
                             "max_speed_px_s": round(max_speed, 1),
                             "confidence": round(conf, 3)},
                    confidence=conf,
                ))
        self.buf.prune({tr.track_id for tr in state.tracks}, t)
        cutoff = t - 120.0
        self._pairs = {k: v for k, v in self._pairs.items() if v["last_ok"] >= cutoff}
        return alerts
