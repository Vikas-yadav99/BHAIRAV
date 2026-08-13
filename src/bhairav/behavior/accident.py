"""Accident detection (Phase 10): a vehicle that stops hard in a driving area
with a downed/motionless person right next to it.

Signals:
  - a vehicle that was recently moving fast and is now (nearly) stopped
    ("hard stop" - e.g. a brake in the middle of the road)
  - a person track within `impact_dist` of the stopped vehicle
  - that person is DOWN (flat bbox / horizontal pose) - the victim

Red severity. Confidence combines the stop severity (peak speed), the
impact gap, and how long the situation has persisted.

Vehicles that were never moving (parked from the start) are ignored - only a
moving vehicle that suddenly stops next to a person counts as an accident.
"""
from __future__ import annotations

import numpy as np

from ..types import VEHICLE_CLASS_IDS, Alert, FrameState, Severity
from .kinematics import MotionBuffer


class AccidentRule:
    name = "accident"
    enabled = True

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.severity = Severity(config.get("severity", "red"))
        self.cruise_speed_norm = float(config.get("cruise_speed_norm", 0.06))  # was moving this fast
        self.still_speed_norm = float(config.get("still_speed_norm", 0.02))    # now (nearly) stopped
        self.impact_dist_norm = float(config.get("impact_dist_norm", 0.10))    # x frame diagonal
        self.confirm_sec = float(config.get("confirm_sec", 1.2))
        self.down_aspect = float(config.get("down_aspect", 1.0))   # h/w below this = lying down
        self.lookback_sec = float(config.get("lookback_sec", 4.0))  # peak-speed memory
        self.reset_gap = float(config.get("reset_gap", 1.0))
        self.buf = MotionBuffer(window_sec=1.0)
        self._vehicles: dict[int, dict] = {}   # track_id -> {peak, peak_t, stopped_since}
        self._pairs: dict[tuple[int, int], dict] = {}  # (vehicle, victim) -> state

    def evaluate(self, state: FrameState, zones: list) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        diag = float(np.hypot(state.frame_w, state.frame_h))
        h = float(state.frame_h)
        t = state.timestamp
        poses = {p.track_id: p for p in state.poses}

        info: dict[int, dict] = {}
        for tr in state.tracks:
            cx, cy = tr.centroid
            self.buf.push(tr.track_id, t, cx, cy)
            info[tr.track_id] = {
                "x": cx, "y": cy,
                "speed": self.buf.speed(tr.track_id, t) or 0.0,
            }

        for tr in state.tracks:
            if tr.class_id not in VEHICLE_CLASS_IDS:
                continue
            speed = info[tr.track_id]["speed"]
            st = self._vehicles.setdefault(tr.track_id,
                                           {"peak": 0.0, "peak_t": -1e9,
                                            "stopped_since": None, "last_ok": 0.0})
            st["last_ok"] = t
            moving_now = speed > self.still_speed_norm * h
            if speed > st["peak"]:
                st["peak"] = speed
                st["peak_t"] = t
            elif moving_now and t - st["peak_t"] > self.lookback_sec:
                # decay the "was moving" memory only while the vehicle is
                # still moving - a stopped vehicle keeps it, so a hard stop
                # followed by a slow confirmation (victim collapsing) still
                # counts as an accident minutes later, not just 4 s later.
                st["peak"] = speed
                st["peak_t"] = t
            # hard stop: was cruising recently, now stopped
            stopped = (speed <= self.still_speed_norm * h
                       and st["peak"] >= self.cruise_speed_norm * h)
            if stopped:
                if st["stopped_since"] is None:
                    st["stopped_since"] = t
            else:
                st["stopped_since"] = None
            if st["stopped_since"] is None:
                continue

            # find the nearest person that is down or motionless near the vehicle
            vx, vy = info[tr.track_id]["x"], info[tr.track_id]["y"]
            best: tuple[float, int] | None = None
            for p in state.tracks:
                if not p.is_person:
                    continue
                dx, dy = info[p.track_id]["x"] - vx, info[p.track_id]["y"] - vy
                dist = float(np.hypot(dx, dy))
                if dist > self.impact_dist_norm * diag:
                    continue
                w = p.bbox[2] - p.bbox[0]
                bh = p.bbox[3] - p.bbox[1]
                aspect = bh / w if w > 0 else 0.0
                pose = poses.get(p.track_id)
                pose_flat = bool(pose and (pose.horizontal_angle_deg() or 0) > 60)
                # the victim must be DOWN (lying), not merely standing near
                # the car - a pedestrian resting by a parked car is normal
                down = aspect < self.down_aspect or pose_flat
                if not down:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, p.track_id)
            if best is None:
                continue
            dist, victim = best
            key = (tr.track_id, victim)
            pk = self._pairs.setdefault(key, {"since": t, "last_ok": t})
            if t - pk["last_ok"] > self.reset_gap:
                pk["since"] = t
            pk["last_ok"] = t
            if t - pk["since"] >= self.confirm_sec:
                peak = st["peak"]
                conf = min(0.95, 0.5
                           + 0.2 * min(peak / (3 * self.cruise_speed_norm * h), 1.0)
                           + 0.2 * (1 - dist / (self.impact_dist_norm * diag))
                           + 0.1 * min((t - pk["since"]) / (3 * self.confirm_sec), 1.0))
                alerts.append(Alert(
                    rule=self.name, zone=None, track_id=tr.track_id,
                    severity=self.severity,
                    message=(f"ACCIDENT: vehicle #{tr.track_id} stopped hard "
                             f"next to person #{victim} (gap {dist:.0f}px, "
                             f"peak speed {peak:.0f}px/s)"),
                    frame_id=state.frame_id, timestamp=t,
                    details={"vehicle": tr.track_id, "victim": victim,
                             "gap_px": round(dist, 1),
                             "peak_speed_px_s": round(peak, 1),
                             "stopped_sec": round(t - st["stopped_since"], 2),
                             "confidence": round(conf, 3)},
                    confidence=conf,
                ))
        self.buf.prune({tr.track_id for tr in state.tracks}, t)
        cutoff = t - 120.0
        self._vehicles = {k: v for k, v in self._vehicles.items()
                          if v["last_ok"] >= cutoff}
        self._pairs = {k: v for k, v in self._pairs.items()
                       if v["last_ok"] >= cutoff}
        return alerts
