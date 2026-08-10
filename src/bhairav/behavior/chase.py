"""Chase / pursuit detection (Phase 2).

Signals:
  - a fast runner and a fast follower within a max distance
  - the follower's heading points at the runner (pursuit)
  - the runner is moving AWAY from the follower (flight)
  - sustained over a window

Orange on confirmation; escalates to Red at 2x the duration.
"""
from __future__ import annotations

import numpy as np

from ..types import Alert, FrameState, Severity
from .kinematics import MotionBuffer


class ChaseRule:
    name = "chase"
    enabled = True

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.severity = Severity(config.get("severity", "orange"))
        self.escalate = bool(config.get("escalate", True))
        self.runner_speed_norm = float(config.get("runner_speed_norm", 0.065))
        self.follower_speed_norm = float(config.get("follower_speed_norm", 0.08))
        self.heading_deg = float(config.get("heading_deg", 30.0))
        self.max_dist_norm = float(config.get("max_dist_norm", 0.30))
        self.duration_sec = float(config.get("duration_sec", 2.0))
        self.reset_gap = float(config.get("reset_gap", 1.0))
        self.buf = MotionBuffer(window_sec=1.0)
        self._pairs: dict[tuple[int, int], dict] = {}

    @staticmethod
    def _angle_deg(ax: float, ay: float, bx: float, by: float) -> float:
        """Angle between vectors a and b, in degrees [0, 180]."""
        na = np.hypot(ax, ay)
        nb = np.hypot(bx, by)
        if na < 1e-6 or nb < 1e-6:
            return 180.0
        c = (ax * bx + ay * by) / (na * nb)
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

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
            v = self.buf.velocity(tr.track_id, state.timestamp)
            info[tr.track_id] = {
                "x": cx, "y": cy,
                "vx": v[0] if v else 0.0,
                "vy": v[1] if v else 0.0,
                "speed": self.buf.speed(tr.track_id, state.timestamp) or 0.0,
            }
        persons = [tr for tr in state.tracks if tr.is_person]
        t = state.timestamp
        for runner in persons:
            ir = info[runner.track_id]
            if ir["speed"] < self.runner_speed_norm * h:
                continue
            for follower in persons:
                if follower.track_id == runner.track_id:
                    continue
                ifollower = info[follower.track_id]
                if ifollower["speed"] < self.follower_speed_norm * h:
                    continue
                dx = ir["x"] - ifollower["x"]
                dy = ir["y"] - ifollower["y"]
                dist = np.hypot(dx, dy)
                if dist > self.max_dist_norm * diag or dist < 5.0:
                    continue
                # follower heading must point at the runner
                if self._angle_deg(ifollower["vx"], ifollower["vy"], dx, dy) > self.heading_deg:
                    continue
                # runner must be moving away from the follower
                if self._angle_deg(ir["vx"], ir["vy"], dx, dy) > self.heading_deg:
                    continue
                key = (follower.track_id, runner.track_id)
                st = self._pairs.setdefault(key, {"since": t, "last_ok": t})
                if t - st["last_ok"] > self.reset_gap:
                    st["since"] = t
                st["last_ok"] = t
                dur = t - st["since"]
                if dur >= self.duration_sec:
                    sev = self.severity
                    if self.escalate and dur >= self.duration_sec * 2.0:
                        sev = Severity.RED
                    align = 1.0 - self._angle_deg(ifollower["vx"], ifollower["vy"], dx, dy) / 180.0
                    conf = min(0.95, 0.5
                               + 0.2 * align
                               + 0.15 * min(ir["speed"] / (2 * self.runner_speed_norm * h), 1.0)
                               + 0.1 * min(dur / (3 * self.duration_sec), 1.0))
                    alerts.append(Alert(
                        rule=self.name, zone=None, track_id=follower.track_id,
                        severity=sev,
                        message=f"CHASE: #{follower.track_id} pursuing #{runner.track_id} "
                                f"(gap {dist:.0f}px)",
                        frame_id=state.frame_id, timestamp=t,
                        details={"follower": follower.track_id, "runner": runner.track_id,
                                 "gap_px": round(dist, 1),
                                 "runner_speed_px_s": round(ir["speed"], 1),
                                 "confidence": round(conf, 3)},
                        confidence=conf,
                    ))
        self.buf.prune({tr.track_id for tr in state.tracks}, t)
        cutoff = t - 120.0
        self._pairs = {k: v for k, v in self._pairs.items() if v["last_ok"] >= cutoff}
        return alerts
