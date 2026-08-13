"""Riot / public-order detection (Phase 10).

A large crowd alone is not a riot - the crowd must be *agitated*. This rule
looks for a dense cluster of people whose motion is simultaneously elevated
and erratic (milling, shoving, surging), sustained over a window.

Signals (all must hold for `duration_sec`):
  - at least `min_people` person tracks whose centroids fit within a
    `cluster_radius` of the cluster centroid
  - mean per-step speed above `speed_norm` (they are moving, not standing)
  - mean heading wobble above `wobble_deg` (erratic, not a marching column)

Red severity. Confidence scales with cluster size, agitation and duration.
A walking/marching group fails the wobble gate; a standing crowd fails the
speed gate - only genuinely agitated clusters fire.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..types import Alert, FrameState, Severity
from .kinematics import MotionBuffer


class RiotRule:
    name = "riot"
    enabled = True

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.severity = Severity(config.get("severity", "red"))
        self.min_people = int(config.get("min_people", 4))
        self.cluster_radius_norm = float(config.get("cluster_radius_norm", 0.06))  # x frame diagonal
        self.speed_norm = float(config.get("speed_norm", 0.045))  # x frame height / s
        self.wobble_deg = float(config.get("wobble_deg", 20.0))
        self.duration_sec = float(config.get("duration_sec", 4.5))
        self.reset_gap = float(config.get("reset_gap", 1.0))
        self.buf = MotionBuffer(window_sec=1.0)
        self._clusters: dict[frozenset, dict] = {}

    def evaluate(self, state: FrameState, zones: list) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        diag = float(np.hypot(state.frame_w, state.frame_h))
        h = float(state.frame_h)
        t = state.timestamp

        info: dict[int, dict] = {}
        persons: list = []
        for tr in state.tracks:
            if not tr.is_person:
                continue
            cx, cy = tr.centroid
            self.buf.push(tr.track_id, t, cx, cy)
            info[tr.track_id] = {
                "x": cx, "y": cy,
                "speed": self.buf.mean_speed(tr.track_id, t),
                "wobble": self.buf.wobble_deg(tr.track_id, t),
            }
            persons.append(tr)

        if len(persons) >= self.min_people:
            ids = [tr.track_id for tr in persons]
            members: set[int] = set()
            for subset in itertools.combinations(ids, self.min_people):
                pts = np.array([[info[i]["x"], info[i]["y"]] for i in subset])
                mean = pts.mean(axis=0)
                radii = np.hypot(pts[:, 0] - mean[0], pts[:, 1] - mean[1])
                if radii.max() <= self.cluster_radius_norm * diag:
                    speeds = [info[i]["speed"] for i in subset]
                    wobbles = [info[i]["wobble"] for i in subset]
                    if (np.mean(speeds) >= self.speed_norm * h
                            and np.mean(wobbles) >= self.wobble_deg):
                        members = set(subset)
                        break
            if members:
                key = frozenset(members)
                st = self._clusters.setdefault(key, {"since": t, "last_ok": t})
                if t - st["last_ok"] > self.reset_gap:
                    st["since"] = t
                st["last_ok"] = t
                dur = t - st["since"]
                if dur >= self.duration_sec:
                    speeds = [info[i]["speed"] for i in members]
                    wobbles = [info[i]["wobble"] for i in members]
                    mean_speed = float(np.mean(speeds))
                    mean_wobble = float(np.mean(wobbles))
                    conf = min(0.95, 0.5
                               + 0.2 * min(len(members) / (2 * self.min_people), 1.0)
                               + 0.15 * min(mean_speed / (2 * self.speed_norm * h), 1.0)
                               + 0.1 * min(mean_wobble / (2 * self.wobble_deg), 1.0)
                               + 0.1 * min(dur / (3 * self.duration_sec), 1.0))
                    alerts.append(Alert(
                        rule=self.name, zone=None, track_id=None,
                        severity=self.severity,
                        message=(f"RIOT: {len(members)}-person agitated cluster "
                                 f"(mean speed {mean_speed:.0f}px/s, wobble "
                                 f"{mean_wobble:.0f}deg, {dur:.1f}s)"),
                        frame_id=state.frame_id, timestamp=t,
                        details={"tracks": sorted(members),
                                 "people": len(members),
                                 "mean_speed_px_s": round(mean_speed, 1),
                                 "mean_wobble_deg": round(mean_wobble, 1),
                                 "duration_sec": round(dur, 2),
                                 "confidence": round(conf, 3)},
                        confidence=conf,
                    ))
        self.buf.prune({tr.track_id for tr in state.tracks}, t)
        cutoff = t - 120.0
        self._clusters = {k: v for k, v in self._clusters.items()
                          if v["last_ok"] >= cutoff}
        return alerts
