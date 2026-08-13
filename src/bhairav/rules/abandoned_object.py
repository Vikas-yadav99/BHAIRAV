"""Abandoned-object rule (Phase 10): an unattended bag/suitcase in a monitored zone.

Signals:
  - a baggage-class track (COCO suitcase/backpack/handbag) inside a monitored zone
  - the object has not moved for `abandon_sec`
  - no person is nearby (`owner_dist`) - the owner walked away

Orange at `abandon_sec`; escalates to Red at 2x when `escalate` is enabled
(mirrors the loitering rule's escalation shape).

The owner-distance check uses the *nearest* person track, so a person merely
walking past the object is enough to pause the clock - the object only counts
as abandoned once the area around it is clear.
"""
from __future__ import annotations

from ..behavior.kinematics import MotionBuffer
from ..geometry import point_in_polygon
from ..types import Alert, FrameState, Severity, Zone
from .base import Rule

# COCO class ids treated as portable/abandonable objects.
BAGGAGE_CLASS_IDS = frozenset({24, 26, 28})  # backpack, handbag, suitcase


class AbandonedObjectRule(Rule):
    name = "abandoned_object"

    def __init__(self, config: dict):
        super().__init__(config)
        self.classes = frozenset(int(c) for c in config.get("classes", [28]))
        self.abandon_sec = float(config.get("abandon_sec", 8.0))
        self.escalate = bool(config.get("escalate", True))
        self.severity = Severity(config.get("severity", "orange"))
        self.owner_dist_norm = float(config.get("owner_dist_norm", 0.06))   # x frame diagonal
        self.still_speed_norm = float(config.get("still_speed_norm", 0.02))  # x frame height / s
        self.buf = MotionBuffer(window_sec=1.0)
        self._state: dict[tuple[str, int], dict] = {}  # (zone, track_id) -> state

    def evaluate(self, state: FrameState, zones: list[Zone]) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        diag = float((state.frame_w ** 2 + state.frame_h ** 2) ** 0.5)
        h = float(state.frame_h)
        t = state.timestamp

        # nearest person per baggage track (owner distance)
        persons = [tr for tr in state.tracks if tr.is_person]

        for zone in zones:
            if zone.kind != "monitored" or not self._zone_selected(zone):
                continue
            poly = zone.to_pixels(state.frame_w, state.frame_h)
            for tr in state.tracks:
                if tr.class_id not in self.classes:
                    continue
                cx, cy = tr.centroid
                self.buf.push(tr.track_id, t, cx, cy)
                if not point_in_polygon(cx, cy, poly):
                    continue
                speed = self.buf.mean_speed(tr.track_id, t)
                if speed > self.still_speed_norm * h:
                    continue
                owner_dist = min(
                    (float((cx - p.centroid[0]) ** 2 + (cy - p.centroid[1]) ** 2) ** 0.5)
                    for p in persons) if persons else float("inf")
                if owner_dist <= self.owner_dist_norm * diag:
                    continue  # attended: someone is still close to the object
                key = (zone.name, tr.track_id)
                st = self._state.setdefault(key, {"since": None, "last_ok": 0.0})
                if st["since"] is None:
                    st["since"] = t
                st["last_ok"] = t
                duration = t - st["since"]
                if duration >= self.abandon_sec:
                    if self.escalate and duration >= self.abandon_sec * 2.0:
                        severity = Severity.RED
                    else:
                        severity = self.severity
                    conf = min(0.95, 0.5
                               + 0.25 * min(duration / (2 * self.abandon_sec), 1.0)
                               + 0.2 * min(owner_dist / (3 * self.owner_dist_norm * diag), 1.0))
                    alerts.append(Alert(
                        rule=self.name,
                        zone=zone.name,
                        track_id=tr.track_id,
                        severity=severity,
                        message=(f"Unattended {tr.label} #{tr.track_id} in zone "
                                 f"'{zone.name}' (still {duration:.1f}s, "
                                 f"nearest person {owner_dist:.0f}px away)"),
                        frame_id=state.frame_id,
                        timestamp=t,
                        details={"still_sec": round(duration, 2),
                                 "owner_dist_px": round(owner_dist, 1),
                                 "confidence": round(conf, 3)},
                        confidence=conf,
                    ))
        self.buf.prune({tr.track_id for tr in state.tracks}, t)
        # forget objects that left or were picked up long ago
        cutoff = t - 120.0
        self._state = {k: v for k, v in self._state.items() if v["last_ok"] >= cutoff}
        return alerts
