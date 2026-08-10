"""Trespassing detection (Phase 2).

Complement to zone_crossing: crossing fires immediately on entry; trespass
fires when a person REMAINS inside a restricted zone beyond a dwell window.

Orange at dwell time; escalates to Red at 2x dwell.
"""
from __future__ import annotations

from ..geometry import point_in_polygon
from ..types import Alert, FrameState, Severity, Zone


class TrespassRule:
    name = "trespass"
    enabled = True

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.severity = Severity(config.get("severity", "orange"))
        self.escalate = bool(config.get("escalate", True))
        self.dwell_sec = float(config.get("dwell_sec", 2.5))
        self.zone_names = config.get("zones")
        self._state: dict[tuple[str, int], dict] = {}

    def evaluate(self, state: FrameState, zones: list[Zone]) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        t = state.timestamp
        for zone in zones:
            if zone.kind != "restricted":
                continue
            if self.zone_names is not None and zone.name not in self.zone_names:
                continue
            poly = zone.to_pixels(state.frame_w, state.frame_h)
            for tr in state.tracks:
                if not tr.is_person:
                    continue
                cx, cy = tr.centroid
                key = (zone.name, tr.track_id)
                st = self._state.setdefault(key, {"entered": None, "left_at": None, "last_seen": 0.0})
                st["last_seen"] = t
                inside = point_in_polygon(cx, cy, poly)
                if inside:
                    if st["entered"] is None:
                        st["entered"] = t
                    st["left_at"] = None
                    dwell = t - st["entered"]
                    if dwell >= self.dwell_sec:
                        sev = self.severity
                        if self.escalate and dwell >= self.dwell_sec * 2.0:
                            sev = Severity.RED
                        conf = min(0.9, 0.5 + 0.15 * min(dwell / self.dwell_sec, 1.0)
                                   + 0.2 * min(dwell / (2 * self.dwell_sec), 1.0))
                        alerts.append(Alert(
                            rule=self.name, zone=zone.name, track_id=tr.track_id,
                            severity=sev,
                            message=f"TRESPASS: #{tr.track_id} in '{zone.name}' for {dwell:.1f}s",
                            frame_id=state.frame_id, timestamp=t,
                            details={"dwell_sec": round(dwell, 2), "confidence": round(conf, 3)},
                            confidence=conf,
                        ))
                else:
                    if st["entered"] is not None:
                        if st["left_at"] is None:
                            st["left_at"] = t
                        elif t - st["left_at"] > 1.0:
                            st["entered"] = None
                            st["left_at"] = None
        cutoff = t - 120.0
        self._state = {k: v for k, v in self._state.items() if v["last_seen"] >= cutoff}
        return alerts
