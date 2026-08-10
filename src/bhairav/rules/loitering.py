"""Loitering rule: a person continuously inside a monitored zone for too long.

Yellow at `duration_sec`; escalates to Orange at 2x when `escalate` is enabled.
"""
from __future__ import annotations

from ..geometry import point_in_polygon
from ..types import Alert, FrameState, Severity, Zone
from .base import Rule


class LoiteringRule(Rule):
    name = "loitering"

    def __init__(self, config: dict):
        super().__init__(config)
        self.duration_sec = float(config.get("duration_sec", 8.0))
        self.escalate = bool(config.get("escalate", True))
        self._state: dict[tuple[str, int], dict] = {}  # (zone, track_id) -> state

    def evaluate(self, state: FrameState, zones: list[Zone]) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        for zone in zones:
            if zone.kind != "monitored" or not self._zone_selected(zone):
                continue
            poly = zone.to_pixels(state.frame_w, state.frame_h)
            for tr in state.tracks:
                if not tr.is_person:
                    continue
                cx, cy = tr.centroid
                key = (zone.name, tr.track_id)
                st = self._state.setdefault(key, {"entered": None, "left_at": None, "last_seen": 0.0})
                st["last_seen"] = state.timestamp
                if point_in_polygon(cx, cy, poly):
                    if st["entered"] is None:
                        st["entered"] = state.timestamp
                    st["left_at"] = None
                    duration = state.timestamp - st["entered"]
                    if duration >= self.duration_sec:
                        if self.escalate and duration >= self.duration_sec * 2.0:
                            severity = Severity.ORANGE
                        else:
                            severity = Severity.YELLOW
                        alerts.append(Alert(
                            rule=self.name,
                            zone=zone.name,
                            track_id=tr.track_id,
                            severity=severity,
                            message=f"Loitering {duration:.1f}s in zone '{zone.name}' - track #{tr.track_id}",
                            frame_id=state.frame_id,
                            timestamp=state.timestamp,
                            details={"duration_sec": round(duration, 2)},
                        ))
                else:
                    if st["entered"] is not None:
                        if st["left_at"] is None:
                            st["left_at"] = state.timestamp
                        elif state.timestamp - st["left_at"] > 1.0:
                            # Left for good (grace period of 1s) - reset the clock.
                            st["entered"] = None
                            st["left_at"] = None
        if len(self._state) > 256:
            # Prune state for tracks not seen in a while (long-running feeds).
            cutoff = state.timestamp - 120.0
            self._state = {k: v for k, v in self._state.items() if v["last_seen"] >= cutoff}
        return alerts
