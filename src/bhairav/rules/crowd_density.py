"""Crowd-density rule: too many people inside a monitored zone.

Orange at `min_people`; escalates to Red at 2x when `escalate` is enabled.
"""
from __future__ import annotations

from ..geometry import point_in_polygon
from ..types import Alert, FrameState, Severity, Zone
from .base import Rule


def count_people_in_zone(state: FrameState, zone: Zone) -> int:
    """Number of person tracks whose centroid is inside the zone (public helper)."""
    poly = zone.to_pixels(state.frame_w, state.frame_h)
    return sum(1 for tr in state.tracks
               if tr.is_person and point_in_polygon(*tr.centroid, poly))


class CrowdDensityRule(Rule):
    name = "crowd_density"

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_people = int(config.get("min_people", 4))
        self.severity = Severity(config.get("severity", "orange"))
        self.escalate = bool(config.get("escalate", True))

    def evaluate(self, state: FrameState, zones: list[Zone]) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        for zone in zones:
            if zone.kind != "monitored" or not self._zone_selected(zone):
                continue
            count = count_people_in_zone(state, zone)
            if count >= self.min_people:
                if self.escalate and count >= self.min_people * 2:
                    severity = Severity.RED
                else:
                    severity = self.severity
                alerts.append(Alert(
                    rule=self.name,
                    zone=zone.name,
                    track_id=None,
                    severity=severity,
                    message=f"Crowd of {count} in zone '{zone.name}' (threshold {self.min_people})",
                    frame_id=state.frame_id,
                    timestamp=state.timestamp,
                    details={"people": count},
                ))
        return alerts
