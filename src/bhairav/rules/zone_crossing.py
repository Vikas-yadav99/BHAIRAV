"""Zone-crossing rule: a person/vehicle inside a restricted zone."""
from __future__ import annotations

from ..geometry import point_in_polygon
from ..types import VEHICLE_CLASS_IDS, Alert, FrameState, Severity, Zone
from .base import Rule


class ZoneCrossingRule(Rule):
    name = "zone_crossing"

    def __init__(self, config: dict):
        super().__init__(config)
        self.severity = Severity(config.get("severity", "red"))
        self.include_vehicles = bool(config.get("include_vehicles", True))

    def evaluate(self, state: FrameState, zones: list[Zone]) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        for zone in zones:
            if zone.kind != "restricted" or not self._zone_selected(zone):
                continue
            poly = zone.to_pixels(state.frame_w, state.frame_h)
            for tr in state.tracks:
                is_target = tr.is_person or (self.include_vehicles and tr.class_id in VEHICLE_CLASS_IDS)
                if not is_target:
                    continue
                cx, cy = tr.centroid
                if point_in_polygon(cx, cy, poly):
                    alerts.append(Alert(
                        rule=self.name,
                        zone=zone.name,
                        track_id=tr.track_id,
                        severity=self.severity,
                        message=f"{tr.label} #{tr.track_id} inside restricted zone '{zone.name}'",
                        frame_id=state.frame_id,
                        timestamp=state.timestamp,
                    ))
        return alerts
