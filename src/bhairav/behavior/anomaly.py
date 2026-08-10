"""Anomaly detection (Phase 2, amber-flag layer).

A lightweight "learned normal" model: during a warmup window we collect the
people-count distribution per monitored zone, then flag frames whose count
deviates by more than `z_thresh` standard deviations. Pure numpy baseline -
when torch lands, this is the seam where an autoencoder variant drops in.

Yellow severity. Confidence scales with the z-score.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..types import Alert, FrameState, Severity, Zone


class AnomalyRule:
    name = "anomaly"
    enabled = True

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.severity = Severity(config.get("severity", "yellow"))
        self.z_thresh = float(config.get("z_thresh", 3.0))
        self.min_count = int(config.get("min_count", 2))
        self.warmup_frames = int(config.get("warmup_frames", 60))
        self.zone_names = config.get("zones")
        self._window: dict[str, deque] = {}
        self._baseline: dict[str, tuple[float, float]] = {}

    def evaluate(self, state: FrameState, zones: list[Zone]) -> list[Alert]:
        if not self.enabled:
            return []
        # Deferred import: avoids a module-load cycle (rules <-> behavior).
        from ..rules.crowd_density import count_people_in_zone
        alerts: list[Alert] = []
        for zone in zones:
            if zone.kind != "monitored":
                continue
            if self.zone_names is not None and zone.name not in self.zone_names:
                continue
            count = count_people_in_zone(state, zone)
            dq = self._window.setdefault(zone.name, deque(maxlen=self.warmup_frames))
            dq.append(count)
            if zone.name not in self._baseline:
                if len(dq) >= self.warmup_frames:
                    arr = np.array(dq, dtype=float)
                    self._baseline[zone.name] = (float(arr.mean()), float(arr.std()))
                continue
            mean, std = self._baseline[zone.name]
            z = (count - mean) / max(std, 0.1)
            if z >= self.z_thresh and count >= self.min_count:
                conf = min(0.9, 0.45 + z / 10.0)
                alerts.append(Alert(
                    rule=self.name, zone=zone.name, track_id=None,
                    severity=self.severity,
                    message=f"ANOMALY in '{zone.name}': {count} people (baseline {mean:.1f}+/-{std:.1f})",
                    frame_id=state.frame_id, timestamp=state.timestamp,
                    details={"people": count, "baseline_mean": round(mean, 2),
                             "baseline_std": round(std, 2), "z": round(z, 2),
                             "confidence": round(conf, 3)},
                    confidence=conf,
                ))
        return alerts
