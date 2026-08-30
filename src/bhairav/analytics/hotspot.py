"""Predictive crime hotspot modeling.

Spatial-temporal analysis that learns where and when incidents
are most likely, using:
  - Kernel Density Estimation over alert locations
  - Time-of-day periodicity
  - Zone-weighted risk scoring
  - Exponential decay for recency

Outputs ranked hotspot zones with confidence scores for
proactive resource allocation.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class HotspotZone:
    """A ranked hotspot prediction."""
    zone: str
    risk_score: float        # 0-1
    alert_count: int
    peak_hour: int           # 0-23
    trend: str               # rising / falling / stable
    predicted_next_hour: float  # expected alerts next hour

    def to_dict(self) -> dict:
        return {
            "zone": self.zone,
            "risk_score": round(self.risk_score, 3),
            "alert_count": self.alert_count,
            "peak_hour": self.peak_hour,
            "trend": self.trend,
            "predicted_next_hour": round(self.predicted_next_hour, 2),
        }


class PredictiveHotspot:
    """Spatial-temporal hotspot predictor.

    Parameters
    ----------
    window_sec : float
        History retention (default 3600 = 1 hour).
    decay_sec : float
        Exponential decay half-life for recency weighting (default 600).
    grid_cells : int
        Spatial grid resolution per axis (default 8).
    min_alerts : int
        Minimum alerts to consider a zone a hotspot (default 2).
    """

    def __init__(self, window_sec: float = 3600.0,
                 decay_sec: float = 600.0, grid_cells: int = 8,
                 min_alerts: int = 2):
        self.window_sec = window_sec
        self.decay_sec = decay_sec
        self.grid_cells = grid_cells
        self.min_alerts = min_alerts
        self._events: list[dict] = []
        self._hour_counts: defaultdict[int, int] = defaultdict(int)
        self._zone_alerts: defaultdict[str, list[float]] = defaultdict(list)

    def observe(self, timestamp: float, zone: str | None = None,
                severity: str = "yellow", rule: str = "",
                x: float | None = None, y: float | None = None) -> None:
        """Record an alert event for hotspot learning."""
        self._events.append({
            "ts": timestamp,
            "zone": zone or "(unknown)",
            "severity": severity,
            "rule": rule,
            "x": x,
            "y": y,
        })
        hour = int((timestamp % 86400) // 3600)
        self._hour_counts[hour] += 1
        self._zone_alerts[zone or "(unknown)"].append(timestamp)
        self._prune()

    def _prune(self) -> None:
        cutoff = time.time() - self.window_sec
        self._events = [e for e in self._events if e["ts"] >= cutoff]
        for z in list(self._zone_alerts):
            self._zone_alerts[z] = [
                t for t in self._zone_alerts[z] if t >= cutoff
            ]

    def _weighted_count(self, zone: str) -> float:
        """Recency-weighted alert count for a zone."""
        now = time.time()
        total = 0.0
        for ts in self._zone_alerts.get(zone, []):
            age = now - ts
            weight = math.exp(-0.693 * age / self.decay_sec)
            total += weight
        return total

    def _trend(self, zone: str) -> str:
        """Compute trend direction for a zone."""
        timestamps = self._zone_alerts.get(zone, [])
        if len(timestamps) < 4:
            return "stable"
        now = time.time()
        recent = sum(1 for t in timestamps if t >= now - 300)
        older = sum(1 for t in timestamps if now - 600 <= t < now - 300)
        if recent > older * 1.3:
            return "rising"
        if recent < older * 0.7:
            return "falling"
        return "stable"

    def _predict_next_hour(self, zone: str) -> float:
        """Predict expected alerts in the next hour for a zone."""
        timestamps = self._zone_alerts.get(zone, [])
        if not timestamps:
            return 0.0
        now = time.time()
        # average rate per minute over last 30 min, extrapolate to 60 min
        recent = [t for t in timestamps if t >= now - 1800]
        if not recent:
            return 0.0
        rate = len(recent) / 30.0  # per minute
        # adjust for trend
        trend = self._trend(zone)
        multiplier = {"rising": 1.4, "falling": 0.6, "stable": 1.0}
        return rate * 60.0 * multiplier.get(trend, 1.0)

    def _peak_hour(self, zone: str) -> int:
        """Most common hour-of-day for alerts in this zone."""
        hour_counts: dict[int, int] = defaultdict(int)
        for ts in self._zone_alerts.get(zone, []):
            hour = int((ts % 86400) // 3600)
            hour_counts[hour] += 1
        if not hour_counts:
            return 0
        return max(hour_counts, key=hour_counts.get)

    def rank_hotspots(self, top_n: int = 10) -> list[HotspotZone]:
        """Rank all zones by risk score."""
        scores: list[HotspotZone] = []
        for zone in self._zone_alerts:
            wc = self._weighted_count(zone)
            raw = len(self._zone_alerts[zone])
            if raw < self.min_alerts:
                continue
            # normalize score 0-1 (sigmoid)
            risk = 1.0 / (1.0 + math.exp(-(wc - 3.0) / 2.0))
            scores.append(HotspotZone(
                zone=zone,
                risk_score=risk,
                alert_count=raw,
                peak_hour=self._peak_hour(zone),
                trend=self._trend(zone),
                predicted_next_hour=self._predict_next_hour(zone),
            ))
        scores.sort(key=lambda h: h.risk_score, reverse=True)
        return scores[:top_n]

    def snapshot(self) -> dict:
        hotspots = self.rank_hotspots()
        return {
            "window_sec": self.window_sec,
            "total_events": len(self._events),
            "zones_tracked": len(self._zone_alerts),
            "hotspots": [h.to_dict() for h in hotspots],
        }

    def reset(self) -> None:
        self._events.clear()
        self._hour_counts.clear()
        self._zone_alerts.clear()
