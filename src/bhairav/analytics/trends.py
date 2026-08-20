"""Alert trend analysis: temporal patterns, rule hotspots, severity distribution.

Tracks a rolling history of alerts and computes:
  - alert count per time bucket (1-min, 5-min, 15-min windows)
  - per-rule frequency
  - severity distribution
  - burst detection (rapid alert clusters)
  - hotspot zones (most active zones)
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass


@dataclass
class TrendPoint:
    timestamp: float
    rule: str
    severity: str
    zone: str | None
    camera: str | None = None


class TrendAnalyzer:
    """Rolling alert trend analyzer.

    Parameters
    ----------
    window_sec : float
        Maximum history retention in seconds (default 900 = 15 min).
    bucket_sec : float
        Time bucket granularity for counts (default 60 = 1 min).
    burst_window_sec : float
        Window to detect alert bursts (default 10).
    burst_threshold : int
        Alert count within burst_window to trigger burst (default 5).
    """

    def __init__(self, window_sec: float = 900.0, bucket_sec: float = 60.0,
                 burst_window_sec: float = 10.0, burst_threshold: int = 5):
        self.window_sec = window_sec
        self.bucket_sec = bucket_sec
        self.burst_window_sec = burst_window_sec
        self.burst_threshold = burst_threshold
        self._points: deque[TrendPoint] = deque()
        self._alert_count = 0

    def observe(self, timestamp: float, rule: str, severity: str,
                zone: str | None = None, camera: str | None = None) -> None:
        """Record a fired alert for trend analysis."""
        self._points.append(TrendPoint(timestamp=timestamp, rule=rule,
                                       severity=severity, zone=zone,
                                       camera=camera))
        self._alert_count += 1
        self._prune(timestamp)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._points and self._points[0].timestamp < cutoff:
            self._points.popleft()

    def _buckets(self, bucket_sec: float | None = None) -> dict[str, int]:
        """Count alerts in time buckets."""
        bs = bucket_sec or self.bucket_sec
        buckets: Counter = Counter()
        for p in self._points:
            bkey = str(int(p.timestamp // bs))
            buckets[bkey] += 1
        return dict(sorted(buckets.items()))

    def by_rule(self) -> dict[str, int]:
        """Alert count grouped by rule name."""
        return dict(Counter(p.rule for p in self._points))

    def by_severity(self) -> dict[str, int]:
        """Alert count grouped by severity level."""
        return dict(Counter(p.severity for p in self._points))

    def by_zone(self) -> dict[str, int]:
        """Alert count grouped by zone (includes None for unzoned)."""
        return dict(Counter((p.zone or "(none)") for p in self._points))

    def by_camera(self) -> dict[str, int]:
        """Alert count grouped by camera."""
        return dict(Counter((p.camera or "(none)") for p in self._points))

    def detect_bursts(self, now: float | None = None) -> list[dict]:
        """Detect rapid-fire alert clusters in the recent window."""
        if now is None:
            now = time.time()
        recent = [p for p in self._points
                  if p.timestamp >= now - self.burst_window_sec]
        if len(recent) < self.burst_threshold:
            return []
        # group by rule
        by_rule: dict[str, list] = {}
        for p in recent:
            by_rule.setdefault(p.rule, []).append(p)
        bursts = []
        for rule, pts in by_rule.items():
            if len(pts) >= self.burst_threshold:
                bursts.append({
                    "rule": rule,
                    "count": len(pts),
                    "window_sec": self.burst_window_sec,
                    "severity": max((p.severity for p in pts),
                                    key=lambda s: {"green": 0, "yellow": 1,
                                                   "orange": 2, "red": 3}.get(s, 0)),
                })
        return bursts

    def rate_per_min(self, window_sec: float = 300.0) -> float:
        """Alerts per minute over the last  seconds."""
        now = time.time()
        cutoff = now - window_sec
        recent = sum(1 for p in self._points if p.timestamp >= cutoff)
        mins = window_sec / 60.0
        return round(recent / max(mins, 1.0), 2)

    def snapshot(self) -> dict:
        """Full trend state for the analytics WebSocket push."""
        return {
            "total": self._alert_count,
            "active": len(self._points),
            "by_rule": self.by_rule(),
            "by_severity": self.by_severity(),
            "by_zone": self.by_zone(),
            "by_camera": self.by_camera(),
            "buckets_1m": self._buckets(self.bucket_sec),
            "rate_per_min": self.rate_per_min(),
            "bursts": self.detect_bursts(),
        }
