"""Analytics engine: coordinates forecast, heatmap, and trend modules."""
from __future__ import annotations

import time

from .forecast import CrowdDensityForecast
from .heatmap import SpatialHeatmap
from .trends import TrendAnalyzer


class AnalyticsEngine:
    """Unified analytics facade consumed by the server on_frame callback."""

    def __init__(self, forecast_horizon_sec: float = 10.0,
                 heatmap_grid: tuple[int, int] = (32, 24),
                 heatmap_decay_sec: float = 30.0,
                 trend_window_sec: float = 900.0):
        self.forecast = CrowdDensityForecast(horizon_sec=forecast_horizon_sec)
        self.heatmap = SpatialHeatmap(
            grid_w=heatmap_grid[0], grid_h=heatmap_grid[1],
            decay_sec=heatmap_decay_sec,
        )
        self.trends = TrendAnalyzer(window_sec=trend_window_sec)
        self._frame_count = 0

    def observe_frame(self, timestamp, person_count, tracks, alerts,
                      zone_counts=None, camera=None):
        self._frame_count += 1
        self.forecast.observe(timestamp, person_count)
        if zone_counts:
            for zone_name, count in zone_counts.items():
                self.forecast.observe(timestamp, count, zone=zone_name)
        self.heatmap.observe_tracks(timestamp, tracks, label_filter="person")
        for a in alerts:
            zone = getattr(a, "zone", None)
            sev = a.severity.value if hasattr(a.severity, "value") else str(a.severity)
            self.trends.observe(timestamp, rule=a.rule, severity=sev,
                                zone=zone, camera=camera)

    def update_heatmap(self, now=None):
        self.heatmap.update(now)

    def snapshot(self):
        return {
            "timestamp": time.time(),
            "frame_count": self._frame_count,
            "forecast": self.forecast.snapshot(),
            "heatmap": self.heatmap.snapshot(),
            "trends": self.trends.snapshot(),
        }

    @property
    def frame_count(self):
        return self._frame_count
