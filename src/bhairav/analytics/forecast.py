"""Crowd density forecasting with rolling-window linear regression.

Maintains a fixed-size sliding window of person-count observations per zone
(or a global window when no zones are defined) and fits a weighted linear
regression to extrapolate density over a configurable forecast horizon.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DensitySample:
    timestamp: float
    person_count: int
    zone: str | None = None


class CrowdDensityForecast:
    """Rolling linear-regression crowd density predictor."""

    def __init__(self, window_sec: float = 60.0, horizon_sec: float = 10.0,
                 min_samples: int = 5):
        self.window_sec = window_sec
        self.horizon_sec = horizon_sec
        self.min_samples = min_samples
        self._samples: deque[DensitySample] = deque()
        self._global_counts: deque[tuple[float, int]] = deque()
        self._zone_counts: dict[str, deque[tuple[float, int]]] = {}
        self._last_forecast: dict = {}

    def observe(self, timestamp: float, person_count: int,
                zone: str | None = None) -> None:
        """Record a new person-count observation."""
        self._samples.append(DensitySample(timestamp=timestamp,
                                           person_count=person_count, zone=zone))
        self._global_counts.append((timestamp, person_count))
        if zone is not None:
            if zone not in self._zone_counts:
                self._zone_counts[zone] = deque()
            self._zone_counts[zone].append((timestamp, person_count))
        self._prune(timestamp)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()
        while self._global_counts and self._global_counts[0][0] < cutoff:
            self._global_counts.popleft()
        for zq in self._zone_counts.values():
            while zq and zq[0][0] < cutoff:
                zq.popleft()

    def _predict(self, series: deque[tuple[float, int]]) -> dict:
        """Fit weighted linear regression and extrapolate."""
        if len(series) < self.min_samples:
            return {"status": "insufficient_data",
                    "min_required": self.min_samples}

        ts = np.array([p[0] for p in series], dtype=np.float64)
        counts = np.array([p[1] for p in series], dtype=np.float64)

        t0 = ts[0]
        t = ts - t0

        # exponential weighting
        alpha = 2.0 / (len(t) + 1.0)
        weights = np.array([(1.0 - alpha) ** i for i in range(len(t))][::-1])

        # weighted least-squares: y = a + b*t
        W = np.diag(weights)
        X = np.column_stack([np.ones(len(t)), t])
        beta = np.linalg.lstsq(W @ X, W @ counts, rcond=None)[0]
        intercept, slope = beta

        current = float(intercept + slope * t[-1])
        future_t = t[-1] + self.horizon_sec
        predicted = max(0.0, float(intercept + slope * future_t))

        # R-squared
        fitted = intercept + slope * t
        residuals = counts - fitted
        ss_res = float(np.sum(weights * residuals ** 2))
        mean_y = float(np.sum(weights * counts) / np.sum(weights))
        ss_tot = float(np.sum(weights * (counts - mean_y) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        confidence = min(1.0, max(0.0, r_squared))

        if slope > 0.05:
            trend = "rising"
        elif slope < -0.05:
            trend = "falling"
        else:
            trend = "stable"

        return {
            "status": "ok",
            "current_density": round(current, 2),
            "predicted_density": round(predicted, 2),
            "horizon_sec": self.horizon_sec,
            "slope_per_sec": round(float(slope), 4),
            "trend": trend,
            "confidence": round(confidence, 3),
            "samples": len(series),
            "window_sec": self.window_sec,
        }

    def forecast(self, zone: str | None = None) -> dict:
        """Return the latest forecast for a zone (or global)."""
        if zone is not None:
            series = self._zone_counts.get(zone, deque())
        else:
            series = self._global_counts
        result = self._predict(series)
        result["zone"] = zone
        result["timestamp"] = time.time()
        self._last_forecast = result
        return result

    @property
    def last_forecast(self) -> dict:
        return dict(self._last_forecast)

    @property
    def person_count(self) -> int:
        if self._global_counts:
            return self._global_counts[-1][1]
        return 0

    def snapshot(self) -> dict:
        zones = {}
        for zname in self._zone_counts:
            zones[zname] = self.forecast(zone=zname)
        return {
            "global": self.forecast(),
            "zones": zones,
            "person_count": self.person_count,
        }
