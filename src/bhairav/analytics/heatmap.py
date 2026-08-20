"""Spatial heatmap: track centroids accumulated into a 2D density grid.

Maintains a time-weighted accumulation of person/vehicle centroids
over a configurable grid resolution. Exposes the heatmap as a nested list
suitable for JSON serialisation and direct rendering in the dashboard
(CSS grid or canvas heatmap layer).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class HeatmapPoint:
    timestamp: float
    x: float   # normalised 0..1
    y: float   # normalised 0..1
    weight: float = 1.0


class SpatialHeatmap:
    """Time-weighted spatial density heatmap.

    Parameters
    ----------
    grid_w : int
        Horizontal grid resolution (default 32).
    grid_h : int
        Vertical grid resolution (default 24).
    decay_sec : float
        Half-life for exponential time decay (default 30).  Older points
        contribute exponentially less; the grid is continuously rebalanced.
    """

    def __init__(self, grid_w: int = 32, grid_h: int = 24,
                 decay_sec: float = 30.0):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.decay_sec = decay_sec
        self._points: deque[HeatmapPoint] = deque()
        self._grid: np.ndarray = np.zeros((grid_h, grid_w), dtype=np.float64)
        self._max: float = 1.0

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        gx = min(self.grid_w - 1, max(0, int(x * self.grid_w)))
        gy = min(self.grid_h - 1, max(0, int(y * self.grid_h)))
        return gy, gx

    def observe(self, timestamp: float, x: float, y: float,
                weight: float = 1.0) -> None:
        """Accumulate a track centroid into the heatmap grid."""
        self._points.append(HeatmapPoint(timestamp=timestamp, x=x, y=y,
                                         weight=weight))

    def _rebuild(self, now: float) -> None:
        """Rebuild the grid with exponential time decay."""
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)
        for p in self._points:
            age = now - p.timestamp
            decay = np.exp(-0.693 * age / self.decay_sec)  # half-life decay
            gy, gx = self._cell(p.x, p.y)
            grid[gy, gx] += p.weight * decay
        self._grid = grid
        self._max = max(float(grid.max()), 1.0)

    def _prune(self, now: float) -> None:
        """Remove very old points (10x decay_sec)."""
        cutoff = now - self.decay_sec * 10
        while self._points and self._points[0].timestamp < cutoff:
            self._points.popleft()

    def update(self, now: float | None = None) -> None:
        """Rebuild the heatmap grid.  Call once per frame tick or at a lower
        frequency for performance."""
        if now is None:
            now = time.time()
        self._prune(now)
        self._rebuild(now)

    def observe_tracks(self, timestamp: float,
                       tracks: list, label_filter: str = "person") -> None:
        """Batch-observe centroids from a list of Track objects."""
        for t in tracks:
            if label_filter and getattr(t, "label", "") != label_filter:
                continue
            cx = (t.bbox[0] + t.bbox[2]) / 2.0
            cy = (t.bbox[1] + t.bbox[3]) / 2.0
            self.observe(timestamp, cx, cy)

    @property
    def grid(self) -> list[list[float]]:
        """Normalised 0..1 heatmap as nested list for JSON."""
        if self._max == 0:
            return [[0.0] * self.grid_w for _ in range(self.grid_h)]
        normed = (self._grid / self._max).tolist()
        return [[round(v, 4) for v in row] for row in normed]

    @property
    def raw_grid(self) -> np.ndarray:
        return self._grid

    def snapshot(self) -> dict:
        """Full heatmap state for the analytics WebSocket push."""
        return {
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "grid": self.grid,
            "points": len(self._points),
        }
