"""Metrics registry + Prometheus text exposition (Phase 9 M3).

A tiny dependency-free metrics core: monotonic counters, gauges and bounded
time-series histories, rendered in Prometheus exposition format for a
scraper (prometheus.yml in deploy/) and consumed directly by the dashboard
Status/Health tab via /api/status `series`. No client library, no threads
beyond a lock - the sampler in serve.py drives it.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class History:
    """Bounded (timestamp, value) series for charts (drop-oldest)."""

    def __init__(self, maxlen: int = 600):
        self.maxlen = maxlen
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, value: float, ts: float | None = None) -> None:
        with self._lock:
            self._buf.append((ts if ts is not None else time.time(),
                              float(value)))

    def points(self) -> list:
        with self._lock:
            return list(self._buf)

    def latest(self) -> float | None:
        with self._lock:
            return self._buf[-1][1] if self._buf else None


class MetricsRegistry:
    """Thread-safe counters, gauges and histories, keyed by (name, labels)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict = {}
        self._gauges: dict = {}
        self._histories: dict = {}

    # ---- mutation --------------------------------------------------------
    def inc(self, name: str, labels: dict | None = None, delta: float = 1.0,
            ts: float | None = None) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + delta
            h = self._histories.get(name)
        if h is not None:
            h.append(self._counters[key], ts)

    def set(self, name: str, value: float, labels: dict | None = None,
            ts: float | None = None) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._gauges[key] = float(value)
            h = self._histories.get(name)
        if h is not None:
            h.append(float(value), ts)

    def history(self, name: str, maxlen: int = 600) -> History:
        with self._lock:
            if name not in self._histories:
                self._histories[name] = History(maxlen)
            return self._histories[name]

    # ---- reads -----------------------------------------------------------
    def snapshot(self) -> dict:
        """JSON-safe snapshot for /api/status `series` (charts in the UI)."""
        with self._lock:
            gauges = sorted(
                ({"name": n, "labels": dict(lbl), "value": v}
                 for (n, lbl), v in self._gauges.items()),
                key=lambda g: (g["name"], str(g["labels"])))
            counters = sorted(
                ({"name": n, "labels": dict(lbl), "value": round(v, 3)}
                 for (n, lbl), v in self._counters.items()),
                key=lambda g: (g["name"], str(g["labels"])))
            series = {n: [{"t": t, "v": v} for t, v in h.points()]
                      for n, h in sorted(self._histories.items())}
        return {"gauges": gauges, "counters": counters, "series": series}

    def render(self) -> str:
        """Prometheus text exposition format (for /metrics)."""
        with self._lock:
            counters = sorted(self._counters.items())
            gauges = sorted(self._gauges.items())
        lines: list[str] = []
        for (name, labels), value in counters:
            lines.append(f"# TYPE {name} counter")
            lines.append(_sample(name, labels, value))
        for (name, labels), value in gauges:
            lines.append(f"# TYPE {name} gauge")
            lines.append(_sample(name, labels, value))
        return "\n".join(lines) + ("\n" if lines else "")


def _sample(name: str, labels: tuple, value: float) -> str:
    if labels:
        rendered = ",".join(f'{k}="{v}"' for k, v in labels)
        return f"{name}{{{rendered}}} {value:g}"
    return f"{name} {value:g}"
