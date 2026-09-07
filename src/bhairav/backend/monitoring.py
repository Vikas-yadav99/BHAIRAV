"""BHAIRAV Monitoring — Request metrics, error tracking, deep health checks.

Provides:
- MonitoringMiddleware: tracks request duration, status codes, error rates
- HealthChecker: deep health checks for all subsystems
- MetricsCollector: in-memory rolling window metrics
"""
from __future__ import annotations

import logging
import os
try:
    import resource
except ImportError:
    resource = None  # Windows
import sys
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("bhairav.monitoring")


# ── Metrics Collector ───────────────────────────────────────────────────────

@dataclass
class EndpointStats:
    """Rolling stats for a single endpoint."""
    count: int = 0
    error_count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    status_codes: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_hit: float = 0.0

    def record(self, status_code: int, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.status_codes[status_code] += 1
        if status_code >= 400:
            self.error_count += 1
        self.last_hit = time.time()

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.count if self.count else 0.0

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else 0,
            "max_ms": round(self.max_ms, 2),
            "status_codes": dict(self.status_codes),
            "last_hit": self.last_hit,
        }


class MetricsCollector:
    """In-memory rolling metrics for all endpoints."""

    def __init__(self, max_endpoints: int = 500):
        self._lock = threading.Lock()
        self._endpoints: dict[str, EndpointStats] = {}
        self._max_endpoints = max_endpoints
        self._start_time = time.time()
        self._total_requests = 0
        self._total_errors = 0

    def record(self, method: str, path: str, status_code: int, duration_ms: float) -> None:
        key = f"{method} {path}"
        with self._lock:
            if key not in self._endpoints:
                if len(self._endpoints) >= self._max_endpoints:
                    # Evict least recently hit
                    oldest_key = min(self._endpoints, key=lambda k: self._endpoints[k].last_hit)
                    del self._endpoints[oldest_key]
                self._endpoints[key] = EndpointStats()
            self._endpoints[key].record(status_code, duration_ms)
            self._total_requests += 1
            if status_code >= 500:
                self._total_errors += 1

    def get_summary(self) -> dict[str, Any]:
        """Return full metrics summary."""
        with self._lock:
            uptime = time.time() - self._start_time
            by_status: dict[int, int] = defaultdict(int)
            total_count = 0
            total_errors = 0
            total_ms = 0.0
            for ep in self._endpoints.values():
                total_count += ep.count
                total_errors += ep.error_count
                total_ms += ep.total_ms
                for code, cnt in ep.status_codes.items():
                    by_status[code] += cnt

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": total_count,
                "total_errors": total_errors,
                "overall_error_rate": round(total_errors / total_count, 4) if total_count else 0,
                "avg_response_ms": round(total_ms / total_count, 2) if total_count else 0,
                "status_code_distribution": {str(k): v for k, v in sorted(by_status.items())},
                "endpoints": {
                    k: v.to_dict() for k, v in sorted(
                        self._endpoints.items(),
                        key=lambda x: x[1].count,
                        reverse=True,
                    )[:50]  # top 50 endpoints
                },
            }

    def get_endpoint(self, method: str, path: str) -> dict | None:
        key = f"{method} {path}"
        with self._lock:
            ep = self._endpoints.get(key)
            return ep.to_dict() if ep else None

    def reset(self) -> None:
        with self._lock:
            self._endpoints.clear()
            self._start_time = time.time()
            self._total_requests = 0
            self._total_errors = 0


# ── HTTP Middleware ──────────────────────────────────────────────────────────

class MonitoringMiddleware:
    """ASGI middleware that records request timing and status codes.

    Usage:
        app.add_middleware(MonitoringMiddleware, metrics=collector)
    """

    def __init__(self, app, metrics: MetricsCollector | None = None):
        self.app = app
        self.metrics = metrics or MetricsCollector()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        start = time.time()

        # Track status code from downstream
        status_holder = [200]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder[0] = message.get("status", 200)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            status_holder[0] = 500
            raise
        finally:
            duration_ms = (time.time() - start) * 1000
            self.metrics.record(method, path, status_holder[0], duration_ms)

            # Log slow requests (> 2 seconds)
            if duration_ms > 2000:
                log.warning("SLOW REQUEST: %s %s took %.0fms (status=%d)",
                            method, path, duration_ms, status_holder[0])

            # Log errors
            if status_holder[0] >= 500:
                log.error("ERROR REQUEST: %s %s status=%d (%.0fms)",
                          method, path, status_holder[0], duration_ms)


# ── Deep Health Check ───────────────────────────────────────────────────────

class HealthChecker:
    """Deep health checks for all subsystems."""

    def __init__(self, store=None, audit=None, hub=None, incidents=None,
                 phone=None, analytics=None, safety=None):
        self.store = store
        self.audit = audit
        self.hub = hub
        self.incidents = incidents
        self.phone = phone
        self.analytics = analytics
        self.safety = safety

    def check(self) -> dict[str, Any]:
        """Run all health checks, return summary."""
        checks = {}
        all_healthy = True

        # Memory check
        try:
            mem = resource.getrusage(resource.RUSAGE_SELF)
            mem_mb = mem.ru_maxrss / 1024  # Linux: bytes, Windows: KB
            checks["memory"] = {
                "status": "ok",
                "max_rss_mb": round(mem_mb, 1),
            }
        except Exception as e:
            checks["memory"] = {"status": "error", "error": str(e)}

        # Disk check
        try:
            st = os.statvfs(".")
            free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
            checks["disk"] = {
                "status": "ok" if free_gb > 1 else "warning",
                "free_gb": round(free_gb, 2),
            }
            if free_gb < 1:
                all_healthy = False
        except Exception as e:
            checks["disk"] = {"status": "error", "error": str(e)}

        # Python check
        checks["python"] = {
            "status": "ok",
            "version": sys.version.split()[0],
        }

        # Thread count
        try:
            checks["threads"] = {
                "status": "ok",
                "count": threading.active_count(),
            }
        except Exception as e:
            checks["threads"] = {"status": "error", "error": str(e)}

        # Module checks
        modules = {
            "evidence_store": self.store,
            "audit_log": self.audit,
            "live_hub": self.hub,
            "incident_store": self.incidents,
            "phone_gateway": self.phone,
            "analytics": self.analytics,
            "city_safety": self.safety,
        }
        for name, obj in modules.items():
            if obj is not None:
                checks[name] = {"status": "ok", "type": type(obj).__name__}
            else:
                checks[name] = {"status": "not_configured"}
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "timestamp": time.time(),
        }
