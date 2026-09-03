"""BHAIRAV Analytics — Historical trends, heatmaps, exportable reports.

Phase 6: Deeper analytics beyond live dashboard:
- Time-series trends (incidents per hour/day/week)
- Geographic heatmaps (incident density by zone)
- Category/level breakdowns over time
- Officer performance metrics
- Exportable CSV/JSON reports
- Predictive patterns (peak hours, hotspots)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

log = logging.getLogger("bhairav.analytics")


# ── Time Bucketing ──────────────────────────────────────────────────────────

def _bucket_timestamp(ts: float, bucket_sec: float) -> float:
    """Round timestamp down to the nearest bucket."""
    return (ts // bucket_sec) * bucket_sec


def _format_bucket(ts: float, granularity: str) -> str:
    """Format a bucket timestamp as a human-readable label."""
    import datetime
    dt = datetime.datetime.fromtimestamp(ts)
    if granularity == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    elif granularity == "day":
        return dt.strftime("%Y-%m-%d")
    elif granularity == "week":
        return dt.strftime("%Y-W%W")
    elif granularity == "month":
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ── Geographic Heatmap ──────────────────────────────────────────────────────

@dataclass
class HeatmapCell:
    lat_center: float
    lng_center: float
    count: int = 0
    categories: dict = field(default_factory=dict)
    avg_level: float = 0.0
    total_level: float = 0.0

    def add_incident(self, category: str, level: int):
        self.count += 1
        self.categories[category] = self.categories.get(category, 0) + 1
        self.total_level += level
        self.avg_level = self.total_level / self.count

    def to_dict(self):
        return {
            "lat": self.lat_center,
            "lng": self.lng_center,
            "count": self.count,
            "categories": self.categories,
            "avg_level": round(self.avg_level, 1),
        }


class HeatmapGenerator:
    """Generate geographic heatmaps from incident data."""

    def __init__(self, cell_size_deg: float = 0.005):
        """cell_size_deg: grid cell size in degrees (~500m at equator)."""
        self.cell_size = cell_size_deg

    def generate(self, incidents: list[dict]) -> list[dict]:
        """Generate heatmap cells from incident list."""
        cells: dict[tuple[int, int], HeatmapCell] = {}

        for inc in incidents:
            loc = inc.get("location", {})
            lat = loc.get("lat", 0)
            lng = loc.get("lng", 0)
            cat = inc.get("category", "other")
            level = inc.get("emergency_level", 1)

            # Snap to grid
            gi = int(lat // self.cell_size)
            gj = int(lng // self.cell_size)
            key = (gi, gj)

            if key not in cells:
                cells[key] = HeatmapCell(
                    lat_center=(gi + 0.5) * self.cell_size,
                    lng_center=(gj + 0.5) * self.cell_size,
                )
            cells[key].add_incident(cat, level)

        return sorted([c.to_dict() for c in cells.values()],
                       key=lambda x: x["count"], reverse=True)


# ── Trend Analysis ──────────────────────────────────────────────────────────

class TrendAnalyzer:
    """Analyze incident trends over time."""

    GRANULARITIES = {
        "hour": 3600,
        "day": 86400,
        "week": 604800,
        "month": 2592000,
    }

    def hourly_trend(self, incidents: list[dict], hours: int = 24) -> list[dict]:
        """Incidents per hour for the last N hours."""
        now = time.time()
        bucket_sec = 3600  # 1 hour
        buckets: dict[float, int] = defaultdict(int)

        for inc in incidents:
            ts = inc.get("created_at", 0)
            if now - ts > hours * 3600:
                continue
            b = _bucket_timestamp(ts, bucket_sec)
            buckets[b] += 1

        result = []
        for i in range(hours):
            ts = now - (i * 3600)
            b = _bucket_timestamp(ts, bucket_sec)
            result.append({
                "time": _format_bucket(b, "hour"),
                "timestamp": b,
                "count": buckets.get(b, 0),
            })
        result.reverse()
        return result

    def daily_trend(self, incidents: list[dict], days: int = 30) -> list[dict]:
        """Incidents per day for the last N days."""
        now = time.time()
        bucket_sec = 86400
        buckets: dict[float, dict] = defaultdict(lambda: {"count": 0, "categories": Counter()})

        for inc in incidents:
            ts = inc.get("created_at", 0)
            if now - ts > days * 86400:
                continue
            b = _bucket_timestamp(ts, bucket_sec)
            buckets[b]["count"] += 1
            buckets[b]["categories"][inc.get("category", "other")] += 1

        result = []
        for i in range(days):
            ts = now - (i * 86400)
            b = _bucket_timestamp(ts, bucket_sec)
            data = buckets.get(b, {"count": 0, "categories": Counter()})
            result.append({
                "date": _format_bucket(b, "day"),
                "timestamp": b,
                "count": data["count"],
                "categories": dict(data["categories"]),
            })
        result.reverse()
        return result

    def peak_hours(self, incidents: list[dict]) -> list[dict]:
        """Find peak incident hours across all data."""
        hour_counts = Counter()
        for inc in incidents:
            import datetime
            dt = datetime.datetime.fromtimestamp(inc.get("created_at", 0))
            hour_counts[dt.hour] += 1

        return sorted([
            {"hour": h, "count": c, "label": f"{h:02d}:00"}
            for h, c in hour_counts.items()
        ], key=lambda x: x["count"], reverse=True)

    def category_trend(self, incidents: list[dict], days: int = 30) -> dict:
        """Incident count per category per day."""
        now = time.time()
        result = defaultdict(lambda: defaultdict(int))

        for inc in incidents:
            ts = inc.get("created_at", 0)
            if now - ts > days * 86400:
                continue
            cat = inc.get("category", "other")
            day = _format_bucket(ts, "day")
            result[cat][day] += 1

        return {cat: dict(days) for cat, days in result.items()}


# ── Officer Performance ─────────────────────────────────────────────────────

class OfficerAnalytics:
    """Track and rank officer performance."""

    def __init__(self, store):
        self.store = store

    def get_officer_stats(self, hours: float = 168) -> list[dict]:
        """Get performance stats for all officers (default: last 7 days)."""
        cutoff = time.time() - (hours * 3600)
        officers = self.store.list_officers()
        incidents = self.store.list_incidents()

        result = []
        for off in officers:
            # Find incidents this officer was assigned to
            my_incidents = [
                i for i in incidents
                if off.id in i.assigned_officers and i.created_at > cutoff
            ]
            resolved = [i for i in my_incidents if i.status == "resolved"]
            total_time = sum(
                i.updated_at - i.created_at for i in resolved
            )

            result.append({
                "officer_id": off.id,
                "name": off.name,
                "role": off.role,
                "incidents_assigned": len(my_incidents),
                "incidents_resolved": len(resolved),
                "avg_response_time": round(
                    total_time / len(resolved), 1
                ) if resolved else None,
                "resolution_rate": round(
                    len(resolved) / len(my_incidents) * 100, 1
                ) if my_incidents else 0,
                "current_status": off.status,
            })

        return sorted(result, key=lambda x: x["incidents_resolved"], reverse=True)

    def get_team_summary(self, hours: float = 168) -> dict:
        """Team-level performance summary."""
        stats = self.get_officer_stats(hours)
        active = [s for s in stats if s["current_status"] != "off_duty"]
        resolved_times = [s["avg_response_time"] for s in stats if s["avg_response_time"]]

        return {
            "total_officers": len(stats),
            "active_officers": len(active),
            "total_incidents_handled": sum(s["incidents_assigned"] for s in stats),
            "total_resolved": sum(s["incidents_resolved"] for s in stats),
            "avg_resolution_rate": round(
                sum(s["resolution_rate"] for s in stats) / len(stats), 1
            ) if stats else 0,
            "avg_response_time": round(
                sum(resolved_times) / len(resolved_times), 1
            ) if resolved_times else None,
            "top_performers": stats[:5],
        }


# ── Report Export ───────────────────────────────────────────────────────────

class ReportExporter:
    """Export incident data as CSV or JSON reports."""

    @staticmethod
    def to_csv(incidents: list[dict]) -> str:
        """Export incidents as CSV string."""
        if not incidents:
            return ""

        output = io.StringIO()
        fields = ["id", "category", "emergency_level", "status", "source",
                  "location_name", "description", "crowd_reports",
                  "ai_verified", "created_at", "updated_at"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")

        writer.writeheader()
        for inc in incidents:
            loc = inc.get("location", {})
            row = {
                "id": inc.get("id", ""),
                "category": inc.get("category", ""),
                "emergency_level": inc.get("emergency_level", ""),
                "status": inc.get("status", ""),
                "source": inc.get("source", ""),
                "location_name": inc.get("location_name", ""),
                "description": inc.get("description", ""),
                "crowd_reports": inc.get("crowd_reports", 0),
                "ai_verified": inc.get("ai_verified", False),
                "created_at": inc.get("created_at", ""),
                "updated_at": inc.get("updated_at", ""),
            }
            writer.writerow(row)

        return output.getvalue()

    @staticmethod
    def to_json_report(incidents: list[dict], title: str = "BHAIRAV Report") -> dict:
        """Generate a structured JSON report with summaries."""
        categories = Counter(i.get("category", "other") for i in incidents)
        levels = Counter(str(i.get("emergency_level", 1)) for i in incidents)
        statuses = Counter(i.get("status", "unknown") for i in incidents)
        sources = Counter(i.get("source", "unknown") for i in incidents)

        return {
            "report_title": title,
            "generated_at": time.time(),
            "total_incidents": len(incidents),
            "summary": {
                "by_category": dict(categories),
                "by_level": dict(levels),
                "by_status": dict(statuses),
                "by_source": dict(sources),
            },
            "incidents": incidents,
        }

    @staticmethod
    def to_geojson(incidents: list[dict]) -> dict:
        """Export incidents as GeoJSON for mapping tools."""
        features = []
        for inc in incidents:
            loc = inc.get("location", {})
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [loc.get("lng", 0), loc.get("lat", 0)],
                },
                "properties": {
                    "id": inc.get("id", ""),
                    "category": inc.get("category", ""),
                    "emergency_level": inc.get("emergency_level", 1),
                    "status": inc.get("status", ""),
                    "description": inc.get("description", ""),
                    "source": inc.get("source", ""),
                    "crowd_reports": inc.get("crowd_reports", 0),
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
        }


# ── Unified Analytics Engine ────────────────────────────────────────────────

class AnalyticsEngine:
    """Unified analytics combining all analysis tools.

    Usage:
        engine = AnalyticsEngine(store)
        dashboard = engine.get_full_analytics(hours=24)
        csv = engine.export_csv(hours=7)
    """

    def __init__(self, store):
        self.store = store
        self.trends = TrendAnalyzer()
        self.officer_analytics = OfficerAnalytics(store)
        self.exporter = ReportExporter()
        self.heatmap = HeatmapGenerator()

    def _get_incidents_dicts(self, hours: float | None = None) -> list[dict]:
        """Get all incidents as dicts, optionally filtered by time."""
        incidents = self.store.list_incidents(limit=10000)
        if hours:
            cutoff = time.time() - (hours * 3600)
            incidents = [i for i in incidents if i.created_at > cutoff]
        return [i.to_dict() for i in incidents]

    def get_full_analytics(self, hours: float = 24) -> dict:
        """Complete analytics dashboard data."""
        incidents = self._get_incidents_dicts(hours)

        return {
            "period_hours": hours,
            "total_incidents": len(incidents),
            "hourly_trend": self.trends.hourly_trend(incidents, hours=min(int(hours), 72)),
            "daily_trend": self.trends.daily_trend(incidents, days=min(int(hours // 24), 30)),
            "peak_hours": self.trends.peak_hours(incidents),
            "category_trend": self.trends.category_trend(incidents),
            "heatmap": self.heatmap.generate(incidents),
            "officer_performance": self.officer_analytics.get_officer_stats(hours),
            "team_summary": self.officer_analytics.get_team_summary(hours),
            "category_breakdown": dict(Counter(i.get("category", "other") for i in incidents)),
            "level_breakdown": dict(Counter(str(i.get("emergency_level", 1)) for i in incidents)),
            "source_breakdown": dict(Counter(i.get("source", "unknown") for i in incidents)),
        }

    def export_csv(self, hours: float | None = None) -> str:
        """Export incidents as CSV."""
        incidents = self._get_incidents_dicts(hours)
        return self.exporter.to_csv(incidents)

    def export_json(self, hours: float | None = None, title: str = "BHAIRAV Report") -> dict:
        """Export incidents as structured JSON report."""
        incidents = self._get_incidents_dicts(hours)
        return self.exporter.to_json_report(incidents, title)

    def export_geojson(self, hours: float | None = None) -> dict:
        """Export incidents as GeoJSON."""
        incidents = self._get_incidents_dicts(hours)
        return self.exporter.to_geojson(incidents)

    def get_alert_patterns(self) -> dict:
        """Analyze alert patterns for predictive insights."""
        incidents = self._get_incidents_dicts()
        if not incidents:
            return {"message": "No data yet"}

        # Find recurring hotspots
        hotspot_hours = defaultdict(int)
        for inc in incidents:
            import datetime
            dt = datetime.datetime.fromtimestamp(inc.get("created_at", 0))
            hotspot_hours[dt.hour] += 1

        # Find most common category combinations
        category_pairs = Counter()
        cats_by_hour = defaultdict(Counter)
        for inc in incidents:
            import datetime
            dt = datetime.datetime.fromtimestamp(inc.get("created_at", 0))
            cat = inc.get("category", "other")
            cats_by_hour[dt.hour][cat] += 1

        peak_hour = max(hotspot_hours, key=hotspot_hours.get) if hotspot_hours else 0
        peak_categories = dict(cats_by_hour.get(peak_hour, {}))

        return {
            "peak_hour": f"{peak_hour:02d}:00",
            "peak_hour_incidents": hotspot_hours.get(peak_hour, 0),
            "peak_hour_categories": peak_categories,
            "hotspot_hours": sorted(
                [{"hour": h, "count": c} for h, c in hotspot_hours.items()],
                key=lambda x: x["count"], reverse=True,
            )[:5],
            "quietest_hour": min(hotspot_hours, key=hotspot_hours.get) if hotspot_hours else 0,
        }
