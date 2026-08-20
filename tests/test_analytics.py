"""Tests for Phase 12 - Predictive Analytics."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhairav.analytics.forecast import CrowdDensityForecast
from bhairav.analytics.heatmap import SpatialHeatmap
from bhairav.analytics.trends import TrendAnalyzer
from bhairav.analytics.engine import AnalyticsEngine
from bhairav.types import Alert, Severity, Track


def _track(tid, bbox=(0.3, 0.3, 0.5, 0.5)):
    return Track(track_id=tid, bbox=bbox, label="person", confidence=0.9)


def _alert(rule="fight", sev="red", zone=None, ts=None):
    return Alert(
        rule=rule,
        zone=zone,
        track_id=1,
        severity=Severity(sev),
        message="t",
        frame_id=1,
        timestamp=ts or time.time(),
    )


def _t(off=0):
    return time.time() + off


class TestForecast:
    def test_insufficient(self):
        f = CrowdDensityForecast(min_samples=5)
        for i in range(3):
            f.observe(_t(-3 + i), 2)
        assert f.forecast()["status"] == "insufficient_data"

    def test_rising(self):
        f = CrowdDensityForecast(min_samples=3, horizon_sec=5.0)
        for i in range(10):
            f.observe(_t(-10 + i), i)
        r = f.forecast()
        assert r["status"] == "ok" and r["trend"] == "rising"

    def test_falling(self):
        f = CrowdDensityForecast(min_samples=3, horizon_sec=5.0)
        for i in range(10):
            f.observe(_t(-10 + i), 10 - i)
        assert f.forecast()["trend"] == "falling"

    def test_stable(self):
        f = CrowdDensityForecast(min_samples=3, horizon_sec=5.0)
        for i in range(10):
            f.observe(_t(-10 + i), 5)
        assert f.forecast()["trend"] == "stable"

    def test_zones(self):
        f = CrowdDensityForecast(min_samples=2, horizon_sec=5.0)
        f.observe(_t(-1), 3, zone="plaza")
        f.observe(_t(0), 5, zone="plaza")
        f.observe(_t(-1), 1, zone="gate")
        f.observe(_t(0), 0, zone="gate")
        assert f.forecast(zone="plaza")["trend"] == "rising"
        assert f.forecast(zone="gate")["trend"] == "falling"

    def test_snapshot(self):
        f = CrowdDensityForecast(min_samples=2, horizon_sec=5.0)
        for i in range(5):
            f.observe(_t(-5 + i), 2)
        s = f.snapshot()
        assert "global" in s and "zones" in s and "person_count" in s

    def test_prune(self):
        f = CrowdDensityForecast(window_sec=5.0, min_samples=2, horizon_sec=1.0)
        f.observe(_t(-10), 5)
        f.observe(_t(-9), 10)
        f.observe(_t(-1), 1)
        f.observe(_t(0), 2)
        assert f.forecast()["samples"] == 2


class TestHeatmap:
    def test_point(self):
        h = SpatialHeatmap(grid_w=4, grid_h=4, decay_sec=60)
        h.observe(_t(0), 0.5, 0.5)
        h.update(now=_t(0))
        g = h.grid
        assert len(g) == 4 and any(g[r][c] > 0 for r in range(4) for c in range(4))

    def test_tracks(self):
        h = SpatialHeatmap(grid_w=4, grid_h=4, decay_sec=60)
        h.observe_tracks(_t(0), [_track(1, (0.1, 0.1, 0.3, 0.3)), _track(2, (0.7, 0.7, 0.9, 0.9))])
        h.update(now=_t(0))
        assert h.snapshot()["points"] == 2

    def test_decay(self):
        h = SpatialHeatmap(grid_w=4, grid_h=4, decay_sec=10)
        h.observe(_t(0), 0.5, 0.5)
        h.update(now=_t(0))
        v0 = h.raw_grid.max()
        h.update(now=_t(30))
        assert h.raw_grid.max() < v0

    def test_snap(self):
        h = SpatialHeatmap(grid_w=8, grid_h=6)
        h.observe(_t(0), 0.5, 0.5)
        h.update(now=_t(0))
        s = h.snapshot()
        assert s["grid_w"] == 8 and s["grid_h"] == 6 and len(s["grid"]) == 6

    def test_empty(self):
        h = SpatialHeatmap(grid_w=4, grid_h=4)
        h.update(now=_t(0))
        assert all(v == 0.0 for r in h.grid for v in r)


class TestTrend:
    def test_rule(self):
        t = TrendAnalyzer()
        t.observe(_t(0), "fight", "red")
        t.observe(_t(1), "fight", "orange")
        t.observe(_t(2), "loiter", "yellow")
        c = t.by_rule()
        assert c["fight"] == 2 and c["loiter"] == 1

    def test_severity(self):
        t = TrendAnalyzer()
        t.observe(_t(0), "fight", "red")
        t.observe(_t(1), "loiter", "yellow")
        c = t.by_severity()
        assert c["red"] == 1 and c["yellow"] == 1

    def test_zone(self):
        t = TrendAnalyzer()
        t.observe(_t(0), "fight", "red", zone="plaza")
        t.observe(_t(1), "fight", "red", zone="plaza")
        t.observe(_t(2), "loiter", "yellow", zone="gate")
        c = t.by_zone()
        assert c["plaza"] == 2 and c["gate"] == 1

    def test_burst(self):
        t = TrendAnalyzer(burst_window_sec=10.0, burst_threshold=3)
        for i in range(5):
            t.observe(_t(-5 + i), "fight", "red")
        b = t.detect_bursts(now=_t(0))
        assert len(b) == 1 and b[0]["rule"] == "fight"

    def test_rate(self):
        t = TrendAnalyzer()
        for i in range(10):
            t.observe(_t(-10 + i), "fight", "red")
        assert t.rate_per_min(window_sec=15.0) > 0

    def test_snap(self):
        t = TrendAnalyzer()
        t.observe(_t(0), "fight", "red")
        s = t.snapshot()
        assert all(k in s for k in ["total", "by_rule", "by_severity", "rate_per_min", "bursts"])

    def test_prune(self):
        t = TrendAnalyzer(window_sec=5.0)
        t.observe(_t(-10), "fight", "red")
        t.observe(_t(0), "loiter", "yellow")
        assert t.snapshot()["active"] == 1


class TestEngine:
    def test_observe(self):
        e = AnalyticsEngine(forecast_horizon_sec=5.0, heatmap_grid=(8, 6), trend_window_sec=300)
        tr = [_track(1, (0.3, 0.3, 0.5, 0.5))]
        for i in range(10):
            e.observe_frame(
                _t(-10 + i),
                2,
                tr,
                [_alert(ts=_t(-10))] if i == 0 else [],
                zone_counts={"plaza": 1},
                camera="C1",
            )
        assert e.frame_count == 10
        s = e.snapshot()
        assert "forecast" in s and "heatmap" in s and "trends" in s

    def test_heatmap(self):
        e = AnalyticsEngine(heatmap_grid=(8, 6))
        now = _t(0)
        e.observe_frame(now, 1, [_track(1, (0.4, 0.4, 0.6, 0.6))], [], camera="C1")
        e.update_heatmap(now=now)
        assert e.snapshot()["heatmap"]["points"] == 1

    def test_zones(self):
        e = AnalyticsEngine(forecast_horizon_sec=5.0, trend_window_sec=300)
        for i in range(10):
            e.observe_frame(_t(-10 + i), i, [], [], zone_counts={"plaza": i}, camera="C1")
        assert "plaza" in e.snapshot()["forecast"]["zones"]

    def test_trends(self):
        e = AnalyticsEngine()
        for i in range(5):
            e.observe_frame(_t(-5 + i), 1, [], [_alert(ts=_t(-5 + i))], camera="C1")
        s = e.snapshot()
        assert s["trends"]["total"] == 5 and s["trends"]["by_rule"].get("fight", 0) == 5

    def test_full(self):
        e = AnalyticsEngine(forecast_horizon_sec=5.0, heatmap_grid=(4, 4), trend_window_sec=300)
        tr = [_track(1, (0.2, 0.2, 0.4, 0.4))]
        now = _t(0)
        for i in range(20):
            ts = _t(-20 + i)
            e.observe_frame(
                ts,
                3,
                tr,
                [
                    _alert(
                        rule="fight" if i % 3 == 0 else "loiter",
                        sev="red" if i % 5 == 0 else "orange",
                        ts=ts,
                    )
                ],
                zone_counts={"plaza": 2},
                camera="C1",
            )
        e.update_heatmap(now=now)
        s = e.snapshot()
        assert s["frame_count"] == 20
        assert s["forecast"]["global"]["status"] == "ok"
        assert s["heatmap"]["points"] == 20
        assert s["trends"]["total"] == 20
