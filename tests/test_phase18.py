"""Tests for Phase 18: NL Summaries, Predictive Hotspot, Resource Allocation."""
from __future__ import annotations

import time
import pytest

from bhairav.analytics.summarizer import NLAlertSummarizer, AlertSummary
from bhairav.analytics.hotspot import PredictiveHotspot, HotspotZone
from bhairav.analytics.allocation import ResourceAllocator, ResourceRecommendation


def _now():
    return time.time()

def _alert(rule="intrusion", severity="red", zone="Zone A", camera="CAM-01", ts=None):
    return {"rule": rule, "severity": severity, "zone": zone, "camera": camera, "timestamp": ts or _now()}


class TestSummarizer:
    def test_empty_returns_all_clear(self):
        s = NLAlertSummarizer(window_sec=60)
        result = s.summarize()
        assert result.severity == "green"
        assert result.alert_count == 0

    def test_single_alert(self):
        s = NLAlertSummarizer(window_sec=60)
        s.observe(_alert(zone="Zone A"))
        result = s.summarize()
        assert result.alert_count == 1
        assert result.severity == "red"

    def test_multiple_alerts(self):
        s = NLAlertSummarizer(window_sec=60)
        for _ in range(5):
            s.observe(_alert(rule="fight", severity="orange"))
        result = s.summarize()
        assert result.alert_count == 5
        assert result.severity == "orange"

    def test_highest_severity_wins(self):
        s = NLAlertSummarizer(window_sec=60)
        s.observe(_alert(severity="green"))
        s.observe(_alert(severity="red"))
        assert s.summarize().severity == "red"

    def test_top_rules(self):
        s = NLAlertSummarizer(window_sec=60)
        for _ in range(3):
            s.observe(_alert(rule="intrusion"))
        for _ in range(2):
            s.observe(_alert(rule="loitering"))
        result = s.summarize()
        assert "intrusion" in result.top_rules
        assert "loitering" in result.top_rules

    def test_to_dict(self):
        s = NLAlertSummarizer(window_sec=60)
        s.observe(_alert())
        d = s.summarize().to_dict()
        assert "text" in d and "severity" in d

    def test_observe_batch(self):
        s = NLAlertSummarizer(window_sec=60)
        s.observe_batch([_alert(), _alert(), _alert()])
        assert s.summarize().alert_count == 3

    def test_reset(self):
        s = NLAlertSummarizer(window_sec=60)
        s.observe(_alert())
        s.reset()
        assert s.summarize().alert_count == 0

    def test_snapshot(self):
        s = NLAlertSummarizer(window_sec=60)
        s.observe(_alert())
        assert isinstance(s.snapshot(), dict)

    def test_llm_success(self):
        s = NLAlertSummarizer(window_sec=60, llm_callback=lambda p: "All good.")
        s.observe(_alert())
        result = s.summarize()
        assert "All good" in result.text
        assert result.confidence == 0.8


class TestHotspot:
    def test_empty_no_hotspots(self):
        assert PredictiveHotspot(window_sec=60).rank_hotspots() == []

    def test_single_zone(self):
        h = PredictiveHotspot(window_sec=60, min_alerts=1)
        now = _now()
        for i in range(5):
            h.observe(now - i, zone="Zone A", rule="intrusion")
        hotspots = h.rank_hotspots()
        assert len(hotspots) == 1 and hotspots[0].zone == "Zone A"

    def test_risk_score_range(self):
        h = PredictiveHotspot(window_sec=60, min_alerts=1)
        now = _now()
        for i in range(10):
            h.observe(now - i, zone="Zone B")
        assert 0.0 <= h.rank_hotspots()[0].risk_score <= 1.0

    def test_multi_zone_ranking(self):
        h = PredictiveHotspot(window_sec=60, min_alerts=1)
        now = _now()
        for i in range(10):
            h.observe(now - i, zone="hot")
        for i in range(2):
            h.observe(now - i, zone="cold")
        assert h.rank_hotspots()[0].zone == "hot"

    def test_peak_hour(self):
        now = _now()
        # Pick a real timestamp and compute its hour-of-day
        expected_hour = int((now % 86400) // 3600)
        # Use timestamps very close to now so they stay in window
        h = PredictiveHotspot(window_sec=3600, min_alerts=1)
        for i in range(5):
            h.observe(now - i, zone="Zone C")
        assert h.rank_hotspots()[0].peak_hour == expected_hour

    def test_trend_rising(self):
        h = PredictiveHotspot(window_sec=600, min_alerts=1)
        now = _now()
        for i in range(8):
            h.observe(now - i * 10, zone="Zone E")
        for i in range(2):
            h.observe(now - 400 - i * 10, zone="Zone E")
        assert h.rank_hotspots()[0].trend == "rising"

    def test_predicted_next_hour(self):
        h = PredictiveHotspot(window_sec=3600, min_alerts=1)
        now = _now()
        for i in range(10):
            h.observe(now - i * 60, zone="Zone F")
        assert h.rank_hotspots()[0].predicted_next_hour > 0

    def test_to_dict(self):
        h = PredictiveHotspot(window_sec=60, min_alerts=1)
        h.observe(_now(), zone="Z"); h.observe(_now(), zone="Z")
        assert "risk_score" in h.rank_hotspots()[0].to_dict()

    def test_snapshot(self):
        h = PredictiveHotspot(window_sec=60, min_alerts=1)
        h.observe(_now(), zone="Z"); h.observe(_now(), zone="Z")
        assert "hotspots" in h.snapshot()

    def test_reset(self):
        h = PredictiveHotspot(window_sec=60, min_alerts=1)
        h.observe(_now(), zone="Z"); h.observe(_now(), zone="Z")
        h.reset()
        assert h.rank_hotspots() == []

    def test_min_alerts_filter(self):
        h = PredictiveHotspot(window_sec=60, min_alerts=5)
        for i in range(3):
            h.observe(_now() - i, zone="Z")
        assert h.rank_hotspots() == []

    def test_prune_old_events(self):
        h = PredictiveHotspot(window_sec=5, min_alerts=1)
        h.observe(_now() - 100, zone="Z")
        assert h.rank_hotspots() == []


class TestAllocation:
    def test_no_recommendations_when_quiet(self):
        assert len(ResourceAllocator(officer_pool=10).analyze([], {})) == 0

    def test_deploy_officer_for_high_risk(self):
        a = ResourceAllocator(officer_pool=10)
        result = a.analyze(
            [{"zone": "Z", "risk_score": 0.9, "trend": "stable", "predicted_next_hour": 10}],
            {"Z": 15},
        )
        assert "deploy_officer" in [r.action for r in result]

    def test_reassign_camera_for_moderate_risk(self):
        a = ResourceAllocator(officer_pool=10)
        result = a.analyze(
            [{"zone": "Z", "risk_score": 0.5, "trend": "stable", "predicted_next_hour": 3}],
            {"Z": 5},
        )
        assert "reassign_camera" in [r.action for r in result]

    def test_call_backup_when_pool_low(self):
        a = ResourceAllocator(officer_pool=2)
        a.deploy_officer("A", 2)
        result = a.analyze(
            [{"zone": "B", "risk_score": 0.8, "trend": "rising", "predicted_next_hour": 8}],
            {"B": 12},
        )
        assert "call_backup" in [r.action for r in result]

    def test_burst_escalation(self):
        a = ResourceAllocator(officer_pool=10)
        result = a.analyze([], {}, trend_data={"bursts": [{"rule": "intrusion", "count": 8, "window_sec": 10}]})
        assert "escalate_response" in [r.action for r in result]
        assert result[0].priority == "critical"

    def test_camera_prioritization(self):
        a = ResourceAllocator(officer_pool=10, cameras=["C1", "C2"])
        result = a.analyze([], {"A": 15, "B": 3})
        assert "prioritize_camera" in [r.action for r in result]

    def test_priority_sorting(self):
        a = ResourceAllocator(officer_pool=10)
        result = a.analyze(
            [{"zone": "A", "risk_score": 0.5, "trend": "stable", "predicted_next_hour": 3},
             {"zone": "B", "risk_score": 0.95, "trend": "rising", "predicted_next_hour": 12}],
            {"A": 5, "B": 20},
        )
        prio = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        nums = [prio.get(r.priority, 9) for r in result]
        assert nums == sorted(nums)

    def test_deploy_and_recall(self):
        a = ResourceAllocator(officer_pool=10)
        a.deploy_officer("A", 3)
        assert a.snapshot()["available"] == 7
        a.recall_officer("A", 2)
        assert a.snapshot()["deployed"]["A"] == 1

    def test_to_dict(self):
        r = ResourceRecommendation("deploy_officer", "high", "A", "test", 0.9, _now() + 300)
        assert r.to_dict()["action"] == "deploy_officer"

    def test_snapshot(self):
        assert ResourceAllocator(officer_pool=5).snapshot()["officer_pool"] == 5

    def test_reset(self):
        a = ResourceAllocator(officer_pool=10)
        a.deploy_officer("A", 5)
        a.reset()
        assert a.snapshot()["available"] == 10
