"""Tests for Phase 21-24: 3D Scene, Traffic, Investigation, NLP Query."""
from __future__ import annotations
import time, tempfile, os, pytest
from bhairav.viz3d import Scene3DManager, Camera3D, Person3D, Zone3D
from bhairav.traffic import TrafficAnalyzer, VehicleCount, CongestionLevel
from bhairav.investigation import InvestigationTimeline, CaseFile, TimelineEvent
from bhairav.nlp import NLPQueryEngine, QueryResult

def _now(): return time.time()

class TestScene3D:
    def test_add_camera(self):
        s = Scene3DManager()
        s.add_camera(Camera3D("c1", "C1", x=10, z=10))
        assert len(s.snapshot()["cameras"]) == 1
    def test_update_persons(self):
        s = Scene3DManager()
        s.add_camera(Camera3D("c1", "C1"))
        s.update_persons("c1", [{"track_id": 1, "bbox": [0.3,0.2,0.5,0.8]}])
        assert len(s.snapshot()["persons"]) == 1
    def test_stale_tracks(self):
        s = Scene3DManager()
        s.add_camera(Camera3D("c1", "C1"))
        s.update_persons("c1", [{"track_id": 1, "bbox": [0.3,0.2,0.5,0.8]}])
        s.update_persons("c1", [{"track_id": 2, "bbox": [0.3,0.2,0.5,0.8]}])
        assert len(s.snapshot()["persons"]) == 1
    def test_event(self):
        s = Scene3DManager()
        s.add_event({"type": "alert"})
        assert len(s.snapshot()["recent_events"]) == 1
    def test_reset(self):
        s = Scene3DManager()
        s.add_camera(Camera3D("c", "C"))
        s.update_persons("c", [{"track_id": 1, "bbox": [0.3,0.2,0.5,0.8]}])
        s.reset()
        assert len(s.snapshot()["persons"]) == 0
    def test_snapshot(self):
        assert "cameras" in Scene3DManager().snapshot()

class TestTraffic:
    def test_observe(self):
        t = TrafficAnalyzer()
        now = _now()
        for i in range(5): t.observe(now-i, 1, "L", i*10, 100)
        assert t.get_zone_counts()[0].count == 1
    def test_speed(self):
        t = TrafficAnalyzer()
        now = _now()
        for i in range(5): t.observe(now-i, 1, "L", i*50, 100)
        assert t.get_zone_counts()[0].avg_speed_kmh > 0
    def test_congestion(self):
        t = TrafficAnalyzer()
        assert t._classify_congestion(50) == CongestionLevel.FREE_FLOW
        assert t._classify_congestion(2) == CongestionLevel.GRIDLOCK
    def test_multi_zone(self):
        t = TrafficAnalyzer()
        now = _now()
        t.observe(now, 1, "A", 10, 100)
        t.observe(now, 2, "B", 20, 100)
        assert len(t.get_zone_counts()) == 2
    def test_snapshot(self):
        assert "total_vehicles_tracked" in TrafficAnalyzer().snapshot()
    def test_reset(self):
        t = TrafficAnalyzer()
        t.observe(_now(), 1, "A", 10, 100)
        t.reset()
        assert t.snapshot()["total_vehicles_tracked"] == 0

class TestTimeline:
    def test_add_event(self):
        with tempfile.TemporaryDirectory() as d:
            tl = InvestigationTimeline(store_path=os.path.join(d, "c.json"))
            tl.add_event("alert", "Fight", zone="A")
            assert len(tl.query()) == 1
    def test_add_note(self):
        with tempfile.TemporaryDirectory() as d:
            tl = InvestigationTimeline(store_path=os.path.join(d, "c.json"))
            tl.add_note("test", author="a")
            assert len(tl.query(event_type="note")) == 1
    def test_case(self):
        with tempfile.TemporaryDirectory() as d:
            tl = InvestigationTimeline(store_path=os.path.join(d, "c.json"))
            case = tl.create_case("Case 1")
            assert case.title == "Case 1"
            assert len(tl.list_cases()) == 1
    def test_attach_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            tl = InvestigationTimeline(store_path=os.path.join(d, "c.json"))
            ev = tl.add_event("alert", "Fight")
            case = tl.create_case("C")
            tl.attach_to_case(case.case_id, [ev.event_id])
            assert len(tl.export_case(case.case_id)["events"]) == 1
    def test_query_filters(self):
        with tempfile.TemporaryDirectory() as d:
            tl = InvestigationTimeline(store_path=os.path.join(d, "c.json"))
            tl.add_event("alert", "F", zone="A", severity="red")
            tl.add_event("alert", "F", zone="B", severity="yellow")
            assert len(tl.query(zone="A")) == 1
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, "c.json")
            tl = InvestigationTimeline(store_path=sp)
            tl.add_event("alert", "T")
            assert len(InvestigationTimeline(store_path=sp).query()) == 1
    def test_summary(self):
        with tempfile.TemporaryDirectory() as d:
            tl = InvestigationTimeline(store_path=os.path.join(d, "c.json"))
            tl.add_event("alert", "A")
            assert tl.timeline_summary()["total_events"] == 1

class TestNLP:
    def test_simple(self):
        r = NLPQueryEngine().query("show all alerts")
        assert r.parsed["intent"] == "search"
    def test_count(self):
        r = NLPQueryEngine().query("how many fights")
        assert r.parsed["intent"] == "count"
        assert "fight" in r.parsed["rules"]
    def test_rules(self):
        r = NLPQueryEngine().query("gunshots in Zone A")
        assert "gunshot" in r.parsed["rules"]
        assert "A" in r.parsed["zones"]
    def test_severity(self):
        r = NLPQueryEngine().query("show critical alerts")
        assert r.parsed["severity"] == "red"
    def test_time(self):
        r = NLPQueryEngine().query("what happened today")
        assert r.parsed["time_range"] is not None
    def test_camera(self):
        r = NLPQueryEngine().query("alerts on camera CAM-01")
        assert "CAM-01" in r.parsed["cameras"]
    def test_with_data(self):
        e = NLPQueryEngine(alert_store=lambda f: [{"rule":"fight","severity":"red","zone":"A"}])
        assert e.query("show fights").count >= 1
    def test_explain(self):
        r = NLPQueryEngine().query("count intrusions")
        assert "Found" in r.explanation
    def test_suggest(self):
        assert len(NLPQueryEngine().suggest("show")) > 0
    def test_snapshot(self):
        e = NLPQueryEngine()
        e.query("test")
        assert e.snapshot()["history_count"] == 1
