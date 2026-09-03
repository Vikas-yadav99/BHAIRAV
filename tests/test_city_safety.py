"""Tests for City Safety — Dedup, Notifications, GPS, Metrics, Resolution."""
import sys, os, time, math, json, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from bhairav.incidents import IncidentStore, DispatchEngine, OfficerStatus
from bhairav.city_safety import (
    IncidentDeduplicator, NotificationManager, GPSTracker,
    ResponseMetrics, ResolutionManager, CitySafetyEngine,
    _haversine_m,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = IncidentStore(path=os.path.join(d, "incidents"))
    yield store
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def engine(tmp_store):
    return DispatchEngine(tmp_store)


@pytest.fixture
def seed_officers(tmp_store):
    """Seed 5 officers around Delhi center."""
    officers = []
    positions = [
        ("Raj", "police", "+91-111", 28.6139, 77.2090),
        ("Priya", "medical", "+91-222", 28.6150, 77.2100),
        ("Amit", "fire", "+91-333", 28.6120, 77.2080),
        ("Sunita", "police", "+91-444", 28.6160, 77.2110),
        ("Vikram", "rescue", "+91-555", 28.6110, 77.2070),
    ]
    for name, role, phone, lat, lng in positions:
        off = tmp_store.register_officer(name, role, phone, lat, lng, [role])
        officers.append(off)
    return officers


# ══════════════════════════════════════════════════════════════════════════════
# 1. INCIDENT DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class TestDeduplication:

    def test_new_incident_created(self, tmp_store):
        dedup = IncidentDeduplicator(tmp_store)
        result = dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight near station", emergency_level=2,
        )
        assert not result.is_duplicate
        assert result.new_incident is not None
        assert result.merged_into is None

    def test_duplicate_detected_same_location(self, tmp_store):
        dedup = IncidentDeduplicator(tmp_store)
        # First report
        r1 = dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight", emergency_level=2,
        )
        # Second report 50m away (within 200m radius)
        r2 = dedup.check_and_merge(
            category="crime", lat=28.6142, lng=77.2093,
            description="Fight at station", emergency_level=2,
        )
        assert r2.is_duplicate
        assert r2.merged_into == r1.new_incident["id"]
        assert r2.crowd_incremented

    def test_different_category_not_duplicate(self, tmp_store):
        dedup = IncidentDeduplicator(tmp_store)
        dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight", emergency_level=2,
        )
        r2 = dedup.check_and_merge(
            category="medical", lat=28.6139, lng=77.2090,
            description="Heart attack", emergency_level=4,
        )
        assert not r2.is_duplicate  # different category = not duplicate

    def test_far_location_not_duplicate(self, tmp_store):
        dedup = IncidentDeduplicator(tmp_store)
        dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight", emergency_level=2,
        )
        # 500m away (beyond 200m radius)
        r2 = dedup.check_and_merge(
            category="crime", lat=28.6200, lng=77.2150,
            description="Fight", emergency_level=2,
        )
        assert not r2.is_duplicate

    def test_crowd_auto_verify_at_3(self, tmp_store):
        dedup = IncidentDeduplicator(tmp_store)
        r1 = dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight", emergency_level=2,
        )
        inc_id = r1.new_incident["id"]

        # 2 more crowd reports (total = 3)
        for lat_offset in [0.0001, 0.0002]:
            dedup.check_and_merge(
                category="crime", lat=28.6139 + lat_offset, lng=77.2090,
                emergency_level=2,
            )

        inc = tmp_store.get_incident(inc_id)
        assert inc.crowd_reports == 3
        assert inc.status == "verified"  # auto-verified at 3 reports

    def test_old_incident_not_duplicate(self, tmp_store):
        dedup = IncidentDeduplicator(tmp_store, time_window=1.0)  # 1 second window
        dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight", emergency_level=2,
        )
        time.sleep(1.5)
        r2 = dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight", emergency_level=2,
        )
        assert not r2.is_duplicate

    def test_stats_tracking(self, tmp_store):
        dedup = IncidentDeduplicator(tmp_store)
        dedup.check_and_merge(category="crime", lat=28.6139, lng=77.2090, emergency_level=2)
        dedup.check_and_merge(category="crime", lat=28.6140, lng=77.2091, emergency_level=2)
        dedup.check_and_merge(category="fire", lat=28.6139, lng=77.2090, emergency_level=3)
        s = dedup.stats()
        assert s["reports_received"] == 3
        assert s["duplicates_found"] == 1
        assert s["new_incidents"] == 2

    def test_resolved_not_duplicate(self, tmp_store):
        dedup = IncidentDeduplicator(tmp_store)
        r1 = dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight", emergency_level=2,
        )
        # Resolve it
        tmp_store.update_incident(r1.new_incident["id"], status="resolved")
        # New report should not be duplicate of resolved incident
        r2 = dedup.check_and_merge(
            category="crime", lat=28.6139, lng=77.2090,
            description="Fight again", emergency_level=2,
        )
        assert not r2.is_duplicate


# ══════════════════════════════════════════════════════════════════════════════
# 2. NOTIFICATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class TestNotifications:

    def test_notify_officer(self):
        nm = NotificationManager()
        nm.notify_officer("off-1", "Alert", "Fight detected", priority="high",
                         channels=["websocket"])
        recent = nm.get_recent(limit=5)
        assert len(recent) == 1
        assert recent[0]["title"] == "Alert"
        assert recent[0]["recipient_id"] == "off-1"
        assert recent[0]["priority"] == "high"

    def test_notify_operators_broadcast(self):
        nm = NotificationManager()
        nm.notify_operators("New incident", "Fire at market", priority="critical")
        recent = nm.get_recent()
        assert len(recent) == 1
        assert recent[0]["recipient_id"] == "all_operators"
        assert recent[0]["recipient_type"] == "operator"

    def test_sms_gateway_stub(self):
        nm = NotificationManager(sms_gateway_url="https://api.twilio.com/send")
        nm.send_sms("+91-9876543210", "Emergency at Main St", priority="high")
        recent = nm.get_recent()
        assert len(recent) == 1
        assert recent[0]["channel"] == "sms"
        assert recent[0]["sent_at"] is not None

    def test_sms_no_gateway(self):
        nm = NotificationManager()  # no gateway
        nm.send_sms("+91-9876543210", "Test")
        recent = nm.get_recent()
        assert len(recent) == 1
        assert recent[0]["sent_at"] is not None  # still marks as "sent" (stub)

    def test_mark_read(self):
        nm = NotificationManager()
        nm.notify_officer("off-1", "Test", "Body", channels=["websocket"])
        nid = nm.get_recent()[0]["id"]
        assert len(nm.get_recent(unread_only=True)) == 1
        nm.mark_read(nid)
        assert len(nm.get_recent(unread_only=True)) == 0

    def test_stats(self):
        nm = NotificationManager()
        nm.notify_officer("off-1", "A", "B", priority="high", channels=["websocket"])
        nm.send_sms("+91-123", "C", priority="medium")
        s = nm.stats()
        assert s["total"] == 2
        assert s["by_channel"]["websocket"] == 1
        assert s["by_channel"]["sms"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. GPS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class TestGPSTracker:

    def test_update_position(self, tmp_store, seed_officers):
        tracker = GPSTracker(tmp_store)
        off = seed_officers[0]
        coord = tracker.update_position(off.id, 28.6140, 77.2091, speed=5.0)
        assert coord is not None
        assert coord.lat == 28.6140
        assert coord.speed == 5.0

    def test_invalid_officer(self, tmp_store):
        tracker = GPSTracker(tmp_store)
        coord = tracker.update_position("nonexistent", 28.6140, 77.2091)
        assert coord is None

    def test_trajectory(self, tmp_store, seed_officers):
        tracker = GPSTracker(tmp_store)
        off = seed_officers[0]
        for i in range(5):
            tracker.update_position(off.id, 28.6139 + i * 0.001, 77.2090)
        traj = tracker.get_trajectory(off.id)
        assert len(traj) == 5
        assert traj[0]["lat"] < traj[-1]["lat"]

    def test_all_positions(self, tmp_store, seed_officers):
        tracker = GPSTracker(tmp_store)
        for off in seed_officers:
            tracker.update_position(off.id, off.location_lat, off.location_lng)
        positions = tracker.get_all_positions()
        assert len(positions) == 5
        for oid, pos in positions.items():
            assert "lat" in pos
            assert "lng" in pos

    def test_distance_to_incident(self, tmp_store, seed_officers):
        tracker = GPSTracker(tmp_store)
        off = seed_officers[0]
        tracker.update_position(off.id, 28.6139, 77.2090)
        dist = tracker.get_distance_to_incident(off.id, 28.6140, 77.2091)
        assert dist is not None
        assert 0 < dist < 200  # within 200m

    def test_find_nearest_by_gps(self, tmp_store, seed_officers):
        tracker = GPSTracker(tmp_store)
        for off in seed_officers:
            tracker.update_position(off.id, off.location_lat, off.location_lng)
        nearest = tracker.find_nearest_by_gps(28.6139, 77.2090, radius_km=10)
        assert len(nearest) >= 1
        assert nearest[0]["distance_km"] < nearest[-1]["distance_km"]

    def test_stale_detection(self, tmp_store, seed_officers):
        tracker = GPSTracker(tmp_store)
        off = seed_officers[0]
        # Set old position
        coord = tracker.update_position(off.id, 28.6139, 77.2090)
        # Fake old timestamp
        tracker._positions[off.id][-1].timestamp = time.time() - 300
        tracker._last_update[off.id] = time.time() - 300
        positions = tracker.get_all_positions()
        assert positions[off.id]["stale"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 4. RESPONSE METRICS
# ══════════════════════════════════════════════════════════════════════════════

class TestResponseMetrics:

    def test_record_resolution(self, tmp_store, engine, seed_officers):
        metrics = ResponseMetrics(tmp_store)
        inc = tmp_store.create_incident(
            "crime", 3, 28.6139, 77.2090, "Station", "Fight", source="public",
        )
        engine.dispatch(inc)
        time.sleep(0.1)
        result = metrics.record_resolution(inc.id)
        assert result is not None
        assert result["total_time_sec"] > 0
        assert result["category"] == "crime"
        assert result["emergency_level"] == 3

    def test_sla_tracking(self, tmp_store, engine, seed_officers):
        metrics = ResponseMetrics(tmp_store)
        inc = tmp_store.create_incident(
            "crime", 4, 28.6139, 77.2090, "Station", "Fight", source="public",
        )
        engine.dispatch(inc)
        # Simulate quick ack
        tmp_store.update_incident(
            inc.id, status="dispatched",
            note="dispatched",
        )
        inc.timeline[-1]["time"] = inc.created_at + 10  # 10s response
        inc.timeline.append({"status": "acknowledged", "time": inc.created_at + 10})
        result = metrics.record_resolution(inc.id)
        assert result["sla_met"] is True  # 10s < 180s SLA for level 4

    def test_summary_empty(self, tmp_store):
        metrics = ResponseMetrics(tmp_store)
        s = metrics.get_summary(hours=24)
        assert s["total_resolved"] == 0
        assert s["avg_response_time"] is None

    def test_live_dashboard(self, tmp_store, seed_officers):
        metrics = ResponseMetrics(tmp_store)
        tmp_store.create_incident(
            "crime", 4, 28.6139, 77.2090, "Station", "Fight", source="public",
        )
        tmp_store.create_incident(
            "medical", 2, 28.6150, 77.2100, "Park", "Injury", source="public",
        )
        dash = metrics.get_live_dashboard()
        assert dash["active_incidents"] == 2
        assert dash["critical"] == 1
        assert dash["medium"] == 1
        assert dash["total_officers"] == 5


# ══════════════════════════════════════════════════════════════════════════════
# 5. RESOLUTION WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

class TestResolutionWorkflow:

    def test_upload_proof(self, tmp_store, engine, seed_officers):
        resolver = ResolutionManager(tmp_store)
        inc = tmp_store.create_incident(
            "crime", 3, 28.6139, 77.2090, "Station", "Fight", source="public",
        )
        engine.dispatch(inc)
        off = seed_officers[0]
        result = resolver.upload_proof(
            inc.id, off.id,
            notes="Area secured, no injuries",
            photos=["base64_photo_1"],
        )
        assert "proof" in result
        assert result["proof"]["notes"] == "Area secured, no injuries"

    def test_resolve_incident(self, tmp_store, engine, seed_officers):
        metrics = ResponseMetrics(tmp_store)
        resolver = ResolutionManager(tmp_store, metrics)
        inc = tmp_store.create_incident(
            "crime", 3, 28.6139, 77.2090, "Station", "Fight", source="public",
        )
        engine.dispatch(inc)
        off = seed_officers[0]
        result = resolver.resolve_incident(
            inc.id, off.id,
            resolution_notes="Fight broke up, parties separated",
            photos=["photo_1"],
        )
        assert "incident" in result
        assert result["incident"]["status"] == "resolved"
        assert "response_metrics" in result

    def test_officer_freed_after_resolve(self, tmp_store, engine, seed_officers):
        resolver = ResolutionManager(tmp_store)
        inc = tmp_store.create_incident(
            "crime", 3, 28.6139, 77.2090, "Station", "Fight", source="public",
        )
        engine.dispatch(inc)
        off = seed_officers[0]
        resolver.resolve_incident(inc.id, off.id, "Done")
        updated_off = tmp_store.get_officer(off.id)
        assert updated_off.status == "available"
        assert updated_off.current_incident is None

    def test_unresolved_summary(self, tmp_store):
        resolver = ResolutionManager(tmp_store)
        tmp_store.create_incident("crime", 4, 28.6139, 77.2090, "S1", "F1", source="public")
        tmp_store.create_incident("medical", 2, 28.6150, 77.2100, "S2", "F2", source="public")
        summary = resolver.get_unresolved_summary()
        assert summary["total_active"] == 2
        assert summary["escalated"] == 1

    def test_proof_not_found(self, tmp_store):
        resolver = ResolutionManager(tmp_store)
        result = resolver.upload_proof("bad_id", "bad_officer", "notes")
        assert "error" in result


# ══════════════════════════════════════════════════════════════════════════════
# 6. UNIFIED CITY SAFETY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TestCitySafetyEngine:

    def test_full_lifecycle(self, tmp_store, seed_officers):
        engine = DispatchEngine(tmp_store)
        safety = CitySafetyEngine(tmp_store, dispatch_engine=engine)

        # Report incident
        result = safety.report_incident(
            category="crime", lat=28.6139, lng=77.2090,
            emergency_level=3, description="Fight at station",
            reporter_name="Rahul", source="public",
        )
        assert not result["duplicate"]
        inc_id = result["incident"]["id"]

        # Second report (duplicate)
        result2 = safety.report_incident(
            category="crime", lat=28.6140, lng=77.2091,
            emergency_level=3, description="Fight at station",
            reporter_name="Neha", source="public",
        )
        assert result2["duplicate"]
        assert result2["merged_into"] == inc_id

        # GPS update
        off = seed_officers[0]
        safety.officer_update_gps(off.id, 28.6145, 77.2095, speed=30.0)

        # Resolve
        resolve_result = safety.resolve(inc_id, off.id, "Situation handled")
        assert resolve_result["incident"]["status"] == "resolved"

    def test_dashboard_data(self, tmp_store, seed_officers):
        engine = DispatchEngine(tmp_store)
        safety = CitySafetyEngine(tmp_store, dispatch_engine=engine)
        safety.report_incident(
            category="fire", lat=28.6139, lng=77.2090,
            emergency_level=4, description="Building fire",
        )
        dash = safety.get_operator_dashboard()
        assert "incidents" in dash
        assert "officers" in dash
        assert "metrics" in dash
        assert "notifications" in dash
        assert "unresolved_summary" in dash

    def test_stats(self, tmp_store, seed_officers):
        engine = DispatchEngine(tmp_store)
        safety = CitySafetyEngine(tmp_store, dispatch_engine=engine)
        s = safety.stats()
        assert "dedup" in s
        assert "notifications" in s
        assert "gps" in s
        assert "store" in s


# ══════════════════════════════════════════════════════════════════════════════
# 7. HAVERSINE DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

class TestHaversine:

    def test_same_point(self):
        assert _haversine_m(28.6139, 77.2090, 28.6139, 77.2090) == 0.0

    def test_known_distance(self):
        # ~111km between 1 degree latitude at equator
        d = _haversine_m(0.0, 0.0, 1.0, 0.0)
        assert 110000 < d < 112000

    def test_short_distance(self):
        d = _haversine_m(28.6139, 77.2090, 28.6140, 77.2091)
        assert 0 < d < 200
