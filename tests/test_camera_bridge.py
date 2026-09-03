"""Tests for Camera-to-Incident Bridge."""
import time
import tempfile
import pytest
from bhairav.incidents import IncidentStore, DispatchEngine
from bhairav.camera_bridge import CameraIncidentBridge, RULE_TO_CATEGORY, SEVERITY_TO_LEVEL


@pytest.fixture
def setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = IncidentStore(tmpdir)
        engine = DispatchEngine(store)
        # Register an officer close to CAM-01 position
        store.register_officer("Raj", "police", "+91-1", 28.614, 77.209)
        yield store, engine


class TestRuleMapping:
    def test_fight_maps_to_crime(self):
        assert RULE_TO_CATEGORY["fight"] == "crime"

    def test_fall_maps_to_medical(self):
        assert RULE_TO_CATEGORY["fall"] == "medical"

    def test_accident_maps_to_road_accident(self):
        assert RULE_TO_CATEGORY["accident"] == "road_accident"

    def test_riot_maps_to_crime(self):
        assert RULE_TO_CATEGORY["riot"] == "crime"

    def test_unknown_rule_maps_to_other(self):
        assert RULE_TO_CATEGORY.get("something_new", "other") == "other"


class TestSeverityMapping:
    def test_yellow_is_level_2(self):
        assert SEVERITY_TO_LEVEL["yellow"] == 2

    def test_red_is_level_4(self):
        assert SEVERITY_TO_LEVEL["red"] == 4

    def test_orange_is_level_3(self):
        assert SEVERITY_TO_LEVEL["orange"] == 3


class TestCameraBridge:
    def test_creates_incident_from_alert(self, setup):
        store, engine = setup
        bridge = CameraIncidentBridge(store, engine)
        alert = {"rule": "fight", "severity": "red", "message": "Fight detected",
                 "camera": "CAM-01", "confidence": 0.9, "timestamp": time.time()}
        result = bridge.on_camera_alert(alert, "CAM-01")
        assert result is not None
        assert result["category"] == "crime"
        assert result["emergency_level"] == 4
        assert result["source"] == "camera"

    def test_auto_dispatches_officers(self, setup):
        store, engine = setup
        bridge = CameraIncidentBridge(store, engine)
        alert = {"rule": "fight", "severity": "red", "message": "Fight",
                 "camera": "CAM-01", "confidence": 0.9, "timestamp": time.time()}
        result = bridge.on_camera_alert(alert, "CAM-01")
        assert len(result["dispatched_officers"]) >= 1

    def test_low_severity_filtered(self, setup):
        store, engine = setup
        bridge = CameraIncidentBridge(store, engine, min_severity="yellow")
        alert = {"rule": "loitering", "severity": "green", "message": "Loitering",
                 "camera": "CAM-01", "confidence": 0.5, "timestamp": time.time()}
        result = bridge.on_camera_alert(alert, "CAM-01")
        assert result is None

    def test_cooldown_prevents_duplicates(self, setup):
        store, engine = setup
        bridge = CameraIncidentBridge(store, engine, cooldown_sec=60)
        now = time.time()
        alert = {"rule": "fight", "severity": "red", "message": "Fight",
                 "camera": "CAM-01", "confidence": 0.9, "timestamp": now}
        r1 = bridge.on_camera_alert(alert, "CAM-01")
        assert r1 is not None
        # Same rule+camera within cooldown → skipped
        alert2 = {"rule": "fight", "severity": "red", "message": "Fight 2",
                  "camera": "CAM-01", "confidence": 0.9, "timestamp": now + 5}
        r2 = bridge.on_camera_alert(alert2, "CAM-01")
        assert r2 is None
        # Different camera → should fire
        alert3 = {"rule": "fight", "severity": "red", "message": "Fight 3",
                  "camera": "CAM-02", "confidence": 0.9, "timestamp": now + 5}
        r3 = bridge.on_camera_alert(alert3, "CAM-02")
        assert r3 is not None

    def test_ai_verified_flag(self, setup):
        store, engine = setup
        bridge = CameraIncidentBridge(store, engine)
        alert = {"rule": "fall", "severity": "orange", "message": "Fall detected",
                 "camera": "CAM-01", "confidence": 0.8, "timestamp": time.time()}
        result = bridge.on_camera_alert(alert, "CAM-01")
        assert result["ai_verified"] is True

    def test_stats_tracking(self, setup):
        store, engine = setup
        bridge = CameraIncidentBridge(store, engine)
        alert = {"rule": "fight", "severity": "red", "message": "Fight",
                 "camera": "CAM-01", "confidence": 0.9, "timestamp": time.time()}
        bridge.on_camera_alert(alert, "CAM-01")
        stats = bridge.stats()
        assert stats["alerts_received"] == 1
        assert stats["incidents_created"] == 1
        assert stats["officers_dispatched"] >= 1

    def test_fire_maps_correctly(self, setup):
        store, engine = setup
        bridge = CameraIncidentBridge(store, engine)
        alert = {"rule": "fall", "severity": "orange", "message": "Person fell",
                 "camera": "CAM-01", "confidence": 0.85, "timestamp": time.time()}
        result = bridge.on_camera_alert(alert, "CAM-01")
        assert result["category"] == "medical"
        assert result["emergency_level"] == 3

    def test_cleanup_cooldowns(self, setup):
        store, engine = setup
        bridge = CameraIncidentBridge(store, engine)
        now = time.time()
        bridge._last_fires = {"fight:CAM-01": now - 7200, "fall:CAM-02": now - 10}
        removed = bridge.cleanup_cooldowns(max_age=3600)
        assert removed == 1
        assert "fight:CAM-01" not in bridge._last_fires
        assert "fall:CAM-02" in bridge._last_fires
