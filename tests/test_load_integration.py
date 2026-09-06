"""Load testing and integration tests for all BHAIRAV endpoints.

Simulates:
- 50 concurrent cameras sending alerts
- 100 officers sending GPS heartbeats
- 200 public incident reports
- Full incident lifecycle (report → dispatch → respond → resolve)
"""
from __future__ import annotations

import math
import threading
import time

from bhairav.incidents import (
    IncidentStore, DispatchEngine, OfficerStatus, seed_demo_data,
)
from bhairav._security import (
    validate_phone,
    OfficerTokenManager, EndpointRateLimiter, ValidationError,
    validate_lat, validate_lng, validate_category, validate_emergency_level,
)


class TestIncidentStoreLoad:
    """Stress test IncidentStore with concurrent operations."""

    def _make_store(self, tmp_path):
        return IncidentStore(path=str(tmp_path / "load_test"))

    def test_concurrent_create(self, tmp_path):
        store = self._make_store(tmp_path)
        errors = []

        def create_incident(i):
            try:
                store.create_incident(
                    category="crime", emergency_level=2,
                    lat=28.6 + (i * 0.001), lng=77.2 + (i * 0.001),
                    location_name=f"Test {i}", description=f"Incident {i}",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_incident, args=(i,))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(store.list_incidents(limit=200)) == 20

    def test_concurrent_dispatch(self, tmp_path):
        store = self._make_store(tmp_path)
        seed_demo_data(store)
        engine = DispatchEngine(store)
        errors = []

        def dispatch_incident(i):
            try:
                inc = store.create_incident(
                    category="fire", emergency_level=3,
                    lat=28.613 + (i * 0.001), lng=77.209 + (i * 0.001),
                    location_name=f"Fire {i}", description=f"Fire incident {i}",
                )
                engine.dispatch(inc)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=dispatch_incident, args=(i,))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_50_concurrent_cameras(self, tmp_path):
        """Simulate 50 cameras sending alerts simultaneously."""
        store = self._make_store(tmp_path)
        seed_demo_data(store)
        engine = DispatchEngine(store)
        incidents_created = []
        lock = threading.Lock()

        def camera_alert(cam_id):
            inc = store.create_incident(
                category="crime", emergency_level=2,
                lat=28.613 + (cam_id * 0.001), lng=77.209 + (cam_id * 0.001),
                location_name=f"CAM-{cam_id:02d}", description=f"Fight detected on CAM-{cam_id:02d}",
                source="camera",
            )
            assigned = engine.dispatch(inc)
            with lock:
                incidents_created.append((inc.id, len(assigned)))

        threads = [threading.Thread(target=camera_alert, args=(i,))
                   for i in range(50)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - t0

        assert len(incidents_created) == 50
        # All should have completed in < 5 seconds
        assert elapsed < 5.0

    def test_100_officer_heartbeats(self, tmp_path):
        """Simulate 100 officers updating GPS simultaneously."""
        store = self._make_store(tmp_path)
        seed_demo_data(store)
        errors = []

        officers = store.list_officers()

        def heartbeat(officer_idx):
            try:
                off = officers[officer_idx % len(officers)]
                store.update_officer(
                    off.id,
                    location_lat=28.613 + (officer_idx * 0.0001),
                    location_lng=77.209 + (officer_idx * 0.0001),
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=heartbeat, args=(i,))
                   for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_200_reports(self, tmp_path):
        """Simulate 200 public incident reports."""
        store = self._make_store(tmp_path)
        errors = []

        def report(i):
            try:
                store.create_incident(
                    category="medical" if i % 3 == 0 else "crime",
                    emergency_level=(i % 4) + 1,
                    lat=28.61 + (i * 0.0001), lng=77.20 + (i * 0.0001),
                    location_name=f"Location {i}", description=f"Report {i}",
                    source="public",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=report, args=(i,))
                   for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(store.list_incidents(limit=500)) == 200


class TestEndToEndIncidentLifecycle:
    """Full lifecycle: report → dispatch → respond → resolve."""

    def test_complete_lifecycle(self, tmp_path):
        store = IncidentStore(path=str(tmp_path / "lifecycle"))
        seed_demo_data(store)
        engine = DispatchEngine(store)

        # Ensure all officers are available for dispatch
        for off in store.list_officers():
            store.update_officer(off.id, status="available", current_incident=None)

        # Step 1: Create incident
        inc = store.create_incident(
            category="medical", emergency_level=4,
            lat=28.6139, lng=77.2090,
            location_name="Connaught Place", description="Heart attack",
            reporter_name="Rahul", source="public",
        )
        assert inc.status == "reported"

        # Step 2: Dispatch
        assigned = engine.dispatch(inc)
        assert len(assigned) > 0
        inc = store.get_incident(inc.id)
        assert inc.status == "dispatched"

        # Step 3: Officer accepts
        officer_id = assigned[0].id
        engine.accept_incident(officer_id, inc.id)
        off = store.get_officer(officer_id)
        assert off.status == OfficerStatus.EN_ROUTE.value

        # Step 4: Officer on scene
        store.update_officer(officer_id, status=OfficerStatus.ON_SCENE.value)
        store.update_incident(inc.id, status="on_scene", note="Arrived on scene")
        inc = store.get_incident(inc.id)
        assert inc.status == "on_scene"

        # Step 5: Resolve
        store.update_incident(inc.id, status="resolved", resolution_notes="Patient stabilized")
        store.update_officer(officer_id, status=OfficerStatus.AVAILABLE.value, current_incident=None)
        inc = store.get_incident(inc.id)
        assert inc.status == "resolved"

        off = store.get_officer(officer_id)
        assert off.status == OfficerStatus.AVAILABLE.value
        assert off.current_incident is None

    def test_multi_officer_dispatch(self, tmp_path):
        """Level 4 incident dispatches multiple officers."""
        store = IncidentStore(path=str(tmp_path / "multi"))
        seed_demo_data(store)
        engine = DispatchEngine(store)

        # Ensure all officers are available
        for off in store.list_officers():
            store.update_officer(off.id, status="available", current_incident=None)

        inc = store.create_incident(
            category="disaster", emergency_level=4,
            lat=28.6139, lng=77.2090,
            location_name="Disaster zone", description="Building collapse",
        )
        assigned = engine.dispatch(inc)
        assert len(assigned) >= 3  # Level 4 dispatches up to 5


class TestDeduplicationUnderLoad:
    """Verify dedup works correctly under concurrent reports."""

    def test_concurrent_dedup(self, tmp_path):
        from bhairav.city_safety import IncidentDeduplicator

        store = IncidentStore(path=str(tmp_path / "dedup"))
        dedup = IncidentDeduplicator(store)

        # 10 people report the same incident within 100m
        results = []
        lock = threading.Lock()

        def report_same(i):
            result = dedup.check_and_merge(
                category="fire", lat=28.6139, lng=77.2090,
                description="Fire in building", emergency_level=3,
            )
            with lock:
                results.append(result)

        threads = [threading.Thread(target=report_same, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # First report creates new, rest are duplicates
        new_count = sum(1 for r in results if not r.is_duplicate)
        dup_count = sum(1 for r in results if r.is_duplicate)
        assert new_count == 1  # only first creates new
        assert dup_count == 9  # rest are merged

        # Verify crowd reports
        incs = store.list_incidents()
        assert len(incs) == 1
        assert incs[0].crowd_reports >= 2  # auto-verified at 2+


class TestSecurityIntegration:
    """Integration tests for security features in real endpoint flow."""

    def test_rate_limit_triggers(self):
        limiter = EndpointRateLimiter(limit=3, window_sec=60.0)
        for _ in range(3):
            assert limiter.allow("attacker") is True
        assert limiter.allow("attacker") is False

    def test_token_lifecycle(self):
        mgr = OfficerTokenManager()
        token = mgr.issue_token("off-001")
        assert mgr.validate_token(token) == "off-001"
        # Same token is idempotent
        assert mgr.validate_token(token) == "off-001"

    def test_input_validation_catches_bad_data(self):
        """All bad inputs raise ValidationError, never crash."""
        bad_inputs = [
            lambda: validate_lat(999),
            lambda: validate_lng(999),
            lambda: validate_phone(""),
            lambda: validate_category("bad"),
            lambda: validate_emergency_level(99),
            lambda: validate_emergency_level("abc"),
        ]
        for fn in bad_inputs:
            try:
                fn()
                assert False, f"should have raised: {fn}"
            except ValidationError:
                pass  # expected


class TestIVRExpiry:
    """IVR sessions expire after TTL."""

    def test_expired_session(self):
        from bhairav.phone_gateway import IVRSystem
        ivr = IVRSystem()
        session = ivr.start_call("+91-1234567890")
        call_id = session["call_id"]

        # Directly set session age to > 5 min
        ivr._calls[call_id]["started_at"] = time.time() - 400

        result = ivr.process_input(call_id, "1")
        assert "expired" in result.get("error", "").lower() or "Invalid" in result.get("error", "")
