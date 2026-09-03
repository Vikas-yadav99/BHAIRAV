"""Tests for BHAIRAV Incident Reporting & Dispatch System."""
import time
import tempfile
import pytest
from bhairav.incidents import (
    IncidentStore, DispatchEngine, seed_demo_data,
    haversine_distance, IncidentStatus, OfficerStatus,
    IncidentCategory, EmergencyLevel,
)


class TestHaversineDistance:
    def test_same_point(self):
        assert haversine_distance(28.6139, 77.2090, 28.6139, 77.2090) == 0.0

    def test_known_distance(self):
        # ~156m between these two Delhi points
        d = haversine_distance(28.6139, 77.2090, 28.6150, 77.2100)
        assert 100 < d < 200

    def test_symmetric(self):
        d1 = haversine_distance(28.6139, 77.2090, 28.6150, 77.2100)
        d2 = haversine_distance(28.6150, 77.2100, 28.6139, 77.2090)
        assert abs(d1 - d2) < 0.01


class TestIncidentStore:
    def test_create_incident(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            inc = store.create_incident(
                category="medical", emergency_level=4,
                lat=28.6139, lng=77.2090,
                location_name="Connaught Place",
                description="Heart attack",
                reporter_name="Rahul", source="public",
            )
            assert inc.id
            assert inc.category == "medical"
            assert inc.status == IncidentStatus.REPORTED.value
            assert inc.emergency_level == 4

    def test_get_incident(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            inc = store.create_incident("fire", 3, 28.61, 77.21, "Test", "Desc")
            found = store.get_incident(inc.id)
            assert found is not None
            assert found.id == inc.id

    def test_list_incidents_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            store.create_incident("medical", 1, 28.61, 77.21, "A", "Desc")
            store.create_incident("fire", 2, 28.61, 77.21, "B", "Desc")
            all_inc = store.list_incidents()
            assert len(all_inc) == 2
            medical = store.list_incidents(category="medical")
            assert len(medical) == 1

    def test_update_incident(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            inc = store.create_incident("crime", 1, 28.61, 77.21, "X", "Desc")
            updated = store.update_incident(inc.id, status="dispatched", note="test")
            assert updated.status == "dispatched"
            assert len(updated.timeline) == 2

    def test_crowd_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            inc = store.create_incident("crime", 2, 28.61, 77.21, "X", "Desc")
            assert inc.crowd_reports == 1
            assert inc.status == IncidentStatus.REPORTED.value

            updated = store.add_crowd_report(inc.id)
            assert updated.crowd_reports == 2
            # Auto-verified after 2 reports
            assert updated.status == IncidentStatus.VERIFIED.value

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            inc = store.create_incident("fire", 3, 28.61, 77.21, "Y", "Desc")
            # Reload from disk
            store2 = IncidentStore(tmpdir)
            found = store2.get_incident(inc.id)
            assert found is not None
            assert found.category == "fire"


class TestDispatchEngine:
    def test_dispatch_medical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            store.register_officer("Dr. Rao", "medical", "+91-123", 28.614, 77.209)
            store.register_officer("Insp. Singh", "police", "+91-456", 28.614, 77.209)
            inc = store.create_incident("medical", 4, 28.614, 77.209, "X", "Heart attack")
            engine = DispatchEngine(store)
            assigned = engine.dispatch(inc)
            assert len(assigned) == 1
            assert assigned[0].role == "medical"
            assert inc.status == IncidentStatus.DISPATCHED.value

    def test_dispatch_nearest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            store.register_officer("Near", "police", "+91-1", 28.6140, 77.2090)
            store.register_officer("Far", "police", "+91-2", 28.6200, 77.2150)
            inc = store.create_incident("crime", 3, 28.614, 77.209, "X", "Fight")
            engine = DispatchEngine(store)
            assigned = engine.dispatch(inc)
            assert len(assigned) >= 1
            assert assigned[0].name == "Near"


class TestSeedDemo:
    def test_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(tmpdir)
            result = seed_demo_data(store)
            assert result["officers"] == 10
            assert result["incidents"] == 5
            stats = store.get_stats()
            assert stats["total_officers"] == 10
            assert stats["total_incidents"] == 5
