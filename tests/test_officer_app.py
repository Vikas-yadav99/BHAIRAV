"""Tests for BHAIRAV Officer App endpoints."""
import time
import tempfile
import pytest
from bhairav.incidents import (
    IncidentStore, DispatchEngine, OfficerStatus, IncidentStatus,
)


@pytest.fixture
def store_and_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = IncidentStore(tmpdir)
        engine = DispatchEngine(store)
        yield store, engine


class TestOfficerLogin:
    def test_login_by_phone(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("Raj", "police", "+91-123", 28.61, 77.21)
        # Simulate the login logic from the route
        phone = "+91-123"
        found = None
        for o in store.list_officers():
            if o.phone == phone:
                found = o
                break
        assert found is not None
        assert found.id == off.id

    def test_login_by_id(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("Priya", "medical", "+91-456", 28.61, 77.21)
        found = store.get_officer(off.id)
        assert found is not None
        assert found.name == "Priya"

    def test_login_not_found(self, store_and_engine):
        store, engine = store_and_engine
        found = store.get_officer("nonexistent")
        assert found is None


class TestOfficerHeartbeat:
    def test_heartbeat_updates_location(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("Raj", "police", "+91-123", 28.61, 77.21)
        store.update_officer(off.id, location_lat=28.62, location_lng=77.22)
        updated = store.get_officer(off.id)
        assert updated.location_lat == 28.62
        assert updated.location_lng == 77.22

    def test_heartbeat_returns_pending_incidents(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("Raj", "police", "+91-123", 28.614, 77.209)
        inc = store.create_incident("crime", 3, 28.614, 77.209, "X", "Fight")
        engine.dispatch(inc)
        # Find incidents assigned to this officer
        assigned = [i for i in store.list_incidents(status="dispatched")
                    if off.id in i.assigned_officers]
        assert len(assigned) >= 1


class TestOfficerRespond:
    def test_accept_changes_status(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("Raj", "police", "+91-123", 28.614, 77.209)
        inc = store.create_incident("crime", 3, 28.614, 77.209, "X", "Fight")
        engine.dispatch(inc)
        # Accept
        store.update_officer(off.id, status=OfficerStatus.EN_ROUTE.value, current_incident=inc.id)
        store.update_incident(inc.id, status=IncidentStatus.EN_ROUTE.value, note="Raj accepted")
        updated = store.get_officer(off.id)
        assert updated.status == OfficerStatus.EN_ROUTE.value

    def test_on_scene(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("Raj", "police", "+91-123", 28.614, 77.209)
        inc = store.create_incident("crime", 3, 28.614, 77.209, "X", "Fight")
        store.update_officer(off.id, status=OfficerStatus.EN_ROUTE.value, current_incident=inc.id)
        store.update_incident(inc.id, status=IncidentStatus.ON_SCENE.value, note="On scene")
        updated_off = store.get_officer(off.id)
        updated_inc = store.get_incident(inc.id)
        assert updated_off.status == OfficerStatus.EN_ROUTE.value
        assert updated_inc.status == IncidentStatus.ON_SCENE.value

    def test_resolved_frees_officer(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("Raj", "police", "+91-123", 28.614, 77.209)
        inc = store.create_incident("crime", 3, 28.614, 77.209, "X", "Fight")
        store.update_officer(off.id, status=OfficerStatus.ON_SCENE.value, current_incident=inc.id)
        store.update_incident(inc.id, status=IncidentStatus.RESOLVED.value, note="Done")
        store.update_officer(off.id, status=OfficerStatus.AVAILABLE.value, current_incident=None)
        updated_off = store.get_officer(off.id)
        updated_inc = store.get_incident(inc.id)
        assert updated_off.status == OfficerStatus.AVAILABLE.value
        assert updated_off.current_incident is None
        assert updated_inc.status == IncidentStatus.RESOLVED.value


class TestRegisterOfficer:
    def test_register(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("New Guy", "police", "+91-999", 28.61, 77.21, ["patrol"])
        assert off.id
        assert off.name == "New Guy"
        assert off.status == OfficerStatus.AVAILABLE.value
        assert off.specialty == ["patrol"]

    def test_register_stores_persistently(self, store_and_engine):
        store, engine = store_and_engine
        off = store.register_officer("Test", "medical", "+91-000", 28.61, 77.21)
        store2 = IncidentStore(store.path)
        found = store2.get_officer(off.id)
        assert found is not None
        assert found.role == "medical"
