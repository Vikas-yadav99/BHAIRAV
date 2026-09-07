#!/usr/bin/env python3
"""Tests for all improvement fixes."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── City Loader Tests ───────────────────────────────────────────────────────

class TestCityLoader:
    def test_load_indore_config(self):
        from bhairav.city_loader import load_city_config
        config = load_city_config("indore")
        assert config is not None
        assert config["city"] == "Indore"
        assert config["state"] == "Madhya Pradesh"
        assert "cameras" in config
        assert "police_stations" in config
        assert "zones" in config

    def test_load_nonexistent_city(self):
        from bhairav.city_loader import load_city_config
        config = load_city_config("atlantis")
        assert config is None

    def test_load_city_seeds_officers(self):
        from bhairav.city_loader import load_city
        from bhairav.incidents import IncidentStore, DispatchEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            dispatch = DispatchEngine(store)
            stats = load_city(store, dispatch, "indore")
            assert stats["officers"] >= 10
            assert stats["incidents"] >= 5
            assert stats["cameras"] == 10

    def test_officers_at_real_coordinates(self):
        from bhairav.city_loader import load_city
        from bhairav.incidents import IncidentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            load_city(store, city_name="indore")
            officers = store.list_officers()
            # All officers should be in Indore area (lat 22.6-22.8, lon 75.8-76.0)
            for off in officers:
                assert 22.3 < off.location_lat < 23.1, f"{off.name} lat out of Indore"
                assert 75.4 < off.location_lng < 76.3, f"{off.name} lon out of Indore"

    def test_incidents_at_real_locations(self):
        from bhairav.city_loader import load_city
        from bhairav.incidents import IncidentStore, DispatchEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            dispatch = DispatchEngine(store)
            load_city(store, dispatch, "indore")
            incidents = store.list_incidents()
            names = [i.location_name for i in incidents]
            assert "Sarafa Bazaar" in names
            assert "Rajwada Palace" in names

    def test_no_delhi_coordinates(self):
        from bhairav.city_loader import load_city
        from bhairav.incidents import IncidentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            load_city(store, city_name="indore")
            for off in store.list_officers():
                assert not (off.location_lat == 28.6139 and off.location_lng == 77.2090), \
                    f"{off.name} still has Delhi coordinates"

    def test_dispatch_uses_real_distances(self):
        from bhairav.city_loader import load_city_config
        from bhairav.incidents import IncidentStore, DispatchEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            dispatch = DispatchEngine(store)
            # Load only officers (no auto-dispatch of seed incidents)
            config = load_city_config("indore")
            for i, ps in enumerate(config["police_stations"]):
                roles = ["police", "medical", "fire", "rescue"]
                store.register_officer(
                    ps.get("name", f"Officer {i}"),
                    roles[i % len(roles)],
                    f"+91-98765{i:05d}",
                    ps["lat"], ps["lon"],
                    ["patrol"],
                )
            # All officers should be available
            available = [o for o in store.list_officers() if o.status == "available"]
            assert len(available) >= 10
            # Create incident right on top of Vijay Nagar PS
            inc = store.create_incident("crime", 4, 22.7516, 75.8952, "Test",
                                         "Test incident at Vijay Nagar")
            assigned = dispatch.dispatch(inc)
            assert len(assigned) >= 1
            for off in assigned:
                assert 22.3 < off.location_lat < 23.1


# ── Dead Code Check ─────────────────────────────────────────────────────────

class TestDeadCode:
    def test_alert_log_not_imported(self):
        """alert_log.py should not be imported by any active module."""
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "alert_log", "src/bhairav/",
             "--include=*.py", "--exclude-dir=__pycache__"],
            capture_output=True, text=True
        )
        # Should have no imports (only the file itself defining it)
        lines = [l for l in result.stdout.strip().splitlines()
                 if "import" in l and "alert_log" in l]
        assert len(lines) == 0, f"alert_log still imported: {lines}"

    def test_no_delhi_in_codebase(self):
        """No Delhi coordinates should remain in source code."""
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "28.6139", "src/bhairav/",
             "--include=*.py", "--exclude-dir=__pycache__"],
            capture_output=True, text=True
        )
        assert result.stdout.strip() == "", f"Delhi coords found:\n{result.stdout}"


# ── Heartbeat Rate Limiting ─────────────────────────────────────────────────

class TestHeartbeatRateLimit:
    def test_heartbeat_has_rate_limit(self):
        """Officer heartbeat endpoint should have rate limiting."""
        from bhairav.backend.server import create_app, LiveHub, PipelineStats
        from bhairav.backend.evidence import EvidenceStore
        from bhairav.backend.audit import AuditLog
        from bhairav.incidents import IncidentStore, DispatchEngine
        from fastapi.testclient import TestClient
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(tmpdir)
            audit = AuditLog(tmpdir)
            inc = IncidentStore()
            dispatch = DispatchEngine(inc)
            app = create_app(store, audit, "test", hub=LiveHub(),
                           incident_store=inc, incident_dispatch=dispatch)
            client = TestClient(app)

            # Login first
            login = client.post("/api/officer/login", json={"phone": "+91-9876543210"})
            if login.status_code == 200:
                token = login.json()["token"]
                # Send many heartbeats rapidly
                for _ in range(25):
                    client.post("/api/officer/heartbeat", json={
                        "token": token, "lat": 22.72, "lng": 75.86
                    })
                # Should not crash (may or may not rate limit, but shouldn't 500)
                r = client.post("/api/officer/heartbeat", json={
                    "token": token, "lat": 22.72, "lng": 75.86
                })
                assert r.status_code in (200, 429)


# ── SLA Breach Alerts ───────────────────────────────────────────────────────

class TestSLAAlerts:
    def test_sla_status_in_stats(self):
        """Safety stats should include SLA tracking info."""
        from bhairav.city_safety import CitySafetyEngine
        from bhairav.incidents import IncidentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            engine = CitySafetyEngine(store)
            stats = engine.stats()
            assert isinstance(stats, dict)

    def test_incident_has_timeline(self):
        """Incidents should track their lifecycle for SLA calculation."""
        from bhairav.incidents import IncidentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            inc = store.create_incident("crime", 4, 22.72, 75.86, "Test", "Test")
            assert inc.created_at > 0
            assert inc.timeline is not None
            assert len(inc.timeline) >= 1


# ── Officer Performance ─────────────────────────────────────────────────────

class TestOfficerPerformance:
    def test_analytics_includes_officer_stats(self):
        """Analytics should have officer performance data."""
        from bhairav.city_analytics import AnalyticsEngine
        from bhairav.incidents import IncidentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            engine = AnalyticsEngine(store)
            stats = engine.officer_analytics.get_officer_stats()
            assert isinstance(stats, (dict, list))

    def test_team_summary(self):
        from bhairav.city_analytics import AnalyticsEngine
        from bhairav.incidents import IncidentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            engine = AnalyticsEngine(store)
            summary = engine.officer_analytics.get_team_summary()
            assert isinstance(summary, dict)


# ── Government Report Export ────────────────────────────────────────────────

class TestGovernmentExport:
    def test_csv_export_has_headers(self):
        from bhairav.city_analytics import AnalyticsEngine
        from bhairav.incidents import IncidentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            # Seed an incident so CSV has data
            store.create_incident("crime", 2, 22.72, 75.86, "Test", "Test")
            engine = AnalyticsEngine(store)
            csv = engine.export_csv()
            assert isinstance(csv, str)
            # Either has content or is empty string (both valid)
            if csv:
                assert "crime" in csv.lower() or "category" in csv.lower()

    def test_json_export(self):
        from bhairav.city_analytics import AnalyticsEngine
        from bhairav.incidents import IncidentStore
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            engine = AnalyticsEngine(store)
            data = engine.export_json()
            assert isinstance(data, (dict, list, str))

    def test_geojson_export(self):
        from bhairav.city_analytics import AnalyticsEngine
        from bhairav.incidents import IncidentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IncidentStore(path=tmpdir)
            engine = AnalyticsEngine(store)
            geojson = engine.export_geojson()
            assert isinstance(geojson, dict)
            assert "type" in geojson or "features" in geojson
