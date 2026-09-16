"""Tests for medium priority items: WebSocket auth, concurrent dispatch,
SLA tracking, officer leaderboard."""
from __future__ import annotations

import concurrent.futures
import random
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_and_token():
    from bhairav.backend.server import create_app, LiveHub, PipelineStats
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.rbac import issue_token
    from bhairav.backend.users import UserStore

    tmpdir = tempfile.mkdtemp(prefix="bhairav_med_")
    ed = Path(tmpdir) / "evidence"
    ed.mkdir(exist_ok=True)
    store = EvidenceStore(str(ed))
    audit = AuditLog(str(ed / "audit.jsonl"))
    secret = "test-med-secret"
    users = UserStore(Path(tmpdir) / "users.json")

    app = create_app(
        store=store, audit=audit, secret=secret,
        hub=LiveHub(), stats=PipelineStats(), users=users,
    )
    admin_token = issue_token(secret, "admin", "admin")
    viewer_token = issue_token(secret, "viewer", "viewer")
    return app, secret, admin_token, viewer_token


@pytest.fixture
def client(app_and_token):
    return TestClient(app_and_token[0], raise_server_exceptions=False)


# -- WebSocket Auth Tests ----------------------------------------------------

class TestWebSocketAuth:

    def test_ws_incidents_no_token(self, app_and_token):
        c = TestClient(app_and_token[0], raise_server_exceptions=False)
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/incidents"):
                pass

    def test_ws_incidents_bad_token(self, app_and_token):
        c = TestClient(app_and_token[0], raise_server_exceptions=False)
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/incidents?token=garbage"):
                pass

    def test_ws_incidents_admin_connects(self, app_and_token):
        _, _, admin_token, _ = app_and_token
        c = TestClient(app_and_token[0], raise_server_exceptions=False)
        with c.websocket_connect(f"/ws/incidents?token={admin_token}") as ws:
            data = ws.receive_text()
            assert data

    def test_ws_field_no_token(self, app_and_token):
        c = TestClient(app_and_token[0], raise_server_exceptions=False)
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/field"):
                pass

    def test_ws_field_admin_connects(self, app_and_token):
        _, _, admin_token, _ = app_and_token
        c = TestClient(app_and_token[0], raise_server_exceptions=False)
        with c.websocket_connect(f"/ws/field?token={admin_token}") as ws:
            pass

    def test_ws_analytics_no_token(self, app_and_token):
        c = TestClient(app_and_token[0], raise_server_exceptions=False)
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/analytics"):
                pass

    def test_ws_analytics_admin_connects(self, app_and_token):
        _, _, admin_token, _ = app_and_token
        c = TestClient(app_and_token[0], raise_server_exceptions=False)
        with c.websocket_connect(f"/ws/analytics?token={admin_token}") as ws:
            pass


# -- Concurrent Dispatch Load Test -------------------------------------------

class TestConcurrentDispatch:

    def _create(self, client, lat, lng, cat="crime", lvl=3):
        return client.post("/api/incidents", json={
            "category": cat, "emergency_level": lvl,
            "lat": lat, "lng": lng,
            "location_name": f"Load {lat:.4f}",
            "description": "Concurrent test",
        })

    def test_10_concurrent(self, client):
        def go(i):
            return self._create(client,
                                22.71 + random.uniform(-0.01, 0.01),
                                75.85 + random.uniform(-0.01, 0.01))
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            results = [f.result() for f in
                       concurrent.futures.as_completed([ex.submit(go, i) for i in range(10)])]
        ok = [r for r in results if r.status_code in (200, 201)]
        assert len(ok) >= 5

    def test_no_500_under_load(self, client):
        def go(i):
            return self._create(client,
                                22.71 + random.uniform(-0.02, 0.02),
                                75.85 + random.uniform(-0.02, 0.02),
                                cat="fire", lvl=4).status_code
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            codes = [f.result() for f in
                     concurrent.futures.as_completed([ex.submit(go, i) for i in range(20)])]
        assert 500 not in codes

    def test_stats_under_load(self, client):
        def create_incident(i):
            return self._create(client,
                                22.71 + random.uniform(-0.01, 0.01),
                                75.85 + random.uniform(-0.01, 0.01))
        def get_stats():
            return client.get("/api/incidents/stats")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = []
            for i in range(10):
                futs.append(ex.submit(create_incident, i))
                futs.append(ex.submit(get_stats))
            results = [f.result() for f in concurrent.futures.as_completed(futs)]
        stats_ok = [r for r in results if hasattr(r, "json") and r.status_code == 200]
        assert len(stats_ok) >= 3


# -- SLA Tests ---------------------------------------------------------------

class TestSLAEndpoints:

    def test_sla_breaches(self, client):
        resp = client.get("/api/safety/sla-breaches")
        assert resp.status_code == 200
        data = resp.json()
        assert "breaches" in data

    def test_government_report(self, client):
        resp = client.get("/api/safety/government-report?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert "period" in data
        assert "total_incidents" in data

    def test_sla_has_countdown_fields(self, client):
        resp = client.get("/api/safety/sla-breaches")
        data = resp.json()
        for b in data.get("breaches", [])[:3]:
            assert "remaining_sec" in b
            assert "sla_target_sec" in b
            assert "breached" in b
            assert "percent_used" in b


# -- Officer Performance Tests -----------------------------------------------

class TestOfficerLeaderboard:

    def test_officer_stats(self, client):
        resp = client.get("/api/analytics/officers")
        assert resp.status_code == 200

    def test_analytics_root(self, client):
        resp = client.get("/api/analytics")
        assert resp.status_code == 200

    def test_patterns(self, client):
        resp = client.get("/api/analytics/patterns")
        assert resp.status_code == 200

    def test_incident_has_timeline(self, client):
        resp = client.get("/api/incidents")
        assert resp.status_code == 200
        data = resp.json()
        incidents = data if isinstance(data, list) else data.get("incidents", [])
        for inc in incidents[:3]:
            assert "created_at" in inc

    def test_city_safety_dashboard(self, client):
        resp = client.get("/api/safety/dashboard")
        assert resp.status_code == 200
