#!/usr/bin/env python3
"""FastAPI integration tests using TestClient."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_and_token():
    from bhairav.backend.server import create_app, LiveHub, PipelineStats
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.rbac import issue_token
    from bhairav.backend.users import UserStore

    tmpdir = tempfile.mkdtemp(prefix="bhairav_test_")
    evidence_dir = Path(tmpdir) / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    store = EvidenceStore(str(evidence_dir))
    audit = AuditLog(str(evidence_dir / "audit.jsonl"))
    secret = "test-integration-secret"

    # UserStore auto-seeds admin/admin123, analyst, operator, police, viewer
    users = UserStore(Path(tmpdir) / "users.json")

    app = create_app(
        store=store, audit=audit, secret=secret,
        hub=LiveHub(), stats=PipelineStats(),
        users=users,
    )

    # Use default seeded credentials: admin/admin123, viewer/viewer1
    admin_token = issue_token(secret, "admin", "admin")
    viewer_token = issue_token(secret, "viewer", "viewer")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    viewer_headers = {"Authorization": f"Bearer {viewer_token}"}

    yield app, admin_headers, viewer_headers, secret, users


@pytest.fixture
def client(app_and_token):
    return TestClient(app_and_token[0])

@pytest.fixture
def admin(app_and_token):
    return app_and_token[1]

@pytest.fixture
def viewer(app_and_token):
    return app_and_token[2]


class TestHealth:
    def test_health_no_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_no_auth(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True


class TestAuth:
    def test_login_success(self, client):
        resp = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["role"] == "admin"

    def test_login_wrong_password(self, client):
        resp = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post("/auth/login", json={"username": "nobody", "password": "x"})
        assert resp.status_code == 401

    def test_protected_no_token(self, client):
        assert client.get("/api/status").status_code == 401

    def test_protected_bad_token(self, client):
        resp = client.get("/api/status", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 401


class TestRBAC:
    def test_viewer_can_read_status(self, client, viewer):
        assert client.get("/api/status", headers=viewer).status_code == 200

    def test_viewer_cannot_list_users(self, client, viewer):
        assert client.get("/api/users", headers=viewer).status_code == 403

    def test_admin_can_list_users(self, client, admin):
        resp = client.get("/api/users", headers=admin)
        assert resp.status_code == 200
        assert "users" in resp.json()

    def test_viewer_cannot_delete_evidence(self, client, viewer):
        assert client.delete("/api/evidence/fake-id", headers=viewer).status_code == 403


class TestEvidence:
    def test_search_empty(self, client, admin):
        resp = client.get("/api/evidence", headers=admin)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_get_nonexistent(self, client, admin):
        assert client.get("/api/evidence/no-such-id", headers=admin).status_code == 404


class TestTrajectory:
    def test_tracked_persons_empty(self, client, admin):
        resp = client.get("/api/persons/tracked", headers=admin)
        assert resp.status_code == 200
        assert resp.json()["persons"] == []

    def test_predict_nonexistent_returns_404(self, client, admin):
        assert client.get("/api/persons/nonexistent/predict", headers=admin).status_code == 404

    def test_trajectory_nonexistent(self, client, admin):
        resp = client.get("/api/persons/nonexistent/trajectory", headers=admin)
        assert resp.status_code == 200
        assert resp.json()["person"] is None

    def test_predict_zones_empty(self, client, admin):
        resp = client.get("/api/persons/predict-zones", headers=admin)
        assert resp.status_code == 200
        assert resp.json()["total_tracked"] == 0


class TestFaceMonitor:
    def test_face_monitor_status(self, client, admin):
        resp = client.get("/api/face/live-monitor/status", headers=admin)
        assert resp.status_code == 200
        assert "enabled" in resp.json()


class TestBodyLimits:
    def test_large_body_rejected(self, client, admin):
        huge = {"data": "x" * (3 * 1024 * 1024)}
        assert client.post("/auth/login", json=huge, headers=admin).status_code == 413


class TestAudit:
    def test_audit_log(self, client, admin):
        resp = client.get("/api/audit", headers=admin)
        assert resp.status_code == 200
        assert "entries" in resp.json()


class TestMetrics:
    def test_metrics_no_registry_returns_503(self, client, admin):
        resp = client.get("/metrics", headers=admin)
        assert resp.status_code == 503


class TestRateLimiting:
    def test_login_rate_limit(self, client):
        for _ in range(10):
            client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        resp = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 429
