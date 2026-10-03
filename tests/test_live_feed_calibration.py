"""Tests for live-feed and calibration server endpoints."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SECRET = "test-lf-secret"


@pytest.fixture(scope="module")
def app_and_token():
    tmpdir = Path(tempfile.mkdtemp(prefix="bhairav_lf_test_"))
    evidence_dir = tmpdir / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    from bhairav.backend.server import create_app, LiveHub, PipelineStats
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.users import UserStore
    from bhairav.backend.rbac import issue_token

    store = EvidenceStore(str(evidence_dir))
    audit = AuditLog(str(evidence_dir / "audit.jsonl"))
    users = UserStore(tmpdir / "users.json")
    app = create_app(
        store=store, audit=audit, secret=SECRET,
        hub=LiveHub(), stats=PipelineStats(), users=users,
    )
    admin_token = issue_token(SECRET, "admin", "admin")
    viewer_token = issue_token(SECRET, "viewer", "viewer")
    return app, admin_token, viewer_token


@pytest.fixture()
def client(app_and_token):
    app, admin_token, _ = app_and_token
    c = TestClient(app, raise_server_exceptions=False)
    c.headers["Authorization"] = f"Bearer {admin_token}"
    return c


@pytest.fixture()
def viewer_client(app_and_token):
    app, _, viewer_token = app_and_token
    c = TestClient(app, raise_server_exceptions=False)
    c.headers["Authorization"] = f"Bearer {viewer_token}"
    return c


# ---- Live Feed Tests --------------------------------------------------------

class TestLiveFeedAPI:
    def test_list_streams_empty(self, client):
        r = client.get("/api/live-feed/streams")
        assert r.status_code == 200
        data = r.json()
        assert "streams" in data
        assert isinstance(data["streams"], list)

    def test_list_streams_viewer(self, viewer_client):
        r = viewer_client.get("/api/live-feed/streams")
        assert r.status_code == 200

    def test_start_stream(self, client):
        r = client.post("/api/live-feed/start", json={
            "id": "test-stream-1",
            "url": "https://example.com/test.m3u8",
            "name": "Test Stream",
            "fps": 1,
            "min_persons": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "started"
        assert data["stream"]["id"] == "test-stream-1"

    def test_stop_stream(self, client):
        client.post("/api/live-feed/start", json={
            "id": "test-stream-2",
            "url": "https://example.com/test.m3u8",
        })
        r = client.post("/api/live-feed/stop/test-stream-2")
        assert r.status_code == 200
        assert r.json()["status"] == "stopped"

    def test_list_streams_after_start(self, client):
        client.post("/api/live-feed/start", json={
            "id": "test-stream-3",
            "url": "https://example.com/test.m3u8",
        })
        r = client.get("/api/live-feed/streams")
        streams = r.json()["streams"]
        ids = [s["id"] for s in streams]
        assert "test-stream-3" in ids
        client.post("/api/live-feed/stop/test-stream-3")

    def test_alerts_empty(self, client):
        r = client.get("/api/live-feed/alerts")
        assert r.status_code == 200
        assert "alerts" in r.json()

    def test_start_requires_admin(self, viewer_client):
        r = viewer_client.post("/api/live-feed/start", json={
            "url": "https://example.com/test.m3u8",
        })
        assert r.status_code in (401, 403)

    def test_stop_requires_admin(self, viewer_client):
        r = viewer_client.post("/api/live-feed/stop/nonexistent")
        assert r.status_code in (401, 403)


# ---- Calibration Tests ------------------------------------------------------

class TestCalibrationAPI:
    def test_get_calibration_empty(self, client):
        r = client.get("/api/calibrate/CAM-01")
        assert r.status_code == 200
        data = r.json()
        assert data["camera_id"] == "CAM-01"
        assert data["calibrated"] is False

    def test_add_calibration_points(self, client):
        points = [
            {"pixel_x": 100, "pixel_y": 100, "lat": 22.7185, "lng": 75.8500, "label": "tl"},
            {"pixel_x": 1820, "pixel_y": 100, "lat": 22.7185, "lng": 75.8540, "label": "tr"},
            {"pixel_x": 100, "pixel_y": 980, "lat": 22.7175, "lng": 75.8500, "label": "bl"},
            {"pixel_x": 1820, "pixel_y": 980, "lat": 22.7175, "lng": 75.8540, "label": "br"},
        ]
        for pt in points:
            r = client.post("/api/calibrate/CAM-TEST-1/point", json=pt)
            assert r.status_code == 200

        r = client.get("/api/calibrate/CAM-TEST-1")
        data = r.json()
        assert data["calibrated"] is True
        assert len(data["points"]) == 4

    def test_pixel_to_gps(self, client):
        r = client.post("/api/calibrate/CAM-TEST-1/pixel-to-gps", json={
            "pixel_x": 960, "pixel_y": 540,
        })
        assert r.status_code == 200
        data = r.json()
        assert "lat" in data
        assert "lng" in data
        assert 22.7175 <= data["lat"] <= 22.7185
        assert 75.8500 <= data["lng"] <= 75.8540

    def test_gps_to_pixel(self, client):
        r = client.post("/api/calibrate/CAM-TEST-1/gps-to-pixel", json={
            "lat": 22.7180, "lng": 75.8520,
        })
        assert r.status_code == 200
        data = r.json()
        assert "pixel_x" in data
        assert "pixel_y" in data

    def test_calibration_not_found(self, client):
        r = client.post("/api/calibrate/CAM-UNKNOWN/pixel-to-gps", json={
            "pixel_x": 100, "pixel_y": 100,
        })
        assert r.status_code == 404

    def test_reset_calibration(self, client):
        r = client.post("/api/calibrate/CAM-TEST-1/reset")
        assert r.status_code == 200
        assert r.json()["status"] == "reset"
        r = client.get("/api/calibrate/CAM-TEST-1")
        assert r.json()["calibrated"] is False

    def test_calibrate_requires_admin(self, viewer_client):
        r = viewer_client.post("/api/calibrate/CAM-01/point", json={
            "pixel_x": 100, "pixel_y": 100, "lat": 22.71, "lng": 75.85,
        })
        assert r.status_code in (401, 403)

    def test_reset_requires_admin(self, viewer_client):
        r = viewer_client.post("/api/calibrate/CAM-01/reset")
        assert r.status_code in (401, 403)
