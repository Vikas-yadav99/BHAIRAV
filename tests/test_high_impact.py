"""Tests for high-impact items: SMS gateway, PostgreSQL store, camera manager, migration."""
from __future__ import annotations

import os
import time
import threading
from unittest.mock import patch, MagicMock

import pytest


# ── SMS Gateway Tests ──────────────────────────────────────────────────────

class TestSMSGatewayStub:
    """Test the SMS gateway in stub mode (no credentials)."""

    def test_auto_detects_stub(self):
        from bhairav.phone_gateway import SMSGateway
        gw = SMSGateway()
        assert gw.provider == "stub"
        assert gw.is_real is False

    def test_stub_send_succeeds(self):
        from bhairav.phone_gateway import SMSGateway
        gw = SMSGateway()
        result = gw.send("+919876543210", "Test alert", priority="high")
        assert result["status"] == "sent"
        assert result["provider"] == "stub"
        assert gw.stats()["sent"] == 1

    def test_stub_send_tracks_history(self):
        from bhairav.phone_gateway import SMSGateway
        gw = SMSGateway()
        gw.send("+919876543210", "Alert 1")
        gw.send("+919876543211", "Alert 2")
        recent = gw.get_recent_sent(limit=10)
        assert len(recent) == 2
        assert recent[0]["to"] == "+919876543210"
        assert recent[1]["to"] == "+919876543211"

    def test_receive_parses_sms(self):
        from bhairav.phone_gateway import SMSGateway
        gw = SMSGateway()
        sms = gw.receive("+919876543210", "URGENT crime fight at market")
        assert sms["parsed"]["category"] == "crime"
        assert sms["parsed"]["emergency_level"] >= 3
        assert gw.stats()["received"] == 1


class TestSMSGatewayTwilio:
    """Test the SMS gateway with Twilio provider."""

    def test_auto_detects_twilio(self):
        from bhairav.phone_gateway import SMSGateway
        with patch.dict(os.environ, {
            "TWILIO_ACCOUNT_SID": "AC_test_sid",
            "TWILIO_AUTH_TOKEN": "test_token",
        }, clear=False):
            gw = SMSGateway()
            assert gw.provider == "twilio"
            assert gw.is_real is True

    def test_twilio_send_makes_http_call(self):
        from bhairav.phone_gateway import SMSGateway
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"sid": "SM_test123"}

        with patch.dict(os.environ, {
            "TWILIO_ACCOUNT_SID": "AC_test_sid",
            "TWILIO_AUTH_TOKEN": "test_token",
        }, clear=False):
            with patch("httpx.post", return_value=mock_resp) as mock_post:
                gw = SMSGateway()
                result = gw.send("+919876543210", "Emergency alert", priority="critical")

                assert result["status"] == "sent"
                assert mock_post.called
                # Should retry up to 3 times on failure
                assert gw.stats()["sent"] == 1

    def test_twilio_send_retries_on_failure(self):
        from bhairav.phone_gateway import SMSGateway
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"

        with patch.dict(os.environ, {
            "TWILIO_ACCOUNT_SID": "AC_test_sid",
            "TWILIO_AUTH_TOKEN": "test_token",
        }, clear=False):
            with patch("httpx.post", return_value=mock_resp):
                gw = SMSGateway()
                result = gw.send("+919876543210", "Test")

                assert result["status"] == "failed"
                assert gw.stats()["failed"] == 1


class TestSMSGatewayMSG91:
    """Test MSG91 provider."""

    def test_auto_detects_msg91(self):
        from bhairav.phone_gateway import SMSGateway
        with patch.dict(os.environ, {"MSG91_API_KEY": "test_key"}, clear=False):
            # Remove TWILIO vars if present
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("TWILIO_")}
            env["MSG91_API_KEY"] = "test_key"
            with patch.dict(os.environ, env, clear=True):
                gw = SMSGateway()
                assert gw.provider == "msg91"
                assert gw.is_real is True

    def test_msg91_send_makes_http_call(self):
        from bhairav.phone_gateway import SMSGateway
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"type": "success", "request_id": "req_123"}

        env = {"MSG91_API_KEY": "test_key", "MSG91_FLOW_ID": "flow_123"}
        with patch.dict(os.environ, env, clear=True):
            with patch("httpx.post", return_value=mock_resp) as mock_post:
                gw = SMSGateway()
                result = gw.send("+919876543210", "Test alert")
                assert result["status"] == "sent"
                assert mock_post.called


# ── Camera Manager Tests ──────────────────────────────────────────────────

class TestCameraManager:
    """Test camera management (no real RTSP needed)."""

    def test_add_camera(self):
        from bhairav.camera_manager import CameraManager
        mgr = CameraManager()
        result = mgr.add_camera("TEST-01", "rtsp://fake:554/stream", "Test Cam")
        assert result["ok"] is True
        assert result["camera_id"] == "TEST-01"

        cameras = mgr.list_cameras()
        assert len(cameras) == 1
        assert cameras[0]["id"] == "TEST-01"

    def test_add_duplicate_camera(self):
        from bhairav.camera_manager import CameraManager
        mgr = CameraManager()
        mgr.add_camera("TEST-01", "rtsp://fake:554/stream")
        result = mgr.add_camera("TEST-01", "rtsp://other:554/stream")
        assert result["ok"] is False
        assert "already exists" in result["error"]

    def test_remove_camera(self):
        from bhairav.camera_manager import CameraManager
        mgr = CameraManager()
        mgr.add_camera("TEST-01", "rtsp://fake:554/stream")
        result = mgr.remove_camera("TEST-01")
        assert result["ok"] is True
        assert len(mgr.list_cameras()) == 0

    def test_remove_nonexistent_camera(self):
        from bhairav.camera_manager import CameraManager
        mgr = CameraManager()
        result = mgr.remove_camera("NOPE")
        assert result["ok"] is False

    def test_get_camera_status(self):
        from bhairav.camera_manager import CameraManager
        mgr = CameraManager()
        mgr.add_camera("TEST-01", "rtsp://fake:554/stream")
        cam = mgr.get_camera("TEST-01")
        assert cam is not None
        assert cam["id"] == "TEST-01"
        assert cam["status"] in ("disconnected", "connecting", "error")

    def test_snapshot_returns_none_when_no_frames(self):
        from bhairav.camera_manager import CameraManager
        mgr = CameraManager()
        assert mgr.get_snapshot("NOPE") is None

    def test_singleton(self):
        from bhairav.camera_manager import get_camera_manager
        m1 = get_camera_manager()
        m2 = get_camera_manager()
        assert m1 is m2


# ── PostgreSQL Store Tests ────────────────────────────────────────────────
# These test the schema and logic without requiring a real PostgreSQL instance.

class TestPgIncidentStore:
    """Test PostgreSQL store compilation and schema."""

    def test_module_loads(self):
        from bhairav.backend.pg_incidents import PgIncidentStore, SCHEMA
        assert "incidents" in SCHEMA
        assert "officers" in SCHEMA
        assert "CREATE TABLE IF NOT EXISTS" in SCHEMA

    def test_schema_has_all_columns(self):
        from bhairav.backend.pg_incidents import SCHEMA
        for col in ["id", "category", "emergency_level", "location_lat",
                     "location_lng", "status", "source", "created_at"]:
            assert col in SCHEMA, f"Missing column: {col}"

    def test_schema_has_indexes(self):
        from bhairav.backend.pg_incidents import SCHEMA
        assert "CREATE INDEX" in SCHEMA
        assert "idx_incidents_status" in SCHEMA
        assert "idx_officers_status" in SCHEMA


# ── Migration Script Tests ────────────────────────────────────────────────

class TestMigrationScript:
    """Test the migration script logic."""

    def test_script_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "migrate_to_pg", "scripts/migrate_to_pg.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "migrate_jsonl_to_pg")

    def test_dry_run_no_files(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "migrate_to_pg", "scripts/migrate_to_pg.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Should not crash with empty directory
        mod.migrate_jsonl_to_pg("postgresql://fake", str(tmp_path), dry_run=True)


# ── Server Endpoint Tests ─────────────────────────────────────────────────

class TestCameraEndpoints:
    """Test camera management API endpoints."""

    @pytest.fixture
    def client(self):
        import tempfile
        from pathlib import Path
        from fastapi.testclient import TestClient
        from bhairav.backend.server import create_app, LiveHub, PipelineStats
        from bhairav.backend.evidence import EvidenceStore
        from bhairav.backend.audit import AuditLog
        from bhairav.backend.users import UserStore

        tmpdir = tempfile.mkdtemp(prefix="bhairav_cam_test_")
        evidence_dir = Path(tmpdir) / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        store = EvidenceStore(str(evidence_dir))
        audit = AuditLog(str(evidence_dir / "audit.jsonl"))
        users = UserStore(Path(tmpdir) / "users.json")
        app = create_app(
            store=store, audit=audit, secret="test-cam-secret",
            hub=LiveHub(), stats=PipelineStats(), users=users,
        )
        return TestClient(app, raise_server_exceptions=False)

    def test_list_cameras_empty(self, client):
        resp = client.get("/api/cameras")
        assert resp.status_code == 200
        data = resp.json()
        assert "cameras" in data

    def test_add_camera(self, client):
        resp = client.post("/api/cameras", json={
            "id": "TEST-CAM-01",
            "rtsp_url": "rtsp://fake:554/stream",
            "name": "Test Camera",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    def test_add_camera_missing_fields(self, client):
        resp = client.post("/api/cameras", json={"id": "X"})
        assert resp.status_code == 400

    def test_get_camera(self, client):
        client.post("/api/cameras", json={
            "id": "TEST-CAM-02",
            "rtsp_url": "rtsp://fake:554/stream",
        })
        resp = client.get("/api/cameras/TEST-CAM-02")
        assert resp.status_code == 200
        assert resp.json()["id"] == "TEST-CAM-02"

    def test_get_camera_not_found(self, client):
        resp = client.get("/api/cameras/NOPE")
        assert resp.status_code == 404

    def test_delete_camera(self, client):
        client.post("/api/cameras", json={
            "id": "DEL-CAM",
            "rtsp_url": "rtsp://fake:554/stream",
        })
        resp = client.delete("/api/cameras/DEL-CAM")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_snapshot_no_camera(self, client):
        resp = client.get("/api/cameras/NOPE/snapshot")
        assert resp.status_code == 404
