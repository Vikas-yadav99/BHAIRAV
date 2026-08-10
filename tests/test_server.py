"""Integration tests for the Phase 3-5 FastAPI server + WebSocket live stream."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from bhairav.backend.audit import AuditLog
from bhairav.backend.evidence import EvidenceStore, EventRecorder
from bhairav.backend.server import LiveHub, PipelineStats, create_app
from bhairav.backend.users import UserStore
from bhairav.types import Alert, Severity


@pytest.fixture()
def app_ctx(tmp_path):
    store = EvidenceStore(tmp_path / "evidence", camera="CAM-01", fps=10.0,
                          blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    hub = LiveHub()
    users = UserStore(tmp_path / "users.json")
    stats = PipelineStats()
    app = create_app(store, audit, secret="test-secret", hub=hub, users=users,
                     stats=stats)
    client = TestClient(app)
    # seed one evidence event (with frames so a clip exists)
    import numpy as np
    rec = EventRecorder(store, pre_sec=0.5, post_sec=0.5)
    from bhairav.types import FrameState
    for i in range(6):
        img = np.full((240, 320, 3), 40 + i, np.uint8)
        st = FrameState(frame_id=i, timestamp=i * 0.1, tracks=[], frame_w=320,
                        frame_h=240, frame=img)
        rec.observe(st)
    rec.on_alert(Alert(rule="fight", zone=None, track_id=3, severity=Severity.RED,
                       message="FIGHT detected", frame_id=10, timestamp=1.0))
    eid = rec.flush()[0]
    return {"client": client, "store": store, "audit": audit, "hub": hub,
            "event_id": eid}


def _login(client, username="admin", password="admin123"):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _login_rec(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_public(app_ctx):
    r = app_ctx["client"].get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_login_issues_token(app_ctx):
    rec = _login_rec(app_ctx["client"], "viewer", "viewer123")
    assert rec["role"] == "viewer"
    assert rec["username"] == "viewer"
    assert rec["token"]


def test_login_rejects_bad_credentials(app_ctx):
    c = app_ctx["client"]
    assert c.post("/auth/login",
                  json={"username": "admin", "password": "wrongpw"}).status_code == 401
    assert c.post("/auth/login",
                  json={"username": "ghost", "password": "whatever123"}).status_code == 401
    # failed attempts are audited
    assert any(e["action"] == "login_failed" for e in app_ctx["audit"].read())


def test_evidence_requires_auth(app_ctx):
    client = app_ctx["client"]
    assert client.get("/api/evidence").status_code == 401  # no token
    assert client.get("/api/evidence", headers={"Authorization": "Bearer garbage"
                                                 }).status_code == 401  # bad token


def test_evidence_search_and_get(app_ctx):
    client = app_ctx["client"]
    tok = _login(client, "viewer", "viewer123")
    headers = {"Authorization": f"Bearer {tok}"}
    r = client.get("/api/evidence", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["events"][0]["rule"] == "fight"
    eid = body["events"][0]["event_id"]
    r2 = client.get(f"/api/evidence/{eid}", headers=headers)
    assert r2.status_code == 200
    assert r2.json()["event_id"] == eid
    # snapshot
    r3 = client.get(f"/api/evidence/{eid}/snapshot", headers=headers)
    assert r3.status_code == 200
    assert r3.headers["content-type"] == "image/jpeg"


def test_download_requires_operator(app_ctx):
    client = app_ctx["client"]
    eid = app_ctx["event_id"]
    viewer = _login(client, "viewer", "viewer123")
    h_viewer = {"Authorization": f"Bearer {viewer}"}
    assert client.get(f"/api/evidence/{eid}/clip",
                      headers=h_viewer).status_code == 403
    operator = _login(client, "operator", "operator123")
    r = client.get(f"/api/evidence/{eid}/clip",
                   headers={"Authorization": f"Bearer {operator}"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"


def test_delete_requires_admin(app_ctx):
    client = app_ctx["client"]
    eid = app_ctx["event_id"]
    analyst = _login(client, "analyst", "analyst123")
    r = client.delete(f"/api/evidence/{eid}",
                      headers={"Authorization": f"Bearer {analyst}"})
    assert r.status_code == 403
    admin = _login(client, "admin")
    r2 = client.delete(f"/api/evidence/{eid}",
                       headers={"Authorization": f"Bearer {admin}"})
    assert r2.status_code == 200
    # audited
    entries = app_ctx["audit"].query(action="delete_evidence")
    assert len(entries) == 1


def test_audit_requires_analyst(app_ctx):
    client = app_ctx["client"]
    viewer = _login(client, "viewer", "viewer123")
    assert client.get("/api/audit",
                      headers={"Authorization": f"Bearer {viewer}"}).status_code == 403
    analyst = _login(client, "analyst", "analyst123")
    r = client.get("/api/audit", headers={"Authorization": f"Bearer {analyst}"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_ws_stream_receives_frames(app_ctx):
    client = app_ctx["client"]
    tok = _login(client, "viewer", "viewer123")
    with client.websocket_connect(f"/ws/stream?token={tok}") as ws:
        # publish a frame from the sync side
        app_ctx["hub"].publish_frame(frame_id=1, timestamp=0.5, jpeg_b64="AAAA",
                                     tracks=[{"id": 1}], poses=[], alerts=[])
        msg = ws.receive_json()
        assert msg["type"] == "frame"
        assert msg["frame_id"] == 1
        assert msg["jpeg"] == "AAAA"


def test_ws_stream_rejects_bad_token(app_ctx):
    client = app_ctx["client"]
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/stream?token=garbage") as ws:
            ws.receive_json()


def test_dashboard_served_at_root(app_ctx):
    client = app_ctx["client"]
    r = client.get("/")
    assert r.status_code == 307 or r.status_code == 200
    # follow the redirect to the dashboard html
    r2 = client.get("/dashboard/")
    assert r2.status_code == 200
    assert "BHAIRAV" in r2.text
    assert "ReactDOM.createRoot" in r2.text  # real React SPA served


def test_expire_endpoint(app_ctx):
    client = app_ctx["client"]
    eid = app_ctx["event_id"]
    # backdate the seeded event so retention removes it
    import os
    import time
    old = time.time() - 40 * 86400
    meta = app_ctx["store"].root / eid / "metadata.json"
    os.utime(meta, (old, old))
    admin = _login(client, "admin", "admin123")
    r = client.post("/api/evidence/expire", json={"max_age_days": 30},
                    headers={"Authorization": f"Bearer {admin}"})
    assert r.status_code == 200
    assert r.json()["expired"] == 1


# ---- Phase 5: users, status, export, workflow -----------------------------
def _auth(client, username, password):
    rec = _login_rec(client, username, password)
    return {"Authorization": f"Bearer {rec['token']}"}


def test_status_endpoint(app_ctx):
    client = app_ctx["client"]
    h = _auth(client, "viewer", "viewer123")
    r = client.get("/api/status", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["evidence"]["total"] == 1
    assert body["evidence"]["by_rule"]["fight"] == 1
    assert body["audit"]["ok"] is True
    assert body["users"] == 4
    assert "pipeline" in body and "clients" in body


def test_user_admin_lifecycle(app_ctx):
    client = app_ctx["client"]
    admin = _auth(client, "admin", "admin123")
    # create
    r = client.post("/api/users", json={"username": "carol", "password": "secret99",
                                        "role": "operator"}, headers=admin)
    assert r.status_code == 200 and r.json()["role"] == "operator"
    # duplicate -> 400
    assert client.post("/api/users", json={"username": "carol", "password": "secret99",
                                            "role": "operator"}, headers=admin).status_code == 400
    # new user can log in with their own role
    rec = _login_rec(client, "carol", "secret99")
    assert rec["role"] == "operator"
    # list hides secrets
    names = {u["username"] for u in client.get("/api/users", headers=admin).json()["users"]}
    assert "carol" in names
    # non-admin cannot manage users
    viewer = _auth(client, "viewer", "viewer123")
    assert client.get("/api/users", headers=viewer).status_code == 403
    assert client.post("/api/users", json={"username": "x", "password": "secret99",
                                            "role": "viewer"}, headers=viewer).status_code == 403
    # cannot delete self
    assert client.delete("/api/users/admin", headers=admin).status_code == 400
    # lock then delete
    assert client.post("/api/users/carol/lock", json={"locked": True},
                       headers=admin).status_code == 200
    assert client.post("/auth/login", json={"username": "carol", "password": "secret99"}
                      ).status_code == 401
    assert client.delete("/api/users/carol", headers=admin).json() == {"deleted": "carol"}
    # password reset
    r = client.post("/api/users/viewer/password", json={"password": "brandnew99"},
                    headers=admin)
    assert r.status_code == 200
    assert client.post("/auth/login", json={"username": "viewer", "password": "brandnew99"}
                      ).status_code == 200
    # all admin actions audited
    actions = {e["action"] for e in app_ctx["audit"].read()}
    assert {"create_user", "delete_user", "lock_user", "reset_password"} <= actions


def test_evidence_workflow_status_and_notes(app_ctx):
    client = app_ctx["client"]
    eid = app_ctx["event_id"]
    operator = _auth(client, "operator", "operator123")
    analyst = _auth(client, "analyst", "analyst123")
    viewer = _auth(client, "viewer", "viewer123")
    # viewer cannot change status (download perm is the gate)
    assert client.post(f"/api/evidence/{eid}/status", json={"status": "acknowledged"},
                       headers=viewer).status_code == 403
    # operator can acknowledge but NOT resolve
    r = client.post(f"/api/evidence/{eid}/status", json={"status": "acknowledged"},
                    headers=operator)
    assert r.status_code == 200 and r.json()["status"] == "acknowledged"
    assert client.post(f"/api/evidence/{eid}/status", json={"status": "resolved"},
                       headers=operator).status_code == 403
    # analyst resolves + notes
    r = client.post(f"/api/evidence/{eid}/status", json={"status": "resolved"},
                    headers=analyst)
    assert r.status_code == 200 and r.json()["status"] == "resolved"
    r = client.post(f"/api/evidence/{eid}/notes", json={"text": "check the gate camera"},
                    headers=analyst)
    assert r.status_code == 200
    assert r.json()["notes"][0]["text"] == "check the gate camera"
    # invalid status -> 400
    assert client.post(f"/api/evidence/{eid}/status", json={"status": "bogus"},
                       headers=analyst).status_code == 400
    # persistence through the store
    rec = app_ctx["store"].get(eid)
    assert rec.status == "resolved" and len(rec.notes) == 1


def test_evidence_export_analyst_only(app_ctx):
    client = app_ctx["client"]
    eid = app_ctx["event_id"]
    operator = _auth(client, "operator", "operator123")
    viewer = _auth(client, "viewer", "viewer123")
    assert client.get("/api/evidence/export", headers=operator).status_code == 403
    assert client.get("/api/evidence/export", headers=viewer).status_code == 403
    analyst = _auth(client, "analyst", "analyst123")
    r = client.get("/api/evidence/export", headers=analyst)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    import io
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert any(n.endswith("manifest.json") for n in names)
    assert any(eid in n for n in names)
    # audited
    assert any(e["action"] == "export_evidence" for e in app_ctx["audit"].read())


def test_export_respects_filters(app_ctx):
    client = app_ctx["client"]
    analyst = _auth(client, "analyst", "analyst123")
    r = client.get("/api/evidence/export?rule=fight", headers=analyst)
    assert r.status_code == 200
    import io
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(r.content))
    manifest = json.loads(z.read("manifest.json"))
    assert manifest["count"] == 1
    assert manifest["events"][0]["rule"] == "fight"


def test_locked_user_token_revoked(app_ctx):
    """An existing token stops working the moment the account is locked."""
    client = app_ctx["client"]
    rec = _login_rec(client, "operator", "operator123")
    tok = rec["token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert client.get("/api/evidence", headers=h).status_code == 200
    admin = _auth(client, "admin", "admin123")
    assert client.post("/api/users/operator/lock", json={"locked": True},
                       headers=admin).status_code == 200
    # the outstanding token is now rejected, not just future logins
    assert client.get("/api/evidence", headers=h).status_code == 401
    # unlock restores it (user is active again)
    assert client.post("/api/users/operator/lock", json={"locked": False},
                       headers=admin).status_code == 200
    assert client.get("/api/evidence", headers=h).status_code == 200


def test_create_user_response_has_no_hash(app_ctx):
    client = app_ctx["client"]
    admin = _auth(client, "admin", "admin123")
    r = client.post("/api/users", json={"username": "dana", "password": "secret99",
                                        "role": "viewer"}, headers=admin)
    assert r.status_code == 200
    body = r.json()
    for secret_key in ("hash", "salt", "iterations"):
        assert secret_key not in body


def test_webhook_notify_posts_red_alert():
    """webhook_notify delivers a JSON POST to the configured endpoint."""
    import http.server
    import threading
    from bhairav.backend.server import webhook_notify

    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, int(self.headers.get("Content-Length", 0)),
                             self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.handle_request, daemon=True)
    t.start()
    alert = {"rule": "fight", "severity": "red", "message": "FIGHT"}
    webhook_notify(f"http://127.0.0.1:{port}/hook", alert)
    t.join(timeout=6)
    srv.server_close()
    assert received, "webhook never delivered"
    path, _, payload = received[0]
    assert path == "/hook"
    data = json.loads(payload)
    assert data["type"] == "bhairav_alert"
    assert data["alert"]["severity"] == "red"


def test_webhook_notify_no_url_is_noop():
    from bhairav.backend.server import webhook_notify
    webhook_notify(None, {"rule": "x"})  # must not raise
    webhook_notify("", {"rule": "x"})
