"""Phase 9 M5 - police read-only role + public blurred monitor tests.

Covers the police role (read-only, no management), the token-gated
/api/public/* endpoints, and the privacy sanitizer used for the public feed.
"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from bhairav.backend.rbac import (PERM_AUDIT, PERM_EVIDENCE_DOWNLOAD,
                                  PERM_EVIDENCE_EXPORT, PERM_EVIDENCE_READ,
                                  PERM_STREAM, PERM_USERS, authorize)


def test_police_role_is_read_only():
    assert authorize("police", PERM_STREAM)
    assert authorize("police", PERM_EVIDENCE_READ)
    assert authorize("police", PERM_EVIDENCE_DOWNLOAD)
    assert not authorize("police", PERM_EVIDENCE_EXPORT)
    assert not authorize("police", PERM_AUDIT)
    assert not authorize("police", PERM_USERS)


def _build_app(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.server import LiveHub, PipelineStats, create_app
    from bhairav.backend.users import UserStore

    store = EvidenceStore(tmp_path / "evidence", camera="CAM-01", fps=10.0,
                          blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    hub = LiveHub()
    app = create_app(store, audit, secret="test-secret", hub=hub,
                     users=UserStore(tmp_path / "users.json"),
                     stats=PipelineStats(),
                     cameras=[{"id": "CAM-01", "name": "Entrance"},
                              {"id": "CAM-02", "name": "Parking"}],
                     public_token="pub-secret-123")
    return TestClient(app), hub


def _login(client, username, password):
    r = client.post("/auth/login", json={"username": username,
                                        "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def test_police_user_login_and_read_only_api(tmp_path):
    client, _ = _build_app(tmp_path)
    h = _login(client, "police", "police123")
    assert client.get("/api/evidence", headers=h).status_code == 200
    assert client.get("/api/status", headers=h).status_code == 200
    assert client.post("/api/evidence/export", headers=h).status_code in (403, 405)
    assert client.get("/api/users", headers=h).status_code == 403
    assert client.get("/api/audit", headers=h).status_code == 403
    assert client.post("/api/reid/subjects/P-1/rename", headers=h,
                       json={"name": "x"}).status_code == 403


def test_public_info_is_open(tmp_path):
    client, _ = _build_app(tmp_path)
    r = client.get("/api/public/info")
    assert r.status_code == 200
    d = r.json()
    assert d["blurred"] is True and d["streaming"] is True
    assert d["cameras"] == ["CAM-01", "CAM-02"]


def test_public_stream_rejects_bad_token(tmp_path):
    pytest.importorskip("fastapi")
    from starlette.websockets import WebSocketDisconnect
    client, _ = _build_app(tmp_path)
    with pytest.raises(WebSocketDisconnect) as ex:
        with client.websocket_connect("/api/public/stream?token=wrong"):
            pass
    assert ex.value.code == 4401
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/public/stream"):
            pass


def test_public_stream_forwards_sanitized_frames(tmp_path):
    pytest.importorskip("fastapi")
    client, hub = _build_app(tmp_path)
    with client.websocket_connect(
            "/api/public/stream?token=pub-secret-123") as ws:
        hub.publish_public_frame(frame_id=42, timestamp=1.5,
                                 jpeg_b64="AAAA", camera="__public__")
        msg = ws.receive_json()
        assert msg["type"] == "frame"
        assert msg["jpeg"] == "AAAA" and msg["frame_id"] == 42
        assert not msg.get("tracks") and not msg.get("poses")
        assert not msg.get("alerts")


def test_public_frames_never_reach_all_camera_clients(tmp_path):
    """Sanitized frames go only to the __public__ channel, never to the
    authenticated 'all cameras' subscribers."""
    pytest.importorskip("fastapi")
    client, hub = _build_app(tmp_path)
    token = _login(client, "viewer", "viewer123")["Authorization"].split()[-1]
    with client.websocket_connect(f"/ws/stream?token={token}") as ws:
        hub.publish_public_frame(frame_id=1, timestamp=1.0, jpeg_b64="PUB",
                                 camera="__public__")
        hub.publish_frame(frame_id=2, timestamp=1.1, jpeg_b64="REAL",
                          tracks=[], poses=[], alerts=[], camera=None)
        seen = []
        import time
        deadline = time.time() + 2
        while time.time() < deadline and len(seen) < 2:
            try:
                seen.append(ws.receive_json())
            except Exception:
                break
        jpegs = [m.get("jpeg") for m in seen if m.get("type") == "frame"]
        assert "REAL" in jpegs
        assert "PUB" not in jpegs


def test_public_stream_disabled_without_token(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.server import LiveHub, PipelineStats, create_app
    from bhairav.backend.users import UserStore

    app = create_app(EvidenceStore(tmp_path / "ev", camera="CAM-01", fps=10.0,
                                   blur_faces=False),
                     AuditLog(tmp_path / "audit.jsonl"), secret="s",
                     hub=LiveHub(), users=UserStore(tmp_path / "u.json"),
                     stats=PipelineStats())  # public_token defaults to None
    c = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/api/public/stream?token=anything"):
            pass


def test_public_sanitizer_blurs_heads_and_downscales():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from serve import _public_sanitize

    h, w = 360, 640
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:, :, :] = (80, 90, 100)                      # background
    frame[60:140, 220:420, :] = (30, 40, 180)           # torso (red)
    # high-frequency noise in the head zone -> blur must flatten it
    frame[20:60, 220:420, :] = np.random.default_rng(0).integers(
        0, 255, (40, 200, 3), dtype=np.uint8)

    class _T:
        label = "person"
        bbox = [200, 20, 440, 140]

    b64 = _public_sanitize(frame, [_T()])
    assert b64 and b64.startswith("/9j/")  # JPEG

    import cv2
    jpg = np.frombuffer(base64.b64decode(b64), np.uint8)
    out = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    assert out is not None
    assert out.shape[1] < 300 and out.shape[0] < 200  # downscaled 0.35
    # head zone (scaled coords) vs the unblurred neck zone below it: the
    # sanitizer must meaningfully collapse the head's variance
    sx = lambda v: int(v * 0.35)  # noqa: E731
    hy2 = sx(20) + max(1, int((sx(140) - sx(20)) * 0.25))
    head = out[sx(20):hy2, sx(200):sx(420)]
    neck = out[hy2:sx(60), sx(200):sx(420)]
    assert head.size > 0 and neck.size > 0
    hv, nv = float(head.var()), float(neck.var())
    assert hv < nv / 3.0, (hv, nv)  # blurred head << unblurred noise
