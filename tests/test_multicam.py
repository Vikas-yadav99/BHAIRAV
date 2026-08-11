"""Phase 8 M2 - multi-camera: hub channel routing, per-camera evidence
stamping, camera search filter, and camera registry on /api/status."""
import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from bhairav.backend.evidence import EvidenceStore, EventRecorder
from bhairav.backend.server import LiveHub
from bhairav.config import load_config
from bhairav.types import Alert, FrameState, Severity


def _alert(rule="fight", ts=10.0, track=1):
    return Alert(rule=rule, zone=None, track_id=track, severity=Severity.RED,
                 message=f"{rule} detected", frame_id=int(ts * 15),
                 timestamp=ts, confidence=0.9)


def _frame(ts, fid, color):
    img = np.full((240, 320, 3), color, np.uint8)
    return FrameState(frame_id=fid, timestamp=ts, tracks=[], frame_w=320,
                      frame_h=240, frame=img)


def _run(coro):
    loop = asyncio.new_event_loop()
    err = []
    def target():
        try:
            loop.run_until_complete(coro)
        except BaseException as exc:  # surface thread failures to the test
            err.append(exc)
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join()
    loop.close()
    if err:
        raise err[0]


def _drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def test_livehub_camera_channels_route_frames_but_share_alerts():
    hub = LiveHub()

    async def scenario():
        cam1 = await hub.subscribe("CAM-01")
        cam2 = await hub.subscribe("CAM-02")
        all_cams = await hub.subscribe()  # legacy "all" subscriber

        hub.publish_frame(frame_id=1, timestamp=0.5, jpeg_b64="A",
                          tracks=[], poses=[], alerts=[], camera="CAM-01")
        hub.publish_frame(frame_id=2, timestamp=0.6, jpeg_b64="B",
                          tracks=[], poses=[], alerts=[], camera="CAM-02")
        hub.publish_alert({"rule": "fight", "camera": "CAM-02", "severity": "red"})
        await asyncio.sleep(0.1)

        c1 = _drain(cam1)
        c2 = _drain(cam2)
        allf = _drain(all_cams)

        # camera-scoped frames: CAM-01 sees only its own frame
        assert [m["frame_id"] for m in c1 if m["type"] == "frame"] == [1]
        assert [m["frame_id"] for m in c2 if m["type"] == "frame"] == [2]
        # "all" subscriber sees frames from every camera, tagged
        assert {m["frame_id"] for m in allf if m["type"] == "frame"} == {1, 2}
        assert all(m["camera"] for m in allf if m["type"] == "frame")
        # alerts broadcast to every subscriber regardless of camera
        assert all(any(m["type"] == "alert" for m in got) for got in (c1, c2, allf))
        assert hub.client_count == 3

        hub.unsubscribe(cam1, "CAM-01")
        assert hub.client_count == 2

    _run(scenario())


def test_livehub_unknown_camera_gets_nothing():
    hub = LiveHub()

    async def scenario():
        q = await hub.subscribe("CAM-09")
        hub.publish_frame(frame_id=5, timestamp=0.5, jpeg_b64="A",
                          tracks=[], poses=[], alerts=[], camera="CAM-01")
        await asyncio.sleep(0.1)
        assert _drain(q) == []

    _run(scenario())


def test_recorder_stamps_camera_on_saved_events(tmp_path):
    store = EvidenceStore(tmp_path, camera="CAM-01", fps=10.0, blur_faces=False)
    rec = EventRecorder(store, pre_sec=1.0, post_sec=1.0, camera="CAM-02")
    for i in range(6):
        rec.observe(_frame(i * 0.1, i, 50))
    eid = rec.on_alert(_alert(ts=0.5), frame=_frame(0.5, 5, 60).frame, state=_frame(0.5, 5, 60))
    rec.flush()
    assert store.get(eid).camera == "CAM-02"
    # default recorder camera falls back to the store camera
    rec2 = EventRecorder(store, pre_sec=1.0, post_sec=1.0)
    for i in range(6):
        rec2.observe(_frame(i * 0.1, i, 50))
    eid2 = rec2.on_alert(_alert(ts=0.5), frame=_frame(0.5, 5, 60).frame, state=_frame(0.5, 5, 60))
    rec2.flush()
    assert store.get(eid2).camera == "CAM-01"


def test_store_search_filters_by_camera(tmp_path):
    store = EvidenceStore(tmp_path, camera="CAM-01", fps=10.0, blur_faces=False)
    for cam, color in (("CAM-01", 10), ("CAM-02", 20)):
        rec = EventRecorder(store, pre_sec=1.0, post_sec=1.0, camera=cam)
        for i in range(6):
            rec.observe(_frame(i * 0.1, i, color))
        rec.on_alert(_alert(ts=0.5), frame=_frame(0.5, 5, color).frame,
                     state=_frame(0.5, 5, color))
        rec.flush()
    assert len(store.search(camera="CAM-01")) == 1
    assert len(store.search(camera="CAM-02")) == 1
    assert store.search(camera="CAM-01")[0].camera == "CAM-01"
    assert store.search(camera="NOPE") == []


def test_config_parses_cameras(tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "cameras": [
            {"id": "GATE", "name": "Main Gate", "source": "rtsp://cam/gate"},
            {"id": "LOT", "source": "blob"},
        ],
    }), encoding="utf-8")
    cfg = load_config(p)
    assert [c.id for c in cfg.cameras] == ["GATE", "LOT"]
    assert cfg.cameras[0].name == "Main Gate"
    assert cfg.cameras[0].detector == "auto"
    assert cfg.cameras[1].name == "LOT"

    p2 = tmp_path / "d.yaml"
    p2.write_text("detector: blob\n", encoding="utf-8")
    assert load_config(p2).cameras == []


def test_status_reports_cameras_and_evidence_filters_by_camera(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from bhairav.backend.audit import AuditLog
    from bhairav.backend.server import PipelineStats, create_app
    from bhairav.backend.users import UserStore

    store = EvidenceStore(tmp_path / "evidence", camera="CAM-01", fps=10.0,
                          blur_faces=False)
    for cam, color in (("CAM-01", 10), ("CAM-02", 20)):
        rec = EventRecorder(store, pre_sec=1.0, post_sec=1.0, camera=cam)
        for i in range(6):
            rec.observe(_frame(i * 0.1, i, color))
        rec.on_alert(_alert(ts=0.5), frame=_frame(0.5, 5, color).frame,
                     state=_frame(0.5, 5, color))
        rec.flush()

    audit = AuditLog(tmp_path / "audit.jsonl")
    users = UserStore(tmp_path / "users.json")
    stats = PipelineStats()
    app = create_app(store, audit, secret="test-secret", users=users,
                     stats=stats, cameras=[{"id": "CAM-01", "name": "Plaza"},
                                           {"id": "CAM-02", "name": "Room"}])
    client = TestClient(app)

    r = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    token = r.json()["token"]
    h = {"Authorization": f"Bearer {token}"}

    st = client.get("/api/status", headers=h).json()
    assert [c["id"] for c in st["cameras"]] == ["CAM-01", "CAM-02"]

    ev = client.get("/api/evidence?camera=CAM-02", headers=h).json()
    assert ev["total"] == 1
    assert ev["events"][0]["camera"] == "CAM-02"
    assert client.get("/api/evidence?camera=CAM-01", headers=h).json()["total"] == 1
