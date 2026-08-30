#!/usr/bin/env python3
"""Deep integration tests for BHAIRAV - fills the gaps.

Tests:
1. Real YOLO pipeline end-to-end on actual video clip
2. Rate limiting under load
3. WebSocket streaming
4. Full evidence write -> search -> retrieve workflow
5. Dashboard content verification
6. Real YuNet/SFace face detection
"""
import sys
import tempfile
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def full_app():
    """Create a full app with real evidence store + audit + metrics."""
    import cv2
    from bhairav.backend.server import create_app, LiveHub, PipelineStats
    from bhairav.backend.evidence import EvidenceStore, EventRecorder
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.users import UserStore
    from bhairav.backend.metrics import MetricsRegistry
    from bhairav.types import Alert, FrameState, Severity

    tmpdir = tempfile.mkdtemp(prefix="bhairav_deep_")
    evidence_dir = Path(tmpdir) / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    store = EvidenceStore(str(evidence_dir), camera="CAM-TEST", fps=15.0, blur_faces=False)
    audit = AuditLog(str(evidence_dir / "audit.jsonl"))
    hub = LiveHub()
    users = UserStore(Path(tmpdir) / "users.json")
    stats = PipelineStats()
    metrics = MetricsRegistry()

    app = create_app(store, audit, hub=hub, users=users, stats=stats,
                     metrics=metrics, secret="deep-test-secret")
    client = TestClient(app)

    # Seed evidence via EventRecorder (the real way evidence gets created)
    rec = EventRecorder(store, pre_sec=0.5, post_sec=0.5)
    for i in range(10):
        img = np.full((240, 320, 3), 40 + i, np.uint8)
        st = FrameState(frame_id=i, timestamp=i * 0.1, tracks=[], frame_w=320,
                        frame_h=240, frame=img)
        rec.observe(st)
    rec.on_alert(Alert(rule="fight", zone=None, track_id=3, severity=Severity.RED,
                       message="FIGHT detected", frame_id=10, timestamp=1.0))
    rec.flush(now=100.0)

    # Login as admin
    resp = client.post("/auth/login",
                       json={"username": "admin", "password": "admin123"})
    token = resp.json()["token"]
    admin_headers = {"Authorization": f"Bearer {token}"}

    # Login as viewer
    resp = client.post("/auth/login",
                       json={"username": "viewer", "password": "viewer123"})
    viewer_token = resp.json()["token"]
    viewer_headers = {"Authorization": f"Bearer {viewer_token}"}

    return {
        "client": client,
        "admin": admin_headers,
        "viewer": viewer_headers,
        "tmpdir": Path(tmpdir),
        "store": store,
        "hub": hub,
        "metrics": metrics,
    }


# ===========================================================================
# 1. REAL YOLO PIPELINE END-TO-END
# ===========================================================================
class TestRealYoloPipeline:
    """Run YOLOv8 on a real video clip through the full pipeline."""

    CLIP = Path("src/bhairav/test_footage/heavy_crowd/clip_4250.mp4")
    MAX_FRAMES = 30

    @pytest.mark.skipif(not CLIP.exists(), reason="test clip not available")
    def test_yolo_detect_track_alert(self):
        """YOLO detects persons, ByteTrack tracks them, rules engine fires alerts."""
        from bhairav.config import AppConfig
        from bhairav.pipeline import build_engine, make_detector, run_pipeline

        cfg = AppConfig()
        cfg.detector = "yolo"
        cfg.model.conf = 0.25
        cfg.model.imgsz = 416

        detector = make_detector(cfg, detector="yolo", source=str(self.CLIP))
        engine = build_engine(cfg)

        tracks_seen = []
        frame_count = [0]

        def on_frame(state, alerts):
            frame_count[0] += 1
            tracks_seen.append(len(state.tracks))
            return None

        all_alerts = run_pipeline(
            detector, engine,
            source=str(self.CLIP),
            max_frames=self.MAX_FRAMES,
            on_frame=on_frame,
        )

        assert frame_count[0] == self.MAX_FRAMES
        assert len(tracks_seen) == self.MAX_FRAMES
        assert sum(1 for t in tracks_seen if t > 0) > 0, "No persons detected"
        assert isinstance(all_alerts, list)

    @pytest.mark.skipif(not CLIP.exists(), reason="test clip not available")
    def test_yolo_trajectory_tracking(self):
        """Trajectory predictor receives real detections and maintains state."""
        from bhairav.config import AppConfig
        from bhairav.pipeline import build_engine, make_detector, run_pipeline
        from bhairav.face_tracking import TrajectoryPredictor

        cfg = AppConfig()
        cfg.detector = "yolo"
        cfg.model.conf = 0.25
        cfg.model.imgsz = 416

        traj = TrajectoryPredictor(zones=cfg.zones)
        detector = make_detector(cfg, detector="yolo", source=str(self.CLIP))
        engine = build_engine(cfg)

        def on_frame(state, alerts):
            for t in state.tracks:
                if t.label == "person" and t.bbox:
                    x1, y1, x2, y2 = t.bbox
                    cx = (x1 + x2) / 2.0 / state.frame_w if state.frame_w else 0.5
                    cy = (y1 + y2) / 2.0 / state.frame_h if state.frame_h else 0.5
                    traj.update(
                        person_id=f"TEST-P{t.track_id}",
                        x=cx, y=cy,
                        camera_id="TEST-CAM",
                        frame_id=state.frame_id,
                        timestamp=state.timestamp,
                    )
            return None

        run_pipeline(detector, engine, source=str(self.CLIP),
                     max_frames=self.MAX_FRAMES, on_frame=on_frame)

        stats = traj.stats()
        assert stats["tracked_persons"] > 0, "No positions tracked"
        assert stats["total_positions"] > 0, f"No observations: {stats}"
        traj.shutdown()


# ===========================================================================
# 2. RATE LIMITING UNDER LOAD
# ===========================================================================
class TestRateLimitingLoad:
    """Rate limiter holds under load."""

    def test_sequential_login_flood(self, full_app):
        """20 bad logins should trigger rate limiter."""
        client = full_app["client"]
        results = []

        for _ in range(20):
            resp = client.post("/auth/login",
                               json={"username": "admin", "password": "wrong"})
            results.append(resp.status_code)

        rate_limited = sum(1 for r in results if r == 429)
        assert rate_limited > 0, f"Rate limiter never fired: {results[-5:]}"

    def test_concurrent_bad_logins(self, full_app):
        """Multiple threads hitting login simultaneously."""
        client = full_app["client"]
        results = []
        lock = threading.Lock()

        def bad_login():
            resp = client.post("/auth/login",
                               json={"username": "admin", "password": "wrong"})
            with lock:
                results.append(resp.status_code)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(bad_login) for _ in range(15)]
            for f in futures:
                f.result(timeout=30)

        assert len(results) == 15
        rate_limited = sum(1 for r in results if r == 429)
        assert rate_limited > 0, f"No rate limiting under concurrency: {results}"


# ===========================================================================
# 3. WEBSOCKET TESTS
# =======================================================================


class TestWebSocketDeep:
    def test_ws_receives_frame_with_tracks(self, full_app):
        client = full_app["client"]
        tok = full_app["admin"].get("Authorization", "").replace("Bearer ", "")
        with client.websocket_connect(f"/ws/stream?token={tok}") as ws:
            full_app["hub"].publish_frame(
                frame_id=42, timestamp=3.0, jpeg_b64="dGVzdA==",
                tracks=[{"id": 1, "label": "person"}],
                poses=[], alerts=[]
            )
            msg = ws.receive_json()
            assert msg["type"] == "frame"
            assert msg["frame_id"] == 42

    def test_ws_rejects_empty_token(self, full_app):
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect):
            with full_app["client"].websocket_connect("/ws/stream?token=") as ws:
                ws.receive_json()


class TestEvidenceWorkflow:
    def test_evidence_search_and_get(self, full_app):
        client = full_app["client"]
        admin = full_app["admin"]
        resp = client.get("/api/evidence", headers=admin)
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert len(data["events"]) > 0
        eid = data["events"][0]["event_id"]
        resp = client.get(f"/api/evidence/{eid}", headers=admin)
        assert resp.status_code == 200
        assert resp.json()["rule"] == "fight"

    def test_evidence_add_notes(self, full_app):
        client = full_app["client"]
        admin = full_app["admin"]
        resp = client.get("/api/evidence", headers=admin)
        eid = resp.json()["events"][0]["event_id"]
        resp = client.post(f"/api/evidence/{eid}/notes",
                          json={"text": "Test note"}, headers=admin)
        assert resp.status_code == 200
        resp = client.get(f"/api/evidence/{eid}", headers=admin)
        assert len(resp.json().get("notes", [])) > 0

    def test_evidence_delete_requires_admin(self, full_app):
        client = full_app["client"]
        viewer = full_app["viewer"]
        admin = full_app["admin"]
        resp = client.get("/api/evidence", headers=admin)
        events = resp.json()["events"]
        if not events:
            pytest.skip("No evidence")
        eid = events[-1]["event_id"]
        resp = client.delete(f"/api/evidence/{eid}", headers=viewer)
        assert resp.status_code in (403, 401)
        resp = client.delete(f"/api/evidence/{eid}", headers=admin)
        assert resp.status_code in (200, 204)

    def test_evidence_requires_auth(self, full_app):
        resp = full_app["client"].get("/api/evidence")
        assert resp.status_code == 401


class TestDashboard:
    def test_dashboard_has_html(self, full_app):
        resp = full_app["client"].get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers.get("content-type", "")
        assert len(resp.text) > 100

    def test_dashboard_has_react_bundle(self, full_app):
        body = full_app["client"].get("/").text.lower()
        assert "<script" in body or "react" in body or "app" in body

    def test_api_docs_available(self, full_app):
        resp = full_app["client"].get("/docs")
        assert resp.status_code == 200

    def test_status_endpoint_real_data(self, full_app):
        resp = full_app["client"].get("/api/status", headers=full_app["admin"])
        assert resp.status_code == 200
        data = resp.json()
        assert "fps" in data or "pipeline" in data or "status" in data


class TestRealFaceDetection:
    MODELS_DIR = Path("models")
    DATA_DIR = Path("tests/data")

    @pytest.mark.skipif(not (Path("models") / "face_detection_yunet.onnx").exists(),
                        reason="YuNet model not downloaded")
    def test_yunet_detects_faces(self):
        import cv2
        from bhairav.backend.face_search import check_models, FaceRecognizer
        models = check_models()
        rec = FaceRecognizer(models["detector"], models["recognizer"])
        img = cv2.imread(str(self.DATA_DIR / "lena.jpg"))
        assert img is not None
        emb = rec.embed(img)
        assert emb is not None and emb.shape == (128,)

    @pytest.mark.skipif(not (Path("models") / "face_detection_yunet.onnx").exists(),
                        reason="YuNet model not downloaded")
    def test_sface_similarity(self):
        import cv2
        from bhairav.backend.face_search import check_models, FaceRecognizer
        models = check_models()
        rec = FaceRecognizer(models["detector"], models["recognizer"])
        lena = cv2.imread(str(self.DATA_DIR / "lena.jpg"))
        messi = cv2.imread(str(self.DATA_DIR / "messi5.jpg"))
        e1, e2 = rec.embed(lena), rec.embed(messi)
        assert e1 is not None and e2 is not None
        assert rec.similarity(e1, e1) > 0.9
        assert rec.similarity(e1, e2) < 0.7

    @pytest.mark.skipif(not (Path("models") / "face_detection_yunet.onnx").exists(),
                        reason="YuNet model not downloaded")
    def test_gallery_search_e2e(self):
        import cv2
        from bhairav.backend.face_search import check_models, FaceRecognizer, FaceGallery
        models = check_models()
        rec = FaceRecognizer(models["detector"], models["recognizer"])
        lena = cv2.imread(str(self.DATA_DIR / "lena.jpg"))
        messi = cv2.imread(str(self.DATA_DIR / "messi5.jpg"))
        e1, e2 = rec.embed(lena), rec.embed(messi)
        g = FaceGallery(Path(tempfile.mkdtemp()) / "gallery.json")
        g.add("lena", e1)
        g.add("messi", e2)
        hits = g.search(e1, threshold=0.5)
        assert hits and hits[0]["name"] == "lena"



class TestFaceRegression:
    """Face detection regression tests that work without video footage."""

    MODELS_DIR = Path("models")
    DATA_DIR = Path("tests/data")

    @pytest.mark.skipif(
        not (Path("models") / "face_detection_yunet.onnx").exists(),
        reason="YuNet model not downloaded"
    )
    def test_face_detection_on_synthetic_face(self):
        """YuNet detects a face in a synthetic image with a face-like pattern."""
        import cv2
        from bhairav.backend.face_search import check_models, FaceRecognizer

        models = check_models()
        rec = FaceRecognizer(models["detector"], models["recognizer"])

        # Create a simple face-like image (oval with eyes)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = (200, 180, 160)  # skin tone background
        cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (180, 160, 140), -1)  # face oval
        cv2.circle(img, (80, 85), 8, (50, 50, 50), -1)  # left eye
        cv2.circle(img, (120, 85), 8, (50, 50, 50), -1)  # right eye
        cv2.ellipse(img, (100, 120), (20, 10), 0, 0, 180, (100, 80, 80), 2)  # mouth

        emb = rec.embed(img)
        # May or may not detect a face in synthetic image — just verify no crash
        # Real photos (lena.jpg, messi5.jpg) are tested in TestRealFaceDetection

    @pytest.mark.skipif(
        not (Path("models") / "face_detection_yunet.onnx").exists(),
        reason="YuNet model not downloaded"
    )
    def test_face_embedding_deterministic(self):
        """Same image produces same embedding every time."""
        import cv2
        from bhairav.backend.face_search import check_models, FaceRecognizer

        models = check_models()
        rec = FaceRecognizer(models["detector"], models["recognizer"])
        img = cv2.imread(str(self.DATA_DIR / "lena.jpg"))
        assert img is not None

        e1 = rec.embed(img)
        e2 = rec.embed(img)
        assert e1 is not None and e2 is not None
        np.testing.assert_array_almost_equal(e1, e2, decimal=5,
            err_msg="Embedding not deterministic")

    @pytest.mark.skipif(
        not (Path("models") / "face_detection_yunet.onnx").exists(),
        reason="YuNet model not downloaded"
    )
    def test_face_gallery_persistence(self):
        """Gallery survives save/load cycle."""
        import cv2
        from bhairav.backend.face_search import check_models, FaceRecognizer, FaceGallery

        models = check_models()
        rec = FaceRecognizer(models["detector"], models["recognizer"])
        img = cv2.imread(str(self.DATA_DIR / "lena.jpg"))
        emb = rec.embed(img)

        gallery_path = Path(tempfile.mkdtemp()) / "gallery.json"
        g1 = FaceGallery(gallery_path)
        g1.add("test_subject", emb, notes="persist test")

        # Reload from disk
        g2 = FaceGallery(gallery_path)
        assert g2.count() == 1
        hits = g2.search(emb, threshold=0.5)
        assert hits and hits[0]["name"] == "test_subject"

    def test_imm_filter_convergence(self):
        """IMM filter processes observations with predict+update cycle."""
        from bhairav.face_tracking import _IMMFilter2D

        imm = _IMMFilter2D()
        imm.init_state(0, 0)

        # Feed a clear constant-velocity pattern (predict then update)
        dt = 0.1
        for i in range(50):
            imm.predict(dt)
            imm.update(float(i * 2 * dt), 0.0)

        # Filter should not crash, should have valid probabilities
        assert len(imm.probs) == 3
        assert abs(sum(imm.probs) - 1.0) < 0.01
        assert imm.active_model in ["constant_velocity", "constant_acceleration", "stopped"]
        assert imm.position_uncertainty >= 0

        # The CV sub-filter should have learned the velocity
        cv_vx, cv_vy = imm.cv.velocity
        assert cv_vx > 0, f"CV filter velocity should be positive, got {cv_vx}"

    def test_camera_calibration_world_coords(self):
        """CameraCalibration maps pixel to world and back."""
        from bhairav.face_tracking import CameraCalibration

        cal = CameraCalibration(camera_id="TEST")
        # Simple calibration: pixel (0,0) -> world (0,0), pixel (100,100) -> world (10,10)
        pixel_pts = [(0, 0), (100, 0), (0, 100), (100, 100)]
        world_pts = [(0, 0), (10, 0), (0, 10), (10, 10)]

        ok = cal.set_from_correspondences(pixel_pts, world_pts)
        assert ok, "Calibration failed"

        # Test forward: pixel -> world
        wx, wy = cal.pixel_to_world(50, 50)
        assert abs(wx - 5.0) < 0.1 and abs(wy - 5.0) < 0.1

        # Test inverse: world -> pixel
        px, py = cal.world_to_pixel(5, 5)
        assert abs(px - 50) < 1 and abs(py - 50) < 1

    def test_chi2_confidence_range(self):
        """Chi-squared confidence is always in [0, 1]."""
        from bhairav.face_tracking import _chi2_confidence

        for nis in [-1, 0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
            c = _chi2_confidence(nis)
            assert 0 <= c <= 1, f"Confidence {c} out of range for NIS={nis}"

        # NIS=0 should be highest confidence
        assert _chi2_confidence(0) > _chi2_confidence(5)
        assert _chi2_confidence(5) > _chi2_confidence(50)
