"""Tests for the Phase 6 face-search pipeline (gallery, evidence index, API)."""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from bhairav.backend.face_search import (FaceGallery, FaceRecognizer,
                                         check_models)

DATA = Path(__file__).resolve().parent / "data"


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(128).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# FaceGallery
# ---------------------------------------------------------------------------
def test_gallery_add_list_remove(tmp_path):
    g = FaceGallery(tmp_path / "gallery.json")
    g.add("alice", _emb(1), notes="missing since June")
    g.add("bob", _emb(2))
    assert g.count() == 2
    assert {s["name"] for s in g.list()} == {"alice", "bob"}
    g2 = FaceGallery(tmp_path / "gallery.json")   # persists
    assert g2.count() == 2
    assert g2.remove("bob") is True
    assert g2.remove("nobody") is False
    assert g2.count() == 1


def test_gallery_rejects_bad_names(tmp_path):
    g = FaceGallery(tmp_path / "gallery.json")
    with pytest.raises(ValueError):
        g.add("", _emb(1))
    with pytest.raises(ValueError):
        g.add("x" * 100, _emb(1))


def test_gallery_search_ranks_correct_subject(tmp_path):
    g = FaceGallery(tmp_path / "gallery.json")
    g.add("alice", _emb(1))
    g.add("bob", _emb(2))
    hits = g.search(_emb(1), top_k=2, threshold=0.0)
    assert hits[0]["name"] == "alice"
    assert hits[0]["similarity"] > hits[1]["similarity"]


def test_gallery_search_threshold(tmp_path):
    g = FaceGallery(tmp_path / "gallery.json")
    g.add("alice", _emb(1))
    assert g.search(_emb(1), threshold=0.99)          # near-identical -> hit
    assert g.search(_emb(50), threshold=0.99) == []   # unrelated -> no hit


def test_similarity_symmetric():
    a, b = _emb(1), _emb(2)
    assert FaceRecognizer.similarity(a, b) == pytest.approx(
        FaceRecognizer.similarity(b, a))
    assert FaceRecognizer.similarity(a, a) > 0.99

# ---------------------------------------------------------------------------
# EvidenceFaceIndex (with a stubbed recognizer)
# ---------------------------------------------------------------------------
class _StubRecognizer:
    def __init__(self, vec=None):
        self._vec = vec

    def faces(self, img):
        _ = img  # deterministic embedding, independent of pixels
        return [{"bbox": (0, 0, 10, 10), "score": 0.9,
                 "embedding": self._vec if self._vec is not None else _emb(7)}]


def test_evidence_index_search_and_stats(tmp_path):
    from bhairav.backend.evidence import EvidenceStore, ActiveEvent
    from bhairav.backend.face_search import EvidenceFaceIndex
    from bhairav.types import Alert, Severity
    import cv2

    store = EvidenceStore(tmp_path / "ev", fps=10.0, blur_faces=False)
    img = np.full((120, 160, 3), 100, np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    ev = ActiveEvent("a" * 12, Alert("fight", None, 1, Severity.RED, "x", 0, 0.0),
                     pre_frames=[], during_frames=[jpg.tobytes()],
                     first_ts=1.0, last_ts=1.0)
    store.save(ev)

    index = EvidenceFaceIndex(store, _StubRecognizer())
    stats = index.index()
    assert stats["total_events"] == 1
    assert index.stats()["events_indexed"] == 1
    hits = index.search(_emb(7), threshold=0.0)      # same embedding -> hit
    assert hits and hits[0]["event_id"] == "a" * 12
    assert index.search(_emb(99), threshold=0.99) == []


# ---------------------------------------------------------------------------
# API integration (fake service injected into create_app)
# ---------------------------------------------------------------------------
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from bhairav.backend.audit import AuditLog
from bhairav.backend.evidence import EvidenceStore
from bhairav.backend.server import create_app
from bhairav.backend.users import UserStore


def _b64_img(img) -> str:
    import cv2
    ok, jpg = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(jpg.tobytes()).decode()


def _make_app(tmp_path, face=None):
    store = EvidenceStore(tmp_path / "evidence", fps=10.0, blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    users = UserStore(tmp_path / "users.json")
    app = create_app(store, audit, secret="test-secret", users=users, face=face)
    return TestClient(app), store, audit


def test_search_503_when_face_disabled(tmp_path):
    c, _, _ = _make_app(tmp_path, face=None)
    tok = c.post("/auth/login", json={"username": "admin", "password": "admin123"}).json()["token"]
    r = c.get("/api/search/status", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 503

def test_search_api_lifecycle_with_fake_service(tmp_path):
    from bhairav.backend.face_search import EvidenceFaceIndex, FaceGallery

    vec = _emb(3)
    recognizer = type("R", (), {
        "embed": lambda self, img: vec,
        "faces": lambda self, img: [{"bbox": (0, 0, 10, 10), "score": 0.9,
                                     "embedding": vec}],
    })()
    gallery = FaceGallery(tmp_path / "gallery.json")
    store = EvidenceStore(tmp_path / "evidence", fps=10.0, blur_faces=False)
    index = EvidenceFaceIndex(store, recognizer)
    face = {"recognizer": recognizer, "gallery": gallery, "index": index}
    c, _, audit = _make_app(tmp_path, face=face)

    tok = c.post("/auth/login", json={"username": "admin", "password": "admin123"}).json()["token"]
    ah = {"Authorization": f"Bearer {tok}"}
    analyst = c.post("/auth/login", json={"username": "analyst", "password": "analyst123"}).json()["token"]
    viewer = c.post("/auth/login", json={"username": "viewer", "password": "viewer123"}).json()["token"]

    # RBAC: viewer cannot query
    assert c.post("/api/search/query", headers={"Authorization": f"Bearer {viewer}"},
                  json={"image_b64": _b64_img(np.zeros((32, 32, 3), np.uint8))}).status_code == 403
    # register (admin only)
    img = _b64_img(np.zeros((64, 64, 3), np.uint8))
    r = c.post("/api/search/register", headers=ah,
               json={"name": "alice", "image_b64": img, "notes": "missing"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "alice"
    # analyst can list + query
    assert c.get("/api/search/subjects",
                 headers={"Authorization": f"Bearer {analyst}"}).json()["subjects"][0]["name"] == "alice"
    q = c.post("/api/search/query", headers={"Authorization": f"Bearer {analyst}"},
               json={"image_b64": img, "top_k": 5, "threshold": 0.0})
    assert q.status_code == 200, q.text
    assert q.json()["subjects"][0]["name"] == "alice"
    # no face -> 400
    empty = type("R2", (), {"embed": lambda self, img: None})()
    c2, _, _ = _make_app(tmp_path, face={"recognizer": empty, "gallery": gallery, "index": index})
    tok2 = c2.post("/auth/login", json={"username": "admin", "password": "admin123"}).json()["token"]
    r = c2.post("/api/search/query", headers={"Authorization": f"Bearer {tok2}"},
                json={"image_b64": img})
    assert r.status_code == 400 and "no face" in r.json()["detail"]
    # delete (admin) + audit trail
    r = c.delete("/api/search/subjects/alice", headers=ah)
    assert r.status_code == 200
    assert any(e["action"] == "register_subject" for e in audit.read())


# ---------------------------------------------------------------------------
# Real end-to-end with the actual YuNet + SFace models (skipped if absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet.onnx").exists(),
                    reason="face models not downloaded (run scripts/fetch_models.py)")
def test_real_face_recognition_end_to_end(tmp_path):
    import cv2
    if not (DATA / "lena.jpg").exists() or not (DATA / "messi5.jpg").exists():
        pytest.skip("test face images missing from tests/data")
    models = check_models()
    recognizer = FaceRecognizer(models["detector"], models["recognizer"])
    lena = cv2.imread(str(DATA / "lena.jpg"))
    messi = cv2.imread(str(DATA / "messi5.jpg"))
    el, em = recognizer.embed(lena), recognizer.embed(messi)
    assert el is not None and em is not None, "real faces must be detected"
    assert recognizer.similarity(el, el) > 0.9
    assert recognizer.similarity(el, em) < 0.5
    g = FaceGallery(tmp_path / "gallery.json")
    g.add("lena", el)
    hits = g.search(el, threshold=0.5)
    assert hits and hits[0]["name"] == "lena"
    hits2 = g.search(em, threshold=0.5)
    assert hits2 == [] or hits2[0]["name"] != "lena"
