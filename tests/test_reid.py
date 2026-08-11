"""Phase 9 M4 - person re-identification unit tests (descriptor, store, service)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from bhairav.reid import (AppearanceExtractor, ReidService, ReidStore,
                          cosine)


def _person(shirt, h=140, w=50):
    """Synthetic person: dark head + colored torso + dark legs."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:30, :, :] = (40, 40, 180)
    img[30:85, :, :] = shirt
    img[85:, :, :] = (30, 30, 60)
    return img


class _Track:
    is_person = True

    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.track_id = track_id


def _state(tracks, ts=1.0, frame_id=1):
    from types import SimpleNamespace
    return SimpleNamespace(tracks=tracks, timestamp=ts, frame_id=frame_id)


# ---- descriptor -----------------------------------------------------------
def test_descriptor_same_person_more_similar_than_different():
    ex = AppearanceExtractor()
    a1 = ex.embed(_person((40, 160, 240)))
    a2 = ex.embed(_person((40, 160, 240)))
    b = ex.embed(_person((20, 200, 40)))
    assert a1 is not None and a2 is not None and b is not None
    same = cosine(a1, a2)
    diff = cosine(a1, b)
    assert same > 0.95
    assert same > diff + 0.2, (same, diff)


def test_descriptor_rejects_tiny_crop():
    ex = AppearanceExtractor()
    assert ex.embed(np.zeros((20, 20, 3), np.uint8)) is None
    # crop smaller than the 40px floor is rejected too
    assert ex.extract_from_frame(np.zeros((100, 100, 3), np.uint8),
                                 (0, 0, 30, 30)) is None


def test_thumbnail_blurs_head_and_is_jpeg():
    ex = AppearanceExtractor()
    thumb = ex.crop_thumbnail(_person((40, 160, 240)), (0, 0, 50, 140))
    assert thumb is not None and thumb.startswith("/9j/")  # JPEG base64


# ---- store ----------------------------------------------------------------
def test_store_roundtrip_and_trail(tmp_path):
    store = ReidStore(tmp_path / "reid")
    rec = store.create_subject("alice", [0.5, 0.5, 0.0, 1.0])
    store.record_sighting(rec["id"], "CAM-01", 1, ts=1.0, frame_id=5,
                          score=0.9, bbox=[1, 2, 3, 4], thumb_b64=None)
    store.record_sighting(rec["id"], "CAM-02", 7, ts=2.5, frame_id=9,
                          score=0.95, bbox=[1, 2, 3, 4], thumb_b64=None)
    # reload from disk
    store2 = ReidStore(tmp_path / "reid")
    assert store2.get(rec["id"])["cameras"] == ["CAM-01", "CAM-02"]
    trail = store2.trail(rec["id"])
    assert [t["camera"] for t in trail] == ["CAM-01", "CAM-02"]
    assert store2.stats()["subjects"] == 1
    assert store2.stats()["sightings"] == 2


def test_store_merge_observation_updates_mean():
    store = ReidStore(__import__("tempfile").mkdtemp())
    rec = store.create_subject("", [1.0, 0.0])
    store.merge_observation(rec["id"], [0.0, 1.0], "CAM-01", 0.7)
    got = store.get(rec["id"])
    assert got["count"] == 2
    assert abs(got["embedding"][0] - 0.5) < 1e-6


def test_store_rename_and_remove(tmp_path):
    store = ReidStore(tmp_path / "reid")
    rec = store.create_subject("", [1.0, 0.0])
    store.record_sighting(rec["id"], "CAM-01", 1, ts=1.0, frame_id=1,
                          score=0.8, bbox=[1, 2, 3, 4], thumb_b64=None)
    assert store.rename(rec["id"], "known-person")
    assert store.get(rec["id"])["name"] == "known-person"
    assert store.remove(rec["id"])
    assert store.get(rec["id"]) is None
    assert store.stats()["sightings"] == 0


def test_best_match_threshold(tmp_path):
    store = ReidStore(tmp_path / "reid")
    store.create_subject("red", [1.0, 0.0])
    assert store.best_match([1.0, 0.0], 0.9) is not None
    assert store.best_match([0.0, 1.0], 0.9) is None


# ---- service --------------------------------------------------------------
def test_service_cross_camera_same_identity(tmp_path):
    store = ReidStore(tmp_path / "reid")
    svc = ReidService(store, assign_threshold=0.6, sighting_gap_sec=0.0)
    alice = _person((40, 160, 240))
    tr = _Track([10, 10, 60, 150], 1)
    r1 = svc.observe(alice, _state([tr], ts=1.0, frame_id=1), "CAM-01")
    r2 = svc.observe(alice, _state([tr], ts=1.1, frame_id=2), "CAM-02")
    assert r1 and r2
    assert r2[0]["subject_id"] == r1[0]["subject_id"]
    assert r2[0]["score"] > 0.8
    assert store.get(r1[0]["subject_id"])["cameras"] == ["CAM-01", "CAM-02"]
    trail = store.trail(r1[0]["subject_id"])
    assert [t["camera"] for t in trail] == ["CAM-01", "CAM-02"]


def test_service_distinguishes_different_people(tmp_path):
    store = ReidStore(tmp_path / "reid")
    svc = ReidService(store, assign_threshold=0.6, sighting_gap_sec=0.0)
    tr = _Track([10, 10, 60, 150], 1)
    r1 = svc.observe(_person((40, 160, 240)), _state([tr]), "CAM-01")
    r2 = svc.observe(_person((20, 200, 40)), _state([tr]), "CAM-01")
    assert r1[0]["subject_id"] != r2[0]["subject_id"]
    assert store.stats()["subjects"] == 2


def test_service_throttles_sightings(tmp_path):
    store = ReidStore(tmp_path / "reid")
    svc = ReidService(store, assign_threshold=0.6, sighting_gap_sec=10.0)
    tr = _Track([10, 10, 60, 150], 1)
    for ts in (1.0, 1.1, 1.2):
        svc.observe(_person((40, 160, 240)), _state([tr], ts=ts), "CAM-01")
    assert store.stats()["sightings"] == 1  # throttled despite 3 observations


# ---- API ------------------------------------------------------------------
def _build_app(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.server import LiveHub, PipelineStats, create_app
    from bhairav.backend.users import UserStore

    store = EvidenceStore(tmp_path / "evidence", camera="CAM-01", fps=10.0,
                          blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    users = UserStore(tmp_path / "users.json")
    rstore = ReidStore(tmp_path / "reid")
    svc = ReidService(rstore, assign_threshold=0.6, sighting_gap_sec=0.0)
    app = create_app(store, audit, secret="test-secret", hub=LiveHub(),
                     users=users, stats=PipelineStats(), reid=svc)
    return TestClient(app), rstore


def test_reid_api_cross_camera_trail(tmp_path):
    client, rstore = _build_app(tmp_path)
    r = client.post("/auth/login", json={"username": "analyst",
                                        "password": "analyst123"})
    assert r.status_code == 200
    token = r.json()["token"]
    h = {"Authorization": "Bearer " + token}
    # rename/delete need PERM_USERS (admin only)
    ra = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    ha = {"Authorization": "Bearer " + ra.json()["token"]}

    tr = _Track([10, 10, 60, 150], 1)
    svc = ReidService(rstore, assign_threshold=0.6, sighting_gap_sec=0.0)
    r1 = svc.observe(_person((40, 160, 240)), _state([tr], ts=1.0), "CAM-01")
    r2 = svc.observe(_person((40, 160, 240)), _state([tr], ts=2.0), "CAM-02")
    sid = r1[0]["subject_id"]
    assert r2[0]["subject_id"] == sid

    stats = client.get("/api/reid/stats", headers=h)
    assert stats.status_code == 200 and stats.json()["subjects"] == 1

    subs = client.get("/api/reid/subjects", headers=h).json()["subjects"]
    assert subs[0]["id"] == sid and subs[0]["cameras"] == ["CAM-01", "CAM-02"]

    trail = client.get(f"/api/reid/subjects/{sid}/trail", headers=h)
    assert trail.status_code == 200
    assert [t["camera"] for t in trail.json()["trail"]] == ["CAM-01", "CAM-02"]

    seen = client.get("/api/reid/sightings", headers=h).json()["sightings"]
    assert len(seen) == 2

    rn = client.post(f"/api/reid/subjects/{sid}/rename", headers=ha,
                     json={"name": "suspect-alpha"})
    assert rn.status_code == 200
    assert client.get("/api/reid/subjects", headers=h).json()["subjects"][0]["name"] == "suspect-alpha"

    rm = client.delete(f"/api/reid/subjects/{sid}", headers=ha)
    assert rm.status_code == 200
    assert client.get("/api/reid/subjects", headers=h).json()["subjects"] == []


def test_reid_api_requires_auth(tmp_path):
    client, _ = _build_app(tmp_path)
    assert client.get("/api/reid/stats").status_code == 401
    # any authenticated role with evidence_read (incl. viewer) can read
    r = client.post("/auth/login", json={"username": "viewer",
                                        "password": "viewer123"})
    assert r.status_code == 200
    vh = {"Authorization": "Bearer " + r.json()["token"]}
    assert client.get("/api/reid/stats", headers=vh).status_code == 200
    # but rename is admin-only (PERM_USERS): viewer is forbidden
    assert client.post("/api/reid/subjects/P-123/rename", headers=vh,
                       json={"name": "x"}).status_code == 403


def test_reid_api_503_without_service(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.server import LiveHub, PipelineStats, create_app
    from bhairav.backend.users import UserStore

    store = EvidenceStore(tmp_path / "ev", camera="CAM-01", fps=10.0,
                          blur_faces=False)
    app = create_app(store, AuditLog(tmp_path / "audit.jsonl"),
                     secret="test-secret", hub=LiveHub(),
                     users=UserStore(tmp_path / "users.json"),
                     stats=PipelineStats())
    c = TestClient(app)
    r = c.post("/auth/login", json={"username": "admin", "password": "admin123"})
    h = {"Authorization": "Bearer " + r.json()["token"]}
    assert c.get("/api/reid/stats", headers=h).status_code == 503


# ---- Phase 9 M5.5: physical-description search ----------------------------
def _make_subject(desc, name="auto"):
    import time
    return {"id": "P-12345678", "name": name, "auto": not name,
            "description": desc, "last_seen": time.time()}


def test_parse_query_colors_and_height():
    from bhairav.describe import parse_query
    assert parse_query("red shirt, tall") == (["red"], "tall")
    assert parse_query("navy jacket medium") == (["blue"], "medium")
    assert parse_query("grey hoodie short person") == (["gray"], "short")
    assert parse_query("") == ([], None)
    assert parse_query("random words") == ([], None)


def test_match_color_present_and_absent():
    from bhairav.describe import match
    desc = {"colors": [{"name": "red", "fraction": 0.6},
                       {"name": "black", "fraction": 0.3}],
            "height_class": "tall"}
    assert match(desc, ["red"]) > match(desc, ["black"])
    assert match(desc, ["green"]) == 0.0  # absent color -> no match
    assert match(desc, None, "tall") > 0.0
    assert match(desc, None, "short") < 1.0  # height mismatch only
    assert match(None, ["red"]) == 0.0


def test_search_subjects_ranks_by_color():
    from bhairav.describe import search_subjects
    red = _make_subject({"colors": [{"name": "red", "fraction": 0.8}],
                         "height_class": "tall"})
    red["id"] = "P-red000001"
    blue = _make_subject({"colors": [{"name": "blue", "fraction": 0.9}],
                          "height_class": "medium"})
    blue["id"] = "P-blue00002"
    hits = search_subjects([blue, red], ["red"])
    # blue has no red in its palette -> excluded entirely, red ranks first
    assert [h["subject"]["id"] for h in hits] == [red["id"]]
    assert hits[0]["score"] > 0.5


def test_describe_person_detects_torso_color():
    from bhairav.describe import describe_person
    img = np.zeros((200, 60, 3), np.uint8)
    img[50:160, :, :] = (40, 40, 240)  # BGR red torso
    d = describe_person(img, frame_h=480)
    assert d is not None
    names = [c["name"] for c in d["colors"]]
    assert "red" in names
    assert d["height_class"] == "medium"  # 200/480


def test_reid_api_search_by_description(tmp_path):
    client, rstore = _build_app(tmp_path)
    r = client.post("/auth/login", json={"username": "analyst",
                                        "password": "analyst123"})
    h = {"Authorization": "Bearer " + r.json()["token"]}
    # two gallery subjects: red shirt vs blue shirt
    svc = ReidService(rstore, assign_threshold=0.99, sighting_gap_sec=0.0)
    red = svc.observe(_person((40, 160, 240)), _state([_Track([10, 10, 60, 150], 1)]),
                      "CAM-01")[0]["subject_id"]
    blue = svc.observe(_person((230, 120, 20)), _state([_Track([10, 10, 60, 150], 2)]),
                       "CAM-01")[0]["subject_id"]
    assert red != blue
    # subjects carry a description from the torso
    subs = client.get("/api/reid/subjects", headers=h).json()["subjects"]
    assert all(s["description"] is not None for s in subs)

    hit = client.get("/api/reid/search?q=red", headers=h)
    assert hit.status_code == 200
    body = hit.json()
    assert body["colors"] == ["red"]
    assert body["results"][0]["subject"]["id"] == red
    assert body["results"][0]["score"] > 0.5

    miss = client.get("/api/reid/search?q=green", headers=h).json()
    assert miss["results"] == []
    # empty query returns no results without erroring
    assert client.get("/api/reid/search?q=", headers=h).json()["results"] == []
