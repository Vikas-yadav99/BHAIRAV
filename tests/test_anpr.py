"""Tests for the Phase 6 ANPR / stolen-vehicle watchlist."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from bhairav.backend.anpr import (PlateReader, PlateRegistry,
                                  StolenVehicleRule)
from bhairav.config import SyntheticConfig
from bhairav.detectors import BlobDetector, default_scenario
from bhairav.types import Severity

# ---------------------------------------------------------------------------
# PlateRegistry (watchlist + read log)
# ---------------------------------------------------------------------------
def test_registry_watch_unwatch_persist(tmp_path):
    p = tmp_path / "plates.json"
    reg = PlateRegistry(p)
    reg.watch(" mh12ab1234 ", reason="stolen from mall")
    assert reg.is_watched("MH12AB1234")      # normalized uppercase
    assert reg.list_watch()[0]["reason"] == "stolen from mall"
    reg2 = PlateRegistry(p)                   # reloads from disk
    assert reg2.is_watched("mh12ab1234")
    assert reg2.unwatch("MH12AB1234") is True
    assert reg2.unwatch("MH12AB1234") is False
    assert reg2.list_watch() == []

def test_registry_rejects_bad_plates(tmp_path):
    reg = PlateRegistry(tmp_path / "plates.json")
    with pytest.raises(ValueError):
        reg.watch("")
    with pytest.raises(ValueError):
        reg.watch("A" * 20)

def test_registry_read_log(tmp_path):
    reg = PlateRegistry(tmp_path / "plates.json")
    reg.add_read("MH12AB1234", 1.5)
    reg.add_read("DL8CAF0001", 2.5)
    reads = reg.recent_reads()
    assert reads[0]["plate"] == "DL8CAF0001"  # newest first
    assert len(reg.recent_reads()) == 2

# ---------------------------------------------------------------------------
# PlateReader on the synthetic scene (exact, deterministic)
# ---------------------------------------------------------------------------
def test_plate_reader_reads_synthetic_plate():
    cfg = SyntheticConfig()
    det = BlobDetector(default_scenario(cfg), fps=cfg.fps)
    reader = PlateReader()
    seen = set()
    for state in det.stream():
        for tr in state.tracks:
            if tr.is_person or tr.class_id not in (2, 5, 7):
                continue
            plate, conf = reader.read(state.frame, tr.bbox)
            if plate:
                seen.add(plate)
    assert "MH12AB1234" in seen, f"plate not read, got {seen}"

def test_plate_reader_rejects_garbage():
    reader = PlateReader()
    blank = np.full((40, 80, 3), 230, np.uint8)
    assert reader.read(blank, (0, 0, 80, 40))[0] is None

# ---------------------------------------------------------------------------
# StolenVehicleRule
# ---------------------------------------------------------------------------
def test_stolen_vehicle_rule_fires_on_watchlist(tmp_path):
    cfg = SyntheticConfig()
    det = BlobDetector(default_scenario(cfg), fps=cfg.fps)
    reg = PlateRegistry(tmp_path / "plates.json")
    reg.watch("MH12AB1234", reason="stolen test")
    rule = StolenVehicleRule({"enabled": True, "severity": "red"})
    rule.registry = reg
    fired = []
    for state in det.stream():
        for a in rule.evaluate(state, []):
            fired.append(a)
    assert fired, "stolen-vehicle alert never fired"
    a = fired[0]
    assert a.rule == "stolen_vehicle"
    assert a.severity == Severity.RED
    assert a.details["plate"] == "MH12AB1234"
    assert a.details["reason"] == "stolen test"

def test_stolen_vehicle_rule_silent_when_not_watched(tmp_path):
    cfg = SyntheticConfig()
    det = BlobDetector(default_scenario(cfg), fps=cfg.fps)
    reg = PlateRegistry(tmp_path / "plates.json")   # empty watchlist
    rule = StolenVehicleRule({"enabled": True})
    rule.registry = reg
    n = sum(len(rule.evaluate(state, [])) for state in det.stream())
    assert n == 0

# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from bhairav.backend.audit import AuditLog
from bhairav.backend.evidence import EvidenceStore
from bhairav.backend.server import create_app
from bhairav.backend.users import UserStore


def _make_app(tmp_path, plates=None):
    store = EvidenceStore(tmp_path / "evidence", fps=10.0, blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    users = UserStore(tmp_path / "users.json")
    app = create_app(store, audit, secret="test-secret", users=users,
                     plates=plates)
    return TestClient(app), audit


def test_vehicle_api_lifecycle(tmp_path):
    reg = PlateRegistry(tmp_path / "plates.json")
    c, audit = _make_app(tmp_path, plates=reg)
    tok = c.post("/auth/login",
                 json={"username": "admin", "password": "admin123"}).json()["token"]
    ah = {"Authorization": f"Bearer {tok}"}
    viewer = c.post("/auth/login",
                    json={"username": "viewer", "password": "viewer123"}).json()["token"]

    # RBAC: viewer cannot manage the watchlist
    assert c.post("/api/vehicles/watch", headers={"Authorization": f"Bearer {viewer}"},
                  json={"plate": "MH12AB1234"}).status_code == 403
    # admin adds + lists
    r = c.post("/api/vehicles/watch", headers=ah,
               json={"plate": "mh12ab1234", "reason": "stolen"})
    assert r.status_code == 200, r.text
    assert r.json()["plate"] == "MH12AB1234"
    assert c.get("/api/vehicles/watch", headers=ah).json()["watch"][0]["plate"] == "MH12AB1234"
    # reads endpoint
    reg.add_read("MH12AB1234", 1.0)
    reads = c.get("/api/vehicles/reads", headers=ah).json()["reads"]
    assert reads and reads[0]["plate"] == "MH12AB1234"
    # delete + audit trail
    assert c.delete("/api/vehicles/watch/MH12AB1234", headers=ah).json()["removed"] == "MH12AB1234"
    assert any(e["action"] == "watch_plate" for e in audit.read())


def test_vehicle_api_503_without_registry(tmp_path):
    c, _ = _make_app(tmp_path, plates=None)
    tok = c.post("/auth/login",
                 json={"username": "admin", "password": "admin123"}).json()["token"]
    r = c.get("/api/vehicles/watch", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 503

# ---------------------------------------------------------------------------
# PlateReader backends (template vs easyocr)
# ---------------------------------------------------------------------------
def test_template_reader_reads_synthetic_plate():
    """Template backend is exact on the synthetic scene (deterministic)."""
    cfg = SyntheticConfig()
    det = BlobDetector(default_scenario(cfg), fps=cfg.fps,
                       width=cfg.width, height=cfg.height)
    reader = PlateReader(backend="template")
    for st in det.stream(source="blob", max_frames=400):
        for tr in st.tracks:
            if tr.class_id in (2, 5, 7) and tr.bbox[3] > 60:
                txt, conf = reader.read(st.frame, tr.bbox)
                if txt == "MH12AB1234":
                    assert conf >= 0.5
                    return
    pytest.fail("no full plate read on synthetic scene")


def test_easyocr_backend_falls_back_to_template(monkeypatch):
    """backend='easyocr' without easyocr installed degrades to template."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "easyocr":
            raise ImportError("easyocr not installed (test)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = SyntheticConfig()
    det = BlobDetector(default_scenario(cfg), fps=cfg.fps,
                       width=cfg.width, height=cfg.height)
    reader = PlateReader(backend="easyocr")
    for st in det.stream(source="blob", max_frames=400):
        for tr in st.tracks:
            if tr.class_id in (2, 5, 7) and tr.bbox[3] > 60:
                txt, conf = reader.read(st.frame, tr.bbox)
                if txt == "MH12AB1234":      # fell back to template path
                    return
    pytest.fail("no full plate read via fallback")


def test_stolen_vehicle_rule_reads_backend_from_config(tmp_path):
    """backend key in the rule config selects the PlateReader backend."""
    rule = StolenVehicleRule({"enabled": True, "backend": "template"})
    assert rule.reader.backend == "template"
    rule2 = StolenVehicleRule({"enabled": True, "backend": "easyocr"})
    assert rule2.reader.backend == "easyocr"


@pytest.mark.skipif(True, reason="real-plate fixtures fetched by scripts/fetch_real_plate_samples.py")
def _unused_real_plate_placeholder():
    pass
