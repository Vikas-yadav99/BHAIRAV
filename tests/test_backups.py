"""Unit tests for Phase 9 M3: backup file naming, retention, verify, and the
/api/ops/backups REST surface (a stub service, no database needed)."""
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from bhairav.backend.audit import AuditLog
from bhairav.backend.backups import (BACKUP_FORMAT, BACKUP_VERSION,
                                     _parse_stamp, list_backups, prune, verify)
from bhairav.backend.evidence import EvidenceStore
from bhairav.backend.server import create_app


def _write_backup(dirpath: Path, name: str, payload: dict) -> Path:
    p = dirpath / name
    p.write_bytes(gzip.compress(
        json.dumps(payload, sort_keys=True).encode("utf-8")))
    return p


def test_parse_stamp_utc():
    ts = _parse_stamp("bhairav_20260811_125200.backup.json.gz")
    assert abs(ts - 1786452720.0) < 2  # 2026-08-11 12:52:00 UTC
    assert _parse_stamp("garbage.txt") == 0.0


def test_list_backups_newest_first(tmp_path):
    t = tmp_path / "b"
    t.mkdir()
    for name, payload in [
        ("bhairav_20260810_000000.backup.json.gz", {"format": BACKUP_FORMAT}),
        ("bhairav_20260811_000000.backup.json.gz", {"format": BACKUP_FORMAT}),
    ]:
        _write_backup(t, name, payload)
    rows = list_backups(t)
    assert [r["name"] for r in rows][0].startswith("bhairav_20260811")
    assert rows[0]["size_bytes"] > 0 and rows[0]["age_sec"] >= 0


def test_prune_keeps_retention(tmp_path):
    t = tmp_path / "b"
    t.mkdir()
    for i in range(6):
        _write_backup(t, f"bhairav_2026081{i}_000000.backup.json.gz",
                      {"format": BACKUP_FORMAT})
    removed = prune(t, 2)
    assert len(removed) == 4
    assert len(list_backups(t)) == 2


def test_verify_valid_and_corrupt(tmp_path):
    good = _write_backup(tmp_path, "bhairav_20260811_000000.backup.json.gz",
                         {"format": BACKUP_FORMAT, "version": BACKUP_VERSION,
                          "created_at": 1.0,
                          "tables": [{"name": "evidence", "columns": [],
                                      "rows": []}]})
    assert verify(good)["ok"] is True
    bad = tmp_path / "broken.backup.json.gz"
    bad.write_bytes(b"not gzip")
    v = verify(bad)
    assert v["ok"] is False and v["error"]


class StubBackupMgr:
    def __init__(self, dirpath):
        self.out_dir = dirpath
        self._created = []

    def list(self):
        return list_backups(self.out_dir)

    def create(self):
        name = f"bhairav_{int(time.time())}.backup.json.gz"
        _write_backup(self.out_dir, name,
                      {"format": BACKUP_FORMAT, "version": BACKUP_VERSION,
                       "created_at": time.time(), "tables": []})
        self._created.append(name)
        return {"path": str(self.out_dir / name), "size_bytes": 10,
                "tables": 0, "pruned": []}

    def read(self, name):
        p = self.out_dir / name
        return p.read_bytes() if p.exists() else None

    def latest(self):
        rows = self.list()
        return rows[0] if rows else None


@pytest.fixture()
def ctx(tmp_path):
    store = EvidenceStore(tmp_path / "evidence", fps=10.0, blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    mgr = StubBackupMgr(tmp_path / "backups")
    mgr.out_dir.mkdir(exist_ok=True)
    app = create_app(store, audit, secret="test-secret", backup_mgr=mgr)
    return TestClient(app), mgr, audit


def _admin(client):
    r = client.post("/auth/login",
                    json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    return r.json()["token"]


def test_backups_endpoints_roundtrip(ctx):
    client, mgr, audit = ctx
    tok = _admin(client)
    h = {"Authorization": "Bearer " + tok}

    r = client.post("/api/ops/backups", headers=h)
    assert r.status_code == 200
    assert mgr._created

    r = client.get("/api/ops/backups", headers=h)
    assert r.status_code == 200
    assert len(r.json()["backups"]) == 1

    name = r.json()["backups"][0]["name"]
    r = client.get("/api/ops/backups/" + name, headers=h)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/gzip")

    # path traversal / bad names rejected
    r = client.get("/api/ops/backups/..%2F..%2Fetc%2Fpasswd", headers=h)
    assert r.status_code == 404

    # download is audited
    assert any(e["action"] == "download_backup" for e in audit.read())


def test_backups_admin_only(ctx):
    client, _, _ = ctx
    r = client.post("/auth/login",
                    json={"username": "viewer", "password": "viewer123"})
    tok = r.json()["token"]
    r = client.get("/api/ops/backups", headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 403


def test_backups_503_without_service(tmp_path):
    store = EvidenceStore(tmp_path / "e", fps=10.0, blur_faces=False)
    audit = AuditLog(tmp_path / "a.jsonl")
    app = create_app(store, audit, secret="test-secret")
    client = TestClient(app)
    tok = client.post("/auth/login",
                      json={"username": "admin", "password": "admin123"}).json()["token"]
    assert client.get("/api/ops/backups",
                      headers={"Authorization": "Bearer " + tok}).status_code == 503
