"""Tests for the Phase 8 PostgreSQL evidence store (pg_store.py).

Pure-logic unit tests (SQL builder, row mapping, id validation, missing
driver) run everywhere. The integration tests need a real PostgreSQL and are
gated behind $BHAIRAV_TEST_DB_URL:

    BHAIRAV_TEST_DB_URL=postgresql://bhairav:bhairav@localhost:5432/bhairav         python -m pytest tests/test_evidence_pg.py
"""
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import cv2
import numpy as np

from bhairav.backend.evidence import ActiveEvent
from bhairav.backend.pg_store import (PostgresEvidenceStore, build_search_query,
                                      load_driver, row_to_record)
from bhairav.types import Alert, Severity

TEST_DB_URL = os.environ.get("BHAIRAV_TEST_DB_URL")
PSYCOPG_INSTALLED = importlib.util.find_spec("psycopg") is not None


def _make_event(event_id="abcd12345678", rule="fight", severity=Severity.RED,
                ts=100.0, n_frames=1) -> ActiveEvent:
    img = np.full((48, 64, 3), 120, np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    assert ok
    return ActiveEvent(
        event_id, Alert(rule, "plaza", 7, severity, "fight in plaza", 0, ts,
                        details={"plate": "MH12AB1234"}, confidence=0.9),
        pre_frames=[], during_frames=[jpg.tobytes()] * n_frames,
        first_ts=ts, last_ts=ts)


def _fresh_store(**kw) -> PostgresEvidenceStore:
    store = PostgresEvidenceStore(TEST_DB_URL, camera="CAM-01", fps=10.0,
                                  blur_faces=False, **kw)
    store._cursor().execute("DELETE FROM evidence")  # isolation per test
    return store


# ---------------------------------------------------------------------------
# Pure-logic unit tests (no database required)
# ---------------------------------------------------------------------------
def test_build_search_query_all_filters():
    sql, params = build_search_query(rule="fight", severity="red",
                                     camera="CAM-01", t0=1.0, t1=9.0,
                                     q="Plaza", limit=10)
    assert "rule = %s" in sql and "severity = %s" in sql and "camera = %s" in sql
    assert "start_ts >= %s" in sql and "start_ts <= %s" in sql
    assert "lower(message) LIKE %s" in sql
    assert params == ["fight", "red", "CAM-01", 1.0, 9.0,
                      "%plaza%", "%plaza%", "%plaza%", 10]


def test_build_search_query_no_filters():
    sql, params = build_search_query()
    assert sql.startswith("SELECT * FROM evidence")
    assert "WHERE" not in sql
    assert params == [50]


def test_row_to_record_maps_columns():
    rec = row_to_record({
        "event_id": "ab12", "rule": "fall", "severity": "orange", "message": "m",
        "zone": None, "track_id": None, "confidence": 0.8, "details": {"k": 1},
        "start_ts": 1.0, "end_ts": 2.0, "frame_count": 3, "camera": "CAM-01",
        "created": 5.0, "blurred": True, "encrypted": False, "status": "new",
        "notes": [],
    }, "/data")
    assert rec.rule == "fall"
    assert rec.to_dict()["severity"] == "orange"
    assert rec.to_dict()["details"] == {"k": 1}


def test_invalid_event_id_rejected_without_db():
    # __new__ skips __init__ (which would connect); validation short-circuits
    # before any database access, so these must not raise or connect
    store = object.__new__(PostgresEvidenceStore)
    assert store.get("..") is None
    assert store.delete("../etc/passwd") is False
    assert store.snapshot_bytes("") is None
    assert store.clip_bytes("'; DROP TABLE evidence; --") is None
    assert store.update_status("..", "resolved", "admin") is None
    assert store.add_note("..", "x", "admin") is None


@pytest.mark.skipif(PSYCOPG_INSTALLED,
                    reason="psycopg installed here; error path covered on fresh envs")
def test_missing_driver_raises_clear_error():
    import bhairav.backend.pg_store as pg
    pg._DRIVER = None  # force the import path
    with pytest.raises(RuntimeError, match="psycopg"):
        load_driver()


# ---------------------------------------------------------------------------
# Integration tests (require BHAIRAV_TEST_DB_URL + psycopg)
# ---------------------------------------------------------------------------
needs_db = pytest.mark.skipif(
    not TEST_DB_URL or PSYCOPG_INSTALLED is False,
    reason="set BHAIRAV_TEST_DB_URL and install psycopg to run these")


@needs_db
def test_save_get_roundtrip():
    store = _fresh_store()
    rec = store.save(_make_event(n_frames=2))
    got = store.get(rec.event_id)
    assert got is not None
    assert got.rule == "fight"
    assert got.to_dict()["severity"] == "red"
    assert got.details["plate"] == "MH12AB1234"
    assert got.frame_count == 2
    assert store.snapshot_bytes(rec.event_id) is not None
    clip = store.clip_bytes(rec.event_id)
    assert clip is not None and len(clip) > 0
    assert store.get("ffffffffffff") is None


@needs_db
def test_save_is_idempotent_and_preserves_workflow():
    store = _fresh_store()
    rec = store.save(_make_event("abcd12345678"))
    store.update_status("abcd12345678", "acknowledged", "alice")
    store.save(_make_event("abcd12345678"))   # re-save must not reset status
    assert store.get("abcd12345678").status == "acknowledged"


@needs_db
def test_encrypted_clip_requires_right_key():
    import os
    key = os.urandom(32)
    store = _fresh_store(encrypt=True, key=key)
    rec = store.save(_make_event())
    # the API decrypts transparently (like the file store), so callers get a
    # usable mp4 back...
    clip = store.clip_bytes(rec.event_id)
    assert clip and b"ftyp" in clip[:16]
    # ...but the bytes at rest are AES-GCM ciphertext, not the mp4
    cur = store._cursor()
    cur.execute("SELECT clip FROM evidence WHERE event_id = %s", (rec.event_id,))
    raw = bytes(cur.fetchone()["clip"])
    assert b"ftyp" not in raw[:16]
    # wrong key -> unreadable (decrypt fails -> None)
    wrong = PostgresEvidenceStore(TEST_DB_URL, encrypt=True,
                                  key=os.urandom(32))
    assert wrong.clip_bytes(rec.event_id) is None     # wrong key -> unreadable


@needs_db
def test_search_filters_and_counts():
    store = _fresh_store()
    store.save(_make_event("aaaa00000001", rule="fight", severity=Severity.RED, ts=10.0))
    store.save(_make_event("bbbb00000002", rule="fall", severity=Severity.ORANGE, ts=20.0))
    store.save(_make_event("cccc00000003", rule="chase", severity=Severity.YELLOW, ts=30.0))
    assert len(store.search(rule="fall")) == 1
    assert len(store.search(severity="red")) == 1
    assert len(store.search(q="plaza")) == 3
    assert len(store.search(t0=25.0)) == 1
    assert len(store.search(limit=2)) == 2
    c = store.counts()
    assert c["total"] == 3
    assert c["by_rule"] == {"fight": 1, "fall": 1, "chase": 1}
    assert c["by_severity"] == {"red": 1, "orange": 1, "yellow": 1}
    assert c["cameras"] == ["CAM-01"]


@needs_db
def test_workflow_status_and_notes():
    store = _fresh_store()
    rec = store.save(_make_event("abcd12345678"))
    r = store.update_status("abcd12345678", "acknowledged", "alice", now=50.0)
    assert r.status == "acknowledged"
    r = store.add_note("abcd12345678", "check the plate", "bob", now=60.0)
    assert r.notes and r.notes[-1]["text"] == "check the plate"
    assert store.get("abcd12345678").status == "acknowledged"
    with pytest.raises(ValueError):
        store.update_status("abcd12345678", "bogus", "alice")


@needs_db
def test_delete_and_expire():
    store = _fresh_store()
    store.save(_make_event("aaaa00000001", ts=10.0))
    store.save(_make_event("bbbb00000002", ts=20.0))
    assert store.delete("aaaa00000001") is True
    assert store.delete("aaaa00000001") is False
    import time
    now = time.time()
    assert store.expire(max_age_days=365.0, now=now) == 0    # too young to expire
    assert store.expire(max_age_days=0.0, now=now + 10) == 1  # everything older than now+10
    assert store.counts()["total"] == 0


@needs_db
def test_max_events_prunes_oldest():
    store = _fresh_store(max_events=2)
    store.save(_make_event("aaaa00000001", ts=10.0))
    store.save(_make_event("bbbb00000002", ts=20.0))
    store.save(_make_event("cccc00000003", ts=30.0))
    assert store.counts()["total"] == 2
    assert store.get("aaaa00000001") is None     # oldest pruned
    assert store.get("cccc00000003") is not None


@needs_db
def test_export_zip():
    store = _fresh_store()
    rec = store.save(_make_event())
    zf = store.export_zip([rec])
    import zipfile, io
    with zipfile.ZipFile(io.BytesIO(zf)) as z:
        names = z.namelist()
        assert any(n.endswith("snapshot.jpg") for n in names)
        assert any(n.endswith("metadata.json") for n in names)
