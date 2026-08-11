"""Phase 9 M4 - PostgreSQL re-id store integration tests.

Gated behind $BHAIRAV_TEST_DB_URL (skip cleanly when unset). Uses a
temporary schema created on connect and verifies the PostgresReidStore
twin behaves like the file ReidStore (subjects, sightings, trails).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib.util

import pytest

PSYCOPG_INSTALLED = importlib.util.find_spec("psycopg") is not None
if PSYCOPG_INSTALLED:
    import psycopg

from bhairav.backend.pg_reid import PostgresReidStore

TEST_DB_URL = os.environ.get("BHAIRAV_TEST_DB_URL")

pytestmark = pytest.mark.skipif(
    not PSYCOPG_INSTALLED or not TEST_DB_URL,
    reason="needs BHAIRAV_TEST_DB_URL and psycopg")


def _unique_url():
    """Fresh database name per run so tests are isolated & idempotent."""
    import uuid
    base = TEST_DB_URL
    admin = base.rsplit("/", 1)[0] + "/postgres"
    new_db = f"reid_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin) as conn:
        conn.autocommit = True
        conn.execute(f'CREATE DATABASE "{new_db}"')
    url = base[:base.rfind("/") + 1] + new_db
    return url, admin, new_db


def test_pg_reid_store_roundtrip():
    url, admin, dbname = _unique_url()
    try:
        store = PostgresReidStore(url)
        desc = {"colors": [{"name": "red", "fraction": 0.7}],
                "height_class": "tall"}
        rec = store.create_subject("alice", [0.5, 0.5, 0.0, 1.0],
                                   description=desc)
        store.record_sighting(rec["id"], "CAM-01", 1, ts=1.0, frame_id=5,
                              score=0.9, bbox=[1, 2, 3, 4], thumb_b64="thumb1")
        store.record_sighting(rec["id"], "CAM-02", 7, ts=2.5, frame_id=9,
                              score=0.95, bbox=[1, 2, 3, 4], thumb_b64=None)
        # fresh connection -> data persisted
        store2 = PostgresReidStore(url)
        assert store2.get(rec["id"])["description"] == desc
        assert store2.get(rec["id"])["cameras"] == ["CAM-01", "CAM-02"]
        assert [t["camera"] for t in store2.trail(rec["id"])] == \
            ["CAM-01", "CAM-02"]
        assert store2.stats()["subjects"] == 1
        assert store2.stats()["sightings"] == 2
        # adaptive mean + rename + remove
        store2.merge_observation(rec["id"], [0.5, 0.5, 0.0, 1.0], "CAM-02", 0.8)
        assert store2.get(rec["id"])["count"] == 2
        assert store2.rename(rec["id"], "known-person")
        assert store2.get(rec["id"])["name"] == "known-person"
        assert store2.best_match([0.5, 0.5, 0.0, 1.0], 0.9) == \
            (rec["id"], pytest.approx(1.0, abs=0.01))
        assert store2.remove(rec["id"])
        assert store2.get(rec["id"]) is None
    finally:
        with psycopg.connect(admin) as conn:
            conn.autocommit = True
            # store connections hold the DB open; terminate then drop
            conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s", (dbname,))
            conn.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
