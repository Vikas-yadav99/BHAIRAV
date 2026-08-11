"""Phase 9 M3 - PostgreSQL backup/restore integration tests.

Gated behind $BHAIRAV_TEST_DB_URL (skip cleanly when unset). Creates two
temporary databases, seeds one with a bytea/jsonb table, dumps it with the
pure-Python backup, restores into the other, and verifies the round trip.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib.util

import pytest

PSYCOPG_INSTALLED = importlib.util.find_spec("psycopg") is not None
if PSYCOPG_INSTALLED:
    import psycopg
    from psycopg.types.json import Jsonb

from bhairav.backend.backups import (BackupService, dump, pg_metrics,
                                     restore, verify)

TEST_DB_URL = os.environ.get("BHAIRAV_TEST_DB_URL")

pytestmark = pytest.mark.skipif(
    not PSYCOPG_INSTALLED or not TEST_DB_URL,
    reason="needs BHAIRAV_TEST_DB_URL and psycopg")

_ADMIN_URL = (TEST_DB_URL.rsplit("/", 1)[0] + "/postgres"
              if TEST_DB_URL else None)


@pytest.fixture()
def scratch_dbs():
    """Two throwaway databases (source + restore target), dropped after."""
    stamp = int(time.time() * 1000)
    names = [f"bhairav_bk_a_{stamp}", f"bhairav_bk_b_{stamp}"]
    admin = psycopg.connect(_ADMIN_URL, autocommit=True)
    for n in names:
        admin.execute(f'DROP DATABASE IF EXISTS "{n}"')
        admin.execute(f'CREATE DATABASE "{n}"')
    admin.close()
    base = TEST_DB_URL.rsplit("/", 1)[0]
    yield [f"{base}/{n}" for n in names]
    admin = psycopg.connect(_ADMIN_URL, autocommit=True)
    for n in names:
        admin.execute(f'DROP DATABASE IF EXISTS "{n}"')
    admin.close()


def _seed(url):
    conn = psycopg.connect(url)
    with conn:
        cur = conn.cursor()
        cur.execute("CREATE TABLE probe (id text PRIMARY KEY, n bigint, "
                    "blob bytea, meta jsonb, f double precision)")
        cur.execute("INSERT INTO probe VALUES (%s, %s, %s, %s, %s)",
                    ("e1", 42, b"\x00\x01\x02blob",
                     Jsonb({"plate": "MH12AB1234", "tags": ["a", "b"]}), 0.5))
        cur.execute("INSERT INTO probe VALUES (%s, %s, %s, %s, %s)",
                    ("e2", -7, None, Jsonb([]), 1.25))
    conn.close()


def test_backup_restore_roundtrip(scratch_dbs):
    url_a, url_b = scratch_dbs
    _seed(url_a)
    with tempfile.TemporaryDirectory() as tmp:
        res = dump(url_a, tmp, retention=2, tables=["probe"])
        assert res["tables"] == 1
        v = verify(res["path"])
        assert v["ok"] and v["tables"][0]["rows"] == 2

        r = restore(url_b, res["path"], wipe=True, tables=["probe"])
        assert r["total_rows"] == 2

        conn = psycopg.connect(url_b)
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT id, n, blob, meta, f FROM probe ORDER BY id")
            rows = cur.fetchall()
        conn.close()
        assert rows[0][2] == b"\x00\x01\x02blob"
        assert rows[0][3] == {"plate": "MH12AB1234", "tags": ["a", "b"]}
        assert rows[1][1] == -7 and rows[1][2] is None


def test_backup_service_and_pg_metrics(scratch_dbs):
    url_a, url_b = scratch_dbs
    _seed(url_a)
    with tempfile.TemporaryDirectory() as tmp:
        svc = BackupService(url_a, tmp, retention=2)
        created = svc.create()
        name = Path(created["path"]).name
        assert svc.latest()["name"] == name
        v = svc.verify(name)
        assert v["ok"] and any(t["name"] == "probe" for t in v["tables"])
        assert svc.list() and svc.list()[0]["size_bytes"] > 0
        assert svc.read("..%2F..%2Fetc%2Fpasswd") is None
        raw = svc.read(svc.latest()["name"])
        assert raw and raw[:2] == b"\x1f\x8b"  # gzip magic

    m = pg_metrics(url_a)
    assert m["reachable"] and m["db_size_bytes"] > 0
    assert m["table_rows"].get("probe") == 2


def test_restore_is_idempotent(scratch_dbs):
    url_a, url_b = scratch_dbs
    _seed(url_a)
    with tempfile.TemporaryDirectory() as tmp:
        res = dump(url_a, tmp, tables=["probe"])
        restore(url_b, res["path"], wipe=True, tables=["probe"])
        # restore again (truncate path) must not duplicate rows
        restore(url_b, res["path"], wipe=False, tables=["probe"])
        conn = psycopg.connect(url_b)
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM probe")
            assert cur.fetchone()[0] == 2
        conn.close()
