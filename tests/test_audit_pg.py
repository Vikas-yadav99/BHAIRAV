"""Tests for the Phase 8 PostgreSQL audit log (pg_audit.py).

Chain-logic unit tests run everywhere; integration tests need a real
PostgreSQL and are gated behind $BHAIRAV_TEST_DB_URL, mirroring
tests/test_evidence_pg.py.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from bhairav.backend.pg_audit import PostgresAuditLog, _canonical, _line_hash

TEST_DB_URL = os.environ.get("BHAIRAV_TEST_DB_URL")
PSYCOPG_INSTALLED = importlib.util.find_spec("psycopg") is not None


# ---------------------------------------------------------------------------
# Pure-logic unit tests (no database required)
# ---------------------------------------------------------------------------
def test_canonical_hashes_exactly_like_file_backend():
    # the file backend hashes json.dumps(entry-without-_hash, sort_keys,
    # separators=(",", ":")) - byte-identical chains are a hard requirement
    entry = {"ts": 1.0, "actor": "alice", "action": "login",
             "target": "x", "detail": {"k": 1}, "prev_hash": "0" * 64,
             "_hash": "ignored"}
    line = json.dumps({k: v for k, v in entry.items() if k != "_hash"},
                      sort_keys=True, separators=(",", ":"))
    assert _line_hash(_canonical(entry)) == hashlib.sha256(line.encode()).hexdigest()


def test_line_hash_is_sha256_hex():
    assert len(_line_hash("x")) == 64


# ---------------------------------------------------------------------------
# Integration tests (require BHAIRAV_TEST_DB_URL + psycopg)
# ---------------------------------------------------------------------------
needs_db = pytest.mark.skipif(
    not TEST_DB_URL or not PSYCOPG_INSTALLED,
    reason="set BHAIRAV_TEST_DB_URL and install psycopg to run these")


@needs_db
def test_append_read_roundtrip_and_chain():
    audit = PostgresAuditLog(TEST_DB_URL)
    audit._cursor().execute("TRUNCATE audit_log RESTART IDENTITY")
    e1 = audit.append("alice", "login", "", now=10.0)
    e2 = audit.append("bob", "export_evidence", "count=5",
                      detail={"n": 5}, now=20.0)
    e3 = audit.append("alice", "delete_evidence", "ev1", now=30.0)
    assert e1["prev_hash"] == "0" * 64
    assert e2["prev_hash"] == e1["_hash"]
    assert e3["prev_hash"] == e2["_hash"]
    rows = audit.read()
    assert [r["action"] for r in rows] == ["login", "export_evidence",
                                           "delete_evidence"]
    assert rows[1]["detail"] == {"n": 5}
    ok, problems = audit.verify()
    assert ok and problems == []


@needs_db
def test_chain_identical_to_file_backend(tmp_path):
    # the same appends must produce the same chain through either backend,
    # so a DB can pick up where a JSONL log left off (or vice versa)
    from bhairav.backend.audit import AuditLog
    fl = AuditLog(tmp_path / "audit.jsonl")
    pg = PostgresAuditLog(TEST_DB_URL)
    pg._cursor().execute("TRUNCATE audit_log RESTART IDENTITY")
    appends = [
        ("alice", "login", "", None, 1.0),
        ("bob", "export_evidence", "count=2", {"n": 2}, 2.0),
        ("alice", "delete_evidence", "ev9", None, 3.0),
    ]
    for actor, action, target, detail, now in appends:
        fl.append(actor, action, target, detail=detail, now=now)
        pg.append(actor, action, target, detail=detail, now=now)
    assert fl.read() == pg.read()
    assert fl.verify() == pg.verify() == (True, [])


@needs_db
def test_verify_detects_tampered_row():
    audit = PostgresAuditLog(TEST_DB_URL)
    cur = audit._cursor()
    cur.execute("TRUNCATE audit_log RESTART IDENTITY")
    for i in range(4):
        audit.append("alice", "login", f"t{i}", now=float(i))
    ok, problems = audit.verify()
    assert ok
    # rewrite a middle row's action: its hash no longer matches and every
    # later row's prev_hash points at a hash that no longer exists
    cur.execute("UPDATE audit_log SET action = 'HACKED' WHERE id = 2")
    ok, problems = audit.verify()
    assert not ok
    assert any("hash mismatch" in pr or "broken chain" in pr for pr in problems)


@needs_db
def test_verify_detects_deleted_row():
    audit = PostgresAuditLog(TEST_DB_URL)
    cur = audit._cursor()
    cur.execute("TRUNCATE audit_log RESTART IDENTITY")
    for i in range(4):
        audit.append("alice", "login", f"t{i}", now=float(i))
    cur.execute("DELETE FROM audit_log WHERE id = 2")
    ok, problems = audit.verify()
    assert not ok
    assert any("broken chain" in pr for pr in problems)


@needs_db
def test_query_filters_and_limit_order():
    audit = PostgresAuditLog(TEST_DB_URL)
    audit._cursor().execute("TRUNCATE audit_log RESTART IDENTITY")
    for i in range(6):
        actor = "alice" if i % 2 == 0 else "bob"
        audit.append(actor, "login" if i % 3 else "logout",
                     f"t{i}", now=float(i))
    assert [r["target"] for r in audit.query(actor="alice")] == ["t0", "t2", "t4"]
    assert [r["target"] for r in audit.query(action="logout")] == ["t0", "t3"]
    assert [r["target"] for r in audit.query(actor="bob", limit=1)] == ["t5"]
    # limit returns the most recent N in chronological order
    assert [r["target"] for r in audit.query(limit=3)] == ["t3", "t4", "t5"]
