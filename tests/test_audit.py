"""Unit tests for the Phase 3 tamper-evident audit log."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhairav.backend.audit import AuditLog


def test_append_and_query(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("alice", "login", "ui", now=1.0)
    log.append("alice", "export", "ev1", now=2.0)
    log.append("bob", "login", "ui", now=3.0)
    rows = log.query()
    assert len(rows) == 3
    assert len(log.query(actor="alice")) == 2
    assert len(log.query(action="login")) == 2
    assert len(log.query(target="ev1")) == 1


def test_chain_verifies_clean(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.append("alice", "action", f"t{i}", now=float(i))
    ok, problems = log.verify()
    assert ok, problems
    assert not problems


def test_chain_detects_tamper(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("alice", "login", "ui", now=1.0)
    log.append("bob", "export", "ev1", now=2.0)
    path = tmp_path / "audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["actor"] = "mallory"  # tamper
    lines[0] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, problems = log.verify()
    assert not ok
    assert any("hash mismatch" in p for p in problems)


def test_chain_detects_deleted_entry(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("alice", "a1", "t", now=1.0)
    log.append("alice", "a2", "t", now=2.0)
    log.append("alice", "a3", "t", now=3.0)
    path = tmp_path / "audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]  # remove the middle entry
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, problems = log.verify()
    assert not ok
    assert any("broken chain link" in p for p in problems)
