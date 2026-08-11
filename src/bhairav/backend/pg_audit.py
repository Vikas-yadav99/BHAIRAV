"""PostgreSQL-backed tamper-evident audit log (Phase 8 M1.5).

Same interface as the file-based ``AuditLog`` (audit.py) - ``append`` /
``read`` / ``query`` / ``verify`` - with the identical SHA-256 hash chain, so
the server and its tests work against either backend:

    file store:  backend.db null  -> AuditLog (JSONL on disk)
    pg store:    backend.db set   -> PostgresAuditLog (audit_log table)

The chain semantics are byte-for-byte the same as the file version: each row
stores sha256 of the canonical JSON of the previous row's content (excluding
its own hash), and ``verify()`` replays the chain and reports any broken link.
Insertion order is guaranteed by a BIGSERIAL id, which plays the role of the
file's line order. The table is created idempotently on first connect and the
driver (psycopg 3) is imported lazily, mirroring pg_store.py.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id        BIGSERIAL PRIMARY KEY,
    ts        DOUBLE PRECISION NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    target    TEXT NOT NULL DEFAULT '',
    detail    JSONB NOT NULL DEFAULT '{}'::jsonb,
    prev_hash TEXT NOT NULL,
    hash      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_actor  ON audit_log (actor);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log (action);
"""

_DRIVER = None  # cached psycopg module; None until first load


def _load_driver():
    """Import psycopg 3 lazily; raise a clear error with the install hint."""
    global _DRIVER
    if _DRIVER is None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                'PostgreSQL audit log requires psycopg 3. '
                'Install it with:  pip install "psycopg[binary]==3.3.4"') from exc
        _DRIVER = psycopg
    return _DRIVER


def _canonical(entry: dict) -> str:
    """The exact JSON line the file backend hashes - keep byte-identical so
    chains written by either backend are interchangeable."""
    body = {k: v for k, v in entry.items() if k != "_hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _to_entry(row: dict) -> dict:
    """Map an audit_log row back to the file backend's entry shape."""
    return {
        "ts": row["ts"], "actor": row["actor"], "action": row["action"],
        "target": row["target"], "detail": dict(row["detail"] or {}),
        "prev_hash": row["prev_hash"], "_hash": row["hash"],
    }


class PostgresAuditLog:
    """Tamper-evident, append-only audit log on PostgreSQL (see module docstring)."""

    def __init__(self, url: str):
        self.url = url
        self._lock = threading.RLock()
        self._conn = None
        # fail fast, same as the evidence store: a dead DB must stop serve.py
        # at startup, not surface as a runtime error minutes later
        with self._lock:
            self._conn = self._connect()
            cur = self._conn.cursor()
            cur.execute(SCHEMA)

    # ---- plumbing ---------------------------------------------------------
    def _connect(self):
        psycopg = _load_driver()
        try:
            conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
        except Exception as exc:  # bad URL, down server, wrong credentials
            raise RuntimeError(
                f"cannot connect to PostgreSQL ({self.url}): {exc}") from exc
        conn.row_factory = psycopg.rows.dict_row
        from psycopg.types.json import register_default_adapters
        register_default_adapters(conn)
        return conn

    def _cursor(self):
        if self._conn is None:
            self._conn = self._connect()
        return self._conn.cursor()

    # ---- write path -------------------------------------------------------
    def append(self, actor: str, action: str, target: str = "",
               detail: dict | None = None, now: float | None = None) -> dict:
        """Append one entry and return it (already persisted)."""
        now = time.time() if now is None else now
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            prev = row["hash"] if row else "0" * 64
            entry = {
                "ts": round(now, 3), "actor": actor, "action": action,
                "target": target, "detail": detail or {}, "prev_hash": prev,
            }
            h = _line_hash(_canonical(entry))
            entry["_hash"] = h
            from psycopg.types.json import Jsonb
            cur.execute(
                "INSERT INTO audit_log (ts, actor, action, target, detail, "
                "prev_hash, hash) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)",
                (entry["ts"], entry["actor"], entry["action"], entry["target"],
                 Jsonb(entry["detail"]), entry["prev_hash"], h))
            return entry

    # ---- read / verify ----------------------------------------------------
    def read(self) -> list[dict]:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT * FROM audit_log ORDER BY id")
            return [_to_entry(r) for r in cur.fetchall()]

    def verify(self) -> tuple[bool, list[str]]:
        """Replay the hash chain; return (ok, problems)."""
        problems: list[str] = []
        prev = "0" * 64
        for i, entry in enumerate(self.read()):
            recomputed = _line_hash(_canonical(entry))
            if recomputed != entry.get("_hash"):
                problems.append(f"row {i}: hash mismatch")
            if entry.get("prev_hash") != prev:
                problems.append(f"row {i}: broken chain link")
            prev = entry.get("_hash", prev)
        return (not problems, problems)

    def query(self, actor: str | None = None, action: str | None = None,
              target: str | None = None, limit: int = 100) -> list[dict]:
        conds: list[str] = []
        params: list = []
        if actor is not None:
            conds.append("actor = %s")
            params.append(actor)
        if action is not None:
            conds.append("action = %s")
            params.append(action)
        if target is not None:
            conds.append("target = %s")
            params.append(target)
        where = f" WHERE {' AND '.join(conds)}" if conds else ""
        params.append(int(limit))
        with self._lock:
            cur = self._cursor()
            cur.execute(
                f"SELECT * FROM audit_log{where} ORDER BY id DESC LIMIT %s",
                params)
            rows = cur.fetchall()
        # last N in chronological order, matching the file backend's
        # rows[-limit:] behavior
        return [_to_entry(r) for r in reversed(rows)]
