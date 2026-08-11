"""PostgreSQL-backed plate watchlist + read log (Phase 8 M3).

Drop-in replacement for the file-based ``PlateRegistry`` (anpr.py) behind the
same interface, so the ANPR rule and the /api/vehicles endpoints work
identically in DB mode. The schema is created idempotently on first connect;
the driver is imported lazily (psycopg 3), same as pg_store / pg_audit.
"""
from __future__ import annotations

import threading
import time

from .pg_store import load_driver

SCHEMA = """
CREATE TABLE IF NOT EXISTS plates_watch (
    plate  TEXT PRIMARY KEY,
    reason TEXT NOT NULL DEFAULT '',
    actor  TEXT NOT NULL DEFAULT 'admin',
    added  DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS plate_reads (
    id    BIGSERIAL PRIMARY KEY,
    plate TEXT NOT NULL,
    ts    DOUBLE PRECISION NOT NULL,
    bbox  JSONB
);
CREATE INDEX IF NOT EXISTS idx_plate_reads_ts ON plate_reads (ts DESC);
"""

MAX_READS = 500


class PostgresPlateRegistry:
    """PlateRegistry-compatible backend on PostgreSQL (see module docstring)."""

    def __init__(self, url: str, max_reads: int = MAX_READS):
        self.url = url
        self.max_reads = max_reads
        self._lock = threading.RLock()
        self._conn = None
        with self._lock:
            self._conn = self._connect()
            cur = self._conn.cursor()
            cur.execute(SCHEMA)

    # ---- plumbing ---------------------------------------------------------
    def _connect(self):
        psycopg = load_driver()
        try:
            conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
        except Exception as exc:
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

    def _prune_reads(self) -> None:
        """Keep the read log bounded (mirrors the file store's deque)."""
        cur = self._cursor()
        cur.execute("SELECT COUNT(*) AS n FROM plate_reads")
        n = cur.fetchone()["n"]
        if n > self.max_reads:
            cur.execute(
                "DELETE FROM plate_reads WHERE id IN ("
                " SELECT id FROM plate_reads ORDER BY ts ASC LIMIT %s)",
                (n - self.max_reads,))

    # ---- watchlist --------------------------------------------------------
    def watch(self, plate: str, reason: str = "", actor: str = "admin",
              now: float | None = None) -> dict:
        plate = (plate or "").strip().upper()
        if not plate or len(plate) > 16:
            raise ValueError("plate must be 1-16 alphanumeric characters")
        with self._lock:
            cur = self._cursor()
            ts = time.time() if now is None else now
            cur.execute(
                "INSERT INTO plates_watch (plate, reason, actor, added)"
                " VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (plate) DO UPDATE SET reason = EXCLUDED.reason,"
                " actor = EXCLUDED.actor, added = EXCLUDED.added",
                (plate, (reason or "")[:200], actor, ts))
            cur.execute("SELECT * FROM plates_watch WHERE plate = %s", (plate,))
            return dict(cur.fetchone())

    def unwatch(self, plate: str) -> bool:
        with self._lock:
            cur = self._cursor()
            cur.execute("DELETE FROM plates_watch WHERE plate = %s", (plate.upper(),))
            return cur.rowcount > 0

    def is_watched(self, plate: str) -> bool:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT 1 FROM plates_watch WHERE plate = %s",
                        ((plate or "").upper(),))
            return cur.fetchone() is not None

    def list_watch(self) -> list[dict]:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT * FROM plates_watch ORDER BY plate")
            return [dict(r) for r in cur.fetchall()]

    # ---- read log ---------------------------------------------------------
    def add_read(self, plate: str, ts: float, bbox=None) -> None:
        from psycopg.types.json import Jsonb
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "INSERT INTO plate_reads (plate, ts, bbox) VALUES (%s, %s, %s)",
                ((plate or "").upper(), round(ts, 3),
                 Jsonb(list(bbox)) if bbox is not None else None))
            self._prune_reads()

    def recent_reads(self, limit: int = 50) -> list[dict]:
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "SELECT plate, ts, bbox FROM plate_reads"
                " ORDER BY ts DESC LIMIT %s", (int(limit),))
            return [dict(r) for r in cur.fetchall()]
