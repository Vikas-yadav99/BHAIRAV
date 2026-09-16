"""PostgreSQL-backed IncidentStore — drop-in replacement for JSONL persistence.

Usage:
    store = PgIncidentStore("postgresql://user:pass@localhost/bhairav")
    # Same API as IncidentStore from incidents.py

Requires: pip install "psycopg[binary]"
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid

log = logging.getLogger("bhairav.pg_incidents")

# Idempotent schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    emergency_level INTEGER NOT NULL DEFAULT 1,
    location_lat    DOUBLE PRECISION NOT NULL DEFAULT 0,
    location_lng    DOUBLE PRECISION NOT NULL DEFAULT 0,
    location_name   TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    reporter_phone  TEXT NOT NULL DEFAULT '',
    reporter_name   TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'reported',
    source          TEXT NOT NULL DEFAULT 'public',
    created_at      DOUBLE PRECISION NOT NULL,
    updated_at      DOUBLE PRECISION NOT NULL,
    assigned_officers JSONB NOT NULL DEFAULT '[]'::jsonb,
    resolution_notes  TEXT NOT NULL DEFAULT '',
    crowd_reports   INTEGER NOT NULL DEFAULT 1,
    ai_verified     BOOLEAN NOT NULL DEFAULT FALSE,
    timeline        JSONB NOT NULL DEFAULT '[]'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents (status);
CREATE INDEX IF NOT EXISTS idx_incidents_category ON incidents (category);
CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_emergency ON incidents (emergency_level DESC);

CREATE TABLE IF NOT EXISTS officers (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'responder',
    phone           TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'available',
    location_lat    DOUBLE PRECISION NOT NULL DEFAULT 0,
    location_lng    DOUBLE PRECISION NOT NULL DEFAULT 0,
    current_incident TEXT,
    specialty       JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_seen       DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_officers_status ON officers (status);
CREATE INDEX IF NOT EXISTS idx_officers_role ON officers (role);
"""


def _load_driver():
    """Lazy-load psycopg."""
    try:
        import psycopg
        return psycopg
    except ImportError:
        raise ImportError(
            "psycopg is required for PostgreSQL backend. "
            'Install: pip install "psycopg[binary]"'
        )


class PgIncidentStore:
    """PostgreSQL-backed incident and officer storage.

    Drop-in replacement for incidents.IncidentStore with the same public API.
    """

    def __init__(self, url: str):
        self._url = url
        self._lock = threading.RLock()
        self._connect()
        self._init_schema()

    def _connect(self):
        psycopg = _load_driver()
        self._conn = psycopg.connect(self._url, autocommit=True)

    def _init_schema(self):
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA)

    def _row_to_incident(self, row: dict):
        from bhairav.incidents import Incident
        return Incident(
            id=row["id"], category=row["category"],
            emergency_level=row["emergency_level"],
            location_lat=row["location_lat"], location_lng=row["location_lng"],
            location_name=row["location_name"], description=row["description"],
            reporter_phone=row.get("reporter_phone", ""),
            reporter_name=row.get("reporter_name", ""),
            status=row["status"], source=row["source"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            assigned_officers=row.get("assigned_officers") or [],
            resolution_notes=row.get("resolution_notes", ""),
            crowd_reports=row.get("crowd_reports", 1),
            ai_verified=row.get("ai_verified", False),
            timeline=row.get("timeline") or [],
        )

    def _row_to_officer(self, row: dict):
        from bhairav.incidents import Officer
        return Officer(
            id=row["id"], name=row["name"], role=row["role"],
            phone=row.get("phone", ""), status=row["status"],
            location_lat=row["location_lat"], location_lng=row["location_lng"],
            current_incident=row.get("current_incident"),
            specialty=row.get("specialty") or [],
            last_seen=row.get("last_seen", 0),
        )

    # ── Incidents ──────────────────────────────────────────────────────────

    def create_incident(self, category, emergency_level, lat, lng, location_name,
                        description, reporter_phone="", reporter_name="", source="public"):
        now = time.time()
        inc_id = uuid.uuid4().hex[:12]
        sql = """INSERT INTO incidents
            (id, category, emergency_level, location_lat, location_lng,
             location_name, description, reporter_phone, reporter_name,
             status, source, created_at, updated_at, assigned_officers,
             resolution_notes, crowd_reports, ai_verified, timeline)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, (
                    inc_id, category, emergency_level, lat, lng,
                    location_name, description, reporter_phone, reporter_name,
                    "reported", source, now, now, [], "", 1, False, []
                ))
        log.info("Incident created: %s (%s L%d) at %s", inc_id, category, emergency_level, location_name)
        return inc_id

    def get_incident(self, inc_id: str):
        sql = "SELECT * FROM incidents WHERE id = %s"
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, (inc_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_incident(dict(row))

    def list_incidents(self, status=None, category=None, limit=50):
        clauses = []
        params = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        if category:
            clauses.append("category = %s")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM incidents {where} ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return [self._row_to_incident(dict(r)) for r in cur.fetchall()]

    def update_incident(self, inc_id: str, **kwargs):
        allowed = {"status", "emergency_level", "location_name", "description",
                    "assigned_officers", "resolution_notes", "ai_verified", "timeline"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        values = list(updates.values()) + [inc_id]
        sql = f"UPDATE incidents SET {set_clause} WHERE id = %s"
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, values)
        return True

    def count_incidents(self, status=None):
        if status:
            sql = "SELECT COUNT(*) FROM incidents WHERE status = %s"
            params = (status,)
        else:
            sql = "SELECT COUNT(*) FROM incidents"
            params = ()
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()[0]

    # ── Officers ───────────────────────────────────────────────────────────

    def add_officer(self, officer_id: str, name: str, role: str = "responder",
                    phone: str = "", lat: float = 0, lng: float = 0,
                    specialty: list | None = None):
        now = time.time()
        sql = """INSERT INTO officers
            (id, name, role, phone, status, location_lat, location_lng,
             current_incident, specialty, last_seen)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING"""
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, (
                    officer_id, name, role, phone, "available",
                    lat, lng, None, specialty or [], now
                ))

    def get_officer(self, officer_id: str):
        sql = "SELECT * FROM officers WHERE id = %s"
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, (officer_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_officer(dict(row))

    def list_officers(self, status=None, role=None):
        clauses = []
        params = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        if role:
            clauses.append("role = %s")
            params.append(role)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM officers {where} ORDER BY name"
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return [self._row_to_officer(dict(r)) for r in cur.fetchall()]

    def update_officer(self, officer_id: str, **kwargs):
        allowed = {"status", "location_lat", "location_lng", "current_incident",
                    "specialty", "last_seen"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        values = list(updates.values()) + [officer_id]
        sql = f"UPDATE officers SET {set_clause} WHERE id = %s"
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, values)
        return True

    def get_available_officers(self, role=None):
        clauses = ["status = 'available'"]
        params = []
        if role:
            clauses.append("role = %s")
            params.append(role)
        where = f"WHERE {' AND '.join(clauses)}"
        sql = f"SELECT * FROM officers {where}"
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return [self._row_to_officer(dict(r)) for r in cur.fetchall()]

    def close(self):
        if self._conn:
            self._conn.close()
