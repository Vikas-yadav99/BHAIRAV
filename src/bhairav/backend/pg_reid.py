"""PostgreSQL-backed person re-identification store (Phase 9 M4).

Drop-in twin of the file-based ``ReidStore`` (reid.py) with the same
interface: gallery subjects with adaptive-mean embeddings and a bounded
sightings log. Like the other PG stores it is enabled by ``backend.db`` /
``BHAIRAV_DB_URL``, creates its schema idempotently on connect, and loads
psycopg lazily.
"""
from __future__ import annotations

import threading
import time

from .pg_store import load_driver
from ..reid import _new_id, cosine

from psycopg.types.json import Jsonb

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS reid_subjects (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    embedding  JSONB NOT NULL,
    count      BIGINT NOT NULL DEFAULT 1,
    notes      TEXT NOT NULL DEFAULT '',
    cameras    JSONB NOT NULL DEFAULT '[]'::jsonb,
    auto       BOOLEAN NOT NULL DEFAULT TRUE,
    description JSONB,
    first_seen DOUBLE PRECISION,
    last_seen  DOUBLE PRECISION,
    created    DOUBLE PRECISION NOT NULL
);
ALTER TABLE reid_subjects ADD COLUMN IF NOT EXISTS description JSONB;
CREATE TABLE IF NOT EXISTS reid_sightings (
    id         TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    camera     TEXT NOT NULL,
    track_id   BIGINT NOT NULL,
    ts         DOUBLE PRECISION NOT NULL,
    frame_id   BIGINT NOT NULL,
    score      DOUBLE PRECISION NOT NULL DEFAULT 0,
    bbox       JSONB NOT NULL DEFAULT '[]'::jsonb,
    thumb      TEXT
);
CREATE INDEX IF NOT EXISTS idx_reid_sightings_subject
    ON reid_sightings(subject_id, ts);
"""


class PostgresReidStore:
    """ReidStore-compatible backend on PostgreSQL (see module docstring)."""

    def __init__(self, url: str, max_sightings: int = 2000):
        self.url = url
        self.max_sightings = max_sightings
        self._lock = threading.RLock()
        self._conn = None
        with self._lock:
            self._conn = self._connect()
            cur = self._conn.cursor()
            cur.execute(SCHEMA)

    def _connect(self):
        from psycopg.types.json import register_default_adapters
        psycopg = load_driver()
        conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
        conn.row_factory = psycopg.rows.dict_row
        register_default_adapters(conn)
        return conn

    def _cursor(self):
        if self._conn is None:
            self._conn = self._connect()
        return self._conn.cursor()

    # ---- subjects ---------------------------------------------------------
    def create_subject(self, name: str, embedding: list,
                       notes: str = "", description: dict | None = None) -> dict:
        sid = _new_id("P")
        now = time.time()
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "INSERT INTO reid_subjects (id, name, embedding, count, notes,"
                " cameras, auto, description, first_seen, last_seen, created)"
                " VALUES (%s, %s, %s, 1, %s, '[]', %s, %s, NULL, NULL, %s)",
                [sid, name or f"auto-{sid}",
                 Jsonb([round(float(v), 6) for v in embedding]), notes,
                 not name, Jsonb(description or None), now])
        return self.get(sid)

    def get(self, sid: str) -> dict | None:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT * FROM reid_subjects WHERE id = %s", [sid])
            row = cur.fetchone()
            if row is None:
                return None
            return self._public(row)

    def _public(self, row) -> dict:
        return {"id": row["id"], "name": row["name"],
                "embedding": list(row["embedding"]),
                "count": int(row["count"]), "notes": row["notes"],
                "description": row["description"],
                "cameras": list(row["cameras"]),
                "auto": bool(row["auto"]),
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "created": row["created"]}

    def list(self) -> list[dict]:
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "SELECT s.*, (SELECT count(*) FROM reid_sightings g "
                "WHERE g.subject_id = s.id) AS sightings "
                "FROM reid_subjects s ORDER BY s.last_seen DESC NULLS LAST")
            out = []
            for row in cur.fetchall():
                rec = self._public(row)
                rec["sightings"] = int(row["sightings"])
                rec.pop("embedding", None)
                out.append(rec)
            return out

    def remove(self, sid: str) -> bool:
        with self._lock:
            cur = self._cursor()
            cur.execute("DELETE FROM reid_sightings WHERE subject_id = %s",
                        [sid])
            cur.execute("DELETE FROM reid_subjects WHERE id = %s", [sid])
            return cur.rowcount > 0

    def rename(self, sid: str, name: str) -> bool:
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "UPDATE reid_subjects SET name = %s, auto = FALSE "
                "WHERE id = %s", [name, sid])
            return cur.rowcount > 0

    def merge_observation(self, sid: str, embedding: list,
                          camera: str, score: float) -> dict | None:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT embedding, count, cameras FROM reid_subjects "
                        "WHERE id = %s", [sid])
            row = cur.fetchone()
            if row is None:
                return None
            n = int(row["count"])
            mean = np.array(row["embedding"], dtype=np.float64)
            vec = np.array(embedding, dtype=np.float64)
            new_emb = [round(float(v), 6) for v in (mean * n + vec) / (n + 1)]
            cameras = list(row["cameras"])
            if camera and camera not in cameras:
                cameras = cameras + [camera]
            cur.execute(
                "UPDATE reid_subjects SET embedding = %s, count = %s,"
                " cameras = %s, last_seen = %s WHERE id = %s",
                [Jsonb(new_emb), n + 1, Jsonb(cameras), time.time(), sid])
        return self.get(sid)

    def record_sighting(self, sid: str, camera: str, track_id: int,
                        ts: float, frame_id: int, score: float,
                        bbox: list, thumb_b64: str | None) -> dict:
        gid = _new_id("S")
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "INSERT INTO reid_sightings (id, subject_id, camera,"
                " track_id, ts, frame_id, score, bbox, thumb)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [gid, sid, camera, int(track_id), round(float(ts), 3),
                 int(frame_id), round(float(score), 3),
                 Jsonb([round(float(v), 1) for v in bbox]), thumb_b64])
            cur.execute(
                "UPDATE reid_subjects SET last_seen = %s,"
                " first_seen = COALESCE(first_seen, %s),"
                " cameras = CASE WHEN cameras @> %s THEN cameras"
                "                ELSE cameras || %s END"
                " WHERE id = %s",
                [round(float(ts), 3), round(float(ts), 3),
                 Jsonb([camera]), Jsonb([camera]), sid])
            self._trim(sid)
        return {"id": gid, "subject_id": sid, "camera": camera,
                "track_id": int(track_id), "ts": round(float(ts), 3),
                "frame_id": int(frame_id), "score": round(float(score), 3),
                "bbox": [round(float(v), 1) for v in bbox],
                "thumb": thumb_b64}

    def _trim(self, sid: str) -> None:
        cur = self._cursor()
        cur.execute(
            "DELETE FROM reid_sightings WHERE id IN ("
            " SELECT id FROM reid_sightings WHERE subject_id = %s"
            " ORDER BY ts DESC OFFSET %s)", [sid, self.max_sightings])

    def sightings(self, subject_id: str | None = None,
                  camera: str | None = None, since: float | None = None,
                  limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM reid_sightings WHERE TRUE"
        params: list = []
        if subject_id:
            sql += " AND subject_id = %s"
            params.append(subject_id)
        if camera:
            sql += " AND camera = %s"
            params.append(camera)
        if since is not None:
            sql += " AND ts >= %s"
            params.append(since)
        sql += " ORDER BY ts ASC LIMIT %s"
        params.append(int(limit))
        with self._lock:
            cur = self._cursor()
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def trail(self, sid: str) -> list[dict]:
        rows = self.sightings(subject_id=sid, limit=10000)
        seen: list[dict] = []
        for r in rows:
            if (seen and seen[-1]["camera"] == r["camera"]
                    and seen[-1]["track_id"] == r["track_id"]
                    and r["ts"] - seen[-1]["ts"] < 60):
                seen[-1]["ts"] = r["ts"]
                continue
            seen.append({"camera": r["camera"], "track_id": r["track_id"],
                         "ts": r["ts"], "score": r["score"],
                         "sighting_id": r["id"]})
        return seen

    def best_match(self, embedding, threshold: float) -> tuple[str, float] | None:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT id, embedding FROM reid_subjects")
            best_id: str | None = None
            best = 0.0
            for row in cur.fetchall():
                sc = cosine(embedding, np.array(row["embedding"],
                                                dtype=np.float64))
                if sc > best:
                    best_id, best = row["id"], sc
        if best_id is not None and best >= threshold:
            return best_id, round(best, 4)
        return None

    def stats(self) -> dict:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT count(*) AS n FROM reid_subjects")
            subjects = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM reid_sightings")
            sightings = cur.fetchone()["n"]
            cur.execute("SELECT DISTINCT camera FROM reid_sightings")
            cameras = sorted(r["camera"] for r in cur.fetchall())
            cur.execute("SELECT count(*) AS n FROM reid_subjects WHERE auto")
            unidentified = cur.fetchone()["n"]
        return {"subjects": int(subjects), "sightings": int(sightings),
                "cameras": cameras, "unidentified": int(unidentified)}
