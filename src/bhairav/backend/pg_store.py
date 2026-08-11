"""PostgreSQL-backed evidence store (Phase 8).

Drop-in replacement for the file-based ``EvidenceStore`` (evidence.py) behind
the same interface, so the server, the ``EventRecorder`` and the face-search
index work against either backend:

    file store:  ``backend.db: null`` (default)  -> EvidenceStore on disk
    pg store:    ``backend.db: postgresql://...`` -> PostgresEvidenceStore
                 (also $BHAIRAV_DB_URL or ``serve.py --db-url``)

Everything that made the file store safe carries over:

  * parameterized SQL everywhere (values are never interpolated into SQL), so
    crafted event ids / search terms / notes cannot inject SQL;
  * event ids are validated against the same ``EVENT_ID_RE`` before any query;
  * evidence media (snapshot, clip) lives in BYTEA columns; with
    ``evidence.encrypt: true`` the clip is AES-256-GCM encrypted with
    ``BHAIRAV_EVIDENCE_KEY``, keeping the "wrong key -> unreadable" property;
  * workflow state (status / notes / history) is JSONB columns rather than a
    plaintext sidecar, so searches can filter on it directly.

The schema is created idempotently on first connect (``CREATE TABLE IF NOT
EXISTS``). The driver (psycopg 3) is imported lazily so the rest of the
package stays importable and testable on a minimal install; DB mode needs::

    pip install "psycopg[binary]==3.3.4"
"""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile

import cv2
import numpy as np

from .evidence import (EVENT_ID_RE, ActiveEvent, EvidenceRecord,
                       _frames_to_mp4)
from .privacy import Encryptor

# Single connection, autocommit, guarded by a lock: the app is one process
# (pipeline thread + FastAPI workers) and psycopg connections are not
# thread-safe, so every public method takes the RLock before touching it.
SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    event_id     TEXT PRIMARY KEY,
    rule         TEXT NOT NULL,
    severity     TEXT NOT NULL,
    message      TEXT NOT NULL,
    zone         TEXT,
    track_id     BIGINT,
    confidence   DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    details      JSONB NOT NULL DEFAULT '{}'::jsonb,
    start_ts     DOUBLE PRECISION NOT NULL,
    end_ts       DOUBLE PRECISION NOT NULL,
    frame_count  INTEGER NOT NULL DEFAULT 0,
    camera       TEXT NOT NULL,
    created      DOUBLE PRECISION NOT NULL,
    blurred      BOOLEAN NOT NULL DEFAULT TRUE,
    encrypted    BOOLEAN NOT NULL DEFAULT FALSE,
    status       TEXT NOT NULL DEFAULT 'new',
    notes        JSONB NOT NULL DEFAULT '[]'::jsonb,
    history      JSONB NOT NULL DEFAULT '[]'::jsonb,
    snapshot     BYTEA,
    clip         BYTEA
);
CREATE INDEX IF NOT EXISTS idx_evidence_start_ts ON evidence (start_ts DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_created  ON evidence (created);
CREATE INDEX IF NOT EXISTS idx_evidence_rule     ON evidence (rule);
CREATE INDEX IF NOT EXISTS idx_evidence_severity ON evidence (severity);
CREATE INDEX IF NOT EXISTS idx_evidence_status   ON evidence (status);
"""

# Columns written by save(); workflow columns (status/notes/history) are
# deliberately excluded so a re-save never clobbers workflow state.
_SAVE_COLS = ("event_id", "rule", "severity", "message", "zone", "track_id",
              "confidence", "details", "start_ts", "end_ts", "frame_count",
              "camera", "created", "blurred", "encrypted", "snapshot", "clip")


class _DriverMissing(RuntimeError):
    pass


_DRIVER = None  # cached psycopg module; None until first load


def load_driver():
    """Import psycopg 3 lazily; raise a clear error with the install hint."""
    global _DRIVER
    if _DRIVER is None:
        try:
            import psycopg
        except ImportError as exc:
            raise _DriverMissing(
                'PostgreSQL evidence store requires psycopg 3. '
                'Install it with:  pip install "psycopg[binary]==3.3.4"') from exc
        _DRIVER = psycopg
    return _DRIVER


def row_to_record(row: dict, root: str) -> EvidenceRecord:
    """Map a dict_row from the evidence table to an EvidenceRecord."""
    return EvidenceRecord(
        event_id=row["event_id"], rule=row["rule"], severity=row["severity"],
        message=row["message"], zone=row.get("zone"), track_id=row.get("track_id"),
        start_ts=row["start_ts"], end_ts=row["end_ts"],
        frame_count=row["frame_count"], camera=row["camera"],
        created=row["created"], blurred=row["blurred"], encrypted=row["encrypted"],
        dir=root, confidence=float(row.get("confidence") or 1.0),
        details=dict(row.get("details") or {}),
        status=row.get("status") or "new",
        notes=list(row.get("notes") or []))


def build_search_query(rule=None, severity=None, camera=None, t0=None, t1=None,
                       q=None, limit: int = 50) -> tuple[str, list]:
    """Build (SQL, params) for a search; pure function so it is unit-testable
    without a database. All values go through %s placeholders."""
    conds: list[str] = []
    params: list = []
    if rule is not None:
        conds.append("rule = %s")
        params.append(rule)
    if severity is not None:
        conds.append("severity = %s")
        params.append(severity)
    if camera is not None:
        conds.append("camera = %s")
        params.append(camera)
    if t0 is not None:
        conds.append("start_ts >= %s")
        params.append(float(t0))
    if t1 is not None:
        conds.append("start_ts <= %s")
        params.append(float(t1))
    if q:
        like = f"%{str(q).lower()}%"
        conds.append("(lower(message) LIKE %s OR lower(rule) LIKE %s "
                     "OR lower(coalesce(zone, '')) LIKE %s)")
        params += [like, like, like]
    where = f" WHERE {' AND '.join(conds)}" if conds else ""
    params.append(int(limit))
    return (f"SELECT * FROM evidence{where} "
            f"ORDER BY start_ts DESC LIMIT %s", params)


class PostgresEvidenceStore:
    """EvidenceStore-compatible backend on PostgreSQL (see module docstring)."""

    def __init__(self, url: str, camera: str = "CAM-01", fps: float = 15.0,
                 blur_faces: bool = True, encrypt: bool = False,
                 key: bytes | None = None, now: float | None = None,
                 max_events: int = 0, root: str = "output/evidence"):
        self.url = url
        self.root = root
        self.camera = camera
        self.fps = fps
        self.blur_faces = blur_faces
        self.encrypt = encrypt
        self._encryptor = Encryptor(key) if encrypt else None
        self._now = now
        self.max_events = max_events
        self._lock = threading.RLock()
        self._conn = None
        # fail fast: a typo'd URL or down DB should stop serve.py at startup
        # with a clear message, not crash the pipeline thread minutes later
        with self._lock:
            self._conn = self._connect()
            cur = self._conn.cursor()
            cur.execute(SCHEMA)

    # ---- plumbing ---------------------------------------------------------
    def _connect(self):
        psycopg = load_driver()
        try:
            conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
        except Exception as exc:  # bad URL, down server, wrong credentials
            raise RuntimeError(
                f"cannot connect to PostgreSQL ({self.url}): {exc}") from exc
        conn.row_factory = psycopg.rows.dict_row
        # psycopg3 needs explicit jsonb registration: without it jsonb
        # columns come back as raw text instead of dict/list, and dict
        # params (evidence.details) raise ProgrammingError on INSERT
        from psycopg.types.json import register_default_adapters
        register_default_adapters(conn)
        return conn

    def _cursor(self):
        if self._conn is None:
            self._conn = self._connect()
        return self._conn.cursor()

    def _validate(self, event_id: str) -> bool:
        return bool(EVENT_ID_RE.fullmatch(event_id or ""))

    # ---- write path -------------------------------------------------------
    def save(self, event: ActiveEvent) -> EvidenceRecord:
        frames: list[np.ndarray] = []
        for jpg in event.pre_frames + event.during_frames:
            arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                frames.append(arr)
        now = self._now if self._now is not None else time.time()

        if frames:
            snap_arr = frames[-1]
        else:
            snap_arr = np.zeros((240, 320, 3), dtype=np.uint8)
        ok, snap = cv2.imencode(".jpg", snap_arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        snapshot = snap.tobytes() if ok else b""

        clip = _frames_to_mp4(frames, self.fps)
        if self.encrypt and self._encryptor is not None:
            clip = self._encryptor.encrypt(clip)

        meta = {
            "event_id": event.event_id,
            "rule": event.alert.rule,
            "severity": event.alert.severity.value,
            "message": event.alert.message,
            "zone": event.alert.zone,
            "track_id": event.alert.track_id,
            "confidence": event.alert.confidence,
            "details": event.alert.details,
            "start_ts": event.first_ts,
            "end_ts": event.last_ts,
            "frame_count": len(frames),
            "camera": event.camera or self.camera,
            "created": now,
            "blurred": self.blur_faces,
            "encrypted": self.encrypt,
            "snapshot": snapshot,
            "clip": clip,
        }
        cols = ", ".join(_SAVE_COLS)
        from psycopg.types.json import Jsonb
        params = [Jsonb(meta[c]) if isinstance(meta[c], (dict, list))
                  else meta[c] for c in _SAVE_COLS]
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _SAVE_COLS[1:])
        with self._lock:
            cur = self._cursor()
            cur.execute(
                f"INSERT INTO evidence ({cols}) VALUES ({', '.join(['%s'] * len(_SAVE_COLS))}) "
                f"ON CONFLICT (event_id) DO UPDATE SET {updates}",
                params)
            if self.max_events > 0:
                cur.execute("SELECT count(*) AS n FROM evidence")
                total = cur.fetchone()["n"]
                excess = total - self.max_events
                if excess > 0:
                    cur.execute(
                        "WITH doomed AS (SELECT event_id FROM evidence "
                        "ORDER BY start_ts ASC LIMIT %s) "
                        "DELETE FROM evidence "
                        "WHERE event_id IN (SELECT event_id FROM doomed)",
                        [excess])
        meta.pop("snapshot")
        meta.pop("clip")
        meta["status"] = "new"
        meta["notes"] = []
        return row_to_record(meta, self.root)

    # ---- read / search path ----------------------------------------------
    def list_all(self) -> list[EvidenceRecord]:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT * FROM evidence ORDER BY start_ts DESC")
            return [row_to_record(r, self.root) for r in cur.fetchall()]

    def get(self, event_id: str) -> EvidenceRecord | None:
        if not self._validate(event_id):
            return None
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT * FROM evidence WHERE event_id = %s", (event_id,))
            row = cur.fetchone()
            return row_to_record(row, self.root) if row else None

    def search(self, rule=None, severity=None, camera=None, t0=None, t1=None,
               q=None, limit: int = 50) -> list[EvidenceRecord]:
        sql, params = build_search_query(rule=rule, severity=severity,
                                         camera=camera, t0=t0, t1=t1, q=q,
                                         limit=limit)
        with self._lock:
            cur = self._cursor()
            cur.execute(sql, params)
            return [row_to_record(r, self.root) for r in cur.fetchall()]

    def delete(self, event_id: str) -> bool:
        if not self._validate(event_id):
            return False
        with self._lock:
            cur = self._cursor()
            cur.execute("DELETE FROM evidence WHERE event_id = %s RETURNING event_id",
                        (event_id,))
            return cur.fetchone() is not None

    def refresh(self) -> None:
        """No-op: PostgreSQL has no stale in-memory index to rebuild."""
        return None

    def expire(self, max_age_days: float, now: float | None = None) -> int:
        cutoff = (now if now is not None else time.time()) - max_age_days * 86400.0
        with self._lock:
            cur = self._cursor()
            cur.execute("DELETE FROM evidence WHERE created < %s", (cutoff,))
            return int(cur.rowcount or 0)


    # ---- workflow state ---------------------------------------------------
    def update_status(self, event_id: str, status: str, actor: str,
                      now: float | None = None) -> EvidenceRecord | None:
        if status not in ("new", "acknowledged", "resolved"):
            raise ValueError(f"invalid status {status!r}")
        if not self._validate(event_id):
            return None
        entry = json.dumps([{"ts": round(self._now if self._now is not None
                                         else (now or time.time()), 3),
                             "actor": actor, "to": status}])
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "UPDATE evidence SET status = %s, "
                "history = coalesce(history, '[]'::jsonb) || %s::jsonb "
                "WHERE event_id = %s RETURNING *",
                (status, entry, event_id))
            row = cur.fetchone()
            return row_to_record(row, self.root) if row else None

    def add_note(self, event_id: str, text: str, actor: str,
                 now: float | None = None) -> EvidenceRecord | None:
        text = (text or "").strip()
        if not self._validate(event_id):
            return None
        if not text:
            return self.get(event_id)
        entry = json.dumps([{"ts": round(self._now if self._now is not None
                                         else (now or time.time()), 3),
                             "actor": actor, "text": text[:2000]}])
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "UPDATE evidence SET notes = coalesce(notes, '[]'::jsonb) || %s::jsonb "
                "WHERE event_id = %s RETURNING *",
                (entry, event_id))
            row = cur.fetchone()
            return row_to_record(row, self.root) if row else None

    # ---- media ------------------------------------------------------------
    def snapshot_bytes(self, event_id: str) -> bytes | None:
        if not self._validate(event_id):
            return None
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT snapshot FROM evidence WHERE event_id = %s",
                        (event_id,))
            row = cur.fetchone()
            return bytes(row["snapshot"]) if row and row["snapshot"] is not None else None

    def clip_bytes(self, event_id: str) -> bytes | None:
        if not self._validate(event_id):
            return None
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT clip, encrypted FROM evidence WHERE event_id = %s",
                        (event_id,))
            row = cur.fetchone()
        if row is None or row["clip"] is None:
            return None
        data = bytes(row["clip"])
        if row["encrypted"]:
            if self._encryptor is None:
                return None
            try:
                return self._encryptor.decrypt(data)
            except Exception:  # wrong key / corrupt ciphertext
                return None
        return data

    # ---- ops / analytics --------------------------------------------------
    def counts(self) -> dict:
        with self._lock:
            cur = self._cursor()
            cur.execute(
                "SELECT count(*) AS total, "
                "coalesce(sum(octet_length(snapshot)), 0) + "
                "coalesce(sum(octet_length(clip)), 0) AS storage FROM evidence")
            row = cur.fetchone()
            total, storage = row["total"], row["storage"]

            def buckets(col: str) -> dict:
                cur.execute(f"SELECT {col} AS k, count(*) AS n FROM evidence "
                            f"GROUP BY {col}")
                return {r["k"]: r["n"] for r in cur.fetchall()}

            cur.execute("SELECT DISTINCT camera FROM evidence ORDER BY camera")
            cameras = [r["camera"] for r in cur.fetchall()]
        return {
            "total": total,
            "by_rule": buckets("rule"),
            "by_severity": buckets("severity"),
            "by_status": buckets("status"),
            "storage_bytes": storage,
            "cameras": cameras,
        }

    def export_zip(self, rows: list[EvidenceRecord]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = {"exported_at": round(time.time(), 3), "count": len(rows),
                        "events": [r.to_dict() for r in rows]}
            zf.writestr("manifest.json",
                        json.dumps(manifest, indent=2, sort_keys=True))
            for r in rows:
                prefix = f"events/{r.event_id}"
                zf.writestr(f"{prefix}/metadata.json",
                            json.dumps(r.to_dict(), indent=2, sort_keys=True))
                snap = self.snapshot_bytes(r.event_id)
                if snap:
                    zf.writestr(f"{prefix}/snapshot.jpg", snap)
        return buf.getvalue()
