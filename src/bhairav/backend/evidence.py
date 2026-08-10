"""Evidence pipeline (Phase 3): pre/during/post-event capture + storage.

    PreEventBuffer - rolling ring of recent frames (JPEG-compressed in memory)
                     kept so the N seconds *before* an alert are preserved.
    EventRecorder  - watches the frame stream; when an alert fires it opens an
                     event, captures the pre-event buffer plus the during-event
                     frames, and after `post_sec` of quiet finalizes the event
                     into the store. Frames are face-blurred *at capture time*
                     (with the real tracks/poses) so stored media is genuinely
                     privacy-safe, not just marked so.
    EvidenceStore  - on-disk storage: one directory per event with
                     metadata.json (searchable), snapshot.jpg, and clip.mp4
                     (optionally encrypted at rest).
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import uuid
import zipfile
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..types import Alert, FrameState
from .privacy import Encryptor, FaceBlur

# Event ids are uuid4 hex, 12 chars; validate before touching the filesystem
# so a crafted id like ".." can never escape the evidence root (path traversal).
EVENT_ID_RE = re.compile(r"^[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# Pre-event ring buffer
# ---------------------------------------------------------------------------
class PreEventBuffer:
    """Keeps up to `duration_sec` of recent (timestamp, jpeg-bytes) frames.

    JPEG-compressed so a 30s buffer at 720p stays well under ~50MB in memory.
    """

    def __init__(self, duration_sec: float, fps: float, jpeg_quality: int = 85):
        self.duration_sec = duration_sec
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        maxlen = max(1, int(duration_sec * fps) + 2)
        self._buf: deque[tuple[float, bytes]] = deque(maxlen=maxlen)

    def push(self, timestamp: float, frame: np.ndarray) -> None:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            self._buf.append((timestamp, jpg.tobytes()))

    def frames_before(self, timestamp: float) -> list[bytes]:
        """All buffered frames with t <= timestamp, oldest first."""
        return [jpg for t, jpg in self._buf if t <= timestamp]

    def clear(self) -> None:
        self._buf.clear()


# ---------------------------------------------------------------------------
# Event recorder
# ---------------------------------------------------------------------------
@dataclass
class ActiveEvent:
    event_id: str
    alert: Alert
    pre_frames: list[bytes]
    during_frames: list[bytes] = field(default_factory=list)
    first_ts: float = 0.0
    last_ts: float = 0.0
    # frames are stored blurred at capture time so privacy holds at rest too


class EventRecorder:
    """Turns the alert stream into finalized evidence events.

    Lifecycle per (rule, zone, track) key:
        on_alert -> opens a new event (pre-frames frozen, during starts)
        subsequent alerts with the same key extend the same event
        per-frame finalize_due() -> closes events `post_sec` after the last alert
        flush() (end of stream) -> closes everything still open

    Privacy: every stored frame is face-blurred *here* using the real
    FrameState (tracks + poses), so the JPEGs on disk are genuinely blurred -
    not just flagged as such.
    """

    def __init__(self, store: "EvidenceStore", pre_sec: float = 5.0,
                 post_sec: float = 5.0, min_gap_sec: float = 10.0,
                 blur_faces: bool = True):
        self.store = store
        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self.min_gap_sec = min_gap_sec  # min time between independent events of same key
        self._blur = FaceBlur(strength=41 if blur_faces else 0)
        self._buffer = PreEventBuffer(pre_sec, store.fps)
        self._active: dict[tuple, ActiveEvent] = {}
        self._last_event_at: dict[tuple, float] = {}

    def _key(self, alert: Alert) -> tuple:
        return (alert.rule, alert.zone, alert.track_id)

    def observe(self, state: FrameState, frame: np.ndarray | None = None) -> None:
        """Feed every frame so the pre-event buffer stays warm."""
        img = frame if frame is not None else state.frame
        if img is None:
            return
        img = self._blur.blur_frame(img, state)  # privacy at capture time
        self._buffer.push(state.timestamp, img)

    def on_alert(self, alert: Alert, frame: np.ndarray | None = None,
                 state: FrameState | None = None) -> str | None:
        """Register a new alert; returns the event_id it belongs to, if any."""
        img = frame if frame is not None else None
        if img is not None and state is not None:
            img = self._blur.blur_frame(img, state)
        key = self._key(alert)
        if key in self._active:
            ev = self._active[key]
            ev.last_ts = max(ev.last_ts, alert.timestamp)
            if img is not None:
                ev.during_frames.append(self._encode(img))
            return ev.event_id
        # cooldown between independent events of the same kind
        last = self._last_event_at.get(key)
        if last is not None and alert.timestamp - last < self.min_gap_sec:
            return None
        ev = ActiveEvent(
            event_id=uuid.uuid4().hex[:12],
            alert=alert,
            pre_frames=self._buffer.frames_before(alert.timestamp),
            first_ts=alert.timestamp,
            last_ts=alert.timestamp,
        )
        if img is not None:
            ev.during_frames.append(self._encode(img))
        self._active[key] = ev
        self._last_event_at[key] = alert.timestamp
        return ev.event_id

    def _encode(self, frame: np.ndarray) -> bytes:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes() if ok else b""

    def finalize_due(self, now: float) -> list[str]:
        """Finalize events whose last alert is older than `post_sec`.

        Call once per frame with the current timestamp so events close shortly
        after they go quiet (and the same key can open a new, independent
        event once `min_gap_sec` has passed).
        """
        done: list[str] = []
        for key, ev in list(self._active.items()):
            if now - ev.last_ts >= self.post_sec:
                self.store.save(ev)
                del self._active[key]
                done.append(ev.event_id)
        return done

    def flush(self, now: float | None = None) -> list[str]:
        """Finalize every open event (end of stream)."""
        done: list[str] = []
        for key, ev in list(self._active.items()):
            self.store.save(ev)
            del self._active[key]
            done.append(ev.event_id)
        return done

    def reset(self) -> None:
        """Clear per-replay state (for a looping/live source): the scene clock
        restarts at 0 each pass, so cooldown timestamps must reset too, or the
        second pass would suppress every event."""
        self._buffer.clear()
        self._active.clear()
        self._last_event_at.clear()

    def active_count(self) -> int:
        return len(self._active)


# ---------------------------------------------------------------------------
# Evidence store
# ---------------------------------------------------------------------------
@dataclass
class EvidenceRecord:
    event_id: str
    rule: str
    severity: str
    message: str
    zone: str | None
    track_id: int | None
    start_ts: float
    end_ts: float
    frame_count: int
    camera: str
    created: float
    blurred: bool
    encrypted: bool
    dir: str
    confidence: float = 1.0
    details: dict = field(default_factory=dict)
    status: str = "new"          # new | acknowledged | resolved (workflow)
    notes: list[dict] = field(default_factory=list)  # [{ts, actor, text}]

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "zone": self.zone,
            "track_id": self.track_id,
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "frame_count": self.frame_count,
            "camera": self.camera,
            "created": round(self.created, 3),
            "blurred": self.blurred,
            "encrypted": self.encrypted,
            "confidence": round(self.confidence, 3),
            "details": self.details,
            "status": self.status,
            "notes": self.notes,
        }


class EvidenceStore:
    """Persists events as directories under `root`:

        <root>/<event_id>/metadata.json
        <root>/<event_id>/snapshot.jpg
        <root>/<event_id>/clip.mp4        (all during+pre frames)
        <root>/<event_id>/clip.bin        (encrypted variant, if encrypt=True)
    """

    def __init__(self, root: str | Path, camera: str = "CAM-01", fps: float = 15.0,
                 blur_faces: bool = True, encrypt: bool = False,
                 key: bytes | None = None, now: float | None = None,
                 max_events: int = 0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.fps = fps
        self.blur_faces = blur_faces
        self.encrypt = encrypt
        self._encryptor = Encryptor(key) if encrypt else None
        self._now = now  # injectable clock for deterministic tests
        # Ops index: the dashboard /api/status polls counts() constantly, and a
        # full disk walk was taking ~10s on a long-lived store. Instead keep an
        # in-memory index, warmed once (lazily) and updated on every mutation.
        self.max_events = max_events  # 0 = unlimited; oldest pruned when over
        self._lock = threading.RLock()
        self._counts: dict | None = None       # None until first warm
        self._storage_bytes: int | None = None
        self._order: list[str] = []            # event ids, oldest first (pruning)
        self._order_built = False
        self._recs: dict[str, EvidenceRecord] = {}   # warm in-memory record index
        self._recs_built = False

    @staticmethod
    def _validate_event_id(event_id: str) -> str:
        """Reject anything that isn't a canonical event id (path traversal)."""
        if not EVENT_ID_RE.fullmatch(event_id or ""):
            raise ValueError(f"invalid event_id {event_id!r}")
        return event_id

    def _event_dir(self, event_id: str) -> Path:
        return self.root / self._validate_event_id(event_id)

    # ---- write path -------------------------------------------------------
    # ---- ops index --------------------------------------------------------
    def _build_order(self) -> None:
        """Event ids oldest-first (by metadata mtime) for oldest-first pruning."""
        entries = []
        try:
            for d in self.root.iterdir():
                if d.is_dir():
                    meta = d / "metadata.json"
                    try:
                        mtime = meta.stat().st_mtime if meta.exists() else 0.0
                    except OSError:
                        mtime = 0.0
                    entries.append((mtime, d.name))
        except OSError:
            pass
        entries.sort()
        self._order = [name for _, name in entries]
        self._order_built = True

    def _warm_recs(self) -> None:
        """One-time full scan building the in-memory record index (search/list)."""
        if self._recs_built:
            return
        for ev_dir in sorted(self.root.iterdir()):
            if not ev_dir.is_dir():
                continue
            rec = self.get(ev_dir.name)
            if rec is not None:
                self._recs[rec.event_id] = rec
        self._recs_built = True

    def _warm_counts(self) -> None:
        """Full scan once; afterwards counts stay in sync via _index_add/_remove."""
        rows = self.list_all()
        storage = 0
        try:
            for p in self.root.rglob("*"):
                if p.is_file():
                    try:
                        storage += p.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        self._counts = {
            "total": len(rows),
            "by_rule": dict(Counter(r.rule for r in rows)),
            "by_severity": dict(Counter(r.severity for r in rows)),
            "by_status": dict(Counter(r.status for r in rows)),
            "cameras": sorted({r.camera for r in rows}),
        }
        self._storage_bytes = storage
        if not self._order_built:
            self._build_order()
        for r in rows:
            self._recs[r.event_id] = r
        self._recs_built = True

    def _index_add(self, rec: EvidenceRecord, bytes_added: int) -> None:
        with self._lock:
            if not self._order_built:
                self._build_order()
            # order tracks every saved event regardless of whether counts have
            # been warmed yet, so max_events pruning works from the first save
            if rec.event_id not in self._order:
                self._order.append(rec.event_id)
            if self._recs_built:
                self._recs[rec.event_id] = rec
            if self._counts is None:
                return
            c = self._counts
            c["total"] += 1
            c["by_rule"][rec.rule] = c["by_rule"].get(rec.rule, 0) + 1
            c["by_severity"][rec.severity] = c["by_severity"].get(rec.severity, 0) + 1
            c["by_status"][rec.status] = c["by_status"].get(rec.status, 0) + 1
            if rec.camera not in c["cameras"]:
                c["cameras"].append(rec.camera)
                c["cameras"].sort()
            self._storage_bytes = (self._storage_bytes or 0) + bytes_added

    def _index_remove(self, rec: EvidenceRecord, bytes_removed: int) -> None:
        with self._lock:
            if rec.event_id in self._order:
                self._order.remove(rec.event_id)
            self._recs.pop(rec.event_id, None)
            if self._counts is None:
                return
            c = self._counts
            c["total"] = max(0, c["total"] - 1)
            for bucket, key in ((c["by_rule"], rec.rule),
                                (c["by_severity"], rec.severity),
                                (c["by_status"], rec.status)):
                if bucket.get(key, 0) > 1:
                    bucket[key] -= 1
                else:
                    bucket.pop(key, None)
            # cameras: recomputed lazily when the index next warms
            self._storage_bytes = max(0, (self._storage_bytes or 0) - bytes_removed)

    def _index_status_move(self, event_id: str, old_status: str,
                           new_status: str) -> None:
        """Keep the by_status bucket accurate when workflow status changes."""
        with self._lock:
            if self._counts is None or old_status == new_status:
                return
            c = self._counts
            if c["by_status"].get(old_status, 0) > 1:
                c["by_status"][old_status] -= 1
            else:
                c["by_status"].pop(old_status, None)
            c["by_status"][new_status] = c["by_status"].get(new_status, 0) + 1

    def _prune_oldest(self) -> int:
        """Delete oldest events until under max_events; returns count removed.

        Only forgets an event in the in-memory index once its directory is
        *really* gone. A transient lock (OneDrive/Windows Defender scanning a
        just-written clip) can make rmtree fail silently; the id stays in the
        prune order and is retried on the next pass instead of being lost,
        which would otherwise let disk grow without bound.
        """
        if self.max_events <= 0:
            return 0
        import shutil
        with self._lock:
            if not self._order_built:
                self._build_order()
            removed = 0
            while len(self._order) > self.max_events and self._order:
                oldest = self._order[0]  # peek; only pop once deletion succeeds
                try:
                    ev_dir = self._event_dir(oldest)
                except ValueError:
                    self._order.pop(0)
                    continue
                if not ev_dir.exists():
                    self._order.pop(0)
                    removed += 1
                    continue
                rec = self.get(oldest)
                size = _dir_size(ev_dir)
                try:
                    shutil.rmtree(ev_dir)
                except OSError:
                    pass
                if ev_dir.exists():
                    # locked: stop this pass; the id stays at the head of the
                    # order and the next save() retries it
                    break
                self._order.pop(0)
                if rec is not None:
                    self._index_remove(rec, size)
                removed += 1
            return removed

    def save(self, event: ActiveEvent) -> EvidenceRecord:
        self._validate_event_id(event.event_id)  # defensive; ids come from uuid4
        ev_dir = self.root / event.event_id
        ev_dir.mkdir(parents=True, exist_ok=True)
        now = self._now if self._now is not None else time.time()

        frames: list[np.ndarray] = []
        for jpg in event.pre_frames + event.during_frames:
            arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                frames.append(arr)

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
            "camera": self.camera,
            "created": now,
            "blurred": self.blur_faces,
            "encrypted": self.encrypt,
        }

        # snapshot = last frame (already blurred at capture time by the
        # recorder), else blank
        if frames:
            snap = frames[-1]
        else:
            snap = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(ev_dir / "snapshot.jpg"), snap)

        if frames:
            clip_path = ev_dir / ("clip.bin" if self.encrypt else "clip.mp4")
            if self.encrypt and self._encryptor is not None:
                clip_path.write_bytes(self._encryptor.encrypt(_frames_to_mp4(frames, self.fps)))
            else:
                _frames_to_mp4(frames, self.fps, clip_path)

        # atomic writes (tmp + rename): a concurrent REST reader can never
        # observe a partially-written metadata.json
        if self.encrypt and self._encryptor is not None:
            _atomic_write(ev_dir / "metadata.enc",
                          self._encryptor.encrypt_json(meta), binary=True)
            _atomic_write(ev_dir / "metadata.json",
                          json.dumps({"encrypted": True, "event_id": event.event_id})
                          .encode("utf-8"), binary=True)
        else:
            _atomic_write(ev_dir / "metadata.json",
                          json.dumps(meta, sort_keys=True).encode("utf-8"), binary=True)

        rec = EvidenceRecord(
            event_id=event.event_id, rule=meta["rule"], severity=meta["severity"],
            message=meta["message"], zone=meta["zone"], track_id=meta["track_id"],
            start_ts=meta["start_ts"], end_ts=meta["end_ts"],
            frame_count=meta["frame_count"], camera=meta["camera"],
            created=meta["created"], blurred=meta["blurred"], encrypted=meta["encrypted"],
            dir=str(ev_dir), confidence=meta["confidence"], details=meta["details"])
        self._index_add(rec, _dir_size(ev_dir))
        self._prune_oldest()
        return rec

    # ---- read / search path ----------------------------------------------
    def list_all(self) -> list[EvidenceRecord]:
        """All events, newest-first (sorts a warm in-memory index; the one-time
        full scan happens lazily, then stays in sync with save/delete/status)."""
        with self._lock:
            self._warm_recs()
            return sorted(self._recs.values(), key=lambda r: r.start_ts, reverse=True)

    def get(self, event_id: str) -> EvidenceRecord | None:
        try:
            ev_dir = self._event_dir(event_id)
        except ValueError:
            return None
        meta_path = ev_dir / "metadata.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if meta.get("encrypted"):
            enc_path = ev_dir / "metadata.enc"
            if not enc_path.exists() or self._encryptor is None:
                return None
            try:
                meta = self._encryptor.decrypt_json(enc_path.read_bytes())
            except Exception:  # wrong key / corrupt ciphertext
                return None
        state = self._read_state(event_id)
        return EvidenceRecord(
            event_id=meta["event_id"], rule=meta["rule"], severity=meta["severity"],
            message=meta["message"], zone=meta.get("zone"), track_id=meta.get("track_id"),
            start_ts=meta["start_ts"], end_ts=meta["end_ts"],
            frame_count=meta["frame_count"], camera=meta["camera"],
            created=meta["created"], blurred=meta["blurred"], encrypted=meta["encrypted"],
            dir=str(ev_dir), confidence=meta.get("confidence", 1.0),
            details=meta.get("details", {}),
            status=state.get("status", "new"),
            notes=state.get("notes", []))

    def search(self, rule: str | None = None, severity: str | None = None,
               camera: str | None = None, t0: float | None = None,
               t1: float | None = None, q: str | None = None,
               limit: int = 50) -> list[EvidenceRecord]:
        """Search evidence by any metadata field (+ free-text on message)."""
        rows = self.list_all()
        if rule is not None:
            rows = [r for r in rows if r.rule == rule]
        if severity is not None:
            rows = [r for r in rows if r.severity == severity]
        if camera is not None:
            rows = [r for r in rows if r.camera == camera]
        if t0 is not None:
            rows = [r for r in rows if r.start_ts >= t0]
        if t1 is not None:
            rows = [r for r in rows if r.start_ts <= t1]
        if q:
            ql = q.lower()
            rows = [r for r in rows
                    if ql in r.message.lower() or ql in r.rule.lower()
                    or ql in (r.zone or "").lower()]
        rows.sort(key=lambda r: r.start_ts, reverse=True)
        return rows[:limit]

    def delete(self, event_id: str) -> bool:
        try:
            ev_dir = self._event_dir(event_id)
        except ValueError:
            return False
        if not (ev_dir / "metadata.json").exists():
            return False
        rec = self.get(event_id)
        self._index_remove(rec, _dir_size(ev_dir)) if rec else None
        import shutil
        shutil.rmtree(ev_dir, ignore_errors=True)
        return True

    def refresh(self) -> None:
        """Rebuild the warm index from disk (after external edits)."""
        with self._lock:
            self._counts = None
            self._storage_bytes = None
            self._order_built = False
            self._recs = {}
            self._recs_built = False

    def expire(self, max_age_days: float, now: float | None = None) -> int:
        """Delete events older than max_age_days; returns count removed."""
        from .privacy import expire_evidence_dir
        removed = expire_evidence_dir(self.root, max_age_days, now=now)
        with self._lock:
            # external deletion happened under us: rebuild the index on next read
            self._counts = None
            self._storage_bytes = None
            self._order_built = False
            self._recs = {}
            self._recs_built = False
        return removed

    # ---- clip access ------------------------------------------------------
    def clip_bytes(self, event_id: str) -> bytes | None:
        rec = self.get(event_id)
        if rec is None:
            return None
        try:
            ev_dir = self._event_dir(event_id)
        except ValueError:
            return None
        if rec.encrypted:
            clip = ev_dir / "clip.bin"
            if not clip.exists() or self._encryptor is None:
                return None
            try:
                return self._encryptor.decrypt(clip.read_bytes())
            except Exception:  # wrong key / corrupt ciphertext
                return None
        clip = ev_dir / "clip.mp4"
        if not clip.exists():
            return None
        return clip.read_bytes()

    def snapshot_bytes(self, event_id: str) -> bytes | None:
        rec = self.get(event_id)
        if rec is None:
            return None
        try:
            snap = self._event_dir(event_id) / "snapshot.jpg"
        except ValueError:
            return None
        if not snap.exists():
            return None
        return snap.read_bytes()

    # ---- workflow state (plaintext sidecar; sealed media is untouched) ----
    def _state_path(self, event_id: str) -> Path:
        return self._event_dir(event_id) / "state.json"

    def _read_state(self, event_id: str) -> dict:
        p = self._state_path(event_id)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return {}

    def _write_state(self, event_id: str, state: dict) -> None:
        _atomic_write(self._state_path(event_id),
                      json.dumps(state, sort_keys=True).encode("utf-8"), binary=True)

    def update_status(self, event_id: str, status: str, actor: str,
                      now: float | None = None) -> EvidenceRecord | None:
        """Set workflow status; returns the updated record or None."""
        if status not in ("new", "acknowledged", "resolved"):
            raise ValueError(f"invalid status {status!r}")
        before = self.get(event_id)
        if before is None:
            return None
        state = self._read_state(event_id)
        state["status"] = status
        state.setdefault("history", []).append({
            "ts": round(self._now if self._now is not None else (now or time.time()), 3),
            "actor": actor, "to": status,
        })
        self._write_state(event_id, state)
        self._index_status_move(event_id, before.status, status)
        with self._lock:
            if self._recs_built:
                self._recs[event_id] = self.get(event_id)
        return self.get(event_id)

    def add_note(self, event_id: str, text: str, actor: str,
                 now: float | None = None) -> EvidenceRecord | None:
        """Append an analyst note; returns the updated record or None."""
        text = (text or "").strip()
        if not text:
            return self.get(event_id)
        if self.get(event_id) is None:
            return None
        state = self._read_state(event_id)
        state.setdefault("notes", []).append({
            "ts": round(self._now if self._now is not None else (now or time.time()), 3),
            "actor": actor, "text": text[:2000],
        })
        self._write_state(event_id, state)
        with self._lock:
            if self._recs_built:
                self._recs[event_id] = self.get(event_id)
        return self.get(event_id)

    # ---- ops / analytics --------------------------------------------------
    def counts(self) -> dict:
        """Evidence totals: by rule, by severity, by status, storage bytes.

        O(1) after the first call: the index is warmed once (full scan) and
        then maintained incrementally by save/delete/expire, so the dashboard
        Status tab stays fast no matter how long the server has been running.
        """
        with self._lock:
            if self._counts is None:
                self._warm_counts()
            c = self._counts
            return {
                "total": c["total"],
                "by_rule": dict(c["by_rule"]),
                "by_severity": dict(c["by_severity"]),
                "by_status": dict(c["by_status"]),
                "storage_bytes": self._storage_bytes or 0,
                "cameras": list(c["cameras"]),
            }

    def export_zip(self, rows: list[EvidenceRecord]) -> bytes:
        """Package search results as an in-memory zip: manifest + metadata +
        snapshot per event (clip excluded to keep exports lean)."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                "exported_at": round(time.time(), 3),
                "count": len(rows),
                "events": [r.to_dict() for r in rows],
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            for r in rows:
                prefix = f"events/{r.event_id}"
                zf.writestr(f"{prefix}/metadata.json",
                            json.dumps(r.to_dict(), indent=2, sort_keys=True))
                snap = self.snapshot_bytes(r.event_id)
                if snap:
                    zf.writestr(f"{prefix}/snapshot.jpg", snap)
        return buf.getvalue()


def _atomic_write(path: Path, data: bytes, binary: bool = True) -> None:
    """Write data atomically: write to a sibling temp file, then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _dir_size(path: Path) -> int:
    """Total bytes of the files directly under `path` (one event dir)."""
    total = 0
    try:
        for p in path.iterdir():
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _frames_to_mp4(frames: list[np.ndarray], fps: float,
                   path: Path | None = None) -> bytes:
    """Encode frames to an mp4 in memory (or to `path`), returning bytes."""
    if not frames:
        return b""
    h, w = frames[0].shape[:2]
    out = path
    tmp = None
    if out is None:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out = Path(tmp.name)
        tmp.close()
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    data = out.read_bytes()
    if tmp is not None:
        try:
            out.unlink()
        except OSError:
            pass
    return data
