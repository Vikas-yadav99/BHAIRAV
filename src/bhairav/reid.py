"""Person re-identification across cameras (Phase 9 M4).

Appearance-based ReID with zero new dependencies (numpy + opencv, both core
deps): a deterministic per-person descriptor built from an HSV histogram, a
2x2 spatial RGB pyramid and a small HOG, L2-normalized and compared by
cosine similarity. A shared gallery gives every camera the same identity
namespace, so the same person observed on CAM-01 and CAM-02 keeps one
``reid_id`` and builds a cross-camera trail.

Privacy: stored thumbnails blur the head band of the crop, matching the
evidence pipeline's default face blur.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

# Descriptor constants (deterministic, resolution-independent after resize)
_CROP_W, _CROP_H = 64, 128
_HUE_BINS, _SAT_BINS, _VAL_BINS = 8, 8, 8
_SPATIAL_ROWS = 3          # 3x2 HSV pyramid (top: shoulders/torso,
_SPATIAL_COLS = 2          #   middle: waist, bottom: legs)
_SPATIAL_BINS = 4          # 4x4x4 per cell
_MATCH_THRESHOLD = 0.72    # cosine similarity to count as "same person"
_SPAWN_THRESHOLD = 0.55    # above this an unknown links to an auto identity
_MIN_CROP_PX = 40          # below this the crop is too small to describe
_MAX_SIGHTINGS = 2000      # bounded sightings history (file store)

SUBJECT_ID_RE = re.compile(r"^P-[0-9a-f]{8}$")
SIGHTING_ID_RE = re.compile(r"^S-[0-9a-f]{8}$")


def _hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=(_CROP_W, _CROP_H), _blockSize=(32, 32),
        _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9)


class AppearanceExtractor:
    """Deterministic full-body appearance descriptor for one person crop."""

    def __init__(self):
        self._hog = _hog()

    def embed(self, crop: np.ndarray) -> np.ndarray | None:
        """Describe a BGR person crop; None when the crop is unusable."""
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if min(h, w) < _MIN_CROP_PX:
            return None
        resized = cv2.resize(crop, (_CROP_W, _CROP_H),
                             interpolation=cv2.INTER_AREA)
        parts: list[np.ndarray] = []
        # Each part is normalized on its own so the fixed weights below
        # actually control the balance (color dominates; HOG is texture only
        # and is down-weighted because two people often share a silhouette).
        # 1) HSV histogram (color identity)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [_HUE_BINS, _SAT_BINS, _VAL_BINS],
            [0, 180, 0, 256, 0, 256])
        parts.append(_normalized(hist))
        # 2) HSV spatial pyramid (where the color is: shoulders/torso/legs
        #    rows x left/right columns), the strongest discriminative cue
        cell_h, cell_w = _CROP_H // _SPATIAL_ROWS, _CROP_W // _SPATIAL_COLS
        for i in range(_SPATIAL_ROWS):
            for j in range(_SPATIAL_COLS):
                cell = resized[i * cell_h:(i + 1) * cell_h,
                               j * cell_w:(j + 1) * cell_w]
                ch = cv2.calcHist([cell], [0, 1, 2], None,
                                  [_SPATIAL_BINS] * 3,
                                  [0, 180, 0, 256, 0, 256])
                parts.append(_normalized(ch))
        # 3) HOG (texture) - a modest fixed weight
        hog_feat = self._hog.compute(resized)
        if hog_feat is not None and hog_feat.size:
            parts.append(_normalized(hog_feat) * 0.3)
        vec = np.concatenate(parts).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return None
        return vec / norm

    def extract_from_frame(self, frame: np.ndarray,
                           bbox) -> np.ndarray | None:
        """Crop a bbox out of a frame (clamped) and describe it."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
            return None
        return self.embed(frame[y1:y2, x1:x2])

    def crop_thumbnail(self, frame: np.ndarray, bbox,
                       blur_head: bool = True, size: int = 96) -> str | None:
        """Base64 JPEG of the person crop (head band blurred), for the UI."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2].copy()
        if blur_head and crop.shape[0] > 8:
            head = crop[:max(2, crop.shape[0] // 4), :]
            crop[:max(2, crop.shape[0] // 4), :] = cv2.GaussianBlur(
                head, (31, 31), 0)
        if crop.shape[1] > size:
            crop = cv2.resize(crop, (size, int(size * crop.shape[0]
                                               / crop.shape[1])))
        ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        return base64.b64encode(jpg.tobytes()).decode("ascii")


def _normalized(arr: np.ndarray) -> np.ndarray:
    flat = arr.flatten().astype(np.float64)
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 1e-12 else flat


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class ReidStore:
    """File-backed gallery + sightings (JSON + JSONL under a directory).

    Subjects keep an adaptive mean embedding (count = how many observations
    it averages), the set of cameras it has been seen on, and first/last
    seen timestamps. Sightings are an append-only, bounded log.
    """

    def __init__(self, path: str | Path, max_sightings: int = _MAX_SIGHTINGS):
        self.path = Path(path)
        self.max_sightings = max_sightings
        self._lock = threading.RLock()
        self._subjects: dict[str, dict] = {}
        self._sightings: list[dict] = []
        self._load()

    # ---- persistence ------------------------------------------------------
    def _load(self) -> None:
        gpath = self.path / "gallery.json"
        if gpath.exists():
            try:
                data = json.loads(gpath.read_text(encoding="utf-8"))
                self._subjects = {s["id"]: s for s in data.get("subjects", [])}
            except (ValueError, OSError):
                self._subjects = {}
        spath = self.path / "sightings.jsonl"
        if spath.exists():
            try:
                for line in spath.read_text(encoding="utf-8").splitlines():
                    try:
                        self._sightings.append(json.loads(line))
                    except ValueError:
                        continue
            except OSError:
                pass

    def _save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "subjects": sorted(
                self._subjects.values(), key=lambda s: s["id"])},
            sort_keys=True).encode("utf-8")
        # unique tmp name + retries: on Windows the target can be briefly
        # held open (OneDrive/AV) -> os.replace raises WinError 5
        gpath = self.path / "gallery.json"
        for attempt in range(5):
            try:
                tmp = self.path / f"gallery.json.tmp{os.getpid()}.{attempt}"
                tmp.write_bytes(payload)
                tmp.replace(gpath)
                break
            except OSError:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
        spath = self.path / "sightings.jsonl"
        with spath.open("w", encoding="utf-8") as fh:
            for row in self._sightings[-self.max_sightings:]:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    # ---- subjects ---------------------------------------------------------
    def create_subject(self, name: str, embedding: list,
                       notes: str = "") -> dict:
        with self._lock:
            sid = _new_id("P")
            rec = {"id": sid, "name": name or f"auto-{sid}",
                   "embedding": [round(float(v), 6) for v in embedding],
                   "count": 1, "notes": notes,
                   "cameras": [], "auto": not name,
                   "first_seen": None, "last_seen": None,
                   "created": time.time()}
            self._subjects[sid] = rec
            self._save()
            return dict(rec)

    def get(self, sid: str) -> dict | None:
        with self._lock:
            return dict(self._subjects[sid]) if sid in self._subjects else None

    def list(self) -> list[dict]:
        with self._lock:
            out = []
            for s in self._subjects.values():
                rec = dict(s)
                rec["sightings"] = sum(
                    1 for g in self._sightings if g["subject_id"] == s["id"])
                rec.pop("embedding", None)
                out.append(rec)
            return sorted(out, key=lambda s: (s.get("last_seen") or 0),
                          reverse=True)

    def remove(self, sid: str) -> bool:
        with self._lock:
            if sid not in self._subjects:
                return False
            del self._subjects[sid]
            self._sightings = [g for g in self._sightings
                               if g["subject_id"] != sid]
            self._save()
            return True

    def rename(self, sid: str, name: str) -> bool:
        with self._lock:
            rec = self._subjects.get(sid)
            if rec is None:
                return False
            rec["name"] = name
            rec["auto"] = False
            self._save()
            return True

    def merge_observation(self, sid: str, embedding: list,
                          camera: str, score: float) -> dict | None:
        """Adaptive mean update: fold a matched observation into the subject."""
        with self._lock:
            rec = self._subjects.get(sid)
            if rec is None:
                return None
            n = rec["count"]
            mean = np.array(rec["embedding"], dtype=np.float64)
            vec = np.array(embedding, dtype=np.float64)
            rec["embedding"] = [round(float(v), 6)
                                for v in (mean * n + vec) / (n + 1)]
            rec["count"] = n + 1
            rec["last_seen"] = time.time()
            if camera and camera not in rec["cameras"]:
                rec["cameras"] = rec["cameras"] + [camera]
            self._save()
            return dict(rec)

    def record_sighting(self, sid: str, camera: str, track_id: int,
                        ts: float, frame_id: int, score: float,
                        bbox: list, thumb_b64: str | None) -> dict:
        with self._lock:
            gid = _new_id("S")
            row = {"id": gid, "subject_id": sid, "camera": camera,
                   "track_id": track_id, "ts": round(ts, 3),
                   "frame_id": frame_id, "score": round(float(score), 3),
                   "bbox": [round(float(v), 1) for v in bbox],
                   "thumb": thumb_b64}
            self._sightings.append(row)
            del self._sightings[:-self.max_sightings]
            rec = self._subjects.get(sid)
            if rec is not None:
                if rec.get("first_seen") is None:
                    rec["first_seen"] = row["ts"]
                rec["last_seen"] = row["ts"]
                if camera and camera not in rec["cameras"]:
                    rec["cameras"] = rec["cameras"] + [camera]
            self._save()
            return dict(row)

    def sightings(self, subject_id: str | None = None,
                  camera: str | None = None, since: float | None = None,
                  limit: int = 100) -> list[dict]:
        with self._lock:
            rows = [dict(g) for g in self._sightings
                    if (subject_id is None or g["subject_id"] == subject_id)
                    and (camera is None or g["camera"] == camera)
                    and (since is None or g["ts"] >= since)]
        rows.sort(key=lambda r: r["ts"])
        return rows[-limit:]

    def trail(self, sid: str) -> list[dict]:
        """Chronological sightings for a subject (cross-camera trail)."""
        rows = self.sightings(subject_id=sid, limit=10000)
        seen: list[dict] = []
        for r in rows:
            # collapse consecutive sightings on the same camera+track
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
        """Highest-scoring subject above `threshold`; (subject_id, score)."""
        with self._lock:
            best_id: str | None = None
            best = 0.0
            for sid, s in self._subjects.items():
                sc = cosine(embedding, np.array(s["embedding"], dtype=np.float64))
                if sc > best:
                    best_id, best = sid, sc
        if best_id is not None and best >= threshold:
            return best_id, round(best, 4)
        return None

    def stats(self) -> dict:
        with self._lock:
            cameras = {g["camera"] for g in self._sightings}
            return {"subjects": len(self._subjects),
                    "sightings": len(self._sightings),
                    "cameras": sorted(cameras),
                    "unidentified": sum(1 for s in self._subjects.values()
                                        if s.get("auto"))}


class ReidService:
    """Shared, thread-safe ReID matcher driven by every camera pipeline.

    ``observe(frame, state, camera)`` embeds each person track, assigns it to
    a gallery subject (creating an auto identity when nothing matches), folds
    the observation into the subject's adaptive mean, and records a
    (throttled) sighting with a blurred thumbnail - all under one lock so
    parallel camera threads cannot interleave.
    """

    def __init__(self, store: ReidStore, extractor: AppearanceExtractor | None = None,
                 assign_threshold: float = 0.60,
                 sighting_gap_sec: float = 3.0):
        self.store = store
        self.extractor = extractor or AppearanceExtractor()
        self.assign_threshold = assign_threshold
        self.sighting_gap_sec = sighting_gap_sec
        self._lock = threading.RLock()
        self._last_sighting: dict[tuple, float] = {}

    def observe(self, frame, state, camera: str) -> list[dict]:
        """Process one frame's person tracks; returns per-track assignments."""
        if frame is None:
            return []
        out: list[dict] = []
        for tr in state.tracks:
            if not tr.is_person:
                continue
            emb = self.extractor.extract_from_frame(frame, tr.bbox)
            if emb is None:
                continue
            with self._lock:
                match = self.store.best_match(emb, self.assign_threshold)
                if match is None:
                    rec = self.store.create_subject("", emb.tolist())
                    sid, score = rec["id"], 0.0
                else:
                    sid, score = match
                    self.store.merge_observation(sid, emb.tolist(), camera,
                                                 score)
                thumb = self.extractor.crop_thumbnail(frame, tr.bbox)
                key = (sid, camera, tr.track_id)
                now = time.time()
                if (now - self._last_sighting.get(key, 0.0)
                        >= self.sighting_gap_sec):
                    self._last_sighting[key] = now
                    self.store.record_sighting(
                        sid, camera, tr.track_id, state.timestamp,
                        state.frame_id, score, list(tr.bbox), thumb)
                rec = self.store.get(sid)
            out.append({"track_id": tr.track_id, "subject_id": sid,
                        "name": rec["name"] if rec else sid,
                        "score": score})
        return out
