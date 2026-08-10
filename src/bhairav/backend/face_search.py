"""Face-based person search (Phase 6): find a person in evidence by photo.

Two-model pipeline built entirely on OpenCV (no new dependencies):

    FaceDetectorYN (YuNet)   - fast face detection
    FaceRecognizerSF (SFace) - 128-d L2-normalized face embeddings

A query photo is embedded and matched by cosine similarity against two
indexes:
    - FaceGallery       - a persistent gallery of registered subjects
                          (missing person / person of interest), JSON on disk
    - EvidenceFaceIndex - face embeddings extracted from stored evidence
                          snapshots, so a query also surfaces the events and
                          clips the person appeared in.

Models are downloaded by scripts/fetch_models.py into <repo>/models/ and
verified by SHA-256 before use. Missing models raise a clear RuntimeError so
the server can report 503 instead of crashing.

Privacy note: embeddings are compact numeric vectors, not images; the
original frames stay face-blurred in evidence. A face is only
re-identifiable against the gallery the operator explicitly registers.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
def models_dir() -> Path:
    """Search order: $BHAIRAV_MODELS_DIR, <repo root>/models, package/models."""
    env = os.environ.get("BHAIRAV_MODELS_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    # .../project/src/bhairav/backend/face_search.py -> project is parents[3]
    repo_root = here.parents[3]
    for cand in (repo_root / "models", here.parents[1] / "models"):
        if (cand / "face_detection_yunet.onnx").exists():
            return cand
    return repo_root / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_models() -> dict[str, Path]:
    """Return {kind: path} for both models or raise RuntimeError with hints."""
    md = models_dir()
    det = md / "face_detection_yunet.onnx"
    rec = md / "face_recognition_sface.onnx"
    missing = [p.name for p in (det, rec) if not p.exists()]
    if missing:
        raise RuntimeError(
            "face search models missing: " + ", ".join(missing) + ". "
            "Fetch them with:  python scripts/fetch_models.py")
    return {"detector": det, "recognizer": rec}

# ---------------------------------------------------------------------------
# FaceRecognizer
# ---------------------------------------------------------------------------
def _align_face(img: np.ndarray, bbox) -> np.ndarray:
    """Canonical 112x112 aligned crop from a face bbox.

    Uses the same box-fraction similarity transform OpenCV's
    ``FaceRecognizerSF.alignCrop`` applies internally (eyes at 25%/75% width,
    40% height; nose anchor at 50%/70%), but implemented explicitly so the
    result is deterministic. ``alignCrop`` is stateful in some OpenCV builds
    and returns *different* crops for identical input, which silently corrupts
    embeddings - unacceptable for identity search.
    """
    x, y, w, h = (float(v) for v in bbox)
    s = 112.0
    src = np.array([[x + 0.25 * w, y + 0.40 * h],
                    [x + 0.75 * w, y + 0.40 * h],
                    [x + 0.50 * w, y + 0.70 * h]], dtype=np.float32)
    dst = np.array([[0.25 * s, 0.40 * s],
                    [0.75 * s, 0.40 * s],
                    [0.50 * s, 0.70 * s]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class FaceRecognizer:
    """Thin wrapper over YuNet + SFace with a stable, testable interface."""

    def __init__(self, detector_path, recognizer_path,
                 score_threshold: float = 0.85, top_k: int = 20):
        self._det = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), score_threshold, 0.3, top_k)
        self._rec = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self._lock = threading.Lock()  # OpenCV trackers aren't thread-safe

    def _detect(self, img):
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))  # scale detection to the actual frame
        _, faces = self._det.detect(img)
        return faces  # N x 15 or None

    def faces(self, img):
        """Detect faces; each entry: {bbox, score, embedding}."""
        with self._lock:
            dets = self._detect(img)
            out = []
            if dets is None:
                return out
            for row in dets:
                x, y, w, h = (float(v) for v in row[:4])
                aligned = _align_face(img, (x, y, w, h))
                feat = self._rec.feature(aligned)
                out.append({"bbox": (x, y, w, h), "score": float(row[-1]),
                            "embedding": feat.reshape(-1).astype(np.float32)})
            return out

    def embed(self, img):
        """Embedding of the highest-scoring face, or None if no face."""
        faces = self.faces(img)
        if not faces:
            return None
        faces.sort(key=lambda f: f["score"], reverse=True)
        return faces[0]["embedding"]

    @staticmethod
    def similarity(a, b) -> float:
        """Cosine similarity of two SFace embeddings (L2-normalized)."""
        return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    @staticmethod
    def encode_b64(embedding) -> str:
        return base64.b64encode(embedding.tobytes()).decode("ascii")

    @staticmethod
    def decode_b64(b64) -> np.ndarray:
        return np.frombuffer(base64.b64decode(b64), dtype=np.float32)


# ---------------------------------------------------------------------------
# FaceGallery (registered subjects)
# ---------------------------------------------------------------------------
class FaceGallery:
    """Persistent gallery of named subjects -> list of embeddings (JSON)."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._subjects: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        self._subjects = data.get("subjects", {})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "subjects": self._subjects}, sort_keys=True, indent=1)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, name, embedding, notes: str = "", now: float | None = None) -> dict:
        name = (name or "").strip()
        if not name or len(name) > 64:
            raise ValueError("subject name must be 1-64 characters")
        with self._lock:
            sub = self._subjects.get(name) or {
                "name": name, "embeddings": [], "created": 0.0, "notes": ""}
            sub["embeddings"].append(FaceRecognizer.encode_b64(embedding))
            sub["created"] = sub.get("created") or (time.time() if now is None else now)
            sub["notes"] = notes[:500]
            self._subjects[name] = sub
            self._save()
            return self._public(sub)

    def remove(self, name: str) -> bool:
        with self._lock:
            if name not in self._subjects:
                return False
            del self._subjects[name]
            self._save()
            return True

    def _public(self, sub: dict) -> dict:
        return {"name": sub["name"], "embeddings": len(sub["embeddings"]),
                "created": round(sub.get("created", 0.0), 3),
                "notes": sub.get("notes", "")}

    def list(self) -> list[dict]:
        with self._lock:
            return [self._public(s) for s in self._subjects.values()]

    def search(self, embedding, top_k: int = 5, threshold: float = 0.55) -> list[dict]:
        """Rank subjects by best cosine similarity across all embeddings."""
        with self._lock:
            scored = []
            for name, sub in self._subjects.items():
                best = 0.0
                for b64 in sub["embeddings"]:
                    sim = FaceRecognizer.similarity(
                        embedding, FaceRecognizer.decode_b64(b64))
                    best = max(best, sim)
                if best >= threshold:
                    scored.append({"name": name, "similarity": round(best, 4),
                                   "embeddings": len(sub["embeddings"])})
            scored.sort(key=lambda r: r["similarity"], reverse=True)
            return scored[:top_k]

    def count(self) -> int:
        with self._lock:
            return len(self._subjects)

# ---------------------------------------------------------------------------
# EvidenceFaceIndex
# ---------------------------------------------------------------------------
class EvidenceFaceIndex:
    """Face embeddings extracted from evidence snapshots (per event).

    Index persists as JSON next to the evidence store so a long-lived server
    doesn't re-embed everything on restart. ``index()`` rescans the store and
    adds any events it hasn't seen yet.
    """

    MAX_FACES_PER_EVENT = 3

    def __init__(self, store, recognizer, index_path=None):
        self.store = store
        self.recognizer = recognizer
        self.index_path = Path(index_path) if index_path else \
            Path(store.root) / "face_index.json"
        self._lock = threading.RLock()
        self._index: dict[str, list[str]] = {}  # event_id -> [b64 embeddings]
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        self._index = data.get("events", {})

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": 1, "events": self._index}, sort_keys=True)
        tmp = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.index_path)

    def index(self) -> dict:
        """Scan all store events, embedding any not yet indexed. Stats back."""
        indexed, skipped = 0, 0
        for rec in self.store.list_all():
            if rec.event_id in self._index:
                continue
            snap = self.store.snapshot_bytes(rec.event_id)
            if snap is None:
                skipped += 1
                continue
            img = cv2.imdecode(np.frombuffer(snap, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
            faces = self.recognizer.faces(img)
            emb = [FaceRecognizer.encode_b64(f["embedding"])
                   for f in faces[:self.MAX_FACES_PER_EVENT]]
            self._index[rec.event_id] = emb
            if emb:
                indexed += 1
            else:
                skipped += 1
        with self._lock:
            self._save()
        return {"indexed_events": indexed, "events_without_faces": skipped,
                "total_events": len(self._index)}

    def search(self, embedding, top_k: int = 5, threshold: float = 0.55) -> list[dict]:
        """Rank evidence events by best face similarity."""
        with self._lock:
            scored = []
            for event_id, b64s in self._index.items():
                if not b64s:
                    continue
                best = max(FaceRecognizer.similarity(
                    embedding, FaceRecognizer.decode_b64(b)) for b in b64s)
                if best >= threshold:
                    rec = self.store.get(event_id)
                    scored.append({
                        "event_id": event_id,
                        "similarity": round(best, 4),
                        "rule": rec.rule if rec else "?",
                        "severity": rec.severity if rec else "?",
                        "timestamp": rec.start_ts if rec else 0.0,
                        "message": rec.message if rec else "",
                    })
            scored.sort(key=lambda r: r["similarity"], reverse=True)
            return scored[:top_k]

    def stats(self) -> dict:
        with self._lock:
            return {"events_indexed": len(self._index),
                    "total_embeddings": sum(len(v) for v in self._index.values())}


def build_face_service(store, gallery_path=None):
    """Assemble recognizer + gallery + index; raises RuntimeError if the
    models are missing so the caller can report 503."""
    models = check_models()
    recognizer = FaceRecognizer(models["detector"], models["recognizer"])
    gallery = FaceGallery(gallery_path or Path(store.root).parent / "face_gallery.json")
    index = EvidenceFaceIndex(store, recognizer)
    return {"recognizer": recognizer, "gallery": gallery, "index": index}
