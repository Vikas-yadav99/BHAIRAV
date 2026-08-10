"""Privacy layer (Phase 3): face blur, encryption at rest, evidence expiry.

    FaceBlur  - blurs the head region of every tracked person in a frame.
                Uses the pose nose keypoint when available (Phase 2), else
                falls back to the top of the bbox.
    Encryptor - AES-256-GCM (via the `cryptography` package, lazily imported)
                so evidence at rest stays confidential. Missing dependency
                raises a clean RuntimeError, mirroring mediapipe/ultralytics.
    expire    - retention policy: delete evidence older than N days.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..types import FrameState

# ---------------------------------------------------------------------------
# Face blur
# ---------------------------------------------------------------------------
# Head box = nose keypoint (normalized) expanded to a face-sized patch, or the
# top ~18% of the person bbox when no pose is available.


class FaceBlur:
    def __init__(self, strength: int = 41):
        self.strength = strength  # odd kernel size; 0 disables

    def blur_frame(self, frame: np.ndarray, state: FrameState) -> np.ndarray:
        if self.strength <= 0:
            return frame
        out = frame.copy()
        w, h = state.frame_w, state.frame_h
        nose_by_track = {p.track_id: p.keypoints[0] for p in state.poses
                         if len(p.keypoints) >= 1 and p.keypoints[0].confidence >= 0.1}
        for tr in state.tracks:
            if not tr.is_person:
                continue
            x1, y1, x2, y2 = (int(v) for v in tr.bbox)
            bw, bh = x2 - x1, y2 - y1
            if bh <= 0 or bw <= 0:
                continue
            nose = nose_by_track.get(tr.track_id)
            if nose is not None:
                # face patch centered on the nose (normalized coords)
                pw = max(bw * 0.42, 24)
                ph = max(bh * 0.30, 22)
                cx, cy = int(nose.x * w), int(nose.y * h)
                rx1 = max(0, cx - int(pw / 2))
                ry1 = max(0, cy - int(ph * 0.45))
                rx2 = min(w, cx + int(pw / 2))
                ry2 = min(h, cy + int(ph * 0.55))
            else:
                # no pose: blur the top 18% band of the person bbox
                rx1, ry1 = max(0, x1), max(0, y1)
                rx2, ry2 = min(w, x2), min(h, y1 + int(bh * 0.18))
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            roi = out[ry1:ry2, rx1:rx2]
            k = self.strength if self.strength % 2 == 1 else self.strength + 1
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            out[ry1:ry2, rx1:rx2] = blurred
        return out


# ---------------------------------------------------------------------------
# Encryption at rest (AES-256-GCM)
# ---------------------------------------------------------------------------
class Encryptor:
    """AES-256-GCM envelope: nonce(12) || ciphertext || tag.

    Lazily imports `cryptography` so the rest of the backend stays importable
    on a minimal install; a missing dependency raises a clean RuntimeError.
    """

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("Encryptor key must be 32 bytes (AES-256)")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "encryption at rest requires the 'cryptography' package; "
                "install it with: pip install cryptography") from exc
        self._aesgcm = AESGCM(key)

    def encrypt(self, plaintext: bytes) -> bytes:
        import os
        nonce = os.urandom(12)
        return nonce + self._aesgcm.encrypt(nonce, plaintext, None)

    def decrypt(self, blob: bytes) -> bytes:
        if len(blob) < 12 + 16:
            raise ValueError("ciphertext too short")
        nonce, ct = blob[:12], blob[12:]
        return self._aesgcm.decrypt(nonce, ct, None)

    def encrypt_json(self, obj: dict) -> bytes:
        return self.encrypt(json.dumps(obj, sort_keys=True).encode("utf-8"))

    def decrypt_json(self, blob: bytes) -> dict:
        return json.loads(self.decrypt(blob).decode("utf-8"))

    @classmethod
    def new_key(cls) -> bytes:
        import os
        return os.urandom(32)


# ---------------------------------------------------------------------------
# Retention / expiry
# ---------------------------------------------------------------------------
def expire_evidence_dir(root: str | Path, max_age_days: float,
                        now: float | None = None) -> int:
    """Delete event subdirectories older than `max_age_days`.

    Returns the number of directories removed. An event dir is any directory
    directly under `root` containing a metadata.json file.
    """
    root = Path(root)
    if not root.exists():
        return 0
    now = time.time() if now is None else now
    cutoff = now - max_age_days * 86400
    removed = 0
    for child in root.iterdir():
        if child.is_dir() and (child / "metadata.json").exists():
            try:
                mtime = (child / "metadata.json").stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                _rmtree(child)
                removed += 1
    return removed


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)
