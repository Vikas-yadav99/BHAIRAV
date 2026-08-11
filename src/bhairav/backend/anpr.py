"""License-plate reading + stolen-vehicle watchlist (Phase 6).

    PlateRegistry  - persisted JSON store of watched plates (with reason) and
                     the recent plate-read log. Shared between the pipeline
                     rule and the REST API.
    PlateReader    - OCR backend for a plate image. The default ``template``
                     backend is calibrated for the synthetic scene (same
                     HERSHEY font the renderer uses), which makes the demo
                     deterministic and dependency-free. Real deployments plug
                     in ``tesseract`` / ``paddle`` backends at the same seam.
    StolenVehicleRule - per-frame: crop each vehicle's plate, OCR it, log the
                     read, and fire a red alert when the plate is watched.

The plate region is derived from the vehicle bbox with the same fractions the
synthetic renderer uses, so reads are exact on the demo scene.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..types import Alert, FrameState, Severity

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MAX_READS = 500


def plate_region(bbox: tuple) -> tuple[int, int, int, int]:
    """Plate crop (x1, y1, x2, y2) inside a vehicle bbox, pixel space.

    Fractions mirror the synthetic renderer: horizontal 7.5%..92.5% of the
    vehicle width (the 0.85-width plate), vertical 30%..95% up from bottom.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    return (int(x1 + 0.075 * w), int(y2 - 0.70 * h),
            int(x1 + 0.925 * w), int(y2 - 0.05 * h))


# ---------------------------------------------------------------------------
# Plate reader (template OCR)
# ---------------------------------------------------------------------------
class PlateReader:
    """Read a plate string from a plate image. `backend` is a seam:

    - 'template': deterministic OCR calibrated to the synthetic scene's font
      (zero extra dependencies; exact on the demo).
    - 'easyocr': deep-learning OCR (detection + recognition) that works on
      real-world plates. Needs `pip install easyocr` (pulls torch); models are
      downloaded on first use. Falls back to 'template' if unavailable.
    """

    def __init__(self, backend: str = "template", min_length: int = 4):
        self.backend = backend
        self.min_length = min_length
        self._templates = self._build_templates()
        self._easy = None
        self._easy_failed = False

    # -- easyocr backend (real-world plates) --------------------------------
    def _easyocr(self):
        """Lazy singleton EasyOCR reader; None when unavailable."""
        if self.backend != "easyocr" or self._easy_failed:
            return None
        if self._easy is None:
            try:
                import easyocr
                self._easy = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception:
                self._easy_failed = True
                print("[anpr] easyocr unavailable - falling back to template "
                      "backend (pip install easyocr to read real plates)")
                self._easy = None
        return self._easy

    def _read_easyocr(self, reader, frame: np.ndarray,
                      bbox: tuple) -> tuple[str | None, float]:
        """OCR the whole vehicle bbox (real plates sit anywhere on the car).

        Upscales the crop for small plates, keeps candidate boxes that look
        like a plate (4+ alnum chars), and returns the best match.
        """
        x1, y1, x2, y2 = (float(v) for v in bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 12 or y2 - y1 < 8:
            return None, 0.0
        region = frame[y1:y2, x1:x2]
        region = cv2.resize(region, None, fx=2.5, fy=2.5,
                            interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(region, detail=1, paragraph=False)
        best, best_conf = None, 0.0
        # real plates can be short (3 chars), so the easyocr gate is looser
        # than the template backend's (which needs 4+ to reject noise)
        min_len = max(3, self.min_length - 1)
        for _box, txt, conf in results:
            cleaned = "".join(ch for ch in txt.upper() if ch in PLATE_CHARS)
            if len(cleaned) < min_len or conf < 0.3:
                continue
            # plates almost always contain a digit: prefer those over longer
            # words that merely sit near the plate (e.g. "ESCOLAR")
            has_digit = any(ch.isdigit() for ch in cleaned)
            score = conf + (0.15 if has_digit else 0.0)
            if score > best_conf:
                best, best_conf = cleaned, float(conf)
        return (best, best_conf) if best else (None, 0.0)

    @staticmethod
    def _build_templates() -> dict[str, np.ndarray]:
        """Render each char in the same font the synthetic renderer uses.

        Keeps the RAW gray glyph (no binary threshold): 1px AA strokes straddle
        any fixed cutoff, which produced empty/solid garbage templates. Both
        this builder and the reader trim ink with ``< 200`` and resize to
        (28, 40), so correlation compares identical gray shapes.
        """
        # must mirror the synthetic renderer's putText params (scale/thickness)
        scale, thickness = 0.5, 1
        tpl: dict[str, np.ndarray] = {}
        for ch in PLATE_CHARS:
            size = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
            w, h = size[0] + 8, size[1] + 10
            img = np.full((h, w), 255, np.uint8)
            cv2.putText(img, ch, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        0, thickness, cv2.LINE_AA)
            # trim to the glyph's ink bbox, like the scene-side crop, so the
            # correlation compares glyph shapes, not canvas padding
            ys, xs = np.where(img < 200)
            if len(xs) == 0:
                continue
            tpl[ch] = cv2.resize(img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], (28, 40))
        return tpl

    def _read_chars(self, plate_img: np.ndarray) -> str:
        """Segments glyphs by vertical projection (robust to touching chars)."""
        gray = plate_img if plate_img.ndim == 2 else \
            cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # ink = pixels clearly darker than the white plate (matches the
        # template builder's < 200 cutoff)
        col_has = (gray < 200).any(axis=0)
        # contiguous columns that contain ink -> candidate char spans
        runs = []
        in_run = False
        for i, has in enumerate(col_has):
            if has and not in_run:
                start, in_run = i, True
            elif not has and in_run:
                runs.append([start, i - 1])
                in_run = False
        if in_run:
            runs.append([start, len(col_has) - 1])
        # merge spans separated by tiny gaps; drop specks
        merged = []
        for r in runs:
            if r[1] - r[0] + 1 < 3:
                continue
            if merged and r[0] - merged[-1][1] <= 2:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        out = []
        for x0, x1 in merged:
            col = gray[:, x0:x1 + 1]
            rows = np.where((col < 200).any(axis=1))[0]
            if rows.size == 0:
                continue
            crop = col[rows[0]:rows[-1] + 1, :]
            resized = cv2.resize(crop, (28, 40))
            best, best_ch = -1.0, "?"
            for ch, t in self._templates.items():
                corr = cv2.matchTemplate(resized, t, cv2.TM_CCOEFF_NORMED)
                _, maxv, _, _ = cv2.minMaxLoc(corr)
                if maxv > best:
                    best, best_ch = float(maxv), ch
            out.append((best_ch, best))
        return "".join(ch for ch, _ in out)

    def read(self, frame: np.ndarray, bbox: tuple) -> tuple[str | None, float]:
        """Read a plate from a frame at a vehicle bbox. Returns (plate, conf)."""
        easy = self._easyocr()
        if easy is not None:
            return self._read_easyocr(easy, frame, bbox)
        return self._read_template(frame, bbox)

    def _read_template(self, frame: np.ndarray,
                       bbox: tuple) -> tuple[str | None, float]:
        """Template OCR path (synthetic-scene calibrated)."""
        x1, y1, x2, y2 = plate_region(bbox)
        h, w = frame.shape[:2]
        if x2 - x1 < 8 or y2 - y1 < 4 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return None, 0.0
        region = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # locate the white plate rectangle (robust to jitter/background): any
        # dark vehicle pixels that leak into the crop would otherwise flood
        # the projection with a full-width blob
        ys, xs = np.where(gray > 190)
        if len(xs) < 40:
            return None, 0.0
        plate = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        plate = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        text = self._read_chars(plate)
        if len(text) < self.min_length:
            return None, 0.0
        return text, 1.0  # template OCR on the synthetic scene is exact


# ---------------------------------------------------------------------------
# Plate registry (watchlist + read log)
# ---------------------------------------------------------------------------
class PlateRegistry:
    """Persisted watchlist of plates + rolling read log (thread-safe)."""

    def __init__(self, path: str | Path, max_reads: int = MAX_READS):
        self.path = Path(path)
        self.max_reads = max_reads
        self._lock = threading.RLock()
        self._watch: dict[str, dict] = {}      # plate -> {reason, actor, added}
        self._reads: deque = deque(maxlen=max_reads)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        self._watch = data.get("watch", {})
        self._reads = deque(data.get("reads", [])[-self.max_reads:], maxlen=self.max_reads)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"watch": self._watch,
                              "reads": list(self._reads)}, sort_keys=True)
        # unique tmp name + retries: on Windows the target can be briefly
        # held open (OneDrive/AV) -> os.replace raises WinError 5
        for attempt in range(5):
            try:
                tmp = self.path.with_suffix(
                    f"{self.path.suffix}.tmp{os.getpid()}.{attempt}")
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, self.path)
                break
            except OSError:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def watch(self, plate: str, reason: str = "", actor: str = "admin",
              now: float | None = None) -> dict:
        plate = (plate or "").strip().upper()
        if not plate or len(plate) > 16:
            raise ValueError("plate must be 1-16 alphanumeric characters")
        with self._lock:
            rec = self._watch.setdefault(plate, {})
            rec.update(plate=plate, reason=(reason or "")[:200], actor=actor,
                       added=time.time() if now is None else now)
            self._save()
            return rec

    def unwatch(self, plate: str) -> bool:
        with self._lock:
            if plate.upper() not in self._watch:
                return False
            del self._watch[plate.upper()]
            self._save()
            return True

    def is_watched(self, plate: str) -> bool:
        with self._lock:
            return (plate or "").upper() in self._watch

    def list_watch(self) -> list[dict]:
        with self._lock:
            return [dict(v) for v in self._watch.values()]

    def add_read(self, plate: str, ts: float, bbox=None) -> None:
        with self._lock:
            self._reads.append({"plate": (plate or "").upper(), "ts": round(ts, 3),
                                "bbox": list(bbox) if bbox else None})
            self._save()

    def recent_reads(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._reads)[-limit:][::-1]

# ---------------------------------------------------------------------------
# Stolen-vehicle rule
# ---------------------------------------------------------------------------
class StolenVehicleRule:
    """Fires a red alert when a watched plate is read on a vehicle.

    Runs per frame on every vehicle track: OCR the plate, log the read, and
    alert when the plate is on the watchlist. Cooldown is handled by the
    rules engine. `registry` may be replaced after construction (the server
    shares one registry with the REST API); a default empty one is used when
    nothing is provided (e.g. the offline demo).
    """

    name = "stolen_vehicle"
    enabled = True

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.severity = Severity(config.get("severity", "red"))
        self.min_confidence = float(config.get("min_confidence", 0.5))
        self.reader = PlateReader(backend=config.get("backend", "template"))
        self.registry = PlateRegistry(Path("output/plates.json"))

    def evaluate(self, state: FrameState, zones: list) -> list[Alert]:
        if not self.enabled or state.frame is None:
            return []
        alerts: list[Alert] = []
        for tr in state.tracks:
            if tr.is_person or tr.class_id not in (2, 5, 7):
                continue
            plate, conf = self.reader.read(state.frame, tr.bbox)
            if plate is None:
                continue
            self.registry.add_read(plate, state.timestamp, bbox=tr.bbox)
            if conf >= self.min_confidence and self.registry.is_watched(plate):
                reason = ""
                for w in self.registry.list_watch():
                    if w["plate"] == plate:
                        reason = w.get("reason", "")
                        break
                alerts.append(Alert(
                    rule=self.name, zone=None, track_id=tr.track_id,
                    severity=self.severity,
                    message=f"STOLEN VEHICLE: plate {plate} spotted - "
                            f"track #{tr.track_id}",
                    frame_id=state.frame_id, timestamp=state.timestamp,
                    details={"plate": plate, "reason": reason,
                             "confidence": round(conf, 3)},
                    confidence=conf,
                ))
        return alerts
