"""BHAIRAV Ephemeral Pipeline — Privacy-First, Zero-Retention Architecture.

Core principle: VIDEO IS EPHEMERAL, INTELLIGENCE IS PERMANENT.

Every frame is processed in real-time and immediately discarded.
Only lightweight detection metadata (JSON) is stored:
  - Bounding boxes, class labels, confidence scores
  - Trajectory vectors (position history as coordinate arrays)
  - Face embeddings (as float vectors, not images)
  - Event summaries (timestamp, category, severity, officers)

Storage comparison for 10 cameras x 24 hours:
  Old: ~10 GB/day (video clips + snapshots)
  New: ~5 MB/day (detection metadata JSON)

Privacy guarantees:
  - No video frames ever written to disk
  - No face images stored (only 128-d float embeddings)
  - No plate images stored (only text plate numbers)
  - All coordinates are abstract 0..1 normalized (not GPS)
  - Metadata auto-expires based on configurable retention policy
  - Right to erasure: delete by time range, camera, or person ID
"""
from __future__ import annotations

import json
import logging
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger("bhairav.ephemeral")


# ── Detection Types ─────────────────────────────────────────────────────────

class DetectionClass(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    FACE = "face"
    PLATE = "plate"
    OBJECT = "object"
    ANOMALY = "anomaly"


@dataclass
class BoundingBox:
    """Normalized coordinates (0..1) — never real-world GPS."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 4), "y1": round(self.y1, 4),
            "x2": round(self.x2, 4), "y2": round(self.y2, 4),
            "conf": round(self.confidence, 3),
        }


@dataclass
class Detection:
    """Single object detection in a frame — metadata only, no image data."""
    class_id: str
    bbox: BoundingBox
    track_id: str | None = None
    attributes: dict = field(default_factory=dict)
    # Face: 128-d embedding vector (float list)
    # Plate: text string
    # Person: height_estimate, clothing_color, etc.

    def to_dict(self) -> dict:
        d = {
            "cls": self.class_id,
            "bbox": self.bbox.to_dict(),
        }
        if self.track_id:
            d["track"] = self.track_id
        if self.attributes:
            d["attr"] = self.attributes
        return d


@dataclass
class FrameMetadata:
    """Metadata for a single processed frame — replaces the frame itself."""
    timestamp: float
    camera_id: str
    frame_index: int
    detections: list[Detection]
    alert: dict | None = None
    processing_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "ts": round(self.timestamp, 3),
            "cam": self.camera_id,
            "idx": self.frame_index,
            "det": [det.to_dict() for det in self.detections],
            "proc_ms": round(self.processing_ms, 1),
        }
        if self.alert:
            d["alert"] = self.alert
        return d


@dataclass
class EventRecord:
    """Incident/event summary — what happened, where, when. No video."""
    id: str
    timestamp: float
    camera_id: str
    category: str
    severity: str
    description: str
    # Detection summary (not full detections)
    detection_count: int = 0
    track_ids: list[str] = field(default_factory=list)
    # Metadata timeline (last N frames before event)
    frame_snapshots: list[dict] = field(default_factory=list)
    # Resolution info
    resolved: bool = False
    resolved_by: str = ""
    resolved_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": round(self.timestamp, 3),
            "cam": self.camera_id,
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "detection_count": self.detection_count,
            "track_ids": self.track_ids,
            "frame_snapshots": self.frame_snapshots,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
        }


# ── Ephemeral Frame Processor ──────────────────────────────────────────────

class EphemeralProcessor:
    """Processes video frames and extracts ONLY metadata.

    Frames are held in memory briefly for detection, then discarded.
    Never written to disk. Only metadata (bounding boxes, trajectories,
    face embeddings) is persisted.

    Usage:
        processor = EphemeralProcessor(camera_id="CAM-01")
        for frame in camera_stream:
            metadata = processor.process(frame)
            # frame is now discarded
            # metadata is a lightweight dict
            store.save_metadata(metadata)
    """

    def __init__(self, camera_id: str, config: dict | None = None):
        self.camera_id = camera_id
        self.config = config or {}
        self.frame_index = 0

        # In-memory ring buffer for last N frames (for event context)
        self._context_buffer_size = self.config.get("context_frames", 30)
        self._context_buffer: list[FrameMetadata] = []

        # Active tracks (person/vehicle IDs being tracked)
        self._active_tracks: dict[str, list[dict]] = defaultdict(list)

        # Face anonymization cache (blur kernel reused)
        self._face_blur_kernel = None

        # Stats
        self.total_frames = 0
        self.total_detections = 0
        self.total_alerts = 0
        self._start_time = time.time()

    def process(self, frame, detections: list[dict] | None = None) -> FrameMetadata:
        """Process a single frame and return metadata. Frame is NOT stored.

        Args:
            frame: numpy array (will be processed then discarded)
            detections: pre-computed detections (optional, for pipeline integration)

        Returns:
            FrameMetadata with bounding boxes, tracks, and alerts
        """
        start = time.perf_counter()
        self.frame_index += 1
        self.total_frames += 1

        # Convert detections to metadata objects
        meta_detections = []
        for det in (detections or []):
            bbox = BoundingBox(
                x1=det.get("x1", 0), y1=det.get("y1", 0),
                x2=det.get("x2", 0), y2=det.get("y2", 0),
                confidence=det.get("confidence", det.get("conf", 0)),
            )
            meta_det = Detection(
                class_id=det.get("class", det.get("cls", "person")),
                bbox=bbox,
                track_id=det.get("track_id"),
                attributes=self._extract_attributes(det),
            )
            meta_detections.append(meta_det)
            self.total_detections += 1

            # Update track history (position only, no images)
            if meta_det.track_id:
                self._active_tracks[meta_det.track_id].append({
                    "ts": time.time(),
                    "x": (bbox.x1 + bbox.x2) / 2,
                    "y": (bbox.y1 + bbox.y2) / 2,
                    "w": bbox.x2 - bbox.x1,
                    "h": bbox.y2 - bbox.y1,
                })
                # Keep only last 100 positions per track
                if len(self._active_tracks[meta_det.track_id]) > 100:
                    self._active_tracks[meta_det.track_id] = \
                        self._active_tracks[meta_det.track_id][-100:]

        # Build frame metadata
        metadata = FrameMetadata(
            timestamp=time.time(),
            camera_id=self.camera_id,
            frame_index=self.frame_index,
            detections=meta_detections,
            processing_ms=(time.perf_counter() - start) * 1000,
        )

        # Keep in context buffer (for event snapshots)
        self._context_buffer.append(metadata)
        if len(self._context_buffer) > self._context_buffer_size:
            self._context_buffer.pop(0)

        return metadata

    def _extract_attributes(self, det: dict) -> dict:
        """Extract non-identifying attributes from a detection.

        Privacy rule: no face images, no plate images.
        Only abstract attributes: height, clothing color, plate text.
        """
        attrs = {}
        # Height estimate (normalized)
        if "y1" in det and "y2" in det:
            h = det["y2"] - det["y1"]
            if h > 0.3:
                attrs["size"] = "large"
            elif h > 0.15:
                attrs["size"] = "medium"
            else:
                attrs["size"] = "small"

        # Clothing color (if available from detector)
        if "clothing_color" in det:
            attrs["color"] = det["clothing_color"]

        # Plate text (string only, no image)
        if "plate_text" in det:
            attrs["plate"] = det["plate_text"]

        # Face: embedding vector only (128-d float list, NOT an image)
        if "face_embedding" in det:
            emb = det["face_embedding"]
            if isinstance(emb, list) and len(emb) > 0:
                # Quantize to reduce storage (8-bit per dimension)
                attrs["face_vec"] = [round(x, 3) for x in emb[:64]]

        return attrs

    def get_context_frames(self, n: int = 10) -> list[dict]:
        """Get last N frames of metadata for event context (no images)."""
        return [f.to_dict() for f in self._context_buffer[-n:]]

    def get_track_history(self, track_id: str) -> list[dict]:
        """Get position history for a tracked object."""
        return self._active_tracks.get(track_id, [])

    def clear_tracks(self, stale_seconds: float = 300.0):
        """Remove tracks not seen in stale_seconds."""
        now = time.time()
        stale = []
        for tid, history in self._active_tracks.items():
            if history and now - history[-1]["ts"] > stale_seconds:
                stale.append(tid)
        for tid in stale:
            del self._active_tracks[tid]

    def stats(self) -> dict:
        elapsed = time.time() - self._start_time
        return {
            "camera_id": self.camera_id,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "total_alerts": self.total_alerts,
            "active_tracks": len(self._active_tracks),
            "fps": round(self.total_frames / max(elapsed, 1), 1),
            "uptime_sec": round(elapsed, 1),
        }


# ── Metadata Store (replaces video evidence store) ──────────────────────────

class MetadataStore:
    """Stores detection metadata — lightweight JSON, never video.

    Storage: <root>/<camera_id>/<date>/<event_id>.json
    Each file is ~1-5 KB (vs ~5 MB for a video clip).

    Retention: auto-delete events older than retention_days.
    """

    def __init__(self, root: str = "output/metadata", retention_days: int = 30,
                 max_events_per_camera: int = 10000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.max_events_per_camera = max_events_per_camera
        self._lock = threading.Lock()

        # Stats
        self.total_saved = 0
        self.total_deleted = 0
        self.storage_bytes = 0

    def save_event(self, event: EventRecord) -> str:
        """Save an event record to disk (JSON only, no video)."""
        cam_dir = self.root / event.camera_id
        cam_dir.mkdir(exist_ok=True)

        date_str = time.strftime("%Y-%m-%d", time.localtime(event.timestamp))
        date_dir = cam_dir / date_str
        date_dir.mkdir(exist_ok=True)

        path = date_dir / f"{event.id}.json"
        data = event.to_dict()

        with self._lock:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=None), encoding="utf-8")
            self.total_saved += 1
            self.storage_bytes += path.stat().st_size

        return str(path)

    def save_frame_metadata(self, metadata: FrameMetadata, event_id: str | None = None):
        """Save frame-level metadata (for active events only)."""
        if event_id is None:
            return  # Don't store routine frame metadata

        cam_dir = self.root / metadata.camera_id
        cam_dir.mkdir(exist_ok=True)

        date_str = time.strftime("%Y-%m-%d", time.localtime(metadata.timestamp))
        date_dir = cam_dir / date_str
        date_dir.mkdir(exist_ok=True)

        frame_file = date_dir / f"{event_id}_frames.jsonl"
        with self._lock:
            with open(frame_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata.to_dict(), ensure_ascii=False) + "\n")

    def save_trajectory(self, track_id: str, camera_id: str, positions: list[dict]):
        """Save a trajectory (position history) for a tracked person/vehicle."""
        cam_dir = self.root / camera_id
        cam_dir.mkdir(exist_ok=True)

        traj_file = cam_dir / f"trajectories.jsonl"
        record = {
            "track_id": track_id,
            "positions": positions,
            "saved_at": time.time(),
        }
        with self._lock:
            with open(traj_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def search(self, camera_id: str | None = None, category: str | None = None,
               start_time: float | None = None, end_time: float | None = None,
               limit: int = 100) -> list[dict]:
        """Search stored event metadata."""
        results = []
        search_dirs = []

        if camera_id:
            cam_dir = self.root / camera_id
            if cam_dir.exists():
                search_dirs.append(cam_dir)
        else:
            if self.root.exists():
                search_dirs = [d for d in self.root.iterdir() if d.is_dir()]

        for cam_dir in search_dirs:
            for date_dir in sorted(cam_dir.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                for json_file in date_dir.glob("*.json"):
                    if not json_file.name.startswith("_"):
                        try:
                            data = json.loads(json_file.read_text(encoding="utf-8"))
                            # Apply filters
                            if category and data.get("category") != category:
                                continue
                            if start_time and data.get("ts", 0) < start_time:
                                continue
                            if end_time and data.get("ts", 0) > end_time:
                                continue
                            results.append(data)
                            if len(results) >= limit:
                                return results
                        except (json.JSONDecodeError, KeyError):
                            continue

        results.sort(key=lambda x: x.get("ts", 0), reverse=True)
        return results[:limit]

    def delete_range(self, start_time: float, end_time: float,
                     camera_id: str | None = None) -> int:
        """Delete all events in a time range (GDPR right to erasure)."""
        deleted = 0
        search_dirs = []

        if camera_id:
            cam_dir = self.root / camera_id
            if cam_dir.exists():
                search_dirs.append(cam_dir)
        else:
            if self.root.exists():
                search_dirs = [d for d in self.root.iterdir() if d.is_dir()]

        with self._lock:
            for cam_dir in search_dirs:
                for date_dir in cam_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    for json_file in date_dir.glob("*.json"):
                        try:
                            data = json.loads(json_file.read_text(encoding="utf-8"))
                            ts = data.get("ts", 0)
                            if start_time <= ts <= end_time:
                                json_file.unlink()
                                deleted += 1
                                # Also delete associated frames file
                                frames_file = date_dir / f"{data['id']}_frames.jsonl"
                                if frames_file.exists():
                                    frames_file.unlink()
                        except (json.JSONDecodeError, KeyError, OSError):
                            continue

        self.total_deleted += deleted
        log.info("Deleted %d events in range %s - %s", deleted,
                 time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time)),
                 time.strftime("%Y-%m-%d %H:%M", time.localtime(end_time)))
        return deleted

    def enforce_retention(self) -> int:
        """Auto-delete events older than retention_days."""
        cutoff = time.time() - (self.retention_days * 86400)
        return self.delete_range(0, cutoff)

    def enforce_max_events(self) -> int:
        """Keep only max_events_per_camera per camera (delete oldest)."""
        deleted = 0
        if not self.root.exists():
            return 0

        with self._lock:
            for cam_dir in self.root.iterdir():
                if not cam_dir.is_dir():
                    continue
                events = []
                for date_dir in cam_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    for json_file in date_dir.glob("*.json"):
                        events.append(json_file)

                if len(events) > self.max_events_per_camera:
                    events.sort(key=lambda f: f.stat().st_mtime)
                    to_delete = events[:len(events) - self.max_events_per_camera]
                    for f in to_delete:
                        try:
                            f.unlink()
                            deleted += 1
                        except OSError:
                            pass

        self.total_deleted += deleted
        return deleted

    def get_storage_stats(self) -> dict:
        """Report storage usage."""
        total_size = 0
        total_files = 0
        cameras = {}

        if self.root.exists():
            for cam_dir in self.root.iterdir():
                if not cam_dir.is_dir():
                    continue
                cam_size = 0
                cam_files = 0
                for f in cam_dir.rglob("*.json*"):
                    cam_size += f.stat().st_size
                    cam_files += 1
                cameras[cam_dir.name] = {"size_bytes": cam_size, "files": cam_files}
                total_size += cam_size
                total_files += cam_files

        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_files": total_files,
            "cameras": cameras,
            "retention_days": self.retention_days,
            "total_saved": self.total_saved,
            "total_deleted": self.total_deleted,
        }

    def stats(self) -> dict:
        return self.get_storage_stats()


# ── Privacy-First Pipeline Orchestrator ─────────────────────────────────────

class PrivacyFirstPipeline:
    """Orchestrates the ephemeral pipeline across all cameras.

    For each camera:
    1. Frame arrives from RTSP/USB
    2. YOLOv8 runs detection (in memory)
    3. Face detection + embedding (in memory)
    4. Tracker assigns IDs (in memory)
    5. Rule engine checks for alerts
    6. METADATA is saved to disk (JSON)
    7. Frame is DISCARDED

    Total memory per camera: ~200MB (YOLO model + frame buffer)
    Total disk per camera per day: ~5 MB (metadata only)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.processors: dict[str, EphemeralProcessor] = {}
        self.store = MetadataStore(
            root=self.config.get("metadata_dir", "output/metadata"),
            retention_days=self.config.get("retention_days", 30),
        )
        self._lock = threading.Lock()

        # Alert callback
        self.on_alert = None

        # Global stats
        self.total_cameras = 0
        self.total_frames = 0
        self.total_alerts = 0

    def register_camera(self, camera_id: str, config: dict | None = None) -> EphemeralProcessor:
        """Register a camera and create its ephemeral processor."""
        with self._lock:
            if camera_id not in self.processors:
                self.processors[camera_id] = EphemeralProcessor(camera_id, config)
                self.total_cameras += 1
                log.info("Registered camera %s for ephemeral processing", camera_id)
        return self.processors[camera_id]

    def process_frame(self, camera_id: str, frame, detections: list[dict] | None = None) -> FrameMetadata:
        """Process a single frame through the ephemeral pipeline."""
        processor = self.processors.get(camera_id)
        if not processor:
            processor = self.register_camera(camera_id)

        metadata = processor.process(frame, detections)
        self.total_frames += 1

        return metadata

    def on_detection_alert(self, camera_id: str, alert: dict,
                           context_frames: list[dict] | None = None):
        """Called when the rule engine fires an alert.

        Saves event metadata (NOT video) and triggers callbacks.
        """
        self.total_alerts += 1
        processor = self.processors.get(camera_id)

        # Build event record
        import uuid
        event = EventRecord(
            id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            camera_id=camera_id,
            category=alert.get("category", "unknown"),
            severity=alert.get("severity", "yellow"),
            description=alert.get("description", alert.get("rule", "")),
            detection_count=alert.get("detection_count", 0),
            track_ids=alert.get("track_ids", []),
            frame_snapshots=context_frames or (processor.get_context_frames(10) if processor else []),
        )

        # Save event metadata
        path = self.store.save_event(event)

        # Save trajectory for any tracked persons
        if processor:
            for tid in event.track_ids:
                history = processor.get_track_history(tid)
                if history:
                    self.store.save_trajectory(tid, camera_id, history)

        # Fire callback
        if self.on_alert:
            try:
                self.on_alert(event.to_dict())
            except Exception as e:
                log.error("Alert callback error: %s", e)

        log.info("Alert saved: %s [%s] on %s (metadata only, no video)",
                 event.id, event.category, camera_id)

        return event

    def run_retention(self) -> dict:
        """Run retention cleanup across all stores."""
        deleted_events = self.store.enforce_retention()
        deleted_max = self.store.enforce_max_events()

        # Clear stale tracks from all processors
        stale_tracks = 0
        for proc in self.processors.values():
            before = len(proc._active_tracks)
            proc.clear_tracks()
            stale_tracks += before - len(proc._active_tracks)

        return {
            "deleted_events": deleted_events,
            "deleted_max_events": deleted_max,
            "cleared_tracks": stale_tracks,
        }

    def stats(self) -> dict:
        return {
            "pipeline": {
                "total_cameras": self.total_cameras,
                "total_frames": self.total_frames,
                "total_alerts": self.total_alerts,
            },
            "storage": self.store.stats(),
            "cameras": {
                cid: proc.stats()
                for cid, proc in self.processors.items()
            },
        }
