"""Unified Person Identity (Group 2 of audit fix).

Replaces the 4 separate ID systems (track_id, reid_id, evidence_id, alert_id)
with one canonical person_id that every subsystem references.

PersonRecord holds all IDs for one person across all cameras and subsystems.
IdentityService maps incoming track_ids to person_ids using re-ID + spatial proximity.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _person_id() -> str:
    """Generate a canonical person ID: P-xxxxxxxx"""
    return f"P-{uuid.uuid4().hex[:8]}"


@dataclass
class PersonRecord:
    """A single person's identity across all cameras and subsystems.

    This is the SINGLE SOURCE OF TRUTH for "who is this person."
    Every other system (trackers, re-ID, evidence, alerts) references
    person_id instead of maintaining its own ID space.
    """
    person_id: str = field(default_factory=_person_id)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Per-camera track IDs: {camera_id: [track_id1, track_id2, ...]}
    track_ids: dict[str, list[int]] = field(default_factory=dict)

    # Re-ID subject ID (from reid gallery)
    reid_subject: str = ""

    # Evidence clip IDs
    evidence_ids: list[str] = field(default_factory=list)

    # Alert IDs this person is associated with
    alert_ids: list[str] = field(default_factory=list)

    # Latest embedding (for re-ID matching)
    embedding: Any = None

    # Metadata
    cameras_seen: list[str] = field(default_factory=list)
    total_frames: int = 0
    notes: list[str] = field(default_factory=list)

    def add_track(self, camera_id: str, track_id: int) -> None:
        """Record that this person was tracked on a camera."""
        if camera_id not in self.track_ids:
            self.track_ids[camera_id] = []
        if track_id not in self.track_ids[camera_id]:
            self.track_ids[camera_id].append(track_id)
        if camera_id not in self.cameras_seen:
            self.cameras_seen.append(camera_id)
        self.last_seen = time.time()
        self.total_frames += 1

    def add_evidence(self, evidence_id: str) -> None:
        if evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)

    def add_alert(self, alert_id: str) -> None:
        if alert_id not in self.alert_ids:
            self.alert_ids.append(alert_id)

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "track_ids": self.track_ids,
            "reid_subject": self.reid_subject,
            "evidence_ids": self.evidence_ids,
            "alert_ids": self.alert_ids,
            "cameras_seen": self.cameras_seen,
            "total_frames": self.total_frames,
            "notes": self.notes,
        }


class IdentityService:
    """Maps track_ids to canonical person_ids.

    When a detector produces a track with track_id=7 on CAM-01, the
    IdentityService resolves it to a person_id (e.g., P-a1b2c3d4).
    If re-ID links that person to a track on CAM-02, both tracks map
    to the same person_id.

    Thread-safe for concurrent camera pipelines.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # camera_id -> track_id -> person_id
        self._track_map: dict[str, dict[int, str]] = {}
        # person_id -> PersonRecord
        self._persons: dict[str, PersonRecord] = {}
        # reid_subject -> person_id
        self._reid_map: dict[str, str] = {}

    def resolve(self, camera_id: str, track_id: int) -> str:
        """Resolve a (camera, track) pair to a person_id.

        Creates a new PersonRecord if this is a fresh track.
        Returns the canonical person_id.
        """
        with self._lock:
            cam_tracks = self._track_map.setdefault(camera_id, {})
            if track_id in cam_tracks:
                return cam_tracks[track_id]

            # New track — create a new person
            pid = _person_id()
            cam_tracks[track_id] = pid
            record = PersonRecord(person_id=pid)
            record.add_track(camera_id, track_id)
            self._persons[pid] = record
            return pid

    def link_reid(self, camera_id: str, track_id: int, reid_subject: str) -> str:
        """Link a track to a re-ID subject, merging if needed.

        If the reid_subject is already known, the track joins the
        existing person. Otherwise a new person is created.

        Returns the canonical person_id.
        """
        with self._lock:
            if reid_subject in self._reid_map:
                # Merge: this track joins an existing person
                pid = self._reid_map[reid_subject]
                record = self._persons[pid]
                record.add_track(camera_id, track_id)
                record.reid_subject = reid_subject

                # Update track map
                cam_tracks = self._track_map.setdefault(camera_id, {})
                cam_tracks[track_id] = pid
                return pid

            # New re-ID subject
            cam_tracks = self._track_map.setdefault(camera_id, {})
            if track_id in cam_tracks:
                pid = cam_tracks[track_id]
            else:
                pid = self.resolve(camera_id, track_id)

            record = self._persons[pid]
            record.reid_subject = reid_subject
            self._reid_map[reid_subject] = pid
            return pid

    def get_person(self, person_id: str) -> PersonRecord | None:
        """Get a person record by ID."""
        return self._persons.get(person_id)

    def get_by_track(self, camera_id: str, track_id: int) -> PersonRecord | None:
        """Get a person record by camera + track ID."""
        with self._lock:
            cam_tracks = self._track_map.get(camera_id, {})
            pid = cam_tracks.get(track_id)
            if pid:
                return self._persons.get(pid)
            return None

    def get_by_reid(self, reid_subject: str) -> PersonRecord | None:
        """Get a person record by re-ID subject."""
        pid = self._reid_map.get(reid_subject)
        if pid:
            return self._persons.get(pid)
        return None

    def add_evidence(self, person_id: str, evidence_id: str) -> bool:
        """Link an evidence clip to a person."""
        record = self._persons.get(person_id)
        if record:
            record.add_evidence(evidence_id)
            return True
        return False

    def add_alert(self, person_id: str, alert_id: str) -> bool:
        """Link an alert to a person."""
        record = self._persons.get(person_id)
        if record:
            record.add_alert(alert_id)
            return True
        return False

    def count(self) -> int:
        """Total number of known persons."""
        return len(self._persons)

    def list_persons(self, limit: int = 100) -> list[dict]:
        """List known persons, most recent first."""
        with self._lock:
            persons = sorted(
                self._persons.values(),
                key=lambda p: p.last_seen,
                reverse=True,
            )
            return [p.to_dict() for p in persons[:limit]]

    def prune(self, max_age_sec: float = 3600) -> int:
        """Remove persons not seen for max_age_sec. Returns count removed."""
        cutoff = time.time() - max_age_sec
        with self._lock:
            to_remove = [pid for pid, p in self._persons.items()
                         if p.last_seen < cutoff]
            for pid in to_remove:
                record = self._persons.pop(pid)
                # Clean up maps
                for cam_tracks in self._track_map.values():
                    to_del = [tid for tid, p in cam_tracks.items() if p == pid]
                    for tid in to_del:
                        del cam_tracks[tid]
                if record.reid_subject and record.reid_subject in self._reid_map:
                    del self._reid_map[record.reid_subject]
            return len(to_remove)
