"""Investigation timeline: chronological evidence browser with case export.

Builds a unified timeline from alerts, evidence clips, re-ID sightings,
and analytics events.  Supports filtering, grouping, and export to
structured case files for law enforcement handoff.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TimelineEvent:
    """A single event on the investigation timeline."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    event_type: str = "alert"  # alert / evidence / sighting / analytics / note
    title: str = ""
    description: str = ""
    severity: str = "yellow"
    zone: str = ""
    camera: str = ""
    evidence_id: str = ""
    reid_subject: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id, "timestamp": self.timestamp,
            "event_type": self.event_type, "title": self.title,
            "description": self.description, "severity": self.severity,
            "zone": self.zone, "camera": self.camera,
            "evidence_id": self.evidence_id,
            "reid_subject": self.reid_subject,
            "metadata": self.metadata,
        }


@dataclass
class CaseFile:
    """An exported case file for law enforcement."""
    case_id: str = field(default_factory=lambda: f"CASE-{uuid.uuid4().hex[:8].upper()}")
    title: str = ""
    created_at: float = field(default_factory=time.time)
    events: list = field(default_factory=list)
    summary: str = ""
    status: str = "open"  # open / closed / archived
    assigned_to: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "title": self.title,
            "created_at": self.created_at, "event_count": len(self.events),
            "summary": self.summary, "status": self.status,
            "assigned_to": self.assigned_to,
        }

    def to_full_dict(self) -> dict:
        d = self.to_dict()
        d["events"] = self.events
        return d


class InvestigationTimeline:
    """Manages the investigation timeline and case files.

    Parameters
    ----------
    store_path : str
        Path to persist case files (JSON).
    max_events : int
        Maximum events to retain (default 10000).
    """

    def __init__(self, store_path: str = "output/cases.json",
                 max_events: int = 10000):
        self._store_path = Path(store_path)
        self._events: list[TimelineEvent] = []
        self._cases: dict[str, CaseFile] = {}
        self.max_events = max_events
        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                for ev in data.get("events", []):
                    self._events.append(TimelineEvent(**{
                        k: v for k, v in ev.items()
                        if k in TimelineEvent.__dataclass_fields__
                    }))
                for c in data.get("cases", []):
                    case = CaseFile(**{
                        k: v for k, v in c.items()
                        if k in CaseFile.__dataclass_fields__
                    })
                    self._cases[case.case_id] = case
            except Exception:
                pass

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "events": [e.to_dict() for e in self._events[-self.max_events:]],
            "cases": [c.to_dict() for c in self._cases.values()],
        }
        self._store_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_event(self, event_type: str, title: str, **kwargs) -> TimelineEvent:
        """Add an event to the timeline."""
        ev = TimelineEvent(event_type=event_type, title=title, **kwargs)
        self._events.append(ev)
        self._events = self._events[-self.max_events:]
        self._save()
        return ev

    def add_alert(self, alert_dict: dict) -> TimelineEvent:
        """Add an alert as a timeline event."""
        return self.add_event(
            "alert",
            title=f"{alert_dict.get('rule', 'unknown')} ({alert_dict.get('severity', 'yellow')})",
            description=alert_dict.get("description", ""),
            severity=alert_dict.get("severity", "yellow"),
            zone=alert_dict.get("zone", ""),
            camera=alert_dict.get("camera", ""),
            evidence_id=alert_dict.get("evidence_id", ""),
        )

    def add_evidence(self, evidence_dict: dict) -> TimelineEvent:
        """Add an evidence clip as a timeline event."""
        return self.add_event(
            "evidence",
            title=f"Evidence: {evidence_dict.get('event_id', 'unknown')}",
            description=evidence_dict.get("description", ""),
            severity=evidence_dict.get("severity", "yellow"),
            zone=evidence_dict.get("zone", ""),
            camera=evidence_dict.get("camera", ""),
            evidence_id=evidence_dict.get("event_id", ""),
        )

    def add_note(self, text: str, author: str = "") -> TimelineEvent:
        """Add an analyst note."""
        return self.add_event("note", title=text, metadata={"author": author})

    def query(self, event_type: str | None = None, zone: str | None = None,
              camera: str | None = None, severity: str | None = None,
              start_time: float | None = None, end_time: float | None = None,
              limit: int = 100) -> list[dict]:
        """Query events with filters."""
        results = self._events
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if zone:
            results = [e for e in results if e.zone == zone]
        if camera:
            results = [e for e in results if e.camera == camera]
        if severity:
            results = [e for e in results if e.severity == severity]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        return [e.to_dict() for e in results[-limit:]]

    def create_case(self, title: str, summary: str = "",
                    assigned_to: str = "") -> CaseFile:
        """Create a new case file."""
        case = CaseFile(title=title, summary=summary, assigned_to=assigned_to)
        self._cases[case.case_id] = case
        self._save()
        return case

    def attach_to_case(self, case_id: str, event_ids: list[str]) -> bool:
        """Attach timeline events to a case."""
        case = self._cases.get(case_id)
        if not case:
            return False
        for ev in self._events:
            if ev.event_id in event_ids:
                case.events.append(ev.to_dict())
        self._save()
        return True

    def export_case(self, case_id: str, output_path: str | None = None) -> dict:
        """Export a case file as structured JSON."""
        case = self._cases.get(case_id)
        if not case:
            raise ValueError(f"Case {case_id} not found")
        data = case.to_full_dict()
        if output_path:
            Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def get_case(self, case_id: str) -> dict | None:
        case = self._cases.get(case_id)
        return case.to_dict() if case else None

    def list_cases(self, status: str | None = None) -> list[dict]:
        cases = list(self._cases.values())
        if status:
            cases = [c for c in cases if c.status == status]
        return [c.to_dict() for c in cases]

    def timeline_summary(self) -> dict:
        """Summary statistics of the timeline."""
        types = {}
        for e in self._events:
            types[e.event_type] = types.get(e.event_type, 0) + 1
        return {
            "total_events": len(self._events),
            "by_type": types,
            "total_cases": len(self._cases),
        }

    def snapshot(self) -> dict:
        return {
            "events": len(self._events),
            "cases": len(self._cases),
            "summary": self.timeline_summary(),
        }
