"""JSON-lines alert persistence + summary helpers.

Persists every fired alert as one JSON object per line, so the file stays
append-only and trivially greppable. The bounded in-memory feed (recent())
is what the dashboard shows in the live wall; summary() powers the
per-camera /api/status cards. Phase 8 notes: alerts are also mirrored into
the evidence store via EventRecorder, and in PostgreSQL mode the audit log
replaces the JSONL audit file, but this alert log remains file-based by
design (it is a rolling human-readable feed, not a source of truth).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .types import Alert


class AlertLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def clear(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.to_dict()) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summary(self) -> dict:
        rows = self.read()
        return {
            "total": len(rows),
            "by_rule": dict(Counter(r["rule"] for r in rows)),
            "by_severity": dict(Counter(r["severity"] for r in rows)),
        }
