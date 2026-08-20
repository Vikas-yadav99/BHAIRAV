"""Local JSONL alert store for the edge agent.

Persists alerts to disk so nothing is lost when the upstream connection
is down. Supports batch read-and-clear for retry, and automatic pruning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock


class LocalAlertStore:
    """Append-only JSONL alert store with batch read/clear."""

    def __init__(self, path, max_age_sec=86400.0):
        self.path = Path(path)
        self.max_age_sec = max_age_sec
        self._lock = Lock()

    def append(self, alert):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert, default=str) + chr(10))

    def read_all(self):
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def read_and_clear(self):
        if not self.path.exists():
            return []
        with self._lock:
            content = self.path.read_text(encoding="utf-8")
            self.path.write_text("", encoding="utf-8")
        return [json.loads(line) for line in content.splitlines() if line.strip()]

    def prune(self):
        if not self.path.exists():
            return 0
        cutoff = time.time() - self.max_age_sec
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            kept = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    alert = json.loads(line)
                    if alert.get("timestamp", 0) >= cutoff:
                        kept.append(line)
                except json.JSONDecodeError:
                    kept.append(line)
            self.path.write_text(chr(10).join(kept) + (chr(10) if kept else ""),
                                encoding="utf-8")
        return len(lines) - len(kept)

    @property
    def count(self):
        if not self.path.exists():
            return 0
        with self._lock:
            return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines()
                       if line.strip())
