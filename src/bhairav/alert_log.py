"""JSON-lines alert persistence + summary helpers."""
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
