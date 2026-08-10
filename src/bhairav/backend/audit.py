"""Tamper-evident, append-only audit log.

Every entry carries a SHA-256 of the *previous* entry's full line, forming a
hash chain: modifying or deleting an entry breaks every subsequent link.
Written as JSON lines so it can be tailed and queried without a database.

Design (Phase 3 privacy layer):
    - append-only: entries are never rewritten in place
    - chained: entry[i].prev_hash == sha256(entry[i-1].line)
    - auditable: verify() replays the chain and reports integrity
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def _line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, actor: str, action: str, target: str = "",
               detail: dict | None = None, now: float | None = None) -> dict:
        """Append one entry and return it (already persisted)."""
        now = time.time() if now is None else now
        prev = self._last_hash()
        entry = {
            "ts": round(now, 3),
            "actor": actor,
            "action": action,
            "target": target,
            "detail": detail or {},
            "prev_hash": prev,
        }
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        entry["_hash"] = _line_hash(line)
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return entry

    def _last_hash(self) -> str:
        """Hash of the most recently written entry (the chain tip)."""
        if not self.path.exists():
            return "0" * 64
        tip = "0" * 64
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if "_hash" in entry:
                tip = entry["_hash"]
        return tip

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    out.append({"_corrupt": True, "line": line})
        return out

    def verify(self) -> tuple[bool, list[str]]:
        """Replay the hash chain; return (ok, problems)."""
        problems: list[str] = []
        prev = "0" * 64
        for i, entry in enumerate(self.read()):
            if "_corrupt" in entry:
                problems.append(f"line {i}: corrupt JSON")
                continue
            line_without_hash = {k: v for k, v in entry.items() if k != "_hash"}
            recomputed = _line_hash(
                json.dumps(line_without_hash, sort_keys=True, separators=(",", ":")))
            if recomputed != entry.get("_hash"):
                problems.append(f"line {i}: hash mismatch")
            if entry.get("prev_hash") != prev:
                problems.append(f"line {i}: broken chain link")
            prev = entry.get("_hash", prev)
        return (not problems, problems)

    def query(self, actor: str | None = None, action: str | None = None,
              target: str | None = None, limit: int = 100) -> list[dict]:
        rows = self.read()
        if actor is not None:
            rows = [r for r in rows if r.get("actor") == actor]
        if action is not None:
            rows = [r for r in rows if r.get("action") == action]
        if target is not None:
            rows = [r for r in rows if r.get("target") == target]
        return rows[-limit:]
