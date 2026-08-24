"""Data retention policies and automated cleanup.

Defines how long different data types are kept, when they
expire, and performs batch deletion of expired records.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RetentionPolicy:
    """Retention rule for a specific data type."""
    data_type: str           # evidence / alerts / reid / analytics / logs
    max_age_days: int        # days to keep (0 = forever)
    max_count: int = 0       # max records (0 = unlimited)
    archive_before_delete: bool = False
    archive_path: str = ""
    delete_pii: bool = True  # scrub PII before deletion

    def is_expired(self, created_at: float, now: float | None = None) -> bool:
        if self.max_age_days <= 0:
            return False
        if now is None:
            now = time.time()
        age_days = (now - created_at) / 86400
        return age_days > self.max_age_days

    def to_dict(self) -> dict:
        return {
            "data_type": self.data_type,
            "max_age_days": self.max_age_days,
            "max_count": self.max_count,
            "archive_before_delete": self.archive_before_delete,
            "delete_pii": self.delete_pii,
        }


DEFAULT_POLICIES = [
    RetentionPolicy("evidence", max_age_days=90, archive_before_delete=True),
    RetentionPolicy("alerts", max_age_days=365, max_count=100000),
    RetentionPolicy("reid", max_age_days=180),
    RetentionPolicy("analytics", max_age_days=30),
    RetentionPolicy("logs", max_age_days=180, max_count=500000),
]


class RetentionManager:
    """Manages data retention policies and enforces cleanup.

    Parameters
    ----------
    data_dir : str
        Root directory for stored data.
    policies : list[RetentionPolicy] | None
        Custom policies (defaults to DEFAULT_POLICIES).
    """

    def __init__(self, data_dir: str = "output",
                 policies: list[RetentionPolicy] | None = None):
        self.data_dir = Path(data_dir)
        self.policies = {p.data_type: p for p in (policies or DEFAULT_POLICIES)}
        self._deletion_log: list[dict] = []

    def check_expiry(self, data_type: str, created_at: float) -> bool:
        """Check if a record is expired."""
        policy = self.policies.get(data_type)
        if not policy:
            return False
        return policy.is_expired(created_at)

    def cleanup(self, data_type: str | None = None) -> dict:
        """Run cleanup for a data type or all types.

        Returns summary of what was cleaned.
        """
        summary = {"deleted": 0, "archived": 0, "skipped": 0}
        types = [data_type] if data_type else list(self.policies.keys())
        now = time.time()

        for dt in types:
            policy = self.policies.get(dt)
            if not policy:
                continue
            dir_path = self.data_dir / dt
            if not dir_path.exists():
                continue
            for f in sorted(dir_path.iterdir()):
                if not f.is_file():
                    continue
                try:
                    created = f.stat().st_mtime
                    if policy.is_expired(created, now):
                        if policy.archive_before_delete and policy.archive_path:
                            archive_dir = Path(policy.archive_path) / dt
                            archive_dir.mkdir(parents=True, exist_ok=True)
                            f.rename(archive_dir / f.name)
                            summary["archived"] += 1
                        else:
                            f.unlink()
                            summary["deleted"] += 1
                        self._deletion_log.append({
                            "data_type": dt,
                            "file": str(f.name),
                            "action": "archived" if policy.archive_before_delete else "deleted",
                            "timestamp": now,
                        })
                    else:
                        summary["skipped"] += 1
                except Exception:
                    summary["skipped"] += 1

        return summary

    def add_policy(self, policy: RetentionPolicy) -> None:
        self.policies[policy.data_type] = policy

    def snapshot(self) -> dict:
        return {
            "data_dir": str(self.data_dir),
            "policies": {k: v.to_dict() for k, v in self.policies.items()},
            "recent_deletions": len(self._deletion_log),
        }
