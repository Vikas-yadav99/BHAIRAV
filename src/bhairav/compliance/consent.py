"""Consent management for GDPR/privacy compliance.

Tracks user consent for data processing, camera monitoring,
and third-party sharing.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConsentRecord:
    """A single consent record."""
    consent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    subject_id: str = ""        # person/camera ID
    subject_type: str = "person"  # person / camera / zone
    consent_type: str = "monitoring"  # monitoring / analytics / sharing / retention
    granted: bool = True
    timestamp: float = field(default_factory=time.time)
    expires_at: float = 0.0     # 0 = no expiry
    source: str = "manual"      # manual / opt-in / legal / court_order
    metadata: dict = field(default_factory=dict)

    def is_valid(self, now: float | None = None) -> bool:
        if not self.granted:
            return False
        if self.expires_at > 0 and (now or time.time()) > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "consent_id": self.consent_id,
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "consent_type": self.consent_type,
            "granted": self.granted,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "source": self.source,
            "metadata": self.metadata,
        }


class ConsentManager:
    """Manages consent records for GDPR compliance.

    Parameters
    ----------
    store_path : str
        Path to persist consent records (JSON).
    """

    def __init__(self, store_path: str = "output/consent.json"):
        self._store_path = Path(store_path)
        self._records: list[ConsentRecord] = []
        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                self._records = [
                    ConsentRecord(**{k: v for k, v in item.items()
                                     if k in ConsentRecord.__dataclass_fields__})
                    for item in data
                ]
            except Exception:
                self._records = []

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._records]
        self._store_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def grant(self, subject_id: str, consent_type: str,
              subject_type: str = "person", expires_in_days: int = 0,
              source: str = "manual", metadata: dict | None = None) -> ConsentRecord:
        """Grant consent for a subject."""
        record = ConsentRecord(
            subject_id=subject_id,
            subject_type=subject_type,
            consent_type=consent_type,
            granted=True,
            expires_at=(time.time() + expires_in_days * 86400) if expires_in_days else 0,
            source=source,
            metadata=metadata or {},
        )
        self._records.append(record)
        self._save()
        return record

    def revoke(self, subject_id: str, consent_type: str) -> bool:
        """Revoke consent for a subject."""
        for r in reversed(self._records):
            if r.subject_id == subject_id and r.consent_type == consent_type and r.granted:
                r.granted = False
                r.timestamp = time.time()
                self._save()
                return True
        return False

    def check(self, subject_id: str, consent_type: str) -> bool:
        """Check if a subject has valid consent."""
        for r in reversed(self._records):
            if r.subject_id == subject_id and r.consent_type == consent_type:
                return r.is_valid()
        return False

    def get_history(self, subject_id: str) -> list[dict]:
        """Get consent history for a subject."""
        return [r.to_dict() for r in self._records
                if r.subject_id == subject_id]

    def list_consents(self, consent_type: str | None = None) -> list[dict]:
        records = self._records
        if consent_type:
            records = [r for r in records if r.consent_type == consent_type]
        return [r.to_dict() for r in records]

    def snapshot(self) -> dict:
        return {
            "total_records": len(self._records),
            "active": sum(1 for r in self._records if r.is_valid()),
            "revoked": sum(1 for r in self._records if not r.granted),
            "consent_types": list(set(r.consent_type for r in self._records)),
        }
