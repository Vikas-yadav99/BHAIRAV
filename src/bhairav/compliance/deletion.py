"""Right-to-deletion (GDPR Art. 17) service.

Allows data subjects to request deletion of their personal data
across all BHAIRAV subsystems.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DeletionRequest:
    """A deletion request from a data subject."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    subject_id: str = ""
    subject_type: str = "person"  # person / camera
    requested_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    status: str = "pending"      # pending / processing / completed / denied
    scope: str = "all"           # all / evidence / reid / alerts
    reason: str = ""
    deleted_items: dict = field(default_factory=dict)  # subsystem -> count
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "requested_at": self.requested_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "scope": self.scope,
            "reason": self.reason,
            "deleted_items": self.deleted_items,
            "error": self.error,
        }


class DeletionService:
    """GDPR right-to-deletion service.

    Scans subsystem data stores and removes all records
    matching the subject ID.

    Parameters
    ----------
    data_dir : str
        Root data directory.
    store_path : str
        Path to persist deletion request history.
    """

    def __init__(self, data_dir: str = "output",
                 store_path: str = "output/deletion_requests.json"):
        self._data_dir = Path(data_dir)
        self._store_path = Path(store_path)
        self._requests: list[DeletionRequest] = []
        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                self._requests = [
                    DeletionRequest(**{k: v for k, v in item.items()
                                       if k in DeletionRequest.__dataclass_fields__})
                    for item in data
                ]
            except Exception:
                self._requests = []

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps([r.to_dict() for r in self._requests], indent=2),
            encoding="utf-8",
        )

    def request_deletion(self, subject_id: str, scope: str = "all",
                         reason: str = "",
                         subject_type: str = "person") -> DeletionRequest:
        """Create a deletion request."""
        req = DeletionRequest(
            subject_id=subject_id,
            subject_type=subject_type,
            scope=scope,
            reason=reason,
        )
        self._requests.append(req)
        self._save()
        return req

    def process(self, request_id: str) -> DeletionRequest:
        """Process a pending deletion request."""
        req = next((r for r in self._requests if r.request_id == request_id), None)
        if not req:
            raise ValueError(f"Request {request_id} not found")
        if req.status != "pending":
            return req

        req.status = "processing"
        deleted = {}

        # Scan evidence files
        if req.scope in ("all", "evidence"):
            count = self._delete_by_subject(req.subject_id, "evidence")
            deleted["evidence"] = count

        # Scan reid data
        if req.scope in ("all", "reid"):
            count = self._delete_from_jsonl(
                self._data_dir / "reid" / "sightings.jsonl",
                req.subject_id,
            )
            deleted["reid_sightings"] = count

        # Scan alert logs
        if req.scope in ("all", "alerts"):
            for f in (self._data_dir / "alerts").glob("*.jsonl") if (self._data_dir / "alerts").exists() else []:
                count = self._delete_from_jsonl(f, req.subject_id)
                deleted[f"alerts_{f.stem}"] = count

        req.deleted_items = deleted
        req.status = "completed"
        req.completed_at = time.time()
        self._save()
        return req

    def _delete_by_subject(self, subject_id: str, subdir: str) -> int:
        """Delete evidence files matching subject_id in filename."""
        dir_path = self._data_dir / subdir
        if not dir_path.exists():
            return 0
        count = 0
        for f in dir_path.iterdir():
            if f.is_file() and subject_id in f.name:
                try:
                    f.unlink()
                    count += 1
                except Exception:
                    pass
        return count

    def _delete_from_jsonl(self, path: Path, subject_id: str) -> int:
        """Remove lines matching subject_id from a JSONL file."""
        if not path.exists():
            return 0
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = []
        removed = 0
        for line in lines:
            try:
                data = json.loads(line)
                if subject_id in json.dumps(data):
                    removed += 1
                else:
                    kept.append(line)
            except Exception:
                kept.append(line)
        if removed:
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return removed

    def deny(self, request_id: str, reason: str = "") -> DeletionRequest:
        req = next((r for r in self._requests if r.request_id == request_id), None)
        if not req:
            raise ValueError(f"Request {request_id} not found")
        req.status = "denied"
        req.error = reason or "Denied by administrator"
        req.completed_at = time.time()
        self._save()
        return req

    def list_requests(self, status: str | None = None) -> list[dict]:
        reqs = self._requests
        if status:
            reqs = [r for r in reqs if r.status == status]
        return [r.to_dict() for r in reqs]

    def snapshot(self) -> dict:
        return {
            "total_requests": len(self._requests),
            "pending": sum(1 for r in self._requests if r.status == "pending"),
            "completed": sum(1 for r in self._requests if r.status == "completed"),
            "denied": sum(1 for r in self._requests if r.status == "denied"),
        }
