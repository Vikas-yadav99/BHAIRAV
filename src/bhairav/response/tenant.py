"""Multi-tenant management (Phase 17.4).

City-level deployment with multiple operators, zones, and camera groups.
Each tenant has isolated data, alert feeds, and dashboard views.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tenant:
    tenant_id: str
    name: str
    role: str = "operator"       # admin | operator | viewer
    cameras: list[str] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    alert_rules: list[str] = field(default_factory=list)
    max_cameras: int = 50
    created_at: float = field(default_factory=time.time)
    active: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "role": self.role,
            "cameras": self.cameras,
            "zones": self.zones,
            "alert_rules": self.alert_rules,
            "max_cameras": self.max_cameras,
            "created_at": round(self.created_at, 3),
            "active": self.active,
            "metadata": self.metadata,
        }


class TenantManager:
    """Manages tenants (operators, zones, camera groups) for city-scale."""

    def __init__(self, store_path: str = "output/tenants.json"):
        self._path = Path(store_path)
        self._tenants: dict[str, Tenant] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for t in data.get("tenants", []):
                    self._tenants[t["tenant_id"]] = Tenant(**t)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"tenants": [t.to_dict() for t in self._tenants.values()]}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def create_tenant(self, tenant_id: str, name: str, **kwargs) -> Tenant:
        with self._lock:
            t = Tenant(tenant_id=tenant_id, name=name, **kwargs)
            self._tenants[tenant_id] = t
            self._save()
            return t

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def list_tenants(self, active_only: bool = True) -> list[dict]:
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.active]
        return [t.to_dict() for t in tenants]

    def update_tenant(self, tenant_id: str, **kwargs) -> bool:
        with self._lock:
            t = self._tenants.get(tenant_id)
            if not t:
                return False
            for k, v in kwargs.items():
                if hasattr(t, k):
                    setattr(t, k, v)
            self._save()
            return True

    def delete_tenant(self, tenant_id: str) -> bool:
        with self._lock:
            if tenant_id not in self._tenants:
                return False
            del self._tenants[tenant_id]
            self._save()
            return True

    def assign_cameras(self, tenant_id: str, camera_ids: list[str]) -> bool:
        with self._lock:
            t = self._tenants.get(tenant_id)
            if not t:
                return False
            t.cameras = list(set(t.cameras + camera_ids))
            self._save()
            return True

    def get_cameras_for_tenant(self, tenant_id: str) -> list[str]:
        t = self._tenants.get(tenant_id)
        return list(t.cameras) if t else []

    def filter_alerts_for_tenant(self, tenant_id: str,
                                 alerts: list[dict]) -> list[dict]:
        t = self._tenants.get(tenant_id)
        if not t:
            return alerts
        cam_set = set(t.cameras)
        zone_set = set(t.zones) if t.zones else None
        return [a for a in alerts
                if a.get("camera") in cam_set
                and (zone_set is None or a.get("zone") in zone_set)]
