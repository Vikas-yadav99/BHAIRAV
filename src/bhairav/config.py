"""Configuration loading with sensible Phase 1 defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .types import Zone

DEFAULTS: dict = {
    "detector": "blob",
    "model": {
        "name": "yolov8n.pt",
        "conf": 0.35,
        "imgsz": 640,
        "classes": [0, 2, 5, 7],
        "tracker": "bytetrack.yaml",
    },
    "synthetic": {"fps": 15, "seed": 7, "duration_sec": 24.0, "width": 1280, "height": 720},
    "alert": {"cooldown_sec": 10.0},
    "zones": [
        {"name": "plaza", "kind": "monitored",
         "points": [[0.30, 0.30], [0.60, 0.30], [0.60, 0.55], [0.30, 0.55]]},
        {"name": "server_room", "kind": "restricted",
         "points": [[0.70, 0.35], [0.92, 0.35], [0.92, 0.62], [0.70, 0.62]]},
    ],
    "rules": {
        # Phase 1 - geometric / statistical
        "loitering": {"enabled": True, "duration_sec": 5.0, "escalate": True, "zones": ["plaza"]},
        "zone_crossing": {"enabled": True, "severity": "red", "include_vehicles": True},
        "crowd_density": {"enabled": True, "min_people": 4, "severity": "orange", "escalate": True},
        # Phase 2 - behavior intelligence
        "fall": {"enabled": True, "severity": "orange", "escalate": True,
                 "vy_thresh_norm": 0.10, "flat_aspect": 1.0, "down_sec": 0.5},
        "fight": {"enabled": True, "severity": "red", "proximity_norm": 0.10,
                   "speed_norm": 0.08, "min_speed_norm": 0.03,
                   "wobble_deg": 25.0, "duration_sec": 1.5},
        "chase": {"enabled": True, "severity": "orange", "escalate": True,
                   "runner_speed_norm": 0.065, "follower_speed_norm": 0.08,
                   "heading_deg": 30.0, "max_dist_norm": 0.30, "duration_sec": 2.0},
        "trespass": {"enabled": True, "severity": "orange", "escalate": True, "dwell_sec": 2.5},
        "anomaly": {"enabled": True, "severity": "yellow", "z_thresh": 3.0,
                     "min_count": 2, "warmup_frames": 45},
        # Phase 6 - ANPR / stolen-vehicle watchlist
        "stolen_vehicle": {"enabled": True, "severity": "red", "min_confidence": 0.5},
    },
    # Phase 3-5 - backend & evidence
    "backend": {
        "host": "127.0.0.1",
        "port": 8000,
        "secret": "dev-secret-change-me",
        "max_recent_alerts": 200,
        "users_file": "output/users.json",
        "webhook_url": None,  # POST red alerts here (Slack-style); None disables
        "db": None,           # Phase 8: PostgreSQL URL (postgresql://...) or None = file store
    },
    "cameras": [],  # Phase 8 M2: multi-camera sources; empty = single --source
    "evidence": {
        "dir": "output/evidence",
        "camera": "CAM-01",
        "fps": 15,
        "pre_sec": 5.0,
        "post_sec": 5.0,
        "min_gap_sec": 10.0,
        "blur_faces": True,
        "encrypt": False,
        "retention_days": 30,
        "max_events": 0,  # cap on stored events; 0 = unlimited (oldest pruned)
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class ModelConfig:
    name: str = "yolov8n.pt"
    conf: float = 0.35
    imgsz: int = 640
    classes: tuple[int, ...] = (0, 2, 5, 7)
    tracker: str = "bytetrack.yaml"

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(name=d.get("name", "yolov8n.pt"), conf=float(d.get("conf", 0.35)),
                   imgsz=int(d.get("imgsz", 640)),
                   classes=tuple(int(c) for c in d.get("classes", [0, 2, 5, 7])),
                   tracker=d.get("tracker", "bytetrack.yaml"))


@dataclass
class SyntheticConfig:
    fps: int = 15
    seed: int = 7
    duration_sec: float = 24.0
    width: int = 1280
    height: int = 720

    @classmethod
    def from_dict(cls, d: dict) -> "SyntheticConfig":
        return cls(fps=int(d.get("fps", 15)), seed=int(d.get("seed", 7)),
                   duration_sec=float(d.get("duration_sec", 24.0)),
                   width=int(d.get("width", 1280)), height=int(d.get("height", 720)))


@dataclass
class AlertConfig:
    cooldown_sec: float = 10.0

    @classmethod
    def from_dict(cls, d: dict) -> "AlertConfig":
        return cls(cooldown_sec=float(d.get("cooldown_sec", 10.0)))


@dataclass
class BackendConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    secret: str = "dev-secret-change-me"
    max_recent_alerts: int = 200
    users_file: str = "output/users.json"
    webhook_url: str | None = None
    db: str | None = None   # Phase 8: PostgreSQL URL; None = file-based store

    @classmethod
    def from_dict(cls, d: dict) -> "BackendConfig":
        return cls(host=d.get("host", "127.0.0.1"),
                   port=int(d.get("port", 8000)),
                   secret=d.get("secret", "dev-secret-change-me"),
                   max_recent_alerts=int(d.get("max_recent_alerts", 200)),
                   users_file=d.get("users_file", "output/users.json"),
                   webhook_url=d.get("webhook_url") or None,
                   db=d.get("db") or None)


@dataclass
class EvidenceConfig:
    dir: str = "output/evidence"
    camera: str = "CAM-01"
    fps: int = 15
    pre_sec: float = 5.0
    post_sec: float = 5.0
    min_gap_sec: float = 10.0
    blur_faces: bool = True
    encrypt: bool = False
    retention_days: float = 30.0
    max_events: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceConfig":
        return cls(dir=d.get("dir", "output/evidence"),
                   camera=d.get("camera", "CAM-01"),
                   fps=int(d.get("fps", 15)),
                   pre_sec=float(d.get("pre_sec", 5.0)),
                   post_sec=float(d.get("post_sec", 5.0)),
                   min_gap_sec=float(d.get("min_gap_sec", 10.0)),
                   blur_faces=bool(d.get("blur_faces", True)),
                   encrypt=bool(d.get("encrypt", False)),
                   retention_days=float(d.get("retention_days", 30.0)),
                   max_events=int(d.get("max_events", 0)))


@dataclass
class CameraConfig:
    """A named video source in the multi-camera setup (Phase 8 / M2).

    `source`/`detector` mirror the single-camera CLI flags; when the
    `cameras` list is empty, serve.py falls back to --source with the
    `evidence.camera` id, so existing single-camera setups keep working.
    """

    id: str = "CAM-01"
    name: str = "Synthetic Plaza"
    source: str = "blob"          # blob | file path | camera index | rtsp://...
    detector: str = "auto"        # blob | yolo | auto

    @classmethod
    def from_dict(cls, d: dict) -> "CameraConfig":
        return cls(id=str(d.get("id", "CAM-01")),
                   name=str(d.get("name", d.get("id", "CAM-01"))),
                   source=str(d.get("source", "blob")),
                   detector=str(d.get("detector", "auto")))


@dataclass
class AppConfig:
    detector: str = "blob"  # blob | yolo | auto
    model: ModelConfig = field(default_factory=ModelConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    zones: list[Zone] = field(default_factory=list)
    rules: dict = field(default_factory=dict)
    backend: BackendConfig = field(default_factory=BackendConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    cameras: list[CameraConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "AppConfig":
        zones = []
        for z in d.get("zones", []):
            zones.append(Zone(name=z["name"], kind=z["kind"],
                              points_norm=[(float(p[0]), float(p[1])) for p in z["points"]]))
        return cls(detector=d.get("detector", "blob"),
                   model=ModelConfig.from_dict(d.get("model", {})),
                   synthetic=SyntheticConfig.from_dict(d.get("synthetic", {})),
                   alert=AlertConfig.from_dict(d.get("alert", {})),
                   zones=zones,
                   rules=dict(d.get("rules", {})),
                   backend=BackendConfig.from_dict(d.get("backend", {})),
                   evidence=EvidenceConfig.from_dict(d.get("evidence", {})),
                   cameras=[CameraConfig.from_dict(c) for c in d.get("cameras", [])])


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load a YAML config, deep-merged over Phase 1 defaults."""
    p = Path(path)
    override: dict = {}
    if p.exists():
        override = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AppConfig.from_dict(_deep_merge(DEFAULTS, override))
