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
    "synthetic": {"fps": 15, "seed": 7, "duration_sec": 32.0, "width": 1280, "height": 720},
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
        # Phase 10 - proactive scene intelligence
        "abandoned_object": {"enabled": True, "severity": "orange", "escalate": True,
                             "classes": [28], "abandon_sec": 8.0,
                             "owner_dist_norm": 0.06, "still_speed_norm": 0.02},
        "accident": {"enabled": True, "severity": "red",
                     "cruise_speed_norm": 0.06, "still_speed_norm": 0.02,
                     "impact_dist_norm": 0.10, "confirm_sec": 1.2,
                     "down_aspect": 1.0, "lookback_sec": 4.0},
        "riot": {"enabled": True, "severity": "red", "min_people": 4,
                 "cluster_radius_norm": 0.10, "speed_norm": 0.04,
                 "wobble_deg": 20.0, "duration_sec": 4.5},
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
        # Phase 9 M5 - read-only public monitor. Set a token (or
        # $BHAIRAV_PUBLIC_TOKEN) to expose a privacy-blurred live view at
        # /?public=<token> with no login. None/empty disables it.
        "public_token": None,
    },
    "cameras": [],
    # Phase 11: audio analytics
    "audio": {
        "enabled": True,
        "sample_rate": 16000,
        "sensitivity": 1.0,
        "cooldown_sec": 15.0,
        "scream_min_dur_sec": 0.4,
    },  # Phase 8 M2: multi-camera sources; empty = single --source
    # Phase 9 M4 - person re-identification across cameras
    "reid": {
        "assign_threshold": 0.60,   # cosine: above this a person links to a known subject
        "sighting_gap_sec": 3.0,    # min seconds between recorded sightings of one track
        "deep_model": None,         # Phase 14: path to ONNX re-ID model (None = HSV+HOG)
        "deep_size": [128, 256],    # Phase 14: model input (W, H)
        "deep_threshold": 0.70,     # Phase 14: cosine threshold for deep matches
    },
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
    """YOLO/ultralytics model settings: weights, confidence, input size, COCO class filter and ByteTrack config."""
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
    """Scripted demo scene: fps, RNG seed, loop duration and frame size. The scene guarantees every alert type fires within `duration_sec` (32 s since Phase 10)."""
    fps: int = 15
    seed: int = 7
    duration_sec: float = 32.0
    width: int = 1280
    height: int = 720

    @classmethod
    def from_dict(cls, d: dict) -> "SyntheticConfig":
        return cls(fps=int(d.get("fps", 15)), seed=int(d.get("seed", 7)),
                   duration_sec=float(d.get("duration_sec", 32.0)),
                   width=int(d.get("width", 1280)), height=int(d.get("height", 720)))


@dataclass
class AlertConfig:
    """Global alert cooldown: the same (rule, zone, track, severity) key cannot refire within `cooldown_sec`."""
    cooldown_sec: float = 10.0

    @classmethod
    def from_dict(cls, d: dict) -> "AlertConfig":
        return cls(cooldown_sec=float(d.get("cooldown_sec", 10.0)))


@dataclass
class AudioConfig:
    """Phase 11 audio analytics settings."""
    enabled: bool = True
    sample_rate: int = 16000
    sensitivity: float = 1.0
    cooldown_sec: float = 15.0
    scream_min_dur_sec: float = 0.4

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConfig":
        return cls(enabled=bool(d.get("enabled", True)),
                   sample_rate=int(d.get("sample_rate", 16000)),
                   sensitivity=float(d.get("sensitivity", 1.0)),
                   cooldown_sec=float(d.get("cooldown_sec", 15.0)),
                   scream_min_dur_sec=float(d.get("scream_min_dur_sec", 0.4)))


@dataclass
class BackendConfig:
    """Server settings: bind host/port, token secret, bounded recent-alert feed, user store path, optional webhook, and the Phase 8 PostgreSQL URL (None = file-based stores)."""
    host: str = "127.0.0.1"
    port: int = 8000
    secret: str = "dev-secret-change-me"
    max_recent_alerts: int = 200
    users_file: str = "output/users.json"
    webhook_url: str | None = None
    db: str | None = None   # Phase 8: PostgreSQL URL; None = file-based store
    public_token: str | None = None  # Phase 9 M5: blurred public monitor token
    # Phase 10 M4: field-officer alert channels (name/url/min_severity/rules/
    # retries); `webhook_url` above is kept as a legacy single channel.
    alert_channels: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "BackendConfig":
        return cls(host=d.get("host", "127.0.0.1"),
                   port=int(d.get("port", 8000)),
                   secret=d.get("secret", "dev-secret-change-me"),
                   max_recent_alerts=int(d.get("max_recent_alerts", 200)),
                   users_file=d.get("users_file", "output/users.json"),
                   webhook_url=d.get("webhook_url") or None,
                   db=d.get("db") or None,
                   public_token=d.get("public_token") or None,
                   alert_channels=list(d.get("alert_channels") or []))


@dataclass
class ReidConfig:
    """Phase 9 M4 + Phase 14 - person re-identification tuning."""
    assign_threshold: float = 0.60
    sighting_gap_sec: float = 3.0
    deep_model: str | None = None       # Phase 14: ONNX model path
    deep_size: list = field(default_factory=lambda: [128, 256])  # Phase 14
    deep_threshold: float = 0.70        # Phase 14: cosine threshold for deep matches

    @classmethod
    def from_dict(cls, d: dict) -> "ReidConfig":
        return cls(assign_threshold=float(d.get("assign_threshold", 0.60)),
                   sighting_gap_sec=float(d.get("sighting_gap_sec", 3.0)),
                   deep_model=d.get("deep_model") or None,
                   deep_size=list(d.get("deep_size", [128, 256])),
                   deep_threshold=float(d.get("deep_threshold", 0.70)))


@dataclass
class EvidenceConfig:
    """Evidence recording: directory, camera tag, fps, pre/during/post windows, face blur + AES-256-GCM encryption at rest, retention policy and the max-events pruning cap."""
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
class AnalyticsConfig:
    """Phase 12 + 18: predictive analytics + NL summaries + hotspot."""
    enabled: bool = True
    forecast_horizon_sec: float = 10.0
    heatmap_grid_w: int = 32
    heatmap_grid_h: int = 24
    heatmap_decay_sec: float = 30.0
    trend_window_sec: float = 900.0
    # Phase 18: NL summaries
    summarizer_window_sec: float = 300.0
    # Phase 18: predictive hotspot
    hotspot_window_sec: float = 3600.0
    hotspot_decay_sec: float = 600.0
    hotspot_min_alerts: int = 2
    # Phase 18: resource allocation
    officer_pool: int = 10
    recommendation_ttl: float = 600.0



@dataclass
class EdgeConfig:
    """Phase 13.1: edge agent settings."""
    enabled: bool = False
    upstream_url: str = ""
    mqtt_broker: str = ""
    mqtt_port: int = 1883
    mqtt_topic: str = "bhairav/alerts"
    store_path: str = "output/edge_alerts.jsonl"
    fps_cap: int = 10
    push_interval_sec: float = 5.0


@dataclass
class FederationConfig:
    """Phase 13.3: multi-site federation settings."""
    enabled: bool = False
    site_id: str = "site-1"
    peers: list = field(default_factory=list)
    secret: str = ""
    push_interval_sec: float = 10.0
@dataclass
class ResponseConfig:
    """Phase 17: threat response settings."""
    ptz_enabled: bool = False
    ptz_protocol: str = "simulated"
    escalation_enabled: bool = True
    escalation_rules: list = field(default_factory=list)
    reports_dir: str = "output/reports"
    tenants_path: str = "output/tenants.json"
    integration_channels: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ResponseConfig":
        ptz = d.get("ptz", {})
        esc = d.get("escalation", {})
        return cls(
            ptz_enabled=bool(ptz.get("enabled", False)),
            ptz_protocol=str(ptz.get("protocol", "simulated")),
            escalation_enabled=bool(esc.get("enabled", True)),
            escalation_rules=list(esc.get("rules", [])),
            reports_dir=str(d.get("reports", {}).get("output_dir", "output/reports")),
            tenants_path=str(d.get("tenants", {}).get("store_path", "output/tenants.json")),
            integration_channels=list(d.get("integrations", {}).get("channels", [])),
        )


@dataclass
class HAConfig:
    """Phase 19: high availability settings."""
    enabled: bool = False
    redis_url: str = ""
    heartbeat_interval: float = 5.0
    expire_after: float = 15.0
    balancer_strategy: str = "least_conn"
    health_check_interval: float = 5.0
    failure_threshold: int = 3


@dataclass
class ComplianceConfig:
    """Phase 20: GDPR/privacy compliance settings."""
    enabled: bool = True
    evidence_retention_days: int = 90
    alert_retention_days: int = 365
    reid_retention_days: int = 180
    analytics_retention_days: int = 30
    logs_retention_days: int = 180
    consent_store_path: str = "output/consent.json"
    deletion_store_path: str = "output/deletion_requests.json"
    auto_cleanup_interval: float = 3600.0


@dataclass
class AppConfig:
    """Top-level configuration tree, built by load_config() from config.yaml deep-merged over DEFAULTS."""
    detector: str = "blob"  # blob | yolo | auto
    model: ModelConfig = field(default_factory=ModelConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    zones: list[Zone] = field(default_factory=list)
    rules: dict = field(default_factory=dict)
    backend: BackendConfig = field(default_factory=BackendConfig)
    reid: ReidConfig = field(default_factory=ReidConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    audio: AudioConfig = field(default_factory=AudioConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    ha: HAConfig = field(default_factory=HAConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

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
                   reid=ReidConfig.from_dict(d.get("reid", {})),
                   evidence=EvidenceConfig.from_dict(d.get("evidence", {})),
                   cameras=[CameraConfig.from_dict(c) for c in d.get("cameras", [])],
                   audio=AudioConfig.from_dict(d.get("audio", {})))


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load a YAML config, deep-merged over Phase 1 defaults."""
    p = Path(path)
    override: dict = {}
    if p.exists():
        override = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AppConfig.from_dict(_deep_merge(DEFAULTS, override))
