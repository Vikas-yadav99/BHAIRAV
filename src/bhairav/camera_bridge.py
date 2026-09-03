"""Camera-to-Incident Bridge — auto-creates incidents from BHAIRAV camera alerts.

When the camera pipeline detects a high-severity event (fight, fall, fire, etc.),
this bridge automatically:
1. Maps the alert rule to an incident category
2. Creates an incident in the IncidentStore
3. Auto-dispatches nearest officers via DispatchEngine
4. Publishes the incident to the operator dashboard in real-time

Severity mapping:
  yellow (1) → Emergency Level 2 (Medium)
  orange (2) → Emergency Level 3 (High)
  red    (3) → Emergency Level 4 (Critical)
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .incidents import IncidentStore, DispatchEngine
    from .backend.server import LiveHub

log = logging.getLogger("bhairav.camera_bridge")

# Map camera alert rules to incident categories
RULE_TO_CATEGORY = {
    "fight": "crime",
    "fall": "medical",
    "chase": "crime",
    "riot": "crime",
    "accident": "road_accident",
    "trespass": "crime",
    "anomaly": "other",
    "loitering": "other",
    "crowd_density": "other",
    "zone_crossing": "other",
    "stolen_vehicle": "crime",
    "abandoned_object": "crime",
}

# Map severity colors to emergency levels
SEVERITY_TO_LEVEL = {
    "green": 1,
    "yellow": 2,
    "orange": 3,
    "red": 4,
}


class CameraIncidentBridge:
    """Listens for camera alerts and auto-creates incidents.

    Usage:
        bridge = CameraIncidentBridge(incident_store, dispatch_engine, hub)
        # In the pipeline thread, when an alert fires:
        bridge.on_camera_alert(alert_dict, camera_id)
    """

    def __init__(self, store: "IncidentStore", dispatch_engine: "DispatchEngine",
                 hub: "LiveHub" | None = None,
                 min_severity: str = "yellow",
                 cooldown_sec: float = 30.0):
        self.store = store
        self.dispatch_engine = dispatch_engine
        self.hub = hub
        self.min_severity = min_severity
        self.cooldown_sec = cooldown_sec
        self._last_fires: dict[str, float] = {}  # rule+camera -> last fire time
        self._stats = {"alerts_received": 0, "incidents_created": 0,
                       "officers_dispatched": 0, "cooled_down": 0}

    def on_camera_alert(self, alert: dict, camera_id: str = "") -> dict | None:
        """Called when the camera pipeline fires an alert.

        Args:
            alert: Alert dict from the pipeline (rule, severity, message, etc.)
            camera_id: Camera identifier

        Returns:
            Created incident dict, or None if skipped (cooldown/low severity).
        """
        self._stats["alerts_received"] += 1
        rule = alert.get("rule", "unknown")
        severity = alert.get("severity", "yellow")
        message = alert.get("message", "")
        confidence = alert.get("confidence", 0.5)
        zone = alert.get("zone", "")
        timestamp = alert.get("timestamp", time.time())

        # Check minimum severity
        sev_order = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
        if sev_order.get(severity, 0) < sev_order.get(self.min_severity, 1):
            return None

        # Check cooldown (same rule + camera within cooldown window)
        cooldown_key = f"{rule}:{camera_id}"
        last = self._last_fires.get(cooldown_key, 0)
        if timestamp - last < self.cooldown_sec:
            self._stats["cooled_down"] += 1
            return None
        self._last_fires[cooldown_key] = timestamp

        # Map to incident
        category = RULE_TO_CATEGORY.get(rule, "other")
        emergency_level = SEVERITY_TO_LEVEL.get(severity, 2)

        # Location: use camera's position (default Delhi center)
        # In production, cameras would have GPS coordinates
        camera_positions = {
            "CAM-01": (28.6139, 77.2090),
            "CAM-02": (28.6150, 77.2100),
            "CAM-03": (28.6120, 77.2080),
            "CAM-04": (28.6160, 77.2110),
            "CAM-05": (28.6110, 77.2070),
            "CAM-06": (28.6180, 77.2130),
        }
        lat, lng = camera_positions.get(camera_id, (28.6139, 77.2090))

        # Create incident
        inc = self.store.create_incident(
            category=category,
            emergency_level=emergency_level,
            lat=lat,
            lng=lng,
            location_name=f"{camera_id or 'Camera'} — {zone or 'Zone'}",
            description=f"[Camera AI] {rule.upper()}: {message}",
            reporter_name="BHAIRAV AI",
            source="camera",
        )

        # Tag with camera metadata
        inc.ai_verified = True
        inc.crowd_reports = 0  # AI detection, not crowd
        self.store._save_incidents()

        # Auto-dispatch based on severity
        assigned = []
        if emergency_level >= 2:
            assigned = self.dispatch_engine.dispatch(inc)

        self._stats["incidents_created"] += 1
        self._stats["officers_dispatched"] += len(assigned)

        result = inc.to_dict()
        result["dispatched_officers"] = [o.to_dict() for o in assigned]

        # Publish to operator dashboard
        if self.hub:
            try:
                self.hub.publish_incident(result)
            except Exception as exc:
                log.warning("Failed to publish incident to dashboard: %s", exc)

        log.info(
            "Camera alert → incident: rule=%s severity=%s camera=%s → incident=%s dispatched=%d",
            rule, severity, camera_id, inc.id, len(assigned),
        )

        return result

    def stats(self) -> dict:
        return dict(self._stats)

    def cleanup_cooldowns(self, max_age: float = 3600) -> int:
        """Remove stale cooldown entries."""
        now = time.time()
        before = len(self._last_fires)
        self._last_fires = {
            k: v for k, v in self._last_fires.items()
            if now - v < max_age
        }
        return before - len(self._last_fires)
