"""Event bus subscribers that wire orphaned modules into the pipeline.

Each subscriber is a thin adapter: it receives Events from the bus and
forwards them to the existing module (EscalationEngine, PTZTracker, etc.).
This replaces the 150-line on_frame callback in serve.py.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .events import Event, EventBus

log = logging.getLogger("bhairav.subscribers")


class EscalationSubscriber:
    """Forwards alert events to the EscalationEngine.

    When a red-severity alert fires, the escalation engine checks its rules
    and triggers callbacks (lockdown, siren, escalation chains).
    """

    def __init__(self, engine, hub=None):
        """
        engine: EscalationEngine instance
        hub: LiveHub for dispatching field alerts (optional)
        """
        self._engine = engine
        self._hub = hub

    def __call__(self, event: Event) -> None:
        alert = event.data
        severity = alert.get("severity", "")
        # Escalation only triggers on high-severity alerts
        if severity not in ("red", "orange"):
            return
        try:
            result = self._engine.evaluate(alert)
            if result and self._hub:
                for escalated in result:
                    self._hub.publish_field_alert(escalated)
        except Exception as exc:
            log.error("EscalationSubscriber error: %s", exc)


class PTZSubscriber:
    """Forwards frame events to the PTZTracker for auto-tracking.

    The tracker receives person detections and issues pan/tilt/zoom commands
    to follow the highest-priority target.
    """

    def __init__(self, tracker):
        """
        tracker: PTZTracker instance
        """
        self._tracker = tracker

    def __call__(self, event: Event) -> None:
        tracks = event.data.get("tracks", [])
        if not tracks:
            return
        try:
            # Convert track dicts to objects the tracker expects
            class SimpleTrack:
                def __init__(self, d):
                    self.track_id = d.get("id", d.get("track_id", 0))
                    self.bbox = tuple(d.get("bbox", [0, 0, 0, 0]))
                    self.label = d.get("label", "person")
                    self.confidence = d.get("conf", d.get("confidence", 1.0))

            track_objs = [SimpleTrack(t) for t in tracks if t.get("label") == "person"]
            if track_objs:
                self._tracker.update(track_objs, event.data.get("timestamp", time.time()))
        except Exception as exc:
            log.error("PTZSubscriber error: %s", exc)


class IntegrationSubscriber:
    """Forwards alert events to external integrations (911, fire, EMS, traffic).

    Each alert is checked against registered channels and forwarded
    if it matches the channel's filter criteria.
    """

    def __init__(self, hub):
        """
        hub: IntegrationHub instance
        """
        self._hub = hub

    def __call__(self, event: Event) -> None:
        alert = event.data
        try:
            self._hub.dispatch(alert)
        except Exception as exc:
            log.error("IntegrationSubscriber error: %s", exc)


class FederationSubscriber:
    """Forwards alert events to peer BHAIRAV servers via federation.

    Alerts are queued and pushed in batches to configured peer servers.
    """

    def __init__(self, client):
        """
        client: FederationClient instance
        """
        self._client = client

    def __call__(self, event: Event) -> None:
        try:
            self._client.send_alert(event.data)
        except Exception as exc:
            log.error("FederationSubscriber error: %s", exc)


class AuditSubscriber:
    """Logs security-relevant events to the SecurityAuditLog.

    Tracks login attempts, alert firings, and data access for compliance.
    """

    def __init__(self, audit_log):
        """
        audit_log: SecurityAuditLog instance from security.py
        """
        self._log = audit_log

    def __call__(self, event: Event) -> None:
        alert = event.data
        severity = alert.get("severity", "info")
        rule = alert.get("rule", "unknown")
        zone = alert.get("zone", "")
        camera = event.source or alert.get("camera", "")
        self._log.log(
            event_type="alert_fired",
            details=f"{rule} in {zone} on {camera} (severity={severity})",
            source=camera,
            severity="warning" if severity in ("red", "orange") else "info",
        )


class EvidenceSubscriber:
    """Notifies the evidence system when alerts fire (for clip recording).

    This is a lightweight adapter — the actual recording is handled by
    EventRecorder in serve.py. This subscriber just ensures the event
    bus path is wired.
    """

    def __init__(self, recorder=None):
        self._recorder = recorder

    def __call__(self, event: Event) -> None:
        # Evidence recording is still handled inline in on_frame
        # because it needs access to the raw frame. This subscriber
        # exists for future decoupling when frames go through the bus.
        pass


def wire_subscribers(bus: EventBus, *, escalation_engine=None,
                     ptz_tracker=None, integration_hub=None,
                     federation_client=None, audit_log=None,
                     live_hub=None) -> dict:
    """Wire all subscribers to the event bus. Returns dict of active subscribers.

    Call this once during startup. Modules that aren't configured are skipped.
    """
    active = {}

    if escalation_engine:
        sub = EscalationSubscriber(escalation_engine, hub=live_hub)
        bus.subscribe("alert", sub)
        active["escalation"] = sub
        log.info("Wired: escalation engine -> event bus")

    if ptz_tracker:
        sub = PTZSubscriber(ptz_tracker)
        bus.subscribe("frame", sub)
        active["ptz"] = sub
        log.info("Wired: PTZ tracker -> event bus")

    if integration_hub:
        sub = IntegrationSubscriber(integration_hub)
        bus.subscribe("alert", sub)
        active["integrations"] = sub
        log.info("Wired: integration hub -> event bus")

    if federation_client:
        sub = FederationSubscriber(federation_client)
        bus.subscribe("alert", sub)
        active["federation"] = sub
        log.info("Wired: federation client -> event bus")

    if audit_log:
        sub = AuditSubscriber(audit_log)
        bus.subscribe("alert", sub)
        active["audit"] = sub
        log.info("Wired: security audit log -> event bus")

    return active
