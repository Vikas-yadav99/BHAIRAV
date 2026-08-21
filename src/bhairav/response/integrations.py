"""Third-party integrations (Phase 17.5).

Pluggable integration hub for external systems:
- Emergency services (911, fire, EMS)
- Traffic management systems
- Building management / SCADA
- Custom webhooks
- SMS / email notifications
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class ChannelType(str, Enum):
    WEBHOOK = "webhook"
    SMS = "sms"
    EMAIL = "email"
    EMERGENCY_911 = "emergency_911"
    FIRE_SYSTEM = "fire_system"
    TRAFFIC_MGMT = "traffic_mgmt"
    SCADA = "scada"
    MQTT = "mqtt"
    CUSTOM = "custom"


@dataclass
class ExternalChannel:
    channel_id: str
    channel_type: str
    name: str
    endpoint: str = ""
    enabled: bool = True
    severity_filter: list[str] = field(default_factory=lambda: ["red"])
    rule_filter: list[str] = field(default_factory=list)
    retry_count: int = 3
    timeout_sec: float = 10.0
    metadata: dict = field(default_factory=dict)


@dataclass
class IntegrationEvent:
    channel_id: str
    channel_type: str
    alert: dict
    status: str  # sent | failed | skipped
    response_code: int | None = None
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


class IntegrationHub:
    """Manages external integrations and dispatches alerts to them.

    Parameters
    ----------
    on_dispatch : callable | None
        ``on_dispatch(event: IntegrationEvent)`` for each dispatch attempt.
    """

    def __init__(self, on_dispatch: Callable | None = None):
        self._channels: dict[str, ExternalChannel] = {}
        self._on_dispatch = on_dispatch
        self._events: list[IntegrationEvent] = []
        self._lock = threading.Lock()

    def register_channel(self, channel: ExternalChannel) -> None:
        with self._lock:
            self._channels[channel.channel_id] = channel
            logger.info("Registered integration channel: %s (%s)", channel.name, channel.channel_type)

    def unregister_channel(self, channel_id: str) -> bool:
        with self._lock:
            if channel_id in self._channels:
                del self._channels[channel_id]
                return True
            return False

    def list_channels(self) -> list[dict]:
        return [c.__dict__.copy() for c in self._channels.values()]

    def dispatch_alert(self, alert: dict) -> list[IntegrationEvent]:
        """Send an alert to all matching channels."""
        events = []
        severity = alert.get("severity", "green")
        rule = alert.get("rule", "")

        for ch in list(self._channels.values()):
            if not ch.enabled:
                continue
            if ch.severity_filter and severity not in ch.severity_filter:
                continue
            if ch.rule_filter and rule not in ch.rule_filter:
                continue
            event = self._send_to_channel(ch, alert)
            events.append(event)
            with self._lock:
                self._events.append(event)
            if self._on_dispatch:
                try:
                    self._on_dispatch(event)
                except Exception:
                    pass
        return events

    def _send_to_channel(self, channel: ExternalChannel, alert: dict) -> IntegrationEvent:
        """Dispatch to a single channel (simulated for now)."""
        try:
            # In production: HTTP POST, MQTT publish, SMTP send, etc.
            logger.info("Dispatching alert to %s (%s): %s",
                        channel.name, channel.channel_type, alert.get("rule"))
            return IntegrationEvent(
                channel_id=channel.channel_id,
                channel_type=channel.channel_type,
                alert=alert,
                status="sent",
                response_code=200,
            )
        except Exception as exc:
            return IntegrationEvent(
                channel_id=channel.channel_id,
                channel_type=channel.channel_type,
                alert=alert,
                status="failed",
                error=str(exc),
            )

    def recent_events(self, limit: int = 50) -> list[dict]:
        return [{"channel": e.channel_id, "type": e.channel_type,
                 "status": e.status, "rule": e.alert.get("rule", ""),
                 "ts": round(e.timestamp, 3)}
                for e in self._events[-limit:]]
