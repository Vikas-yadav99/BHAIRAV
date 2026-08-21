"""Alert escalation workflows (Phase 17.3).

Defines escalation rules that automatically respond to alerts:
- Lockdown zones when red-severity threats detected
- Trigger sirens/alarms
- Escalate through notification chains (operator -> supervisor -> emergency)
- Auto-escalate if alerts within a zone exceed a threshold
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class EscalationAction(str, Enum):
    NOTIFY = "notify"
    LOCKDOWN = "lockdown"
    SIREN = "siren"
    ESCALATE_CHAIN = "escalate_chain"
    DISPATCH = "dispatch"
    RECORD_EVIDENCE = "record_evidence"
    PTZ_TRACK = "ptz_track"


class EscalationLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class EscalationRule:
    name: str
    trigger_rules: list[str]          # which alert rules trigger this
    trigger_severity: str = "red"     # minimum severity
    trigger_count: int = 1            # alerts in window to trigger
    trigger_window_sec: float = 60.0  # time window for counting
    actions: list[str] = field(default_factory=lambda: ["notify"])
    level: str = "critical"
    cooldown_sec: float = 300.0       # minimum seconds between escalations
    zones: list[str] = field(default_factory=list)  # empty = all zones


@dataclass
class EscalationEvent:
    rule_name: str
    level: str
    actions: list[str]
    triggered_by: list[dict]
    timestamp: float
    zone: str | None = None


class EscalationEngine:
    """Processes alerts and fires escalation rules.

    Parameters
    ----------
    rules : list[EscalationRule]
        The escalation rules to evaluate.
    on_escalate : callable | None
        ``on_escalate(event: EscalationEvent)`` called when an escalation fires.
    """

    def __init__(self, rules: list[EscalationRule] | None = None,
                 on_escalate: Callable | None = None):
        self.rules = rules or []
        self._on_escalate = on_escalate
        self._alert_history: list[dict] = []
        self._cooldowns: dict[str, float] = {}
        self._events: list[EscalationEvent] = []

    def process_alert(self, alert: dict) -> list[EscalationEvent]:
        """Feed an alert through escalation rules. Returns any events that fired."""
        self._alert_history.append(alert)
        cutoff = time.time() - 300
        self._alert_history = [a for a in self._alert_history if a.get("timestamp", 0) > cutoff]

        fired = []
        now = time.time()
        zone = alert.get("zone")
        rule_name = alert.get("rule", "")
        severity = alert.get("severity", "green")

        for esc_rule in self.rules:
            # Check if this alert matches the rule
            if esc_rule.trigger_rules and rule_name not in esc_rule.trigger_rules:
                continue
            severity_order = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
            if severity_order.get(severity, 0) < severity_order.get(esc_rule.trigger_severity, 3):
                continue
            if esc_rule.zones and zone not in esc_rule.zones:
                continue
            # Check cooldown
            if now - self._cooldowns.get(esc_rule.name, 0) < esc_rule.cooldown_sec:
                continue
            # Count matching alerts in window
            window_start = now - esc_rule.trigger_window_sec
            matching = [a for a in self._alert_history
                        if a.get("rule") in esc_rule.trigger_rules
                        and a.get("timestamp", 0) >= window_start]
            if len(matching) < esc_rule.trigger_count:
                continue
            # Fire escalation
            event = EscalationEvent(
                rule_name=esc_rule.name,
                level=esc_rule.level,
                actions=list(esc_rule.actions),
                triggered_by=matching[-5:],
                timestamp=now,
                zone=zone,
            )
            self._cooldowns[esc_rule.name] = now
            self._events.append(event)
            fired.append(event)
            logger.warning("Escalation fired: %s (level=%s, actions=%s)",
                           esc_rule.name, esc_rule.level, esc_rule.actions)
            if self._on_escalate:
                try:
                    self._on_escalate(event)
                except Exception as exc:
                    logger.error("Escalation callback failed: %s", exc)
        return fired

    @property
    def events(self) -> list[EscalationEvent]:
        return list(self._events[-100:])

    def recent_events(self, limit: int = 20) -> list[dict]:
        return [{"rule": e.rule_name, "level": e.level, "actions": e.actions,
                 "zone": e.zone, "ts": round(e.timestamp, 3)}
                for e in self._events[-limit:]]
