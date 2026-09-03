"""SmartAlertManager: reduces false-positive alerts."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .types import Alert, FrameState, Severity, SEVERITY_ORDER

log = logging.getLogger("bhairav.alert_manager")


def _severity_rank(s: Severity) -> int:
    return SEVERITY_ORDER.index(s)


@dataclass
class AlertStats:
    total_received: int = 0
    total_emitted: int = 0
    dropped_confidence: int = 0
    dropped_sustained: int = 0
    dropped_cooldown: int = 0
    dropped_ratelimit: int = 0
    merged_dedup: int = 0

    def to_dict(self) -> dict:
        return {
            "total_received": self.total_received,
            "total_emitted": self.total_emitted,
            "dropped_confidence": self.dropped_confidence,
            "dropped_sustained": self.dropped_sustained,
            "dropped_cooldown": self.dropped_cooldown,
            "dropped_ratelimit": self.dropped_ratelimit,
            "merged_dedup": self.merged_dedup,
            "acceptance_rate": round(self.total_emitted / max(1, self.total_received), 3),
        }


@dataclass
class _PendingAlert:
    rule: str
    zone: str | None
    track_id: int | None
    severity: Severity
    first_seen: float
    last_seen: float
    frame_count: int = 1
    best_confidence: float = 0.0
    latest_message: str = ""
    latest_details: dict = field(default_factory=dict)


class SmartAlertManager:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.min_confidence: float = float(cfg.get("min_confidence", 0.3))
        self.sustained_frames: int = int(cfg.get("sustained_frames", 2))
        self.dedup_radius: float = float(cfg.get("dedup_radius", 0.15))
        self.cooldown_sec: float = float(cfg.get("cooldown_sec", 15.0))
        self.max_alerts_per_min: int = int(cfg.get("max_alerts_per_min", 20))
        self.escalate_cooldown_sec: float = float(cfg.get("escalate_cooldown_sec", 5.0))
        self._pending: dict[tuple, _PendingAlert] = {}
        self._cooldowns: dict[tuple, float] = {}
        self._recent_emissions: list[float] = []
        self.stats = AlertStats()

    def process(self, raw_alerts: list[Alert], state: FrameState | None = None) -> list[Alert]:
        self.stats.total_received += len(raw_alerts)
        now = state.timestamp if state else time.time()

        # Step 1: Confidence filtering
        filtered = []
        for alert in raw_alerts:
            if alert.confidence >= self.min_confidence:
                filtered.append(alert)
            else:
                self.stats.dropped_confidence += 1

        # Step 2: Sustained detection
        newly_emitted: list[Alert] = []

        for alert in filtered:
            key = (alert.rule, alert.zone, alert.track_id)

            if key in self._pending:
                p = self._pending[key]
                p.last_seen = now
                p.frame_count += 1
                p.best_confidence = max(p.best_confidence, alert.confidence)
                p.latest_message = alert.message
                p.latest_details = alert.details
                if _severity_rank(alert.severity) > _severity_rank(p.severity):
                    p.severity = alert.severity
            else:
                self._pending[key] = _PendingAlert(
                    rule=alert.rule,
                    zone=alert.zone,
                    track_id=alert.track_id,
                    severity=alert.severity,
                    first_seen=now,
                    last_seen=now,
                    frame_count=1,
                    best_confidence=alert.confidence,
                    latest_message=alert.message,
                    latest_details=alert.details,
                )

            if key in self._pending and self._pending[key].frame_count >= self.sustained_frames:
                newly_emitted.append(self._finalize_pending(self._pending[key], now))
                del self._pending[key]

        # Step 3: Expire stale pending
        stale_keys = [k for k, p in self._pending.items() if now - p.last_seen > 5.0]
        for k in stale_keys:
            self.stats.dropped_sustained += 1
            del self._pending[k]

        # Step 4: Cooldown filtering
        cooldown_filtered = []
        for alert in newly_emitted:
            key = (alert.rule, alert.zone, alert.track_id, alert.severity.value)
            last_fired = self._cooldowns.get(key)
            is_escalation = alert.severity in (Severity.ORANGE, Severity.RED)
            cd = self.escalate_cooldown_sec if is_escalation else self.cooldown_sec
            if last_fired is not None and (now - last_fired) < cd:
                self.stats.dropped_cooldown += 1
                continue
            self._cooldowns[key] = now
            cooldown_filtered.append(alert)

        # Step 5: Global rate limit
        if self.max_alerts_per_min > 0:
            cutoff = now - 60.0
            self._recent_emissions = [t for t in self._recent_emissions if t > cutoff]
            rate_filtered = []
            for alert in cooldown_filtered:
                if len(self._recent_emissions) < self.max_alerts_per_min:
                    self._recent_emissions.append(now)
                    rate_filtered.append(alert)
                else:
                    self.stats.dropped_ratelimit += 1
            cooldown_filtered = rate_filtered

        # Step 6: Spatial deduplication
        if self.dedup_radius > 0 and len(cooldown_filtered) > 1:
            cooldown_filtered = self._dedup(cooldown_filtered)

        self.stats.total_emitted += len(cooldown_filtered)
        return cooldown_filtered

    def _finalize_pending(self, pending: _PendingAlert, now: float) -> Alert:
        return Alert(
            rule=pending.rule,
            zone=pending.zone,
            track_id=pending.track_id,
            severity=pending.severity,
            message=pending.latest_message,
            frame_id=0,
            timestamp=now,
            details={
                **pending.latest_details,
                "sustained_frames": pending.frame_count,
                "first_seen": pending.first_seen,
                "confidence_peak": round(pending.best_confidence, 3),
            },
            confidence=pending.best_confidence,
        )

    def _dedup(self, alerts: list[Alert]) -> list[Alert]:
        result: list[Alert] = []
        seen: dict[tuple, Alert] = {}
        for alert in sorted(alerts, key=lambda a: _severity_rank(a.severity), reverse=True):
            dedup_key = (alert.rule, alert.track_id)
            if dedup_key not in seen:
                seen[dedup_key] = alert
                result.append(alert)
            else:
                existing = seen[dedup_key]
                if self._zones_close(existing.zone, alert.zone):
                    self.stats.merged_dedup += 1
                else:
                    result.append(alert)
                    seen[dedup_key] = alert
        return result

    def _zones_close(self, zone_a: str | None, zone_b: str | None) -> bool:
        if zone_a == zone_b:
            return True
        if zone_a is None or zone_b is None:
            return True
        return zone_a.split("_")[0].split("-")[0].lower() == zone_b.split("_")[0].split("-")[0].lower()

    def reset(self) -> None:
        self._pending.clear()
        self._cooldowns.clear()
        self._recent_emissions.clear()
        self.stats = AlertStats()

    def get_stats(self) -> dict:
        return self.stats.to_dict()

    def pending_count(self) -> int:
        return len(self._pending)
