"""BHAIRAV City Safety — Dedup, Notifications, GPS Tracking, Metrics, Resolution.

This module fills the gaps between camera detection and real-world response:

1. Incident Deduplication — same event from multiple sources → merge into one
2. Notification System — browser push, SMS stub, WebSocket dispatch channel
3. GPS Tracking — officer location streaming to operator dashboard
4. Response Metrics — SLA tracking, avg response time, category breakdown
5. Resolution Workflow — officer uploads proof (photo + notes), marks resolved
"""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger("bhairav.city_safety")


# ── Incident Deduplication ──────────────────────────────────────────────────

# Same incident reported within these thresholds is considered duplicate
DEDUP_RADIUS_METERS = 200.0     # within 200m
DEDUP_TIME_WINDOW_SEC = 1800.0  # within 30 minutes
DEDUP_CATEGORY_MATCH = True     # same category required


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in meters."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class DedupResult:
    """Result of a deduplication check."""
    is_duplicate: bool
    merged_into: str | None = None   # incident ID if merged
    crowd_incremented: bool = False
    new_incident: dict | None = None  # if not duplicate, the created incident


class IncidentDeduplicator:
    """Detects and merges duplicate incident reports.

    When multiple people (or camera + people) report the same event,
    this merges them into one incident and increments the crowd counter.
    """

    def __init__(self, store, dispatch_engine=None,
                 radius_m: float = DEDUP_RADIUS_METERS,
                 time_window: float = DEDUP_TIME_WINDOW_SEC):
        self.store = store
        self.dispatch_engine = dispatch_engine
        self.radius_m = radius_m
        self.time_window = time_window
        self._stats = {
            "reports_received": 0,
            "duplicates_found": 0,
            "incidents_merged": 0,
            "new_incidents": 0,
        }

    def check_and_merge(self, category: str, lat: float, lng: float,
                        description: str = "", reporter_phone: str = "",
                        reporter_name: str = "", source: str = "public",
                        emergency_level: int = 1) -> DedupResult:
        """Check if a new report duplicates an existing incident.

        If duplicate: increments crowd counter, escalates if threshold met.
        If new: creates the incident.
        """
        self._stats["reports_received"] += 1
        now = time.time()

        # Search recent incidents for potential duplicates
        candidates = []
        for inc in self.store.list_incidents(limit=500):
            age = now - inc.created_at
            if age > self.time_window:
                continue
            if inc.status in ("resolved", "cancelled"):
                continue
            dist = _haversine_m(lat, lng, inc.location_lat, inc.location_lng)
            if dist > self.radius_m:
                continue
            if DEDUP_CATEGORY_MATCH and inc.category != category:
                continue
            candidates.append((dist, inc))

        if candidates:
            # Find closest match
            candidates.sort(key=lambda x: x[0])
            dist, best_match = candidates[0]

            # Merge: increment crowd counter
            self._stats["duplicates_found"] += 1
            self._stats["incidents_merged"] += 1

            inc = self.store.add_crowd_report(best_match.id)

            # Add reporter info to timeline
            if inc:
                inc.timeline.append({
                    "status": "crowd_report",
                    "time": now,
                    "note": f"Additional report from {reporter_name or 'anonymous'} "
                            f"({source}), {dist:.0f}m from original. "
                            f"Total crowd reports: {inc.crowd_reports}",
                })
                self.store._save_incidents()

                # If crowd reports >= 3, auto-verify
                if inc.crowd_reports >= 3 and inc.status == "reported":
                    self.store.update_incident(
                        inc.id, status="verified",
                        note=f"Auto-verified: {inc.crowd_reports} independent reports"
                    )

            return DedupResult(
                is_duplicate=True,
                merged_into=best_match.id,
                crowd_incremented=True,
            )

        # New incident — create it
        inc = self.store.create_incident(
            category=category,
            emergency_level=emergency_level,
            lat=lat,
            lng=lng,
            location_name=description[:50] if description else "Reported location",
            description=description,
            reporter_phone=reporter_phone,
            reporter_name=reporter_name,
            source=source,
        )
        self._stats["new_incidents"] += 1

        # Auto-dispatch if severity warrants it
        if self.dispatch_engine and emergency_level >= 2:
            self.dispatch_engine.dispatch(inc)

        return DedupResult(
            is_duplicate=False,
            new_incident=inc.to_dict(),
        )

    def stats(self) -> dict:
        return dict(self._stats)


# ── Notification System ─────────────────────────────────────────────────────

@dataclass
class Notification:
    id: str
    recipient_type: str   # "officer", "operator", "public"
    recipient_id: str     # officer_id, or "all_operators", or phone
    channel: str          # "push", "sms", "websocket", "email"
    title: str
    body: str
    priority: str         # "low", "medium", "high", "critical"
    created_at: float
    sent_at: float | None = None
    read: bool = False
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "recipient_type": self.recipient_type,
            "recipient_id": self.recipient_id,
            "channel": self.channel,
            "title": self.title,
            "body": self.body,
            "priority": self.priority,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
            "read": self.read,
            "data": self.data,
        }


class NotificationManager:
    """Multi-channel notification system.

    Channels:
    - websocket: Real-time push to connected dashboards
    - push: Browser push notifications (service worker)
    - sms: Twilio/similar gateway stub
    - email: Future email gateway
    """

    PRIORITY_LEVELS = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def __init__(self, hub=None, sms_gateway_url: str | None = None):
        self.hub = hub
        self.sms_gateway_url = sms_gateway_url
        self._notifications: list[Notification] = []
        self._subscribers: dict[str, set[Callable]] = defaultdict(set)
        self._stats = {
            "total_sent": 0, "by_channel": Counter(),
            "by_priority": Counter(), "failed": 0,
        }

    def notify_officer(self, officer_id: str, title: str, body: str,
                       priority: str = "high", data: dict | None = None,
                       channels: list[str] | None = None):
        """Send notification to a specific officer."""
        channels = channels or ["websocket", "push"]
        now = time.time()

        for channel in channels:
            n = Notification(
                id=uuid.uuid4().hex[:12],
                recipient_type="officer",
                recipient_id=officer_id,
                channel=channel,
                title=title,
                body=body,
                priority=priority,
                created_at=now,
                data=data or {},
            )
            self._notifications.append(n)
            self._send(n)
            self._stats["total_sent"] += 1
            self._stats["by_channel"][channel] += 1
            self._stats["by_priority"][priority] += 1

    def notify_operators(self, title: str, body: str,
                         priority: str = "high", data: dict | None = None):
        """Broadcast notification to all connected operators."""
        now = time.time()
        n = Notification(
            id=uuid.uuid4().hex[:12],
            recipient_type="operator",
            recipient_id="all_operators",
            channel="websocket",
            title=title,
            body=body,
            priority=priority,
            created_at=now,
            data=data or {},
        )
        self._notifications.append(n)
        self._send(n)
        self._stats["total_sent"] += 1
        self._stats["by_channel"]["websocket"] += 1
        self._stats["by_priority"][priority] += 1

    def send_sms(self, phone: str, message: str, priority: str = "medium"):
        """Send SMS via gateway stub.

        In production, this would call Twilio or similar.
        For now, it logs and stores the intent.
        """
        now = time.time()
        n = Notification(
            id=uuid.uuid4().hex[:12],
            recipient_type="public",
            recipient_id=phone,
            channel="sms",
            title="BHAIRAV Alert",
            body=message,
            priority=priority,
            created_at=now,
        )
        self._notifications.append(n)

        if self.sms_gateway_url:
            # Stub: would POST to Twilio/TextBelt
            log.info("SMS → %s: %s (gateway: %s)", phone, message[:50],
                     self.sms_gateway_url)
            n.sent_at = now
        else:
            log.info("SMS stub → %s: %s (no gateway configured)", phone, message[:50])
            n.sent_at = now

        self._stats["total_sent"] += 1
        self._stats["by_channel"]["sms"] += 1
        self._stats["by_priority"][priority] += 1

    def _send(self, notification: Notification):
        """Dispatch notification through the appropriate channel."""
        try:
            if notification.channel == "websocket" and self.hub:
                # Push to field alert channel
                self.hub.publish_field_alert({
                    "type": "notification",
                    "notification": notification.to_dict(),
                })
                notification.sent_at = time.time()
            elif notification.channel == "push":
                # Browser push via service worker
                # In production: webpush.sendNotification()
                log.info("Push notification → %s: %s",
                         notification.recipient_id, notification.title)
                notification.sent_at = time.time()
            elif notification.channel == "sms":
                self.send_sms(
                    notification.recipient_id,
                    f"{notification.title}\n{notification.body}",
                    notification.priority,
                )
        except Exception as exc:
            log.warning("Failed to send notification %s: %s", notification.id, exc)
            self._stats["failed"] += 1

    def get_recent(self, limit: int = 50, unread_only: bool = False) -> list[dict]:
        """Get recent notifications."""
        notifs = self._notifications[-limit:]
        if unread_only:
            notifs = [n for n in notifs if not n.read]
        return [n.to_dict() for n in reversed(notifs)]

    def mark_read(self, notification_id: str) -> bool:
        for n in self._notifications:
            if n.id == notification_id:
                n.read = True
                return True
        return False

    def stats(self) -> dict:
        return {
            "total": len(self._notifications),
            "sent": self._stats["total_sent"],
            "failed": self._stats["failed"],
            "by_channel": dict(self._stats["by_channel"]),
            "by_priority": dict(self._stats["by_priority"]),
        }


# ── GPS Tracking ────────────────────────────────────────────────────────────

@dataclass
class GPSCoordinate:
    lat: float
    lng: float
    timestamp: float
    speed: float = 0.0      # km/h
    heading: float = 0.0    # degrees (0=N, 90=E)
    accuracy: float = 10.0  # meters


class GPSTracker:
    """Tracks real-time GPS positions for all officers.

    Stores the last N positions per officer for trajectory display.
    Integrates with the operator dashboard via WebSocket.
    """

    MAX_HISTORY_PER_OFFICER = 100
    STALE_THRESHOLD_SEC = 120.0  # officer considered "stale" after 2 min

    def __init__(self, store, hub=None):
        self.store = store
        self.hub = hub
        self._positions: dict[str, list[GPSCoordinate]] = defaultdict(list)
        self._last_update: dict[str, float] = {}

    def update_position(self, officer_id: str, lat: float, lng: float,
                        speed: float = 0.0, heading: float = 0.0,
                        accuracy: float = 10.0) -> GPSCoordinate | None:
        """Update an officer's GPS position."""
        off = self.store.get_officer(officer_id)
        if not off:
            return None

        coord = GPSCoordinate(
            lat=lat, lng=lng, timestamp=time.time(),
            speed=speed, heading=heading, accuracy=accuracy,
        )

        history = self._positions[officer_id]
        history.append(coord)
        if len(history) > self.MAX_HISTORY_PER_OFFICER:
            history.pop(0)

        self._last_update[officer_id] = coord.timestamp

        # Update store
        self.store.update_officer(officer_id, location_lat=lat, location_lng=lng)

        # Publish to operator dashboard
        if self.hub:
            try:
                self.hub.publish_field_alert({
                    "type": "gps_update",
                    "officer_id": officer_id,
                    "lat": lat, "lng": lng,
                    "speed": speed, "heading": heading,
                    "accuracy": accuracy,
                    "timestamp": coord.timestamp,
                })
            except Exception:
                pass

        return coord

    def get_position(self, officer_id: str) -> GPSCoordinate | None:
        """Get latest position for an officer."""
        history = self._positions.get(officer_id, [])
        return history[-1] if history else None

    def get_trajectory(self, officer_id: str, last_n: int = 50) -> list[dict]:
        """Get recent trajectory for map rendering."""
        history = self._positions.get(officer_id, [])[-last_n:]
        return [
            {"lat": c.lat, "lng": c.lng, "ts": c.timestamp,
             "speed": c.speed, "heading": c.heading}
            for c in history
        ]

    def get_all_positions(self) -> dict[str, dict]:
        """Get current positions of all officers (for operator map)."""
        result = {}
        now = time.time()
        for officer_id, history in self._positions.items():
            if not history:
                continue
            latest = history[-1]
            stale = (now - latest.timestamp) > self.STALE_THRESHOLD_SEC
            off = self.store.get_officer(officer_id)
            result[officer_id] = {
                "lat": latest.lat,
                "lng": latest.lng,
                "timestamp": latest.timestamp,
                "speed": latest.speed,
                "heading": latest.heading,
                "stale": stale,
                "name": off.name if off else "",
                "role": off.role if off else "",
            }
        return result

    def get_distance_to_incident(self, officer_id: str,
                                  inc_lat: float, inc_lng: float) -> float | None:
        """Calculate distance from officer to incident in meters."""
        pos = self.get_position(officer_id)
        if not pos:
            return None
        return _haversine_m(pos.lat, pos.lng, inc_lat, inc_lng)

    def find_nearest_by_gps(self, lat: float, lng: float,
                            role: str | None = None,
                            radius_km: float = 10.0,
                            limit: int = 5) -> list[dict]:
        """Find nearest officers using live GPS positions (not last registered)."""
        now = time.time()
        candidates = []

        for officer_id, history in self._positions.items():
            if not history:
                continue
            latest = history[-1]
            stale = (now - latest.timestamp) > self.STALE_THRESHOLD_SEC
            if stale:
                continue

            off = self.store.get_officer(officer_id)
            if not off or off.status != "available":
                continue
            if role and off.role != role:
                continue

            dist = _haversine_m(lat, lng, latest.lat, latest.lng) / 1000.0
            if dist <= radius_km:
                candidates.append({
                    "distance_km": round(dist, 2),
                    "officer": off.to_dict(),
                    "gps": {"lat": latest.lat, "lng": latest.lng,
                            "speed": latest.speed, "heading": latest.heading},
                })

        candidates.sort(key=lambda x: x["distance_km"])
        return candidates[:limit]

    def stats(self) -> dict:
        now = time.time()
        active = sum(
            1 for oid, hist in self._positions.items()
            if hist and (now - hist[-1].timestamp) < self.STALE_THRESHOLD_SEC
        )
        return {
            "total_tracked": len(self._positions),
            "active": active,
            "stale": len(self._positions) - active,
        }


# ── Response Metrics ────────────────────────────────────────────────────────

class ResponseMetrics:
    """Track response times, SLA compliance, and operational analytics.

    SLA targets:
    - Level 4 (Critical): respond within 3 minutes
    - Level 3 (High): respond within 5 minutes
    - Level 2 (Medium): respond within 10 minutes
    - Level 1 (Low): respond within 30 minutes
    """

    SLA_TARGETS = {
        4: 180,   # 3 minutes
        3: 300,   # 5 minutes
        2: 600,   # 10 minutes
        1: 1800,  # 30 minutes
    }

    def __init__(self, store):
        self.store = store
        self._response_times: list[dict] = []  # completed incident metrics

    def record_dispatch(self, incident_id: str):
        """Record when an incident is dispatched (for response time tracking)."""
        inc = self.store.get_incident(incident_id)
        if inc:
            inc.timeline.append({
                "status": "metrics_dispatch",
                "time": time.time(),
                "note": "Response timer started",
            })
            self.store._save_incidents()

    def record_resolution(self, incident_id: str) -> dict | None:
        """Record when an incident is resolved. Returns response time metrics."""
        inc = self.store.get_incident(incident_id)
        if not inc:
            return None

        now = time.time()
        created = inc.created_at

        # Find first dispatch time
        dispatch_time = None
        for entry in inc.timeline:
            if entry["status"] == "dispatched":
                dispatch_time = entry["time"]
                break

        # Find first acknowledgment
        ack_time = None
        for entry in inc.timeline:
            if entry["status"] == "acknowledged":
                ack_time = entry["time"]
                break

        # Calculate metrics
        total_time = now - created
        response_time = (ack_time - created) if ack_time else None
        dispatch_delay = (dispatch_time - created) if dispatch_time else None
        on_scene_time = None
        for entry in inc.timeline:
            if entry["status"] == "on_scene":
                on_scene_time = entry["time"] - created
                break

        sla_target = self.SLA_TARGETS.get(inc.emergency_level, 600)
        sla_met = (response_time is not None and response_time <= sla_target) if response_time else False

        metrics = {
            "incident_id": incident_id,
            "category": inc.category,
            "emergency_level": inc.emergency_level,
            "total_time_sec": round(total_time, 1),
            "response_time_sec": round(response_time, 1) if response_time else None,
            "dispatch_delay_sec": round(dispatch_delay, 1) if dispatch_delay else None,
            "on_scene_time_sec": round(on_scene_time, 1) if on_scene_time else None,
            "sla_target_sec": sla_target,
            "sla_met": sla_met,
            "crowd_reports": inc.crowd_reports,
            "assigned_officers": len(inc.assigned_officers),
            "source": inc.source,
            "resolved_at": now,
        }

        self._response_times.append(metrics)
        return metrics

    def get_summary(self, hours: float = 24.0) -> dict:
        """Get response metrics summary for the last N hours."""
        cutoff = time.time() - (hours * 3600)
        recent = [m for m in self._response_times if m["resolved_at"] > cutoff]

        if not recent:
            return {
                "period_hours": hours,
                "total_resolved": 0,
                "avg_response_time": None,
                "avg_total_time": None,
                "sla_compliance": None,
                "by_category": {},
                "by_level": {},
            }

        response_times = [m["response_time_sec"] for m in recent if m["response_time_sec"] is not None]
        total_times = [m["total_time_sec"] for m in recent]

        return {
            "period_hours": hours,
            "total_resolved": len(recent),
            "avg_response_time": round(sum(response_times) / len(response_times), 1) if response_times else None,
            "avg_total_time": round(sum(total_times) / len(total_times), 1),
            "sla_compliance": round(
                sum(1 for m in recent if m["sla_met"]) / len(recent) * 100, 1
            ),
            "by_category": {
                cat: {
                    "count": sum(1 for m in recent if m["category"] == cat),
                    "avg_response": round(
                        sum(m["response_time_sec"] for m in recent
                            if m["category"] == cat and m["response_time_sec"])
                        / max(1, sum(1 for m in recent
                                      if m["category"] == cat and m["response_time_sec"])),
                        1,
                    ),
                }
                for cat in set(m["category"] for m in recent)
            },
            "by_level": {
                str(lvl): {
                    "count": sum(1 for m in recent if m["emergency_level"] == lvl),
                    "sla_met": sum(1 for m in recent
                                   if m["emergency_level"] == lvl and m["sla_met"]),
                }
                for lvl in set(m["emergency_level"] for m in recent)
            },
        }

    def get_live_dashboard(self) -> dict:
        """Real-time dashboard metrics for operator view."""
        incidents = self.store.list_incidents()
        officers = self.store.list_officers()

        active = [i for i in incidents if i.status not in ("resolved", "cancelled")]
        resolved_today = [
            i for i in incidents
            if i.status == "resolved"
            and (time.time() - i.updated_at) < 86400
        ]

        # Average wait time for active incidents
        now = time.time()
        wait_times = [(now - i.created_at) for i in active]
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0

        return {
            "active_incidents": len(active),
            "critical": sum(1 for i in active if i.emergency_level == 4),
            "high": sum(1 for i in active if i.emergency_level == 3),
            "medium": sum(1 for i in active if i.emergency_level == 2),
            "low": sum(1 for i in active if i.emergency_level == 1),
            "resolved_today": len(resolved_today),
            "total_officers": len(officers),
            "available_officers": sum(1 for o in officers if o.status == "available"),
            "dispatched_officers": sum(1 for o in officers if o.status != "available"),
            "avg_wait_time_sec": round(avg_wait, 1),
            "incidents_by_category": dict(Counter(i.category for i in active)),
        }


# ── Resolution Workflow ─────────────────────────────────────────────────────

@dataclass
class ResolutionProof:
    incident_id: str
    officer_id: str
    notes: str
    photos: list[str] = field(default_factory=list)  # base64 or file paths
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self):
        return {
            "incident_id": self.incident_id,
            "officer_id": self.officer_id,
            "notes": self.notes,
            "photos": self.photos,
            "timestamp": self.timestamp,
        }


class ResolutionManager:
    """Handles incident resolution with proof upload and verification.

    Workflow:
    1. Officer arrives on scene → marks "on_scene"
    2. Officer takes photos / writes notes → uploads proof
    3. Officer marks "resolved" with proof attached
    4. Proof stored for audit trail and training data
    """

    def __init__(self, store, metrics: ResponseMetrics | None = None,
                 proof_dir: str = "output/resolution_proof"):
        self.store = store
        self.metrics = metrics
        self.proof_dir = Path(proof_dir)
        self.proof_dir.mkdir(parents=True, exist_ok=True)
        self._proofs: dict[str, list[ResolutionProof]] = defaultdict(list)

    def upload_proof(self, incident_id: str, officer_id: str,
                     notes: str = "", photos: list[str] | None = None) -> dict:
        """Officer uploads proof (photos + notes) for an incident."""
        inc = self.store.get_incident(incident_id)
        if not inc:
            return {"error": "Incident not found"}
        off = self.store.get_officer(officer_id)
        if not off:
            return {"error": "Officer not found"}

        proof = ResolutionProof(
            incident_id=incident_id,
            officer_id=officer_id,
            notes=notes,
            photos=photos or [],
        )
        self._proofs[incident_id].append(proof)

        # Save proof to disk
        proof_file = self.proof_dir / f"{incident_id}_{officer_id}_{int(proof.timestamp)}.json"
        proof_file.write_text(json.dumps(proof.to_dict(), indent=2), encoding="utf-8")

        # Add to incident timeline
        inc.timeline.append({
            "status": "proof_uploaded",
            "time": proof.timestamp,
            "note": f"Officer {off.name} uploaded proof: {notes[:100]}",
        })
        self.store._save_incidents()

        return {"proof": proof.to_dict(), "message": "Proof uploaded successfully"}

    def resolve_incident(self, incident_id: str, officer_id: str,
                         resolution_notes: str = "",
                         photos: list[str] | None = None) -> dict:
        """Officer marks incident as resolved with proof."""
        inc = self.store.get_incident(incident_id)
        if not inc:
            return {"error": "Incident not found"}
        off = self.store.get_officer(officer_id)
        if not off:
            return {"error": "Officer not found"}

        # Upload final proof if provided
        if photos or resolution_notes:
            self.upload_proof(incident_id, officer_id, resolution_notes, photos)

        # Update incident status
        self.store.update_incident(
            incident_id,
            status="resolved",
            resolution_notes=resolution_notes,
            note=f"Resolved by {off.name}: {resolution_notes[:100]}",
        )

        # Free the officer
        self.store.update_officer(
            officer_id, status="available", current_incident=None,
        )

        # Record metrics
        metrics = None
        if self.metrics:
            metrics = self.metrics.record_resolution(incident_id)

        inc = self.store.get_incident(incident_id)
        result = {
            "incident": inc.to_dict(),
            "officer": self.store.get_officer(officer_id).to_dict(),
        }
        if metrics:
            result["response_metrics"] = metrics

        return result

    def get_proof(self, incident_id: str) -> list[dict]:
        """Get all proof for an incident."""
        return [p.to_dict() for p in self._proofs.get(incident_id, [])]

    def get_unresolved_summary(self) -> dict:
        """Summary of unresolved incidents for operator dashboard."""
        active = [i for i in self.store.list_incidents()
                  if i.status not in ("resolved", "cancelled")]

        by_age = {"under_5m": 0, "5_to_30m": 0, "30m_to_1h": 0, "over_1h": 0}
        now = time.time()

        for inc in active:
            age_min = (now - inc.created_at) / 60
            if age_min < 5:
                by_age["under_5m"] += 1
            elif age_min < 30:
                by_age["5_to_30m"] += 1
            elif age_min < 60:
                by_age["30m_to_1h"] += 1
            else:
                by_age["over_1h"] += 1

        return {
            "total_active": len(active),
            "by_age": by_age,
            "escalated": sum(1 for i in active if i.emergency_level >= 3),
            "unassigned": sum(1 for i in active if not i.assigned_officers),
        }


# ── Unified City Safety Engine ──────────────────────────────────────────────

class CitySafetyEngine:
    """Unified engine combining all city safety components.

    This is the main entry point that wires together:
    - Deduplication
    - Notifications
    - GPS tracking
    - Response metrics
    - Resolution workflow
    """

    def __init__(self, store, hub=None, dispatch_engine=None,
                 sms_gateway_url: str | None = None):
        self.store = store
        self.hub = hub
        self.dispatch_engine = dispatch_engine

        self.deduplicator = IncidentDeduplicator(store, dispatch_engine)
        self.notifications = NotificationManager(hub, sms_gateway_url)
        self.gps_tracker = GPSTracker(store, hub)
        self.metrics = ResponseMetrics(store)
        self.resolver = ResolutionManager(store, self.metrics)

    def report_incident(self, category: str, lat: float, lng: float,
                        emergency_level: int = 1, description: str = "",
                        reporter_phone: str = "", reporter_name: str = "",
                        source: str = "public") -> dict:
        """Public reports an incident — dedup, dispatch, notify."""
        # Step 1: Dedup check
        result = self.deduplicator.check_and_merge(
            category=category, lat=lat, lng=lng,
            description=description, reporter_phone=reporter_phone,
            reporter_name=reporter_name, source=source,
            emergency_level=emergency_level,
        )

        if result.is_duplicate:
            # Notify operators about crowd confirmation
            self.notifications.notify_operators(
                title=f"Crowd Report: {category.upper()}",
                body=f"Another person confirmed the {category} incident. "
                     f"Total reports: check incident {result.merged_into}",
                priority="medium",
                data={"incident_id": result.merged_into, "crowd_report": True},
            )
            return {
                "duplicate": True,
                "merged_into": result.merged_into,
                "message": "Your report has been added to an existing incident",
            }

        # Step 2: New incident created, dispatch if needed
        inc_data = result.new_incident

        # Step 3: Notify nearby officers
        if inc_data and emergency_level >= 2:
            inc_id = inc_data["id"]
            assigned = inc_data.get("dispatched_officers", [])
            for off_data in assigned:
                self.notifications.notify_officer(
                    officer_id=off_data["id"],
                    title=f"🚨 {emergency_level_name(emergency_level)}: {category.upper()}",
                    body=f"Incident at {description or 'reported location'}. "
                         f"Respond immediately.",
                    priority="critical" if emergency_level >= 4 else "high",
                    data={"incident_id": inc_id},
                )

            # Notify operators
            self.notifications.notify_operators(
                title=f"New {category.upper()} incident dispatched",
                body=f"{len(assigned)} officers dispatched to {description or 'location'}",
                priority="high" if emergency_level >= 3 else "medium",
                data={"incident_id": inc_id, "officers_dispatched": len(assigned)},
            )

        return {
            "duplicate": False,
            "incident": inc_data,
            "message": "Incident reported and dispatched" if inc_data else "Incident reported",
        }

    def camera_alert(self, alert: dict, camera_id: str = "") -> dict | None:
        """Camera AI detects an incident — dedup, create, dispatch, notify."""
        rule = alert.get("rule", "unknown")
        severity = alert.get("severity", "yellow")
        message = alert.get("message", "")

        # Map rule to category
        from .camera_bridge import RULE_TO_CATEGORY, SEVERITY_TO_LEVEL
        category = RULE_TO_CATEGORY.get(rule, "other")
        level = SEVERITY_TO_LEVEL.get(severity, 2)

        # Get camera position
        camera_positions = {
            "CAM-01": (28.6139, 77.2090),
            "CAM-02": (28.6150, 77.2100),
            "CAM-03": (28.6120, 77.2080),
            "CAM-04": (28.6160, 77.2110),
            "CAM-05": (28.6110, 77.2070),
            "CAM-06": (28.6180, 77.2130),
        }
        lat, lng = camera_positions.get(camera_id, (28.6139, 77.2090))

        # Use dedup to handle camera + public overlap
        result = self.report_incident(
            category=category,
            lat=lat,
            lng=lng,
            emergency_level=level,
            description=f"[{camera_id}] {rule.upper()}: {message}",
            reporter_name="BHAIRAV AI",
            source="camera",
        )
        return result

    def officer_update_gps(self, officer_id: str, lat: float, lng: float,
                           speed: float = 0.0, heading: float = 0.0) -> dict:
        """Officer sends GPS heartbeat."""
        coord = self.gps_tracker.update_position(officer_id, lat, lng, speed, heading)
        if coord:
            return {"ok": True, "position": {"lat": lat, "lng": lng}}
        return {"error": "Officer not found"}

    def resolve(self, incident_id: str, officer_id: str,
                notes: str = "", photos: list[str] | None = None) -> dict:
        """Officer resolves an incident with proof."""
        return self.resolver.resolve_incident(incident_id, officer_id, notes, photos)

    def get_operator_dashboard(self) -> dict:
        """Full dashboard data for operator view."""
        return {
            "incidents": {
                "active": [i.to_dict() for i in self.store.list_incidents()
                           if i.status not in ("resolved", "cancelled")],
                "recent_resolved": [i.to_dict() for i in self.store.list_incidents(status="resolved")[:10]],
            },
            "officers": {
                "all": [o.to_dict() for o in self.store.list_officers()],
                "gps_positions": self.gps_tracker.get_all_positions(),
            },
            "metrics": self.metrics.get_live_dashboard(),
            "notifications": self.notifications.get_recent(limit=20),
            "unresolved_summary": self.resolver.get_unresolved_summary(),
            "sla_summary": self.metrics.get_summary(hours=24),
        }

    def get_officer_dashboard(self, officer_id: str) -> dict:
        """Dashboard data for a specific officer."""
        off = self.store.get_officer(officer_id)
        if not off:
            return {"error": "Officer not found"}

        assigned = [i for i in self.store.list_incidents()
                    if officer_id in i.assigned_officers
                    and i.status not in ("resolved", "cancelled")]

        return {
            "officer": off.to_dict(),
            "assigned_incidents": [i.to_dict() for i in assigned],
            "position": self.gps_tracker.get_position(officer_id),
            "trajectory": self.gps_tracker.get_trajectory(officer_id),
        }

    def stats(self) -> dict:
        return {
            "dedup": self.deduplicator.stats(),
            "notifications": self.notifications.stats(),
            "gps": self.gps_tracker.stats(),
            "store": self.store.get_stats(),
        }


def emergency_level_name(level: int) -> str:
    return {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}.get(level, "UNKNOWN")
