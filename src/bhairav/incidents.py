"""BHAIRAV City Safety - Incident Reporting & Dispatch System."""
from __future__ import annotations

import json, logging, math, time, uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

log = logging.getLogger("bhairav.incidents")


class IncidentCategory(str, Enum):
    MEDICAL = "medical"
    FIRE = "fire"
    CRIME = "crime"
    ROAD_ACCIDENT = "road_accident"
    DISASTER = "disaster"
    MISSING_PERSON = "missing_person"
    OTHER = "other"


class EmergencyLevel(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class IncidentStatus(str, Enum):
    REPORTED = "reported"
    VERIFIED = "verified"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class OfficerStatus(str, Enum):
    AVAILABLE = "available"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    OFF_DUTY = "off_duty"


@dataclass
class Incident:
    id: str
    category: str
    emergency_level: int
    location_lat: float
    location_lng: float
    location_name: str
    description: str
    reporter_phone: str
    reporter_name: str
    status: str
    source: str
    created_at: float
    updated_at: float
    assigned_officers: list = field(default_factory=list)
    resolution_notes: str = ""
    crowd_reports: int = 1
    ai_verified: bool = False
    timeline: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id, "category": self.category,
            "emergency_level": self.emergency_level,
            "location": {"lat": self.location_lat, "lng": self.location_lng},
            "location_name": self.location_name, "description": self.description,
            "reporter_phone": self.reporter_phone, "reporter_name": self.reporter_name,
            "status": self.status, "source": self.source,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "assigned_officers": self.assigned_officers,
            "resolution_notes": self.resolution_notes,
            "crowd_reports": self.crowd_reports, "ai_verified": self.ai_verified,
            "timeline": self.timeline,
        }


@dataclass
class Officer:
    id: str
    name: str
    role: str
    phone: str
    status: str
    location_lat: float
    location_lng: float
    current_incident: str = None
    specialty: list = field(default_factory=list)
    last_seen: float = 0.0

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "role": self.role,
            "phone": self.phone, "status": self.status,
            "location": {"lat": self.location_lat, "lng": self.location_lng},
            "current_incident": self.current_incident,
            "specialty": self.specialty, "last_seen": self.last_seen,
        }

class IncidentStore:
    def __init__(self, path="output/incidents"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._inc_file = self.path / "incidents.jsonl"
        self._off_file = self.path / "officers.jsonl"
        self._incidents = {}
        self._officers = {}
        self._load()

    def _load(self):
        for f, cls, store in [(self._inc_file, Incident, self._incidents),
                               (self._off_file, Officer, self._officers)]:
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if cls is Incident:
                        obj = Incident(
                            id=d["id"], category=d["category"],
                            emergency_level=d["emergency_level"],
                            location_lat=d["location_lat"], location_lng=d["location_lng"],
                            location_name=d["location_name"], description=d["description"],
                            reporter_phone=d.get("reporter_phone",""),
                            reporter_name=d.get("reporter_name",""),
                            status=d["status"], source=d["source"],
                            created_at=d["created_at"], updated_at=d["updated_at"],
                            assigned_officers=d.get("assigned_officers",[]),
                            resolution_notes=d.get("resolution_notes",""),
                            crowd_reports=d.get("crowd_reports",1),
                            ai_verified=d.get("ai_verified",False),
                            timeline=d.get("timeline",[]),
                        )
                    else:
                        obj = Officer(
                            id=d["id"], name=d["name"], role=d["role"],
                            phone=d.get("phone",""), status=d["status"],
                            location_lat=d["location_lat"], location_lng=d["location_lng"],
                            current_incident=d.get("current_incident"),
                            specialty=d.get("specialty",[]),
                            last_seen=d.get("last_seen",0),
                        )
                    store[obj.id] = obj
                except (json.JSONDecodeError, KeyError):
                    continue

    def _save_incidents(self):
        with self._inc_file.open("w", encoding="utf-8") as f:
            for inc in self._incidents.values():
                f.write(json.dumps({
                    "id": inc.id, "category": inc.category,
                    "emergency_level": inc.emergency_level,
                    "location_lat": inc.location_lat, "location_lng": inc.location_lng,
                    "location_name": inc.location_name, "description": inc.description,
                    "reporter_phone": inc.reporter_phone, "reporter_name": inc.reporter_name,
                    "status": inc.status, "source": inc.source,
                    "created_at": inc.created_at, "updated_at": inc.updated_at,
                    "assigned_officers": inc.assigned_officers,
                    "resolution_notes": inc.resolution_notes,
                    "crowd_reports": inc.crowd_reports, "ai_verified": inc.ai_verified,
                    "timeline": inc.timeline,
                }, ensure_ascii=False) + "\n")

    def _save_officers(self):
        with self._off_file.open("w", encoding="utf-8") as f:
            for off in self._officers.values():
                f.write(json.dumps({
                    "id": off.id, "name": off.name, "role": off.role,
                    "phone": off.phone, "status": off.status,
                    "location_lat": off.location_lat, "location_lng": off.location_lng,
                    "current_incident": off.current_incident,
                    "specialty": off.specialty, "last_seen": off.last_seen,
                }, ensure_ascii=False) + "\n")

    def create_incident(self, category, emergency_level, lat, lng, location_name,
                        description, reporter_phone="", reporter_name="", source="public"):
        now = time.time()
        inc = Incident(
            id=uuid.uuid4().hex[:12], category=category,
            emergency_level=emergency_level,
            location_lat=lat, location_lng=lng,
            location_name=location_name, description=description,
            reporter_phone=reporter_phone, reporter_name=reporter_name,
            status=IncidentStatus.REPORTED.value, source=source,
            created_at=now, updated_at=now,
            timeline=[{"status": "reported", "time": now,
                       "note": f"Reported by {reporter_name or chr(39)+chr(39)} via {source}"}],
        )
        self._incidents[inc.id] = inc
        self._save_incidents()
        return inc

    def get_incident(self, iid):
        return self._incidents.get(iid)

    def list_incidents(self, status=None, category=None, limit=100):
        r = list(self._incidents.values())
        if status:
            r = [i for i in r if i.status == status]
        if category:
            r = [i for i in r if i.category == category]
        r.sort(key=lambda i: i.created_at, reverse=True)
        return r[:limit]

    def update_incident(self, iid, **kw):
        inc = self._incidents.get(iid)
        if not inc:
            return None
        now = time.time()
        for k, v in kw.items():
            if hasattr(inc, k):
                setattr(inc, k, v)
        inc.updated_at = now
        if "status" in kw:
            inc.timeline.append({"status": kw["status"], "time": now, "note": kw.get("note", "")})
        self._save_incidents()
        return inc

    def add_crowd_report(self, iid):
        inc = self._incidents.get(iid)
        if not inc:
            return None
        inc.crowd_reports += 1
        inc.updated_at = time.time()
        if inc.crowd_reports >= 2 and inc.status == IncidentStatus.REPORTED.value:
            inc.status = IncidentStatus.VERIFIED.value
            inc.timeline.append({"status": "verified", "time": time.time(),
                                "note": f"Auto-verified: {inc.crowd_reports} crowd reports"})
        self._save_incidents()
        return inc

    def find_nearby_incidents(self, lat, lng, radius_m=100, time_window_sec=1800):
        now = time.time()
        results = []
        for inc in self._incidents.values():
            if now - inc.created_at > time_window_sec:
                continue
            if haversine_distance(lat, lng, inc.location_lat, inc.location_lng) <= radius_m:
                results.append(inc)
        return results

    def register_officer(self, name, role, phone, lat, lng, specialty=None):
        off = Officer(
            id=uuid.uuid4().hex[:8], name=name, role=role, phone=phone,
            status=OfficerStatus.AVAILABLE.value,
            location_lat=lat, location_lng=lng,
            specialty=specialty or [], last_seen=time.time(),
        )
        self._officers[off.id] = off
        self._save_officers()
        return off

    def get_officer(self, oid):
        return self._officers.get(oid)

    def list_officers(self, status=None):
        r = list(self._officers.values())
        if status:
            r = [o for o in r if o.status == status]
        return r

    def update_officer(self, oid, **kw):
        off = self._officers.get(oid)
        if not off:
            return None
        for k, v in kw.items():
            if hasattr(off, k):
                setattr(off, k, v)
        off.last_seen = time.time()
        self._save_officers()
        return off

    def get_stats(self):
        incs = list(self._incidents.values())
        offs = list(self._officers.values())
        return {
            "total_incidents": len(incs),
            "by_status": dict(Counter(i.status for i in incs)),
            "by_category": dict(Counter(i.category for i in incs)),
            "by_level": dict(Counter(str(i.emergency_level) for i in incs)),
            "active_incidents": len([i for i in incs if i.status not in ("resolved","cancelled")]),
            "total_officers": len(offs),
            "available_officers": len([o for o in offs if o.status == OfficerStatus.AVAILABLE.value]),
            "dispatched_officers": len([o for o in offs if o.status != OfficerStatus.AVAILABLE.value]),
        }

CATEGORY_ROLE_MAP = {
    "medical": ["medical", "rescue"], "fire": ["fire", "rescue"],
    "crime": ["police"], "road_accident": ["police", "medical", "rescue"],
    "disaster": ["police", "medical", "fire", "rescue"],
    "missing_person": ["police"], "other": ["police"],
}
LEVEL_OFFICER_COUNT = {1: 1, 2: 2, 3: 3, 4: 5}
LEVEL_DISPATCH_RADIUS = {1: 5.0, 2: 10.0, 3: 15.0, 4: 25.0}


# Escalation tiers: if no response after timeout_sec, auto-escalate.
ESCALATION_TIERS = [
    {"timeout_sec": 30, "action": "widen_radius", "radius_multiplier": 1.5},
    {"timeout_sec": 60, "action": "notify_all", "radius_multiplier": 2.0},
    {"timeout_sec": 120, "action": "escalate_level", "level_bump": 1},
]

class DispatchEngine:
    def __init__(self, store, on_dispatch=None, on_escalation=None):
        self.store = store
        self.on_dispatch = on_dispatch   # callback(incident_dict, assigned_officers)
        self.on_escalation = on_escalation  # callback(incident_dict, escalation_info)
        self._pending_escalations = {}  # incident_id -> [(tier_index, timer_handle)]
        self._lock = __import__("threading").Lock()

    def dispatch(self, incident):
        """Multi-tier dispatch: specialty match → role match → any available."""
        required = CATEGORY_ROLE_MAP.get(incident.category, ["police"])
        num = LEVEL_OFFICER_COUNT.get(incident.emergency_level, 2)
        radius = LEVEL_DISPATCH_RADIUS.get(incident.emergency_level, 10.0)

        # Tier 1: specialty match (officers whose specialty matches category)
        tier1 = []
        for off in self.store.list_officers(status=OfficerStatus.AVAILABLE.value):
            d = haversine_distance(incident.location_lat, incident.location_lng,
                                   off.location_lat, off.location_lng) / 1000.0
            if d <= radius and off.role in required:
                tier1.append((d, off, "role_match"))

        # Tier 2: specialty overlap (officer specialty intersects category needs)
        tier2 = []
        needed_specialties = set(required)
        for off in self.store.list_officers(status=OfficerStatus.AVAILABLE.value):
            if any(o[1].id == off.id for o in tier1):
                continue
            d = haversine_distance(incident.location_lat, incident.location_lng,
                                   off.location_lat, off.location_lng) / 1000.0
            if d <= radius and set(off.specialty or []) & needed_specialties:
                tier2.append((d, off, "specialty_match"))

        # Tier 3: any available within radius
        tier3 = []
        seen_ids = {o[1].id for o in tier1 + tier2}
        for off in self.store.list_officers(status=OfficerStatus.AVAILABLE.value):
            if off.id in seen_ids:
                continue
            d = haversine_distance(incident.location_lat, incident.location_lng,
                                   off.location_lat, off.location_lng) / 1000.0
            if d <= radius:
                tier3.append((d, off, "any_available"))

        # Merge tiers, sorted by distance within each tier
        all_candidates = []
        for tier in [tier1, tier2, tier3]:
            tier.sort(key=lambda x: x[0])
            all_candidates.extend(tier)

        # Assign officers
        assigned = []
        for d, off, match_type in all_candidates[:num]:
            off.status = OfficerStatus.DISPATCHED.value
            off.current_incident = incident.id
            incident.assigned_officers.append(off.id)
            assigned.append(off)

        if assigned:
            incident.status = IncidentStatus.DISPATCHED.value
            names = ", ".join(o.name for o in assigned)
            incident.timeline.append({
                "status": "dispatched", "time": time.time(),
                "note": f"Dispatched {len(assigned)} officer(s): {names}",
            })
        else:
            # No officers available — mark for escalation
            incident.timeline.append({
                "status": "no_response", "time": time.time(),
                "note": "No available officers within radius",
            })

        self.store._save_incidents()
        self.store._save_officers()

        # Notify callback
        if self.on_dispatch and assigned:
            try:
                self.on_dispatch(incident.to_dict(), [o.to_dict() for o in assigned])
            except Exception:
                pass

        # Schedule escalation checks if no officers found or severity is high
        if not assigned or incident.emergency_level >= 3:
            self._schedule_escalation(incident)

        return assigned

    def _schedule_escalation(self, incident):
        """Schedule escalation tiers for an unresponded incident."""
        import threading
        with self._lock:
            # Cancel any existing escalation timers for this incident
            if incident.id in self._pending_escalations:
                for _, handle in self._pending_escalations[incident.id]:
                    try:
                        handle.cancel()
                    except Exception:
                        pass
            self._pending_escalations[incident.id] = []

        for i, tier in enumerate(ESCALATION_TIERS):
            def make_check(idx=i, t=tier):
                def check():
                    self._apply_escalation(incident, idx, t)
                return check

            timer = threading.Timer(tier["timeout_sec"], make_check())
            timer.daemon = True
            timer.start()
            with self._lock:
                self._pending_escalations.setdefault(incident.id, []).append((i, timer))

    def _apply_escalation(self, incident, tier_idx, tier):
        """Apply an escalation tier to an incident."""
        with self._lock:
            pending = self._pending_escalations.get(incident.id, [])
            # Only run if this tier hasn't been superseded
            active_indices = {idx for idx, _ in pending}
            if tier_idx not in active_indices:
                return

        # Re-fetch incident from store (may have been updated)
        inc = self.store.get_incident(incident.id)
        if not inc or inc.status in ("resolved", "cancelled"):
            return

        action = tier["action"]
        escalation_info = {"tier": tier_idx, "action": action, "time": time.time()}

        if action == "widen_radius" or action == "notify_all":
            # Try dispatching again with a wider radius
            multiplier = tier.get("radius_multiplier", 1.5)
            required = CATEGORY_ROLE_MAP.get(inc.category, ["police"])
            num = LEVEL_OFFICER_COUNT.get(inc.emergency_level, 2) - len(inc.assigned_officers)
            if num <= 0:
                return
            base_radius = LEVEL_DISPATCH_RADIUS.get(inc.emergency_level, 10.0)
            wide_radius = base_radius * multiplier
            additional = []
            assigned_ids = set(inc.assigned_officers)
            for off in self.store.list_officers(status=OfficerStatus.AVAILABLE.value):
                if off.id in assigned_ids:
                    continue
                d = haversine_distance(inc.location_lat, inc.location_lng,
                                       off.location_lat, off.location_lng) / 1000.0
                if d <= wide_radius and (off.role in required or off.specialty):
                    additional.append((d, off))
            additional.sort(key=lambda x: x[0])
            for d, off in additional[:num]:
                off.status = OfficerStatus.DISPATCHED.value
                off.current_incident = inc.id
                inc.assigned_officers.append(off.id)
                additional = additional[1:]  # consume
                num -= 1
                if num <= 0:
                    break

            if len(inc.assigned_officers) > len(incident.assigned_officers):
                new_count = len(inc.assigned_officers) - len(incident.assigned_officers)
                inc.timeline.append({
                    "status": "escalated", "time": time.time(),
                    "note": f"Tier {tier_idx+1}: widened radius to {wide_radius:.1f}km, dispatched {new_count} additional officer(s)",
                })
                self.store._save_incidents()
                self.store._save_officers()
                escalation_info["officers_added"] = new_count

        elif action == "escalate_level":
            # Bump emergency level if possible
            if inc.emergency_level < 4:
                inc.emergency_level = min(inc.emergency_level + tier.get("level_bump", 1), 4)
                inc.timeline.append({
                    "status": "escalated", "time": time.time(),
                    "note": f"Auto-escalated to level {inc.emergency_level}: no response after {tier['timeout_sec']}s",
                })
                self.store._save_incidents()
                escalation_info["new_level"] = inc.emergency_level
                # Try dispatch again at new level
                self.dispatch(inc)

        # Notify callback
        if self.on_escalation and escalation_info.get("officers_added") or escalation_info.get("new_level"):
            try:
                self.on_escalation(inc.to_dict(), escalation_info)
            except Exception:
                pass

        # Clean up completed escalation timers
        if tier_idx == len(ESCALATION_TIERS) - 1:
            with self._lock:
                self._pending_escalations.pop(incident.id, None)

    def accept_incident(self, officer_id, incident_id):
        """Officer accepts/acknowledges an incident."""
        off = self.store.get_officer(officer_id)
        inc = self.store.get_incident(incident_id)
        if not off or not inc:
            return None
        off.status = OfficerStatus.EN_ROUTE.value
        off.current_incident = incident_id
        inc.timeline.append({
            "status": "acknowledged", "time": time.time(),
            "note": f"{off.name} acknowledged the dispatch",
        })
        self.store._save_incidents()
        self.store._save_officers()
        return inc

    def find_nearest_officers(self, lat, lng, role=None, radius_km=10.0, limit=5):
        results = []
        for off in self.store.list_officers(status=OfficerStatus.AVAILABLE.value):
            if role and off.role != role:
                continue
            d = haversine_distance(lat, lng, off.location_lat, off.location_lng) / 1000.0
            if d <= radius_km:
                results.append((d, off))
        results.sort(key=lambda x: x[0])
        return results[:limit]

    def get_escalation_status(self, incident_id):
        """Check escalation status for an incident."""
        inc = self.store.get_incident(incident_id)
        if not inc:
            return None
        age = time.time() - inc.created_at
        active_tiers = []
        with self._lock:
            pending = self._pending_escalations.get(incident_id, [])
            for idx, _ in pending:
                if idx < len(ESCALATION_TIERS):
                    active_tiers.append({
                        "tier": idx + 1,
                        "action": ESCALATION_TIERS[idx]["action"],
                        "trigger_in": max(0, ESCALATION_TIERS[idx]["timeout_sec"] - age),
                    })
        return {
            "incident_id": incident_id,
            "age_sec": round(age, 1),
            "status": inc.status,
            "emergency_level": inc.emergency_level,
            "assigned_count": len(inc.assigned_officers),
            "active_escalations": active_tiers,
        }


def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def seed_demo_data(store):
    for name, role, phone, lat, lng, spec in [
        ("Raj Kumar", "police", "+91-9876543210", 28.6139, 77.2090, ["patrol"]),
        ("Priya Singh", "medical", "+91-9876543211", 28.6150, 77.2100, ["emergency"]),
        ("Amit Verma", "fire", "+91-9876543212", 28.6120, 77.2080, ["hazmat"]),
        ("Sunita Devi", "police", "+91-9876543213", 28.6160, 77.2110, ["investigation"]),
        ("Vikram Patel", "rescue", "+91-9876543214", 28.6110, 77.2070, ["medical","fire"]),
        ("Anita Sharma", "medical", "+91-9876543215", 28.6170, 77.2120, ["trauma"]),
        ("Deepak Joshi", "police", "+91-9876543216", 28.6100, 77.2060, ["cyber"]),
        ("Meena Kumari", "fire", "+91-9876543217", 28.6180, 77.2130, ["rescue"]),
        ("Ravi Shankar", "police", "+91-9876543218", 28.6090, 77.2050, ["patrol"]),
        ("Kavita Reddy", "rescue", "+91-9876543219", 28.6190, 77.2140, ["medical"]),
    ]:
        store.register_officer(name, role, phone, lat, lng, spec)

    for cat, level, lat, lng, name, desc, reporter, source in [
        ("medical", 4, 28.6135, 77.2095, "Connaught Place", "Person collapsed, possible heart attack", "Rahul", "public"),
        ("fire", 3, 28.6155, 77.2105, "Chandni Chowk market", "Smoke from shop building", "Amit", "public"),
        ("crime", 2, 28.6125, 77.2085, "Karol Bagh", "Group fighting near bus stop", "Neha", "public"),
        ("road_accident", 3, 28.6165, 77.2115, "ITO crossing", "Car hit pedestrian", "Vikram", "camera"),
        ("medical", 2, 28.6115, 77.2075, "Lajpat Nagar", "Elderly person needs attention", "Sunita", "sms"),
    ]:
        inc = store.create_incident(cat, level, lat, lng, name, desc,
                                     reporter_name=reporter, source=source)
        if level >= 2:
            DispatchEngine(store).dispatch(inc)

    return {"officers": 10, "incidents": 5}


# ── FastAPI Routes ───────────────────────────────────────────────

def create_incident_routes(app, store: IncidentStore, dispatch_engine: DispatchEngine):
    """Add incident API routes to the FastAPI app."""

    @app.post("/api/incidents")
    def report_incident(body: dict = {}):
        """Public endpoint: report a new incident."""
        cat = body.get("category", "other")
        level = int(body.get("emergency_level", 1))
        lat = float(body.get("lat", 0))
        lng = float(body.get("lng", 0))
        name = body.get("location_name", "Unknown")
        desc = body.get("description", "")
        phone = body.get("reporter_phone", "")
        rname = body.get("reporter_name", "")
        source = body.get("source", "public")

        inc = store.create_incident(cat, level, lat, lng, name, desc,
                                     reporter_phone=phone, reporter_name=rname, source=source)

        # Auto-dispatch based on severity
        if level >= 2:
            assigned = dispatch_engine.dispatch(inc)
        else:
            assigned = []

        return {
            "incident": inc.to_dict(),
            "dispatched_officers": [o.to_dict() for o in assigned],
            "message": f"Incident {inc.id} reported and dispatched" if assigned else f"Incident {inc.id} reported",
        }

    @app.get("/api/incidents")
    def list_incidents(status: str = None, category: str = None, limit: int = 100):
        """List incidents with optional filters."""
        incidents = store.list_incidents(status=status, category=category, limit=limit)
        return {"incidents": [i.to_dict() for i in incidents], "total": len(incidents)}

    @app.get("/api/incidents/{incident_id}")
    def get_incident(incident_id: str):
        """Get a single incident by ID."""
        inc = store.get_incident(incident_id)
        if not inc:
            return {"error": "Incident not found"}, 404
        return {"incident": inc.to_dict()}

    @app.post("/api/incidents/{incident_id}/status")
    def update_incident_status(incident_id: str, body: dict = {}):
        """Update incident status (officer or operator)."""
        new_status = body.get("status", "")
        note = body.get("note", "")
        inc = store.update_incident(incident_id, status=new_status, note=note)
        if not inc:
            return {"error": "Incident not found"}, 404
        return {"incident": inc.to_dict()}

    @app.post("/api/incidents/{incident_id}/crowd")
    def crowd_report(incident_id: str):
        """Another person reports the same incident."""
        inc = store.add_crowd_report(incident_id)
        if not inc:
            return {"error": "Incident not found"}, 404
        return {"incident": inc.to_dict(), "crowd_reports": inc.crowd_reports}

    @app.get("/api/incidents/nearby")
    def nearby_incidents(lat: float, lng: float, radius: float = 100):
        """Find incidents near a location."""
        incs = store.find_nearby_incidents(lat, lng, radius_m=radius)
        return {"incidents": [i.to_dict() for i in incs], "total": len(incs)}

    @app.get("/api/officers")
    def list_officers(status: str = None):
        """List officers."""
        officers = store.list_officers(status=status)
        return {"officers": [o.to_dict() for o in officers], "total": len(officers)}

    @app.get("/api/officers/{officer_id}")
    def get_officer(officer_id: str):
        """Get officer by ID."""
        off = store.get_officer(officer_id)
        if not off:
            return {"error": "Officer not found"}, 404
        return {"officer": off.to_dict()}

    @app.post("/api/officers/{officer_id}/status")
    def update_officer_status(officer_id: str, body: dict = {}):
        """Update officer status."""
        new_status = body.get("status", "")
        off = store.update_officer(officer_id, status=new_status)
        if not off:
            return {"error": "Officer not found"}, 404
        return {"officer": off.to_dict()}

    @app.post("/api/officers/{officer_id}/location")
    def update_officer_location(officer_id: str, body: dict = {}):
        """Update officer GPS location."""
        lat = float(body.get("lat", 0))
        lng = float(body.get("lng", 0))
        off = store.update_officer(officer_id, location_lat=lat, location_lng=lng)
        if not off:
            return {"error": "Officer not found"}, 404
        return {"officer": off.to_dict()}

    @app.get("/api/incidents/stats")
    def incident_stats():
        """Get incident statistics."""
        return store.get_stats()

    @app.post("/api/officers/{officer_id}/accept/{incident_id}")
    def accept_incident(officer_id: str, incident_id: str):
        """Officer accepts/dispatches to an incident."""
        inc = dispatch_engine.accept_incident(officer_id, incident_id)
        if not inc:
            return {"error": "Officer or incident not found"}, 404
        return {"incident": inc.to_dict()}

    @app.get("/api/incidents/{incident_id}/escalation")
    def escalation_status(incident_id: str):
        """Check escalation status for an incident."""
        status = dispatch_engine.get_escalation_status(incident_id)
        if not status:
            return {"error": "Incident not found"}, 404
        return status

    @app.get("/api/dispatch/nearest")
    def find_nearest(lat: float, lng: float, role: str = None, radius: float = 10.0):
        """Find nearest available officers."""
        results = dispatch_engine.find_nearest_officers(lat, lng, role=role, radius_km=radius)
        return {"officers": [{"distance_km": round(d, 2), "officer": o.to_dict()} for d, o in results]}
