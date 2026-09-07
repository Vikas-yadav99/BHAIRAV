"""BHAIRAV City Loader — Loads real city data for pilot deployment.

Reads city config (police stations, cameras, zones) and seeds the
incident system with real coordinates. Currently configured for:
  - Indore, Madhya Pradesh, India (first pilot city)

Usage:
    from bhairav.city_loader import load_city
    load_city(incident_store, dispatch_engine, city_name="indore")
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger("bhairav.city_loader")

# Cities we have data for
CITIES = {
    "indore": {
        "name": "Indore",
        "state": "Madhya Pradesh",
        "country": "India",
        "center": {"lat": 22.7107, "lon": 75.8352},
    },
}


def load_city_config(city_name: str) -> dict | None:
    """Load city config from disk."""
    config_path = Path(__file__).parent.parent.parent / "city" / city_name / "config.json"
    if not config_path.exists():
        # Try relative to project root
        config_path = Path("city") / city_name / "config.json"
    if not config_path.exists():
        log.warning("City config not found for %s", city_name)
        return None
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_city(store, dispatch_engine=None, city_name: str = "indore",
              camera_bridge=None) -> dict:
    """Load real city data and seed the incident system.

    Args:
        store: IncidentStore instance
        dispatch_engine: DispatchEngine instance (optional)
        city_name: city to load ("indore")
        camera_bridge: CameraIncidentBridge (optional, for camera registration)

    Returns:
        Summary of what was seeded
    """
    config = load_city_config(city_name)
    if not config:
        return {"error": f"City config not found: {city_name}"}

    stats = {"city": config["city"], "officers": 0, "incidents": 0, "cameras": 0}

    # ── Seed officers at real police stations ──────────────────────────
    police_stations = config.get("police_stations", [])
    officer_phones = [
        "+91-9876543210", "+91-9876543211", "+91-9876543212",
        "+91-9876543213", "+91-9876543214", "+91-9876543215",
        "+91-9876543216", "+91-9876543217", "+91-9876543218",
        "+91-9876543219", "+91-9876543220", "+91-9876543221",
        "+91-9876543222", "+91-9876543223",
    ]
    roles = ["police", "police", "police", "police", "police", "police",
             "medical", "medical", "fire", "fire", "rescue", "rescue",
             "police", "medical"]
    specialties = [
        ["patrol", "law_enforcement"], ["patrol", "crowd_control"],
        ["investigation", "cyber"], ["patrol", "night_shift"],
        ["traffic", "patrol"], ["special_ops", "law_enforcement"],
        ["emergency", "trauma"], ["cardiac", "emergency"],
        ["hazmat", "rescue"], ["structural", "rescue"],
        ["medical", "fire"], ["disaster", "rescue"],
        ["patrol", "vip_security"], ["ambulance", "emergency"],
    ]

    for i, ps in enumerate(police_stations):
        name = ps.get("name", f"Officer {i+1}")
        role = roles[i % len(roles)]
        phone = officer_phones[i % len(officer_phones)]
        lat = ps["lat"]
        lng = ps["lon"]
        spec = specialties[i % len(specialties)]

        store.register_officer(name, role, phone, lat, lng, spec)
        stats["officers"] += 1

    log.info("Seeded %d officers at %s police stations", stats["officers"], city_name)

    # ── Seed demo incidents at real Indore locations ───────────────────
    incidents = [
        ("medical", 4, 22.7179, 75.8519, "Sarafa Bazaar",
         "Person collapsed near food stalls, possible cardiac event",
         "Rahul", "public"),
        ("crime", 3, 22.7176, 75.8550, "Rajwada Palace",
         "Suspicious group of 4-5 persons near palace entrance",
         "Neha", "public"),
        ("fire", 2, 22.7516, 75.8952, "Vijay Nagar Main Road",
         "Smoke from ground floor shop, electrical fire suspected",
         "Amit", "public"),
        ("road_accident", 3, 22.7050, 75.8600, "MR 10 Bridge",
         "Two-wheeler collision, person injured on road",
         "Vikram", "camera"),
        ("medical", 2, 22.7359, 75.9145, "Khajrana Road",
         "Elderly person needs medical attention, breathing difficulty",
         "Sunita", "sms"),
        ("crime", 2, 22.7251, 75.8878, "Palasia Square",
         "Group argument escalating near bus stop, 6 persons involved",
         "Deepak", "public"),
        ("road_accident", 4, 22.7200, 75.8750, "AB Road Junction",
         "Car hit auto-rickshaw, multiple injuries reported",
         "Meena", "camera"),
    ]

    for cat, level, lat, lng, loc_name, desc, reporter, source in incidents:
        inc = store.create_incident(
            cat, level, lat, lng, loc_name, desc,
            reporter_name=reporter, source=source,
        )
        stats["incidents"] += 1
        if dispatch_engine and level >= 2:
            dispatch_engine.dispatch(inc)

    log.info("Seeded %d incidents at %s locations", stats["incidents"], city_name)

    # ── Register cameras ──────────────────────────────────────────────
    cameras = config.get("cameras", [])
    for cam in cameras:
        if camera_bridge:
            camera_bridge.register_camera(
                cam["id"], name=cam["name"],
                lat=cam["lat"], lon=cam["lon"], zone=cam.get("zone", "default"),
            )
        stats["cameras"] += 1

    log.info("Registered %d cameras for %s", stats["cameras"], city_name)

    # ── Store city config for API/dashboard access ─────────────────────
    stats["config"] = config
    stats["zones"] = config.get("zones", {})
    stats["dispatch_center"] = config.get("dispatch_center", {})

    return stats


def get_city_config(city_name: str = "indore") -> dict | None:
    """Get city config for API responses."""
    return load_city_config(city_name)
