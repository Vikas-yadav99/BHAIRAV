"""Phase 8 M4 - Investigation Assistant: offline NL parser + endpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from bhairav.backend.assistant import parse_query

NOW = 1_800_000_000.0
CTX = {"zones": ["plaza", "server_room"], "cameras": ["CAM-01", "CAM-02"],
       "users": ["admin", "analyst"], "now": NOW}


def test_parses_rule_severity_zone_and_time():
    r = parse_query("show all red fight alerts in the plaza last 7 days", CTX)
    assert r["filters"]["severity"] == "red"
    assert r["filters"]["rule"] == "fight"
    assert r["filters"]["t0"] == pytest.approx(NOW - 7 * 86400)
    assert r["filters"]["t1"] is None
    assert "plaza" in r["search_kwargs"]["q"]  # zone kept for the stores
    assert any("rule = fight" in p for p in r["plan"])
    assert any("severity = red" in p for p in r["plan"])
    assert any("time: last 7 day(s)" in p for p in r["plan"])


def test_parses_camera_and_plate():
    r = parse_query("plate MH12AB1234 seen on CAM-02 yesterday", CTX)
    assert r["plates"] == ["MH12AB1234"]
    assert r["filters"]["camera"] == "CAM-02"
    # "yesterday" is a calendar-day window: a 24h span ending at today's
    # midnight (which may be slightly before NOW)
    t0, t1 = r["filters"]["t0"], r["filters"]["t1"]
    assert t1 - t0 == pytest.approx(86400)
    assert 0 < NOW - t1 < 86400
    assert "plate = MH12AB1234" in r["plan"]


def test_parses_audit_intent_and_actor():
    r = parse_query("who logged in yesterday", CTX)
    assert r["want_audit"] is True
    r2 = parse_query("what did admin change", CTX)
    assert r2["want_audit"] is True
    assert r2["actor"] == "admin"


def test_generic_query_warns_but_runs():
    r = parse_query("latest evidence", CTX)
    assert r["warnings"]  # no filters recognised
    assert r["search_kwargs"]["rule"] is None


def test_bare_vehicle_not_watchlist():
    r = parse_query("vehicles in the plaza", CTX)
    assert r["filters"]["rule"] is None  # no stolen_vehicle from bare "vehicle"


def test_severity_synonyms():
    assert parse_query("critical incidents", CTX)["filters"]["severity"] == "red"
    assert parse_query("amber flags", CTX)["filters"]["severity"] == "yellow"


def test_rule_synonyms():
    assert parse_query("brawl detected", CTX)["filters"]["rule"] == "fight"
    assert parse_query("intruder in the building", CTX)["filters"]["rule"] == "trespass"


def test_endpoint_returns_plan_events_and_audit(tmp_path):
    from fastapi.testclient import TestClient

    from bhairav.backend.audit import AuditLog
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.server import create_app
    from bhairav.backend.users import UserStore

    store = EvidenceStore(tmp_path / "evidence", camera="CAM-01", fps=10.0,
                          blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.append("admin", "login", "role=admin")
    users = UserStore(tmp_path / "users.json")
    app = create_app(store, audit, secret="test-secret", users=users,
                     assistant_ctx={"zones": ["plaza"], "rules": ["fight"]})
    client = TestClient(app)

    r = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    token = r.json()["token"]
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    res = client.post("/api/assistant/query",
                      json={"query": "show fight alerts in the plaza"},
                      headers=h)
    assert res.status_code == 200
    body = res.json()
    assert body["plan"]
    assert "events" in body and "audit" in body and "plate_reads" in body

    res2 = client.post("/api/assistant/query", json={"query": "who logged in"},
                       headers=h)
    assert res2.status_code == 200
    assert res2.json()["audit"]  # the seeded login entry

    # viewer (no PERM_EVIDENCE_EXPORT) is denied
    vtok = client.post("/auth/login", json={"username": "viewer",
                                            "password": "viewer123"}).json()["token"]
    vh = {"Authorization": f"Bearer {vtok}", "Content-Type": "application/json"}
    assert client.post("/api/assistant/query", json={"query": "fight"},
                       headers=vh).status_code == 403
