"""Tests for Phase 9 M3: metrics registry, Prometheus exposition, /metrics
and /ready endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from bhairav.backend.audit import AuditLog
from bhairav.backend.evidence import EvidenceStore
from bhairav.backend.metrics import History, MetricsRegistry
from bhairav.backend.server import create_app


def test_history_bounded_and_ordered():
    h = History(maxlen=3)
    for i in range(5):
        h.append(i, ts=float(i))
    pts = h.points()
    assert [p[1] for p in pts] == [2.0, 3.0, 4.0]
    assert h.latest() == 4.0


def test_registry_counters_gauges_and_series():
    m = MetricsRegistry()
    m.inc("frames_total", {"camera": "CAM-01"}, 3)
    m.inc("frames_total", {"camera": "CAM-01"}, 2)
    m.inc("frames_total")
    m.set("fps", 12.5)
    h = m.history("fps", maxlen=4)
    for v in (10, 11, 12, 13, 14):
        m.set("fps", v, ts=float(v))
    snap = m.snapshot()
    assert snap["counters"][0]["value"] == 5.0
    assert [p["v"] for p in snap["series"]["fps"]] == [11, 12, 13, 14]
    assert h.latest() == 14.0


def test_prometheus_render_format():
    m = MetricsRegistry()
    m.inc("bhairav_alerts_total", {"camera": "CAM-01"}, 2)
    m.set("bhairav_clients", 3)
    text = m.render()
    assert "# TYPE bhairav_alerts_total counter" in text
    assert 'bhairav_alerts_total{camera="CAM-01"} 2' in text
    assert "# TYPE bhairav_clients gauge" in text
    assert "bhairav_clients 3" in text


@pytest.fixture()
def ctx(tmp_path):
    store = EvidenceStore(tmp_path / "evidence", fps=10.0, blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    metrics = MetricsRegistry()
    metrics.history("bhairav_clients")
    metrics.set("bhairav_clients", 1)
    app = create_app(store, audit, secret="test-secret",
                     metrics=metrics, ready_check=lambda: True,
                     metrics_token="scrape-secret-123")
    return TestClient(app)


def _admin(client):
    r = client.post("/auth/login",
                    json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    return r.json()["token"]


def test_ready_public(ctx):
    r = ctx.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_metrics_requires_credentials(ctx):
    assert ctx.get("/metrics").status_code == 401


def test_metrics_scrape_token_and_admin(ctx):
    r = ctx.get("/metrics",
                headers={"Authorization": "Bearer scrape-secret-123"})
    assert r.status_code == 200
    assert "bhairav_clients 1" in r.text
    tok = _admin(ctx)
    r = ctx.get("/metrics", headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 200
    assert "bhairav_clients" in r.text


def test_metrics_wrong_token_rejected(ctx):
    r = ctx.get("/metrics", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_status_includes_series(ctx):
    tok = _admin(ctx)
    d = ctx.get("/api/status",
                headers={"Authorization": "Bearer " + tok}).json()
    assert d["series"]["series"]["bhairav_clients"][0]["v"] == 1.0
