"""Tests for the Phase 6 deployment-hardening layer."""
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from bhairav.backend.hardening import RateLimiter, is_loopback, load_evidence_key


# ---------------------------------------------------------------------------
# is_loopback
# ---------------------------------------------------------------------------
def test_is_loopback():
    assert is_loopback("127.0.0.1")
    assert is_loopback("127.0.0.5")
    assert is_loopback("localhost")
    assert is_loopback("::1")
    assert not is_loopback("")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("192.168.1.10")
    assert not is_loopback("10.0.0.1")
    assert not is_loopback("bhairav.example.com")


# ---------------------------------------------------------------------------
# load_evidence_key
# ---------------------------------------------------------------------------
def test_load_evidence_key_none():
    assert load_evidence_key(None) is None
    assert load_evidence_key("") is None


def test_load_evidence_key_valid():
    key = os.urandom(32)
    b64 = base64.b64encode(key).decode()
    assert load_evidence_key(b64) == key


def test_load_evidence_key_wrong_length():
    key = base64.b64encode(os.urandom(16)).decode()
    with pytest.raises(ValueError, match="expected 32"):
        load_evidence_key(key)


def test_load_evidence_key_bad_base64():
    with pytest.raises(ValueError, match="valid base64"):
        load_evidence_key("!!!not-base64!!!")


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------
def test_rate_limiter_allows_up_to_limit():
    rl = RateLimiter(limit=3, window_sec=60)
    assert rl.allow("k")
    assert rl.allow("k")
    assert rl.allow("k")
    assert not rl.allow("k")          # 4th hit in the window is blocked
    assert rl.remaining("k") == 0


def test_rate_limiter_per_key_isolation():
    rl = RateLimiter(limit=1, window_sec=60)
    assert rl.allow("a")
    assert not rl.allow("a")
    assert rl.allow("b")              # different key unaffected


def test_rate_limiter_window_resets():
    rl = RateLimiter(limit=1, window_sec=10)
    assert rl.allow("k", now=100.0)
    assert not rl.allow("k", now=105.0)
    assert rl.allow("k", now=110.01)  # window elapsed
    assert rl.remaining("k", now=115.0) == 0


def test_rate_limiter_prunes_stale_keys():
    rl = RateLimiter(limit=5, window_sec=10)
    rl.allow("old", now=100.0)
    rl.allow("new", now=1000.0)
    assert "old" not in rl._hits
    assert "new" in rl._hits


# ---------------------------------------------------------------------------
# HTTP edge: login rate limit (429) + oversized body (413)
# ---------------------------------------------------------------------------
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from bhairav.backend.audit import AuditLog
from bhairav.backend.evidence import EvidenceStore
from bhairav.backend.hardening import RateLimiter
from bhairav.backend.server import create_app
from bhairav.backend.users import UserStore


@pytest.fixture()
def app_client(tmp_path):
    store = EvidenceStore(tmp_path / "evidence", fps=10.0, blur_faces=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    users = UserStore(tmp_path / "users.json")
    app = create_app(store, audit, secret="test-secret", users=users,
                     login_limiter=RateLimiter(limit=3, window_sec=60))
    return TestClient(app)


def test_login_rate_limited(app_client):
    for _ in range(3):
        r = app_client.post("/auth/login", json={"username": "admin", "password": "admin123"})
        assert r.status_code == 200, r.text
    r = app_client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 429
    assert "slow down" in r.json()["detail"]


def test_login_oversized_body_rejected(app_client):
    huge = "x" * (2 * 1024 * 1024 + 1)   # > 2 MB
    r = app_client.post("/auth/login", json={"username": huge, "password": "y"})
    assert r.status_code == 413
