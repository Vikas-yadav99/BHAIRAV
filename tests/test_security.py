"""Tests for security hardening: validation, rate limiting, tokens, WebSocket limits."""
from __future__ import annotations

import math
import time
import threading
from bhairav._security import (
    ValidationError,
    validate_lat, validate_lng, validate_phone, validate_category,
    validate_emergency_level, validate_string, safe_float, safe_int,
    validate_proof_photos,
    EndpointRateLimiter, OfficerTokenManager, WSConnectionLimiter,
    RequestLogger,
)


class TestInputValidation:
    """Lat/lng bounds, phone format, category, emergency level."""

    def test_lat_valid(self):
        assert validate_lat(0) == 0.0
        assert validate_lat(28.6139) == 28.6139
        assert validate_lat(-90) == -90.0
        assert validate_lat(90) == 90.0

    def test_lat_invalid(self):
        try:
            validate_lat(91)
            assert False, "should have raised"
        except ValidationError:
            pass
        try:
            validate_lat(-91)
            assert False, "should have raised"
        except ValidationError:
            pass
        try:
            validate_lat(float("nan"))
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_lng_valid(self):
        assert validate_lng(0) == 0.0
        assert validate_lng(77.2090) == 77.2090
        assert validate_lng(-180) == -180.0
        assert validate_lng(180) == 180.0

    def test_lng_invalid(self):
        try:
            validate_lng(181)
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_phone_valid(self):
        assert validate_phone("+91-9876543210") == "+91-9876543210"
        assert validate_phone("+1234567890") == "+1234567890"

    def test_phone_invalid(self):
        try:
            validate_phone("")
            assert False, "should have raised"
        except ValidationError:
            pass
        try:
            validate_phone("123")  # too short
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_category_valid(self):
        assert validate_category("medical") == "medical"
        assert validate_category("CRIME") == "crime"
        assert validate_category("road_accident") == "road_accident"

    def test_category_invalid(self):
        try:
            validate_category("invalid_cat")
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_emergency_level_valid(self):
        assert validate_emergency_level(1) == 1
        assert validate_emergency_level(4) == 4

    def test_emergency_level_invalid(self):
        try:
            validate_emergency_level(5)
            assert False, "should have raised"
        except ValidationError:
            pass
        try:
            validate_emergency_level(0)
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_safe_float(self):
        assert safe_float("3.14") == 3.14
        assert safe_float(None) == 0.0
        assert safe_float(None, 5.0) == 5.0

    def test_safe_float_invalid(self):
        try:
            safe_float("not_a_number", field_name="test")
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_validate_string(self):
        assert validate_string("hello", "test") == "hello"
        assert validate_string("", "test") == ""  # not required
        assert validate_string("  hello  ", "test") == "hello"
        try:
            validate_string("", "test", required=True)
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_validate_string_max_length(self):
        try:
            validate_string("x" * 2001, "test", max_len=2000)
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_validate_proof_photos_count(self):
        assert validate_proof_photos([]) == []
        assert validate_proof_photos(["a", "b"]) == ["a", "b"]
        try:
            validate_proof_photos(["a"] * 6)  # max 5
            assert False, "should have raised"
        except ValidationError:
            pass

    def test_validate_proof_photos_size(self):
        try:
            validate_proof_photos(["x" * 6_000_000])
            assert False, "should have raised"
        except ValidationError:
            pass


class TestRateLimiter:
    """EndpointRateLimiter: fixed-window per-IP rate limiting."""

    def test_allows_within_limit(self):
        limiter = EndpointRateLimiter(limit=5, window_sec=1.0)
        for _ in range(5):
            assert limiter.allow("ip1") is True

    def test_blocks_over_limit(self):
        limiter = EndpointRateLimiter(limit=3, window_sec=60.0)
        for _ in range(3):
            limiter.allow("ip1")
        assert limiter.allow("ip1") is False

    def test_different_ips_independent(self):
        limiter = EndpointRateLimiter(limit=2, window_sec=60.0)
        limiter.allow("ip1")
        limiter.allow("ip1")
        assert limiter.allow("ip1") is False
        assert limiter.allow("ip2") is True  # different IP

    def test_remaining(self):
        limiter = EndpointRateLimiter(limit=5, window_sec=60.0)
        limiter.allow("ip1")
        assert limiter.remaining("ip1") == 4
        limiter.allow("ip1")
        assert limiter.remaining("ip1") == 3

    def test_retry_after(self):
        limiter = EndpointRateLimiter(limit=1, window_sec=10.0)
        limiter.allow("ip1")
        retry = limiter.retry_after("ip1")
        assert 0 < retry <= 10


class TestOfficerTokens:
    """OfficerTokenManager: HMAC-signed, time-limited tokens."""

    def test_issue_and_validate(self):
        mgr = OfficerTokenManager(secret="test-secret-key")
        token = mgr.issue_token("officer-001")
        assert ":" in token
        result = mgr.validate_token(token)
        assert result == "officer-001"

    def test_invalid_token(self):
        mgr = OfficerTokenManager(secret="test-secret-key")
        assert mgr.validate_token("invalid") is None
        assert mgr.validate_token("") is None
        assert mgr.validate_token(None) is None

    def test_wrong_secret(self):
        mgr1 = OfficerTokenManager(secret="secret1")
        mgr2 = OfficerTokenManager(secret="secret2")
        token = mgr1.issue_token("officer-001")
        assert mgr2.validate_token(token) is None  # wrong secret

    def test_expired_token(self):
        mgr = OfficerTokenManager(secret="test-secret-key")
        # Create a token that's 25 hours old
        import hashlib as _hashlib
        ts = str(int(time.time() - 86400 * 25))
        payload = f"officer-001:{ts}"
        sig = _hashlib.new("sha256", payload.encode()).hexdigest()[:32]
        expired_token = f"{payload}:{sig}"
        assert mgr.validate_token(expired_token) is None

    def test_get_officer_id(self):
        mgr = OfficerTokenManager()
        token = mgr.issue_token("off-123")
        assert mgr.get_officer_id(token) == "off-123"


class TestWSConnectionLimiter:
    """WSConnectionLimiter: per-IP and global WebSocket limits."""

    def test_allows_within_limits(self):
        limiter = WSConnectionLimiter(per_ip=3, global_max=10)
        assert limiter.allow("1.2.3.4") is True
        assert limiter.allow("1.2.3.4") is True
        assert limiter.allow("1.2.3.4") is True

    def test_blocks_per_ip(self):
        limiter = WSConnectionLimiter(per_ip=2, global_max=100)
        limiter.allow("1.2.3.4")
        limiter.allow("1.2.3.4")
        assert limiter.allow("1.2.3.4") is False  # per-IP limit

    def test_blocks_global(self):
        limiter = WSConnectionLimiter(per_ip=100, global_max=3)
        limiter.allow("ip1")
        limiter.allow("ip2")
        limiter.allow("ip3")
        assert limiter.allow("ip4") is False  # global limit

    def test_release(self):
        limiter = WSConnectionLimiter(per_ip=1, global_max=1)
        limiter.allow("1.2.3.4")
        assert limiter.allow("1.2.3.4") is False
        limiter.release("1.2.3.4")
        assert limiter.allow("1.2.3.4") is True

    def test_stats(self):
        limiter = WSConnectionLimiter(per_ip=5, global_max=100)
        limiter.allow("1.2.3.4")
        stats = limiter.stats()
        assert stats["total"] == 1
        assert "1.2.3.4" in stats["per_ip"]


class TestRequestLogger:
    """RequestLogger: structured logging with stats."""

    def test_log_and_recent(self):
        logger = RequestLogger(max_entries=100)
        logger.log("GET", "/api/test", 200, "1.2.3.4", 12.5)
        entries = logger.recent(10)
        assert len(entries) == 1
        assert entries[0]["status"] == 200

    def test_rotation(self):
        logger = RequestLogger(max_entries=5)
        for i in range(10):
            logger.log("GET", f"/api/{i}", 200, "1.2.3.4", 1.0)
        assert len(logger.recent(100)) == 5  # rotated

    def test_stats(self):
        logger = RequestLogger()
        logger.log("GET", "/api/test", 200, "1.2.3.4", 1.0)
        logger.log("POST", "/api/test", 500, "1.2.3.4", 50.0)
        stats = logger.stats()
        assert stats["total"] == 2
        assert stats["errors"] == 1


class TestLoadSimulation:
    """Simulate concurrent load on rate limiter and token manager."""

    def test_rate_limiter_under_load(self):
        limiter = EndpointRateLimiter(limit=100, window_sec=1.0)
        blocked = []

        def hit_limiter(key):
            if not limiter.allow(key):
                blocked.append(key)

        threads = [threading.Thread(target=hit_limiter, args=(f"ip{i % 10}",))
                   for i in range(150)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 IPs * 100 limit = up to 1000 allowed, but we only have 150 total
        # so all should pass (15 per IP)
        assert len(blocked) == 0

    def test_rate_limiter_stress(self):
        limiter = EndpointRateLimiter(limit=10, window_sec=60.0)
        blocked_count = 0

        def hit(key):
            nonlocal blocked_count
            if not limiter.allow(key):
                blocked_count += 1

        threads = [threading.Thread(target=hit, args=("attacker",))
                   for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert blocked_count == 40  # 50 - 10 allowed = 40 blocked

    def test_token_manager_thread_safety(self):
        mgr = OfficerTokenManager(secret="test")
        results = []

        def issue_and_validate():
            token = mgr.issue_token("officer-001")
            result = mgr.validate_token(token)
            results.append(result)

        threads = [threading.Thread(target=issue_and_validate) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == "officer-001" for r in results)
        assert len(results) == 50
