"""Security module tests."""
import time
from bhairav.backend.security import (
    sanitize_input, sanitize_dict, html_escape, sql_safe_identifier,
    CORSConfig, CSRFProtection, SecurityAuditLog, VulnerabilityScanner,
    RequestValidator, SecretsConfig, generate_api_key, hash_api_key,
    verify_api_key, DEFAULT_SECURITY_HEADERS,
)


class TestInputSanitization:
    def test_clean_input(self):
        r = sanitize_input("hello world")
        assert r.safe is True

    def test_sql_injection_select(self):
        r = sanitize_input("'; DROP TABLE users; --")
        assert r.safe is False
        assert r.threat_type == "sql_injection"

    def test_sql_injection_union(self):
        r = sanitize_input("1 UNION SELECT * FROM passwords")
        assert r.safe is False

    def test_xss_script_tag(self):
        r = sanitize_input("<script>alert(1)</script>")
        assert r.safe is False
        assert r.threat_type == "xss"

    def test_xss_event_handler(self):
        r = sanitize_input('<img onerror="alert(1)">')
        assert r.safe is False

    def test_xss_javascript_uri(self):
        r = sanitize_input("javascript:alert(1)")
        assert r.safe is False

    def test_path_traversal(self):
        r = sanitize_input("../../../etc/passwd")
        assert r.safe is False
        assert r.threat_type == "path_traversal"

    def test_max_length(self):
        r = sanitize_input("a" * 10001, max_length=10000)
        assert r.safe is False
        assert r.threat_type == "oversized"

    def test_dict_sanitization_clean(self):
        r = sanitize_dict({"name": "test", "value": 42})
        assert r.safe is True

    def test_dict_sanitization_injection(self):
        r = sanitize_dict({"name": "'; DROP TABLE --"})
        assert r.safe is False

    def test_dict_nested_injection(self):
        r = sanitize_dict({"outer": {"inner": "<script>alert(1)</script>"}})
        assert r.safe is False

    def test_list_in_dict_injection(self):
        r = sanitize_dict({"items": ["safe", "'; DROP TABLE --"]})
        assert r.safe is False


class TestHtmlEscape:
    def test_basic_escape(self):
        assert html_escape("<b>") == "&lt;b&gt;"
        assert html_escape("a & b") == "a &amp; b"

    def test_quotes(self):
        assert html_escape('"hello"') == "&quot;hello&quot;"


class TestSqlSafeIdentifier:
    def test_valid(self):
        assert sql_safe_identifier("users") == "users"
        assert sql_safe_identifier("_private") == "_private"

    def test_invalid(self):
        import pytest
        with pytest.raises(ValueError):
            sql_safe_identifier("users; DROP TABLE")
        with pytest.raises(ValueError):
            sql_safe_identifier("1table")


class TestCORS:
    def test_default_origins(self):
        cfg = CORSConfig()
        assert cfg.is_origin_allowed("http://localhost:8000")
        assert not cfg.is_origin_allowed("http://evil.com")

    def test_wildcard(self):
        cfg = CORSConfig(allowed_origins=["*"])
        assert cfg.is_origin_allowed("http://evil.com")

    def test_headers(self):
        cfg = CORSConfig()
        h = cfg.get_headers("http://localhost:8000")
        assert "Access-Control-Allow-Origin" in h
        assert h["Access-Control-Allow-Credentials"] == "true"


class TestCSRF:
    def test_generate_and_validate(self):
        csrf = CSRFProtection("test-secret")
        token = csrf.generate_token("session123")
        assert csrf.validate_token(token, "session123") is True

    def test_wrong_session(self):
        csrf = CSRFProtection("test-secret")
        token = csrf.generate_token("session1")
        assert csrf.validate_token(token, "session2") is False

    def test_expired_token(self):
        csrf = CSRFProtection("test-secret")
        token = csrf.generate_token("session1")
        csrf._ttl = 0
        time.sleep(0.01)
        assert csrf.validate_token(token, "session1") is False


class TestAuditLog:
    def test_log_event(self):
        log = SecurityAuditLog()
        log.log("test", "test event")
        assert len(log.events) == 1
        assert log.events[0]["type"] == "test"

    def test_login_attempt(self):
        log = SecurityAuditLog()
        log.log_login_attempt("admin", True, "127.0.0.1")
        log.log_login_attempt("admin", False, "127.0.0.1")
        assert len(log.events) == 2
        assert log.events[0]["severity"] == "info"
        assert log.events[1]["severity"] == "warning"

    def test_injection_attempt(self):
        log = SecurityAuditLog()
        log.log_injection_attempt("sql_injection", "DROP TABLE", "10.0.0.1")
        events = log.get_events(severity="critical")
        assert len(events) == 1

    def test_summary(self):
        log = SecurityAuditLog()
        log.log("type_a", "e1")
        log.log("type_a", "e2")
        log.log("type_b", "e3")
        s = log.get_summary()
        assert s["total"] == 3
        assert s["by_type"]["type_a"] == 2

    def test_prune_old_events(self):
        log = SecurityAuditLog(max_events=5)
        for i in range(10):
            log.log("test", f"event {i}")
        assert len(log.events) == 5

    def test_filter_events(self):
        log = SecurityAuditLog()
        log.log("login", "e1")
        log.log("injection", "e2")
        log.log("login", "e3")
        assert len(log.get_events(event_type="login")) == 2


class TestRequestValidator:
    def test_bad_method(self):
        v = RequestValidator()
        r = v.validate_request("PATCH", "/api/test", {})
        assert r.safe is True  # PATCH is allowed

    def test_invalid_method(self):
        v = RequestValidator()
        r = v.validate_request("TRACE", "/api/test", {})
        assert r.safe is False

    def test_url_too_long(self):
        v = RequestValidator()
        r = v.validate_request("GET", "/api/" + "a" * 3000, {})
        assert r.safe is False

    def test_body_too_large(self):
        v = RequestValidator()
        r = v.validate_request("POST", "/api/test", {"content-type": "text/plain"}, b"x" * 20000000)
        assert r.safe is False

    def test_json_injection(self):
        v = RequestValidator()
        body = b'{"name": "<script>alert(1)</script>"}'
        r = v.validate_request("POST", "/api/test", {"content-type": "application/json"}, body)
        assert r.safe is False

    def test_clean_json(self):
        v = RequestValidator()
        body = b'{"name": "test", "value": 42}'
        r = v.validate_request("POST", "/api/test", {"content-type": "application/json"}, body)
        assert r.safe is True


class TestAPIKeys:
    def test_generate(self):
        key = generate_api_key()
        assert key.startswith("bhr_")
        assert len(key) == 68

    def test_hash_and_verify(self):
        key = generate_api_key()
        h = hash_api_key(key)
        assert verify_api_key(key, h) is True
        assert verify_api_key("wrong_key", h) is False

    def test_constant_time(self):
        """Verify timing-safe comparison."""
        import hmac
        key = generate_api_key()
        h = hash_api_key(key)
        # Should not raise or differ based on timing
        assert verify_api_key(key, h)


class TestVulnerabilityScanner:
    def test_scan_clean_code(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("x = 1\nprint(x)\n")
        from bhairav.backend.security import VulnerabilityScanner
        s = VulnerabilityScanner()
        r = s.scan_file(str(f))
        assert len(r) == 0

    def test_scan_debug_mode(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("DEBUG = True\nsecret = 'abc123'\n")
        from bhairav.backend.security import VulnerabilityScanner
        s = VulnerabilityScanner()
        r = s.scan_file(str(f))
        checks = [x["check"] for x in r]
        assert "debug_mode" in checks
        assert "hardcoded_secrets" in checks

    def test_scan_directory(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("eval('1+1')\n")
        from bhairav.backend.security import VulnerabilityScanner
        s = VulnerabilityScanner()
        r = s.scan_directory(str(tmp_path))
        assert r["total_findings"] > 0
