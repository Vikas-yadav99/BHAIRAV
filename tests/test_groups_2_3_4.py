"""Tests for Groups 2, 3, and 4 of the audit fix.

Group 2: Unified Identity — PersonRecord + IdentityService
Group 3: Real Detection — import validation, health check
Group 4: Security Wiring — rate limiting, audit logging
"""
import time

from bhairav.identity import PersonRecord, IdentityService, _person_id
from bhairav.detectors.validation import (
    full_health_check, check_yolo_available, check_opencv_available,
    get_detection_status, find_test_video, validate_detection_on_frame,
)
from bhairav.backend.security import (
    SecurityAuditLog, RequestValidator, CORSConfig, CSRFProtection,
    sanitize_input,
)
from bhairav.backend.hardening import RateLimiter


# ============================================================
# Group 2: Unified Identity
# ============================================================

class TestPersonRecord:
    def test_create(self):
        p = PersonRecord()
        assert p.person_id.startswith("P-")
        assert len(p.person_id) == 10
        assert p.first_seen > 0
        assert p.total_frames == 0

    def test_add_track(self):
        p = PersonRecord()
        p.add_track("CAM-01", 7)
        assert "CAM-01" in p.track_ids
        assert 7 in p.track_ids["CAM-01"]
        assert "CAM-01" in p.cameras_seen
        assert p.total_frames == 1

    def test_add_track_dedup(self):
        p = PersonRecord()
        p.add_track("CAM-01", 7)
        p.add_track("CAM-01", 7)  # same track
        assert p.track_ids["CAM-01"] == [7]
        assert p.total_frames == 2  # frames increment, but track_id deduped

    def test_add_evidence(self):
        p = PersonRecord()
        p.add_evidence("EV-001")
        p.add_evidence("EV-001")  # dedup
        assert p.evidence_ids == ["EV-001"]

    def test_add_alert(self):
        p = PersonRecord()
        p.add_alert("AL-001")
        assert p.alert_ids == ["AL-001"]

    def test_to_dict(self):
        p = PersonRecord()
        d = p.to_dict()
        assert "person_id" in d
        assert "track_ids" in d
        assert "cameras_seen" in d


class TestIdentityService:
    def test_resolve_new_track(self):
        svc = IdentityService()
        pid = svc.resolve("CAM-01", 1)
        assert pid.startswith("P-")
        assert svc.count() == 1

    def test_resolve_same_track(self):
        svc = IdentityService()
        pid1 = svc.resolve("CAM-01", 1)
        pid2 = svc.resolve("CAM-01", 1)
        assert pid1 == pid2  # same track -> same person

    def test_resolve_different_tracks(self):
        svc = IdentityService()
        pid1 = svc.resolve("CAM-01", 1)
        pid2 = svc.resolve("CAM-01", 2)
        assert pid1 != pid2  # different tracks -> different people

    def test_link_reid_new(self):
        svc = IdentityService()
        pid = svc.link_reid("CAM-01", 1, "SUBJ-abc")
        assert pid.startswith("P-")
        person = svc.get_person(pid)
        assert person.reid_subject == "SUBJ-abc"

    def test_link_reid_merge(self):
        svc = IdentityService()
        pid1 = svc.link_reid("CAM-01", 1, "SUBJ-abc")
        pid2 = svc.link_reid("CAM-02", 3, "SUBJ-abc")
        assert pid1 == pid2  # same reid subject -> same person
        person = svc.get_person(pid1)
        assert "CAM-01" in person.cameras_seen
        assert "CAM-02" in person.cameras_seen

    def test_get_by_track(self):
        svc = IdentityService()
        pid = svc.resolve("CAM-01", 5)
        person = svc.get_by_track("CAM-01", 5)
        assert person is not None
        assert person.person_id == pid

    def test_get_by_reid(self):
        svc = IdentityService()
        pid = svc.link_reid("CAM-01", 1, "SUBJ-xyz")
        person = svc.get_by_reid("SUBJ-xyz")
        assert person is not None
        assert person.person_id == pid

    def test_add_evidence(self):
        svc = IdentityService()
        pid = svc.resolve("CAM-01", 1)
        assert svc.add_evidence(pid, "EV-100")
        person = svc.get_person(pid)
        assert "EV-100" in person.evidence_ids

    def test_add_alert(self):
        svc = IdentityService()
        pid = svc.resolve("CAM-01", 1)
        assert svc.add_alert(pid, "AL-200")
        person = svc.get_person(pid)
        assert "AL-200" in person.alert_ids

    def test_list_persons(self):
        svc = IdentityService()
        svc.resolve("CAM-01", 1)
        svc.resolve("CAM-01", 2)
        persons = svc.list_persons()
        assert len(persons) == 2

    def test_prune(self):
        svc = IdentityService()
        pid = svc.resolve("CAM-01", 1)
        person = svc.get_person(pid)
        person.last_seen = time.time() - 7200  # 2 hours ago
        removed = svc.prune(max_age_sec=3600)
        assert removed == 1
        assert svc.count() == 0

    def test_cross_camera_identity(self):
        """Same person tracked on two cameras gets one person_id via re-ID."""
        svc = IdentityService()
        # Person appears on CAM-01 as track 3
        pid1 = svc.resolve("CAM-01", 3)
        # Same person appears on CAM-02 as track 7, linked by re-ID
        pid2 = svc.link_reid("CAM-02", 7, "SUBJ-shared")
        # Now link CAM-01 track to same re-ID � merges with CAM-02's person
        pid3 = svc.link_reid("CAM-01", 3, "SUBJ-shared")
        # pid2 and pid3 should be the same (both linked via re-ID)
        assert pid2 == pid3
        person = svc.get_person(pid2)
        assert len(person.cameras_seen) == 2


# ============================================================
# Group 3: Detection Validation
# ============================================================

class TestDetectionValidation:
    def test_full_health_check(self):
        result = full_health_check()
        assert "ready" in result
        assert "checks" in result
        assert "opencv" in result["checks"]
        assert result["checks"]["opencv"]["available"]  # opencv is a core dep

    def test_check_yolo(self):
        result = check_yolo_available()
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_check_opencv(self):
        result = check_opencv_available()
        assert result["available"] is True  # must be installed

    def test_get_detection_status(self):
        status = get_detection_status()
        assert "pipeline_ready" in status
        assert "mode" in status or "error" in status

    def test_validate_frame_none(self):
        result = validate_detection_on_frame(None)
        assert result["ok"] is False

    def test_validate_frame_valid(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = validate_detection_on_frame(frame)
        assert result["ok"] is True
        assert result["frame_shape"] == [480, 640, 3]

    def test_validate_frame_empty(self):
        import numpy as np
        frame = np.array([])
        result = validate_detection_on_frame(frame)
        assert result["ok"] is False

    def test_find_test_video(self):
        # Should return None or a valid path
        result = find_test_video()
        assert result is None or isinstance(result, str)


# ============================================================
# Group 4: Security Wiring
# ============================================================

class TestSecurityWiring:
    def test_rate_limiter_all_endpoints(self):
        """RateLimiter works for any key, not just login."""
        limiter = RateLimiter(limit=3, window_sec=60)
        assert limiter.allow("post:/api/alerts") is True
        assert limiter.allow("post:/api/alerts") is True
        assert limiter.allow("post:/api/alerts") is True
        assert limiter.allow("post:/api/alerts") is False  # limit reached

    def test_rate_limiter_different_endpoints_independent(self):
        limiter = RateLimiter(limit=2, window_sec=60)
        assert limiter.allow("post:/api/alerts") is True
        assert limiter.allow("post:/api/alerts") is True
        assert limiter.allow("post:/api/alerts") is False
        # Different endpoint has its own limit
        assert limiter.allow("delete:/api/evidence/1") is True

    def test_request_validator_blocks_injection(self):
        v = RequestValidator()
        r = v.validate_request("POST", "/api/test", {"content-type": "application/json"},
                              b'{"name": "<script>alert(1)</script>"}')
        assert r.safe is False
        assert r.threat_type == "xss"

    def test_request_validator_allows_clean(self):
        v = RequestValidator()
        r = v.validate_request("GET", "/api/status", {})
        assert r.safe is True

    def test_request_validator_blocks_bad_method(self):
        v = RequestValidator()
        r = v.validate_request("TRACE", "/api/test", {})
        assert r.safe is False

    def test_security_audit_log_records(self):
        log = SecurityAuditLog()
        log.log_login_attempt("admin", True, "127.0.0.1")
        log.log_login_attempt("hacker", False, "10.0.0.1")
        log.log_injection_attempt("sql", "DROP TABLE", "10.0.0.1")
        summary = log.get_summary()
        assert summary["total"] == 3
        assert summary["by_severity"]["warning"] == 1
        assert summary["by_severity"]["critical"] == 1

    def test_cors_blocks_unknown_origin(self):
        cors = CORSConfig(allowed_origins=["https://bhairav.city.gov"])
        assert cors.is_origin_allowed("https://bhairav.city.gov") is True
        assert cors.is_origin_allowed("https://evil.com") is False

    def test_csrf_token_roundtrip(self):
        csrf = CSRFProtection("test-secret")
        token = csrf.generate_token("session-123")
        assert csrf.validate_token(token, "session-123") is True
        assert csrf.validate_token(token, "wrong-session") is False

    def test_sql_injection_detection(self):
        r = sanitize_input("'; DROP TABLE users; --")
        assert r.safe is False
        assert r.threat_type == "sql_injection"

    def test_xss_detection(self):
        r = sanitize_input('<img onerror="alert(1)">')
        assert r.safe is False
        assert r.threat_type == "xss"
