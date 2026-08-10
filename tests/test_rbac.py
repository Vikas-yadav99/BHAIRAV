"""Unit tests for Phase 3 RBAC: roles, permission matrix, signed tokens."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhairav.backend.rbac import (
    Permissions, Role, authorize, issue_token, new_secret, validate_token)


def test_roles_grant_expected_permissions():
    assert authorize("viewer", Permissions.STREAM)
    assert authorize("viewer", Permissions.ALERTS)
    assert not authorize("viewer", Permissions.EVIDENCE_DOWNLOAD)
    assert authorize("operator", Permissions.EVIDENCE_DOWNLOAD)
    assert not authorize("operator", Permissions.AUDIT)
    assert authorize("analyst", Permissions.AUDIT)
    assert authorize("admin", Permissions.EVIDENCE_DELETE)
    assert authorize("admin", Permissions.USERS)


def test_unknown_role_denied_and_role_object_validates():
    assert not authorize("root", Permissions.USERS)
    import pytest
    with pytest.raises(ValueError):
        Role("root")
    assert Role("admin").can(Permissions.EVIDENCE_DELETE)


def test_token_roundtrip_and_expiry():
    secret = new_secret()
    tok = issue_token(secret, "alice", "analyst", ttl_sec=60, now=1000.0)
    claims = validate_token(secret, tok, now=1030.0)
    assert claims["sub"] == "alice"
    assert claims["role"] == "analyst"
    assert validate_token(secret, tok, now=1100.0) is None  # expired


def test_token_tampering_detected():
    secret = new_secret()
    tok = issue_token(secret, "alice", "viewer", ttl_sec=60, now=1000.0)
    # flip a char in the payload
    bad = ("A" if tok[0] != "A" else "B") + tok[1:]
    assert validate_token(secret, bad, now=1000.0) is None
    # wrong secret
    assert validate_token(new_secret(), tok, now=1000.0) is None
    # garbage
    assert validate_token(secret, "not.a.token", now=1000.0) is None


def test_token_rejects_forged_role():
    secret = new_secret()
    tok = issue_token(secret, "bob", "viewer", ttl_sec=60, now=1000.0)
    claims = validate_token(secret, tok, now=1000.0)
    assert claims["role"] == "viewer"  # role is signed, cannot be escalated
