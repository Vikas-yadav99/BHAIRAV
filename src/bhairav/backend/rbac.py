"""Role-based access control: roles, a permission matrix, and signed tokens.

Pure stdlib (hmac/hashlib/secrets) so it works on any install and is fully
unit-testable. Tokens are `payload.signature` where the payload is a
URL-safe base64 JSON blob and the signature is HMAC-SHA256 of it under the
server secret. No external auth dependency.

Roles (least -> most privileged):
    viewer   - watch live stream, browse alert feed
    operator - viewer + download evidence
    analyst  - operator + search/export evidence, view audit trail
    admin    - analyst + delete/expire evidence, manage users
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

TOKEN_TTL_SEC = 12 * 3600  # 12h default; override via issue_token(ttl_sec=...)

# --------------------------------------------------------------------------
# Permission matrix
# --------------------------------------------------------------------------
# Each permission gates a specific action in the API layer.
PERM_STREAM = "stream"                      # consume the live WebSocket feed
PERM_ALERTS = "alerts"                      # read the alert feed
PERM_EVIDENCE_READ = "evidence_read"        # browse/search evidence metadata
PERM_EVIDENCE_DOWNLOAD = "evidence_download"  # download clips/snapshots
PERM_EVIDENCE_EXPORT = "evidence_export"    # bulk export
PERM_EVIDENCE_DELETE = "evidence_delete"    # delete / expire evidence
PERM_AUDIT = "audit"                        # read audit log
PERM_USERS = "users"                        # manage roles / issue tokens


class Permissions:
    """Namespace of all permission constants."""

    STREAM = PERM_STREAM
    ALERTS = PERM_ALERTS
    EVIDENCE_READ = PERM_EVIDENCE_READ
    EVIDENCE_DOWNLOAD = PERM_EVIDENCE_DOWNLOAD
    EVIDENCE_EXPORT = PERM_EVIDENCE_EXPORT
    EVIDENCE_DELETE = PERM_EVIDENCE_DELETE
    AUDIT = PERM_AUDIT
    USERS = PERM_USERS


# role -> set of permissions
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({PERM_STREAM, PERM_ALERTS, PERM_EVIDENCE_READ}),
    "operator": frozenset({PERM_STREAM, PERM_ALERTS, PERM_EVIDENCE_READ,
                           PERM_EVIDENCE_DOWNLOAD}),
    "analyst": frozenset({PERM_STREAM, PERM_ALERTS, PERM_EVIDENCE_READ,
                          PERM_EVIDENCE_DOWNLOAD, PERM_EVIDENCE_EXPORT, PERM_AUDIT}),
    "admin": frozenset({PERM_STREAM, PERM_ALERTS, PERM_EVIDENCE_READ,
                        PERM_EVIDENCE_DOWNLOAD, PERM_EVIDENCE_EXPORT,
                        PERM_EVIDENCE_DELETE, PERM_AUDIT, PERM_USERS}),
    # Phase 9 M5 - police read-only: live feed, alerts, evidence browse +
    # download; NO export/delete/audit/users and no management actions
    "police": frozenset({PERM_STREAM, PERM_ALERTS, PERM_EVIDENCE_READ,
                         PERM_EVIDENCE_DOWNLOAD}),
}

VALID_ROLES = frozenset(ROLE_PERMISSIONS)


@dataclass(frozen=True)
class Role:
    """A named role with its granted permissions."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in ROLE_PERMISSIONS:
            raise ValueError(
                f"unknown role {self.name!r}; expected one of {sorted(VALID_ROLES)}")

    @property
    def permissions(self) -> frozenset[str]:
        return ROLE_PERMISSIONS[self.name]

    def can(self, permission: str) -> bool:
        return permission in self.permissions


def authorize(role: str | Role, permission: str) -> bool:
    """Pure permission check - no side effects, trivially unit-testable."""
    if isinstance(role, Role):
        return role.can(permission)
    if role not in ROLE_PERMISSIONS:
        return False
    return permission in ROLE_PERMISSIONS[role]


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    """URL-safe base64 decode; returns b"" on malformed input (never raises)."""
    try:
        pad = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + pad)
    except (ValueError, TypeError):  # invalid base64 characters/length
        return b""


def issue_token(secret: str, username: str, role: str,
                ttl_sec: float = TOKEN_TTL_SEC, now: float | None = None) -> str:
    """Issue an HMAC-signed capability token for (username, role)."""
    if role not in ROLE_PERMISSIONS:
        raise ValueError(f"cannot issue token for unknown role {role!r}")
    now = time.time() if now is None else now
    payload = _b64e(json.dumps({
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + ttl_sec,
    }, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), payload.encode("ascii"),
                   hashlib.sha256).digest()
    return f"{payload}.{_b64e(sig)}"


def validate_token(secret: str, token: str, now: float | None = None) -> dict | None:
    """Validate a token; return its claims dict, or None if invalid/expired."""
    try:
        payload, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    expected = hmac.new(secret.encode("utf-8"), payload.encode("ascii"),
                        hashlib.sha256).digest()
    if not hmac.compare_digest(_b64d(sig), expected):
        return None
    try:
        claims = json.loads(_b64d(payload).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    now = time.time() if now is None else now
    if claims.get("exp", 0) <= now:
        return None
    if claims.get("role") not in ROLE_PERMISSIONS:
        return None
    return claims


def new_secret() -> str:
    """Generate a fresh server secret (call once at deploy time)."""
    return secrets.token_urlsafe(32)
