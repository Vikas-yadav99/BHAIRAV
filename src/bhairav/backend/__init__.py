"""Phase 3 - product backend & evidence.

FastAPI services + WebSocket live stream, the pre/during/post-event evidence
pipeline with searchable metadata, and the privacy layer (face blur,
encryption at rest, RBAC, audit logs, evidence expiry). Pure-logic modules
(rbac, audit, privacy, evidence) have zero heavy dependencies; only
``server`` imports FastAPI, lazily, so the rest of the package stays
importable (and testable) on a minimal install.
"""
from __future__ import annotations

from .audit import AuditLog
from .evidence import EvidenceStore, EventRecorder, PreEventBuffer
from .privacy import Encryptor, FaceBlur
from .rbac import TOKEN_TTL_SEC, Permissions, Role, authorize, issue_token, validate_token
from .users import UserError, UserStore

__all__ = [
    "AuditLog",
    "Encryptor",
    "EvidenceStore",
    "EventRecorder",
    "FaceBlur",
    "Permissions",
    "PreEventBuffer",
    "Role",
    "TOKEN_TTL_SEC",
    "UserError",
    "UserStore",
    "authorize",
    "issue_token",
    "validate_token",
]
