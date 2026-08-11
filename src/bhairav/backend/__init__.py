"""Phase 3-8 - product backend & evidence.

FastAPI services + WebSocket live stream, the pre/during/post-event evidence
pipeline with searchable metadata, and the privacy layer (face blur,
encryption at rest, RBAC, audit logs, evidence expiry). Pure-logic modules
(rbac, audit, privacy, evidence) have zero heavy dependencies; only
``server`` imports FastAPI, lazily, so the rest of the package stays
importable (and testable) on a minimal install.

Phase 8 adds a drop-in PostgreSQL backend for every persistent store:
  pg_store / pg_audit / pg_users / pg_plates implement the same interfaces
  as the file-based stores, enabled by setting backend.db (BHAIRAV_DB_URL).
  server.py also hosts the Investigation Assistant endpoint and the
  multi-camera LiveHub channels.
"""
from __future__ import annotations

from .audit import AuditLog
from .evidence import EvidenceStore, EventRecorder, PreEventBuffer
from .hardening import RateLimiter, is_loopback, load_evidence_key
from .privacy import Encryptor, FaceBlur
from .rbac import TOKEN_TTL_SEC, Permissions, Role, authorize, issue_token, validate_token
from .users import UserError, UserStore

__all__ = [
    "AuditLog",
    "RateLimiter",
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
    "is_loopback",
    "load_evidence_key",
    "authorize",
    "issue_token",
    "validate_token",
]
