"""Deployment hardening helpers (Phase 6).

- ``load_evidence_key``  - resolve the AES-256 key from a base64 env var with
                           strict validation (32 bytes), so ``encrypt: true``
                           can never silently start with a broken key.
- ``is_loopback``        - is a bind host safe to run with default credentials?
- ``RateLimiter``        - small in-memory fixed-window limiter (stdlib only)
                           used to throttle the login endpoint against
                           credential-stuffing / brute force at the network
                           layer (complements the per-account lockout in
                           users.py).

Pure stdlib + no heavy deps, matching the rest of the backend.
"""
from __future__ import annotations

import base64
import binascii
import threading
import time

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_loopback(host: str) -> bool:
    """True when binding to host exposes the service only on this machine."""
    host = (host or "").strip().lower()
    if host in LOOPBACK_HOSTS:
        return True
    if host.startswith("127."):  # 127.0.0.0/8
        return True
    if host.startswith("0:0:0:0:0:0:0:1") or host == "::1":
        return True
    return False


def load_evidence_key(b64: str | None) -> bytes | None:
    """Decode a base64 32-byte AES-256 key; None in -> None out.

    Raises ValueError with a clear message on malformed input, so callers can
    fail fast at startup instead of discovering a broken key mid-run.
    """
    if not b64:
        return None
    try:
        key = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "BHAIRAV_EVIDENCE_KEY must be valid base64 (a 32-byte AES-256 key). "
            "Generate one with:  python -c \"import os,base64;"
            "print(base64.b64encode(os.urandom(32)).decode())\""
        ) from exc
    if len(key) != 32:
        raise ValueError(
            f"BHAIRAV_EVIDENCE_KEY decodes to {len(key)} bytes; expected 32 "
            "(AES-256). Generate one with: python -c \"import os,base64;"
            "print(base64.b64encode(os.urandom(32)).decode())\""
        )
    return key


class RateLimiter:
    """Fixed-window per-key rate limiter (thread-safe).

    ``allow(key)`` returns False once more than ``limit`` calls happened in the
    last ``window_sec``. Keys with no activity for a full window are dropped so
    the table cannot grow without bound on long-running servers.
    """

    def __init__(self, limit: int, window_sec: float):
        self.limit = limit
        self.window_sec = float(window_sec)
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        stale = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
        for k in stale:
            del self._hits[k]

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            hits = [t for t in self._hits.get(key, []) if now - t < self.window_sec]
            if len(hits) >= self.limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True

    def remaining(self, key: str, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self.window_sec]
            return max(0, self.limit - len(hits))
