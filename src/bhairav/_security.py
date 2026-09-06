"""BHAIRAV Security — Input validation, rate limiting, secure tokens.

Centralizes all input validation and security helpers used across
incidents.py, phone_gateway.py, server.py, and city_safety.py.

Replaces ad-hoc validation scattered across endpoint handlers with
consistent, testable security primitives.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger("bhairav.security")


# ── Input Validation ────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when input validation fails. Endpoints catch this and return 400."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


def validate_lat(lat: float) -> float:
    """Validate latitude is within [-90, 90]."""
    if not isinstance(lat, (int, float)) or math.isnan(lat):
        raise ValidationError("lat", "must be a number")
    if lat < -90 or lat > 90:
        raise ValidationError("lat", f"must be between -90 and 90, got {lat}")
    return float(lat)


def validate_lng(lng: float) -> float:
    """Validate longitude is within [-180, 180]."""
    if not isinstance(lng, (int, float)) or math.isnan(lng):
        raise ValidationError("lng", "must be a number")
    if lng < -180 or lng > 180:
        raise ValidationError("lng", f"must be between -180 and 180, got {lng}")
    return float(lng)


def validate_phone(phone: str) -> str:
    """Validate phone number format (E.164-ish)."""
    phone = (phone or "").strip()
    if not phone:
        raise ValidationError("phone", "is required")
    # Allow + prefix, digits, hyphens, spaces
    cleaned = phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
    if not cleaned.startswith("+") and not cleaned[0].isdigit():
        raise ValidationError("phone", f"invalid format: {phone}")
    digit_count = sum(1 for c in cleaned if c.isdigit())
    if digit_count < 7 or digit_count > 15:
        raise ValidationError("phone", f"must have 7-15 digits, got {digit_count}")
    return phone


def validate_category(category: str) -> str:
    """Validate incident category against allowed values."""
    VALID = {"medical", "fire", "crime", "road_accident", "disaster",
             "missing_person", "other"}
    cat = (category or "").strip().lower()
    if cat not in VALID:
        raise ValidationError("category", f"must be one of {sorted(VALID)}, got '{cat}'")
    return cat


def validate_emergency_level(level: int) -> int:
    """Validate emergency level is 1-4."""
    try:
        level = int(level)
    except (TypeError, ValueError):
        raise ValidationError("emergency_level", "must be an integer 1-4")
    if level < 1 or level > 4:
        raise ValidationError("emergency_level", f"must be 1-4, got {level}")
    return level


def validate_string(value: str, field_name: str, max_len: int = 1000,
                    required: bool = False) -> str:
    """Validate a string field."""
    value = (value or "").strip()
    if required and not value:
        raise ValidationError(field_name, "is required")
    if len(value) > max_len:
        raise ValidationError(field_name, f"max {max_len} characters, got {len(value)}")
    return value


def safe_float(value, default: float = 0.0, field_name: str = "value") -> float:
    """Safely convert to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(field_name, f"must be a number, got {type(value).__name__}")


def safe_int(value, default: int = 0, field_name: str = "value") -> int:
    """Safely convert to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValidationError(field_name, f"must be an integer, got {type(value).__name__}")


# ── Rate Limiting for Public Endpoints ──────────────────────────────────────

class EndpointRateLimiter:
    """Per-IP rate limiter for public endpoints.

    Fixed-window algorithm. Tracks hits per key (typically IP address)
    and rejects when the limit is exceeded within the window.
    """

    def __init__(self, limit: int = 30, window_sec: float = 60.0):
        """
        Args:
            limit: Max requests per window per key
            window_sec: Window duration in seconds
        """
        self.limit = limit
        self.window_sec = window_sec
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Returns True if the request is allowed, False if rate-limited."""
        now = time.time()
        cutoff = now - self.window_sec
        with self._lock:
            # Prune old hits
            self._hits[key] = [t for t in self._hits[key] if t > cutoff]
            if len(self._hits[key]) >= self.limit:
                return False
            self._hits[key].append(now)
            return True

    def remaining(self, key: str) -> int:
        """How many requests remain in the current window."""
        now = time.time()
        cutoff = now - self.window_sec
        with self._lock:
            hits = [t for t in self._hits[key] if t > cutoff]
            return max(0, self.limit - len(hits))

    def retry_after(self, key: str) -> float:
        """Seconds until the oldest hit in the window expires."""
        now = time.time()
        with self._lock:
            hits = self._hits.get(key, [])
            if not hits:
                return 0.0
            return max(0, self.window_sec - (now - hits[0]))


# ── Secure Officer Tokens ──────────────────────────────────────────────────

class OfficerTokenManager:
    """Issue and validate officer authentication tokens.

    Tokens are HMAC-SHA256 signed with a server secret, NOT a hardcoded key.
    Format: {officer_id}:{timestamp}:{hmac_signature}

    Production: server secret comes from env var or generated at startup.
    """

    def __init__(self, secret: str | None = None):
        self.secret = secret or secrets.token_hex(32)

    def issue_token(self, officer_id: str) -> str:
        """Issue a signed token for an officer."""
        ts = str(int(time.time()))
        payload = f"{officer_id}:{ts}"
        sig = hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]
        return f"{payload}:{sig}"

    def validate_token(self, token: str) -> str | None:
        """Validate a token and return the officer_id, or None if invalid.

        Tokens expire after 24 hours.
        """
        if not token or ":" not in token:
            return None

        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            return None

        payload, sig = parts
        expected = hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]

        if not hmac.compare_digest(sig, expected):
            return None

        # Extract officer_id and timestamp
        id_ts = payload.split(":", 1)
        if len(id_ts) != 2:
            return None

        officer_id, ts_str = id_ts
        try:
            ts = float(ts_str)
        except ValueError:
            return None

        # Token expiry: 24 hours
        if time.time() - ts > 86400:
            return None

        return officer_id

    def get_officer_id(self, token: str) -> str:
        """Extract officer_id from token without full validation (for perf).
        Use validate_token() for security-sensitive checks."""
        if not token or ":" not in token:
            return ""
        return token.split(":", 1)[0]


# ── WebSocket Connection Limiter ───────────────────────────────────────────

class WSConnectionLimiter:
    """Limit concurrent WebSocket connections per IP and globally."""

    def __init__(self, per_ip: int = 5, global_max: int = 256):
        self.per_ip = per_ip
        self.global_max = global_max
        self._per_ip: dict[str, int] = defaultdict(int)
        self._total = 0
        self._lock = threading.Lock()

    def allow(self, ip: str) -> bool:
        with self._lock:
            if self._total >= self.global_max:
                return False
            if self._per_ip[ip] >= self.per_ip:
                return False
            self._per_ip[ip] += 1
            self._total += 1
            return True

    def release(self, ip: str) -> None:
        with self._lock:
            self._per_ip[ip] = max(0, self._per_ip[ip] - 1)
            self._total = max(0, self._total - 1)
            if self._per_ip[ip] == 0:
                self._per_ip.pop(ip, None)

    def stats(self) -> dict:
        with self._lock:
            return {"total": self._total, "per_ip": dict(self._per_ip)}


# ── IVR Session Expiry ────────────────────────────────────────────────────

def cleanup_expired_sessions(sessions: dict, ttl_sec: float = 300.0) -> int:
    """Remove expired IVR call sessions. Returns count removed."""
    now = time.time()
    expired = [
        cid for cid, session in sessions.items()
        if now - session.get("started_at", 0) > ttl_sec
    ]
    for cid in expired:
        del sessions[cid]
    return len(expired)


# ── Request Logging ────────────────────────────────────────────────────────

@dataclass
class RequestLog:
    """Structured request log entry."""
    timestamp: float
    method: str
    path: str
    status_code: int
    client_ip: str
    duration_ms: float
    error: str = ""


class RequestLogger:
    """In-memory request log with rotation (keeps last N entries)."""

    def __init__(self, max_entries: int = 10000):
        self._entries: list[RequestLog] = []
        self._max = max_entries
        self._lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "by_status": defaultdict(int),
            "by_path": defaultdict(int),
            "errors": 0,
        }

    def log(self, method: str, path: str, status_code: int,
            client_ip: str, duration_ms: float, error: str = ""):
        entry = RequestLog(
            timestamp=time.time(), method=method, path=path,
            status_code=status_code, client_ip=client_ip,
            duration_ms=duration_ms, error=error,
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]
            self._stats["total_requests"] += 1
            self._stats["by_status"][status_code] += 1
            self._stats["by_path"][path] += 1
            if status_code >= 500:
                self._stats["errors"] += 1

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return [
                {"ts": e.timestamp, "method": e.method, "path": e.path,
                 "status": e.status_code, "ip": e.client_ip,
                 "ms": round(e.duration_ms, 1), "error": e.error}
                for e in self._entries[-limit:]
            ]

    def stats(self) -> dict:
        with self._lock:
            return {
                "total": self._stats["total_requests"],
                "errors": self._stats["errors"],
                "by_status": dict(self._stats["by_status"]),
                "top_paths": dict(
                    sorted(self._stats["by_path"].items(),
                           key=lambda x: -x[1])[:20]
                ),
            }


# ── Proof Upload Size Limiter ─────────────────────────────────────────────

MAX_PROOF_PHOTOS = 5
MAX_PHOTO_SIZE_BASE64 = 5_000_000  # ~3.75MB per photo (base64 is ~1.33x)


def validate_proof_photos(photos: list[str]) -> list[str]:
    """Validate proof photo uploads (count + size)."""
    if not photos:
        return []
    if len(photos) > MAX_PROOF_PHOTOS:
        raise ValidationError("photos", f"max {MAX_PROOF_PHOTOS} photos, got {len(photos)}")
    for i, photo in enumerate(photos):
        if len(photo) > MAX_PHOTO_SIZE_BASE64:
            raise ValidationError(
                f"photos[{i}]",
                f"too large ({len(photo)} bytes, max {MAX_PHOTO_SIZE_BASE64})"
            )
    return photos
