"""Comprehensive Security Module (Phase 26).

Covers: input sanitization, SQL injection prevention, XSS protection,
CORS hardening, secrets management, audit logging, CSRF protection,
security headers, and vulnerability scanning.
"""
from __future__ import annotations

import html
import json
import os
import re
import secrets
import sys
import time
import hashlib
import hmac
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# --- Input Sanitization ---

SQL_INJECTION_PATTERNS = [
    re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|DECLARE|TRUNCATE)\b", re.I),
    re.compile(r"(--|;|/\*|\*/|@@|@)", re.I),
    re.compile(r"\bOR\b\s+\d+\s*=\s*\d+", re.I),
    re.compile(r"\bAND\b\s+\d+\s*=\s*\d+", re.I),
    re.compile(r"""['"]\s*(OR|AND)\s+['"]""", re.I),
]

XSS_PATTERNS = [
    re.compile(r"<\s*script", re.I),
    re.compile(r"javascript\s*:", re.I),
    re.compile(r"on\w+\s*=", re.I),
    re.compile(r"<\s*iframe", re.I),
    re.compile(r"<\s*object", re.I),
    re.compile(r"<\s*embed", re.I),
    re.compile(r"<\s*svg\s+onload", re.I),
    re.compile(r"data\s*:\s*text/html", re.I),
]

PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\."),
    re.compile(r"(%2e%2e|%252e)", re.I),
    re.compile(r"(/etc/passwd|/etc/shadow)", re.I),
]


@dataclass
class SanitizeResult:
    safe: bool = True
    value: Any = None
    threat_type: str = ""
    details: str = ""


def sanitize_input(value: str, max_length: int = 10000) -> SanitizeResult:
    if not isinstance(value, str):
        value = str(value)
    if len(value) > max_length:
        return SanitizeResult(False, value[:max_length], "oversized", f"Input exceeds {max_length} chars")
    for pat in SQL_INJECTION_PATTERNS:
        if pat.search(value):
            return SanitizeResult(False, value, "sql_injection", f"Pattern: {pat.pattern[:50]}")
    for pat in XSS_PATTERNS:
        if pat.search(value):
            return SanitizeResult(False, value, "xss", f"Pattern: {pat.pattern[:50]}")
    for pat in PATH_TRAVERSAL_PATTERNS:
        if pat.search(value):
            return SanitizeResult(False, value, "path_traversal", f"Pattern: {pat.pattern[:50]}")
    return SanitizeResult(True, value)


def sanitize_dict(data: dict, max_depth: int = 10) -> SanitizeResult:
    if max_depth <= 0:
        return SanitizeResult(False, data, "nesting", "Max depth exceeded")
    for key, value in data.items():
        if isinstance(value, str):
            result = sanitize_input(value)
            if not result.safe:
                return SanitizeResult(False, data, result.threat_type, f"Field {key}: {result.details}")
        elif isinstance(value, dict):
            result = sanitize_dict(value, max_depth - 1)
            if not result.safe:
                return result
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    result = sanitize_input(item)
                    if not result.safe:
                        return SanitizeResult(False, data, result.threat_type, f"Field {key}[{i}]: {result.details}")
                elif isinstance(item, dict):
                    result = sanitize_dict(item, max_depth - 1)
                    if not result.safe:
                        return result
    return SanitizeResult(True, data)


def html_escape(value: str) -> str:
    return html.escape(value, quote=True)


def sql_safe_identifier(name: str) -> str:
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


# --- Security Headers ---

DEFAULT_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' ws: wss:; font-src 'self'; object-src 'none'; frame-ancestors 'none'",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


def security_middleware(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        if hasattr(response, "headers"):
            for header, value in DEFAULT_SECURITY_HEADERS.items():
                response.headers[header] = value
        return response
    return wrapper


# --- CORS Hardening ---

@dataclass
class CORSConfig:
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"])
    allowed_methods: list[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    allowed_headers: list[str] = field(default_factory=lambda: ["Authorization", "Content-Type"])
    allow_credentials: bool = True
    max_age: int = 3600

    def is_origin_allowed(self, origin: str) -> bool:
        if not origin:
            return False
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins

    def get_headers(self, origin: str) -> dict:
        headers = {}
        if self.is_origin_allowed(origin):
            headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Max-Age"] = str(self.max_age)
        return headers


# --- Secrets Management ---

@dataclass
class SecretsConfig:
    evidence_key_env: str = "BHAIRAV_EVIDENCE_KEY"
    jwt_secret_env: str = "BHAIRAV_JWT_SECRET"
    webhook_secret_env: str = "BHAIRAV_WEBHOOK_SECRET"

    def get_evidence_key(self) -> bytes | None:
        raw = os.environ.get(self.evidence_key_env)
        if not raw:
            return None
        import base64
        try:
            key = base64.b64decode(raw, validate=True)
        except Exception:
            raise ValueError(f"{self.evidence_key_env} is not valid base64")
        if len(key) != 32:
            raise ValueError(f"{self.evidence_key_env} must be 32 bytes (AES-256), got {len(key)}")
        return key

    def get_jwt_secret(self) -> str:
        secret = os.environ.get(self.jwt_secret_env, "")
        if not secret:
            secret = secrets.token_hex(32)
        return secret

    def get_webhook_secret(self) -> str:
        return os.environ.get(self.webhook_secret_env, secrets.token_hex(16))


def generate_api_key() -> str:
    return f"bhr_{secrets.token_hex(32)}"


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(key: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_api_key(key), hashed)


# --- CSRF Protection ---

class CSRFProtection:
    def __init__(self, secret: str = ""):
        self._secret = secret or secrets.token_hex(32)
        self._tokens: dict[str, float] = {}
        self._ttl = 3600

    def generate_token(self, session_id: str) -> str:
        ts = str(int(time.time()))
        payload = f"{session_id}:{ts}"
        sig = hmac.new(self._secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        token = f"{ts}.{sig}"
        self._tokens[token] = time.time()
        return token

    def validate_token(self, token: str, session_id: str) -> bool:
        if not token or token not in self._tokens:
            return False
        if time.time() - self._tokens[token] > self._ttl:
            del self._tokens[token]
            return False
        parts = token.split(".", 1)
        if len(parts) != 2:
            return False
        ts, sig = parts
        payload = f"{session_id}:{ts}"
        expected = hmac.new(self._secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)

    def _prune(self):
        now = time.time()
        stale = [k for k, v in self._tokens.items() if now - v > self._ttl]
        for k in stale:
            del self._tokens[k]


# --- Rate Limiting ---

@dataclass
class RateLimiter:
    """Per-key token-bucket rate limiter.

    ``allow(key)`` returns True while the budget remains, False once the
    caller has exceeded *limit* calls inside the sliding *window_sec*.
    """
    limit: int = 60
    window_sec: float = 60.0
    _buckets: dict = field(default_factory=dict, repr=False)

    def allow(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_sec
        bucket = self._buckets.setdefault(key, [])
        # Evict entries outside the window
        self._buckets[key] = bucket = [t for t in bucket if t > window_start]
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True

    def reset(self, key: str = "") -> None:
        if key:
            self._buckets.pop(key, None)
        else:
            self._buckets.clear()


# --- Request Validation ---

@dataclass
class RequestValidator:
    max_body_size: int = 10 * 1024 * 1024
    max_url_length: int = 2048
    max_header_count: int = 50
    rate_limiter: Any = None

    def validate_request(self, method: str, url: str, headers: dict, body: bytes = b"") -> SanitizeResult:
        allowed = {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}
        if method.upper() not in allowed:
            return SanitizeResult(False, method, "bad_method", f"Method {method} not allowed")
        if len(url) > self.max_url_length:
            return SanitizeResult(False, url, "url_too_long", f"URL exceeds {self.max_url_length}")
        result = sanitize_input(url)
        if not result.safe:
            return result
        if len(headers) > self.max_header_count:
            return SanitizeResult(False, None, "too_many_headers", f"Header count {len(headers)} exceeds {self.max_header_count}")
        if len(body) > self.max_body_size:
            return SanitizeResult(False, None, "body_too_large", f"Body size {len(body)} exceeds {self.max_body_size}")
        if body and "json" in headers.get("content-type", ""):
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    result = sanitize_dict(data)
                    if not result.safe:
                        return result
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if self.rate_limiter:
            ip = headers.get("x-forwarded-for", "unknown").split(",")[0].strip()
            if not self.rate_limiter.allow(ip):
                return SanitizeResult(False, ip, "rate_limited", "Rate limit exceeded")
        return SanitizeResult(True, None)


# --- Security Audit Log ---

@dataclass
class SecurityAuditLog:
    events: list[dict] = field(default_factory=list)
    max_events: int = 10000

    def log(self, event_type: str, details: str, source: str = "system", severity: str = "info", meta: dict | None = None):
        entry = {"ts": time.time(), "type": event_type, "details": details, "source": source, "severity": severity, "meta": meta or {}}
        self.events.append(entry)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def log_login_attempt(self, username: str, success: bool, ip: str = ""):
        self.log("login_attempt", f"{'SUCCESS' if success else 'FAILED'}: {username}", source=ip, severity="warning" if not success else "info")

    def log_injection_attempt(self, threat_type: str, value: str, ip: str = ""):
        self.log("injection_attempt", f"{threat_type}: {value[:200]}", source=ip, severity="critical")

    def log_privilege_escalation(self, user: str, action: str):
        self.log("privilege_escalation", f"{user} attempted: {action}", severity="critical")

    def log_data_access(self, user: str, resource: str):
        self.log("data_access", f"{user} accessed: {resource}", severity="info")

    def log_config_change(self, user: str, setting: str, old_val: str, new_val: str):
        self.log("config_change", f"{user} changed {setting}: {old_val} -> {new_val}", severity="warning")

    def get_events(self, event_type: str = "", severity: str = "", since: float = 0, limit: int = 100) -> list[dict]:
        filtered = self.events
        if event_type:
            filtered = [e for e in filtered if e["type"] == event_type]
        if severity:
            filtered = [e for e in filtered if e["severity"] == severity]
        if since:
            filtered = [e for e in filtered if e["ts"] >= since]
        return filtered[-limit:]

    def get_summary(self) -> dict:
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        for e in self.events:
            by_type[e["type"]] += 1
            by_severity[e["severity"]] += 1
        return {"total": len(self.events), "by_type": dict(by_type), "by_severity": dict(by_severity), "last_event": self.events[-1] if self.events else None}


# --- Vulnerability Scanner ---

class VulnerabilityScanner:
    CHECKS = [
        ("hardcoded_secrets", re.compile(r"(password|secret|token|key)\s*=\s*['\"][^'\"]+['\"]", re.I)),
        ("debug_mode", re.compile(r"debug\s*=\s*True", re.I)),
        ("eval_usage", re.compile(r"\beval\s*\(", re.I)),
        ("exec_usage", re.compile(r"\bexec\s*\(", re.I)),
        ("pickle_load", re.compile(r"pickle\.loads?", re.I)),
        ("shell_true", re.compile(r"shell\s*=\s*True", re.I)),
        ("wildcard_cors", re.compile(r"allow_origins.*\*", re.I)),
    ]

    def scan_file(self, filepath: str) -> list[dict]:
        findings = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    for name, pattern in self.CHECKS:
                        if pattern.search(line):
                            findings.append({"file": filepath, "line": line_no, "check": name, "code": line.strip()[:200]})
        except OSError:
            pass
        return findings

    def scan_directory(self, directory: str) -> dict:
        import pathlib
        all_findings = []
        files_scanned = 0
        for p in pathlib.Path(directory).rglob("*.py"):
            all_findings.extend(self.scan_file(str(p)))
            files_scanned += 1
        by_check = defaultdict(int)
        for f in all_findings:
            by_check[f["check"]] += 1
        return {"files_scanned": files_scanned, "total_findings": len(all_findings), "by_check": dict(by_check), "findings": all_findings[:50]}
