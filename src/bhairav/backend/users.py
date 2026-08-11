"""User store (Phase 5): real accounts with PBKDF2-hashed passwords.

Replaces the Phase 3/4 "pick any role" login (a real security hole: any client
could mint an admin token). Users live in a JSON file under the evidence dir;
passwords are salted PBKDF2-HMAC-SHA256 hashes (stdlib only), verified with a
constant-time compare. Mutations are serialized under a lock and written
atomically (tmp + rename) so concurrent reads never see partial state.

Account lifecycle is audited at the API layer (rbac + server.py).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

from .rbac import VALID_ROLES

# 200k iterations is the OWASP-ish default for SHA-256 PBKDF2; override with
# the BHAIRAV_PBKDF2_ITERATIONS env var (tests drop it to keep the suite fast).
PBKDF2_ITERATIONS = int(os.environ.get("BHAIRAV_PBKDF2_ITERATIONS", "200000"))
MIN_PASSWORD_LEN = 6
MAX_USERS = 64
AUTH_MAX_FAILURES = 5        # consecutive failures before temporary lockout
AUTH_LOCKOUT_SEC = 300.0     # lockout window (5 min)

_dummy_cache: tuple[str, int, str] | None = None


def _dummy_creds() -> tuple[str, int, str]:
    """Dummy credential for timing equalization (computed once, lazily)."""
    global _dummy_cache
    if _dummy_cache is None:
        _dummy_cache = UserStore._hash_password("timing-equalizer-dummy")
    return _dummy_cache

# Credentials for a fresh install; override via env or the admin API.
# Only seeded when the users file does not exist yet.
DEFAULT_USERS = [
    {"username": "admin", "password": "admin123", "role": "admin"},
    {"username": "operator", "password": "operator123", "role": "operator"},
    {"username": "analyst", "password": "analyst123", "role": "analyst"},
    {"username": "viewer", "password": "viewer123", "role": "viewer"},
    {"username": "police", "password": "police123", "role": "police"},
]


class UserError(Exception):
    """Raised for domain-level user errors (dup, bad role, weak password)."""


class UserStore:
    """JSON-backed user directory with PBKDF2 password hashing.

    Layout::

        {
          "version": 1,
          "users": {
            "admin": {"role": "admin", "salt": "...", "iterations": 200000,
                      "hash": "...", "created": 123.4, "last_login": null,
                      "locked": false}
          }
        }
    """

    def __init__(self, path: str | Path, seed: bool = True,
                 now: float | None = None):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._now = now  # injectable clock for deterministic tests
        self._users: dict[str, dict] = {}
        self._failures: dict[str, list[float]] = {}  # in-memory brute-force counter
        self._load()
        if not self._users and seed:
            self._seed_defaults()

    # ---- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return  # corrupt file -> start empty (next save repairs it)
        self._users = {u["username"]: u for u in data.get("users", [])
                       if isinstance(u, dict) and u.get("username")}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1,
             "users": sorted(self._users.values(), key=lambda u: u["username"])},
            sort_keys=True, indent=2).encode("utf-8")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, self.path)

    def _now_ts(self) -> float:
        return time.time() if self._now is None else self._now

    # ---- seeding ----------------------------------------------------------
    def _seed_defaults(self) -> None:
        env_pw = os.environ.get("BHAIRAV_ADMIN_PW")
        for u in DEFAULT_USERS:
            password = env_pw if u["username"] == "admin" and env_pw else u["password"]
            self.create(u["username"], password, u["role"], seed=True)
        self._save()

    # ---- password hashing -------------------------------------------------
    @staticmethod
    def _hash_password(password: str, iterations: int = PBKDF2_ITERATIONS,
                       salt: bytes | None = None) -> tuple[str, int, str]:
        salt = salt or secrets.token_bytes(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt,
                                 iterations)
        return salt.hex(), iterations, dk.hex()

    @staticmethod
    def _verify_password(password: str, salt_hex: str, iterations: int,
                         expected_hash: str) -> bool:
        try:
            salt = bytes.fromhex(salt_hex)
        except ValueError:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt,
                                 max(1, int(iterations)))
        return hmac.compare_digest(dk.hex(), expected_hash)

    # ---- queries ----------------------------------------------------------
    def get(self, username: str) -> dict | None:
        with self._lock:
            u = self._users.get(username)
            return dict(u) if u else None  # copy; never hand out internal refs

    def public_view(self) -> list[dict]:
        """Users without any password material (safe to expose via the API)."""
        with self._lock:
            return [{"username": u["username"], "role": u["role"],
                     "created": round(u["created"], 3),
                     "last_login": (round(u["last_login"], 3)
                                    if u.get("last_login") else None),
                     "locked": bool(u.get("locked"))}
                    for u in sorted(self._users.values(), key=lambda x: x["username"])]

    def _prune_failures(self, username: str) -> list[float]:
        now = self._now_ts()
        recent = [t for t in self._failures.get(username, [])
                  if now - t < AUTH_LOCKOUT_SEC]
        self._failures[username] = recent
        return recent

    def is_locked_out(self, username: str) -> bool:
        """True if the account is under a temporary brute-force lockout."""
        with self._lock:
            return len(self._prune_failures(username)) >= AUTH_MAX_FAILURES

    def authenticate(self, username: str, password: str) -> dict | None:
        """Verify credentials; returns the public record or None.

        Unknown usernames still burn a full PBKDF2 check (constant-time-ish
        response, no username enumeration via timing), and 5 consecutive
        failures trigger a temporary lockout that auto-expires.
        """
        with self._lock:
            if self.is_locked_out(username):
                return None
            u = self._users.get(username)
            if u is None or u.get("locked"):
                # uniform work for unknown/locked accounts too
                self._verify_password(password, *_dummy_creds())
                return None
            if not self._verify_password(password, u["salt"], u["iterations"],
                                         u["hash"]):
                self._failures.setdefault(username, []).append(self._now_ts())
                return None
            self._failures.pop(username, None)  # success clears the counter
            u["last_login"] = self._now_ts()
            self._save()
            return {"username": u["username"], "role": u["role"]}

    # ---- mutations --------------------------------------------------------
    def create(self, username: str, password: str, role: str,
               seed: bool = False) -> dict:
        """Create a user. Returns the PUBLIC record (never hashes/salts).

        Raises UserError on invalid input / duplicates.
        """
        username = (username or "").strip()
        if not username or len(username) > 32:
            raise UserError("username must be 1-32 characters")
        if role not in VALID_ROLES:
            raise UserError(f"invalid role {role!r}; expected one of {sorted(VALID_ROLES)}")
        if not seed and len(password) < MIN_PASSWORD_LEN:
            raise UserError(f"password must be at least {MIN_PASSWORD_LEN} characters")
        with self._lock:
            if username in self._users:
                raise UserError(f"user {username!r} already exists")
            if len(self._users) >= MAX_USERS:
                raise UserError(f"user limit ({MAX_USERS}) reached")
            salt_hex, iterations, pw_hash = self._hash_password(password)
            self._users[username] = {
                "username": username, "role": role,
                "salt": salt_hex, "iterations": iterations, "hash": pw_hash,
                "created": self._now_ts(), "last_login": None, "locked": False,
            }
            self._save()
        return next(u for u in self.public_view() if u["username"] == username)

    def delete(self, username: str) -> bool:
        with self._lock:
            if username not in self._users:
                return False
            del self._users[username]
            self._save()
            return True

    def set_locked(self, username: str, locked: bool) -> bool:
        with self._lock:
            u = self._users.get(username)
            if u is None:
                return False
            u["locked"] = bool(locked)
            self._save()
            return True

    def set_role(self, username: str, role: str) -> bool:
        if role not in VALID_ROLES:
            raise UserError(f"invalid role {role!r}")
        with self._lock:
            u = self._users.get(username)
            if u is None:
                return False
            u["role"] = role
            self._save()
            return True

    def change_password(self, username: str, new_password: str) -> bool:
        if len(new_password) < MIN_PASSWORD_LEN:
            raise UserError(f"password must be at least {MIN_PASSWORD_LEN} characters")
        with self._lock:
            u = self._users.get(username)
            if u is None:
                return False
            salt_hex, iterations, pw_hash = self._hash_password(new_password)
            u.update(salt=salt_hex, iterations=iterations, hash=pw_hash)
            self._save()
            return True

    def count(self) -> int:
        with self._lock:
            return len(self._users)
