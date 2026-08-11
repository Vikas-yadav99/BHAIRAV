"""PostgreSQL-backed user store (Phase 8 M3).

Drop-in replacement for the file-based ``UserStore`` (users.py) behind the
same interface, so the auth layer and admin APIs work identically. Password
hashing is reused unchanged: salted PBKDF2-HMAC-SHA256 with constant-time
verify, the same lockout policy, and the same seeding of demo accounts on a
fresh install.

DB mode is enabled together with the evidence store when
``backend.db`` / ``BHAIRAV_DB_URL`` / ``--db-url`` is set; the schema is
created idempotently on first connect. The driver is imported lazily
(psycopg 3), same as pg_store / pg_audit.
"""
from __future__ import annotations

import threading
import time

from .pg_store import load_driver
from .users import (AUTH_LOCKOUT_SEC, AUTH_MAX_FAILURES, DEFAULT_USERS,
                    MAX_USERS, MIN_PASSWORD_LEN, UserError, UserStore,
                    VALID_ROLES, _dummy_creds)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username   TEXT PRIMARY KEY,
    role       TEXT NOT NULL,
    salt       TEXT NOT NULL,
    iterations INTEGER NOT NULL,
    hash       TEXT NOT NULL,
    created    DOUBLE PRECISION NOT NULL,
    last_login DOUBLE PRECISION,
    locked     BOOLEAN NOT NULL DEFAULT FALSE
);
"""


class PostgresUserStore:
    """UserStore-compatible backend on PostgreSQL (see module docstring)."""

    def __init__(self, url: str, seed: bool = True, now: float | None = None):
        self.url = url
        self._now = now  # injectable clock for deterministic tests
        self._lock = threading.RLock()
        self._failures: dict[str, list[float]] = {}  # in-memory lockout counter
        self._conn = None
        with self._lock:
            self._conn = self._connect()
            cur = self._conn.cursor()
            cur.execute(SCHEMA)
        if seed:
            self._seed_defaults()

    def _now_ts(self) -> float:
        return time.time() if self._now is None else self._now

    # ---- plumbing ---------------------------------------------------------
    def _connect(self):
        psycopg = load_driver()
        try:
            conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
        except Exception as exc:
            raise RuntimeError(
                f"cannot connect to PostgreSQL ({self.url}): {exc}") from exc
        conn.row_factory = psycopg.rows.dict_row
        from psycopg.types.json import register_default_adapters
        register_default_adapters(conn)
        return conn

    def _cursor(self):
        if self._conn is None:
            self._conn = self._connect()
        return self._conn.cursor()

    def _row(self, username: str) -> dict | None:
        cur = self._cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        return cur.fetchone()

    def _public(self, row: dict) -> dict:
        return {"username": row["username"], "role": row["role"],
                "created": round(row["created"], 3),
                "last_login": (round(row["last_login"], 3)
                               if row.get("last_login") else None),
                "locked": bool(row.get("locked"))}

    # ---- seeding ----------------------------------------------------------
    def _seed_defaults(self) -> None:
        cur = self._cursor()
        cur.execute("SELECT COUNT(*) AS n FROM users")
        if cur.fetchone()["n"] > 0:
            return
        import os
        env_pw = os.environ.get("BHAIRAV_ADMIN_PW")
        for u in DEFAULT_USERS:
            password = env_pw if u["username"] == "admin" and env_pw else u["password"]
            self.create(u["username"], password, u["role"], seed=True)

    # ---- queries ----------------------------------------------------------
    def get(self, username: str) -> dict | None:
        with self._lock:
            row = self._row(username)
            return dict(row) if row else None

    def public_view(self) -> list[dict]:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT * FROM users ORDER BY username")
            return [self._public(r) for r in cur.fetchall()]

    def count(self) -> int:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT COUNT(*) AS n FROM users")
            return cur.fetchone()["n"]

    def _prune_failures(self, username: str) -> list[float]:
        now = self._now_ts()
        recent = [t for t in self._failures.get(username, [])
                  if now - t < AUTH_LOCKOUT_SEC]
        self._failures[username] = recent
        return recent

    def is_locked_out(self, username: str) -> bool:
        with self._lock:
            return len(self._prune_failures(username)) >= AUTH_MAX_FAILURES

    def authenticate(self, username: str, password: str) -> dict | None:
        """Verify credentials; returns the public record or None.

        Mirrors UserStore semantics: unknown/locked accounts burn a dummy
        PBKDF2 check (no username enumeration via timing) and 5 consecutive
        failures trigger a temporary lockout that auto-expires.
        """
        with self._lock:
            if self.is_locked_out(username):
                return None
            row = self._row(username)
            if row is None or row.get("locked"):
                UserStore._verify_password(password, *_dummy_creds())
                return None
            if not UserStore._verify_password(password, row["salt"], row["iterations"],
                                    row["hash"]):
                self._failures.setdefault(username, []).append(self._now_ts())
                return None
            self._failures.pop(username, None)
            cur = self._cursor()
            cur.execute("UPDATE users SET last_login = %s WHERE username = %s",
                        (self._now_ts(), username))
            return self._public(row)

    # ---- mutations --------------------------------------------------------
    def create(self, username: str, password: str, role: str,
               seed: bool = False) -> dict:
        """Create a user. Returns the PUBLIC record (never hashes/salts)."""
        username = (username or "").strip()
        if not username or len(username) > 32:
            raise UserError("username must be 1-32 characters")
        if role not in VALID_ROLES:
            raise UserError(f"invalid role {role!r}; expected one of {sorted(VALID_ROLES)}")
        if not seed and len(password) < MIN_PASSWORD_LEN:
            raise UserError(f"password must be at least {MIN_PASSWORD_LEN} characters")
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                raise UserError(f"user {username!r} already exists")
            cur.execute("SELECT COUNT(*) AS n FROM users")
            if cur.fetchone()["n"] >= MAX_USERS:
                raise UserError(f"user limit ({MAX_USERS}) reached")
            salt_hex, iterations, pw_hash = UserStore._hash_password(password)
            cur.execute(
                "INSERT INTO users (username, role, salt, iterations, hash,"
                " created, last_login, locked) VALUES (%s, %s, %s, %s, %s, %s, NULL, FALSE)",
                (username, role, salt_hex, iterations, pw_hash, self._now_ts()))
            row = self._row(username)
            return self._public(row)

    def delete(self, username: str) -> bool:
        with self._lock:
            cur = self._cursor()
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
            return cur.rowcount > 0

    def set_locked(self, username: str, locked: bool) -> bool:
        with self._lock:
            cur = self._cursor()
            cur.execute("UPDATE users SET locked = %s WHERE username = %s",
                        (bool(locked), username))
            return cur.rowcount > 0

    def set_role(self, username: str, role: str) -> bool:
        if role not in VALID_ROLES:
            raise UserError(f"invalid role {role!r}")
        with self._lock:
            cur = self._cursor()
            cur.execute("UPDATE users SET role = %s WHERE username = %s",
                        (role, username))
            return cur.rowcount > 0

    def change_password(self, username: str, new_password: str) -> bool:
        if len(new_password) < MIN_PASSWORD_LEN:
            raise UserError(f"password must be at least {MIN_PASSWORD_LEN} characters")
        with self._lock:
            row = self._row(username)
            if row is None:
                return False
            salt_hex, iterations, pw_hash = UserStore._hash_password(new_password)
            cur = self._cursor()
            cur.execute(
                "UPDATE users SET salt = %s, iterations = %s, hash = %s"
                " WHERE username = %s",
                (salt_hex, iterations, pw_hash, username))
            return True
