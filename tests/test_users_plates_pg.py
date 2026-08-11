"""Phase 8 M3 - PostgreSQL user store + plate watchlist (pg_users / pg_plates).

Unit tests (validation, missing-driver path) run everywhere; integration
tests need a real PostgreSQL and are gated behind $BHAIRAV_TEST_DB_URL.
"""
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from bhairav.backend import pg_plates as pgp
from bhairav.backend import pg_users as pgu
from bhairav.backend.pg_store import _DriverMissing
from bhairav.backend.users import UserError

TEST_DB_URL = os.environ.get("BHAIRAV_TEST_DB_URL")


# ---------------------------------------------------------------------------
# Unit tests (no database required)
# ---------------------------------------------------------------------------
def test_pg_users_missing_driver_raises_clear_error(monkeypatch):
    def boom(*a, **k):
        raise _DriverMissing("psycopg required")
    monkeypatch.setattr(pgu, "load_driver", boom)
    with pytest.raises(RuntimeError, match="psycopg"):
        pgu.PostgresUserStore("postgresql://x/y")


def test_pg_plates_missing_driver_raises_clear_error(monkeypatch):
    def boom(*a, **k):
        raise _DriverMissing("psycopg required")
    monkeypatch.setattr(pgp, "load_driver", boom)
    with pytest.raises(RuntimeError, match="psycopg"):
        pgp.PostgresPlateRegistry("postgresql://x/y")


def test_pg_users_validation_without_db(monkeypatch):
    class _Fake:
        def __init__(self, *a, **k):
            pass
        def _cursor(self):
            raise AssertionError("no DB should be touched for validation")
        def _seed_defaults(self):
            pass
    fake = _Fake()
    with pytest.raises(UserError):
        pgu.PostgresUserStore.create(fake, "", "x", "admin")
    with pytest.raises(UserError):
        pgu.PostgresUserStore.create(fake, "ok", "x", "nope")
    with pytest.raises(UserError):
        pgu.PostgresUserStore.change_password(fake, "nobody", "short")
    with pytest.raises(UserError):
        pgu.PostgresUserStore.set_role(fake, "u", "badrole")


def test_pg_plates_validation_without_db(monkeypatch):
    class _Fake:
        def __init__(self, *a, **k):
            pass
        def _cursor(self):
            raise AssertionError("no DB should be touched for validation")
    with pytest.raises(ValueError):
        pgp.PostgresPlateRegistry.watch(_Fake(), "", "")
    with pytest.raises(ValueError):
        pgp.PostgresPlateRegistry.watch(_Fake(), "X" * 20, "")


# ---------------------------------------------------------------------------
# Integration tests (gated on a real PostgreSQL)
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not TEST_DB_URL, reason="set BHAIRAV_TEST_DB_URL to run PostgreSQL tests")


@pytest.fixture()
def pg_users():
    # construct once to ensure the schema, wipe, then re-construct so the
    # empty-table seeding path actually runs (seeding happens in __init__)
    pgu.PostgresUserStore(TEST_DB_URL)._cursor().execute("DELETE FROM users")
    store = pgu.PostgresUserStore(TEST_DB_URL)
    yield store


@pytest.fixture()
def pg_plates():
    reg = pgp.PostgresPlateRegistry(TEST_DB_URL)
    cur = reg._cursor()
    cur.execute("DELETE FROM plates_watch")
    cur.execute("DELETE FROM plate_reads")
    yield reg


def test_users_seed_and_authenticate(pg_users):
    assert pg_users.count() == 4  # admin/operator/analyst/viewer seeded
    assert pg_users.authenticate("admin", "admin123")["role"] == "admin"
    assert pg_users.authenticate("admin", "wrong") is None
    assert pg_users.authenticate("ghost", "whatever") is None  # no enumeration
    pub = pg_users.public_view()
    assert all("hash" not in u and "salt" not in u for u in pub)


def test_users_crud_and_lock(pg_users):
    rec = pg_users.create("alice", "secret1", "analyst")
    assert rec["role"] == "analyst"
    with pytest.raises(UserError):
        pg_users.create("alice", "secret1", "viewer")  # duplicate
    assert pg_users.authenticate("alice", "secret1") is not None
    assert pg_users.set_locked("alice", True)
    assert pg_users.authenticate("alice", "secret1") is None  # locked revokes
    assert pg_users.set_locked("alice", False)
    assert pg_users.authenticate("alice", "secret1") is not None
    assert pg_users.change_password("alice", "newpass99")
    assert pg_users.authenticate("alice", "newpass99") is not None
    assert pg_users.authenticate("alice", "secret1") is None
    assert pg_users.set_role("alice", "admin")
    assert pg_users.get("alice")["role"] == "admin"
    assert pg_users.delete("alice")
    assert pg_users.get("alice") is None


def test_users_lockout_after_repeated_failures(pg_users):
    for _ in range(5):
        assert pg_users.authenticate("admin", "bad") is None
    assert pg_users.is_locked_out("admin")
    assert pg_users.authenticate("admin", "admin123") is None  # locked out


def test_plates_watch_and_reads(pg_plates):
    rec = pg_plates.watch("mh12ab1234", reason="stolen", actor="admin")
    assert rec["plate"] == "MH12AB1234"
    assert pg_plates.is_watched("mh12ab1234")
    assert pg_plates.list_watch()[0]["reason"] == "stolen"
    # re-watch updates in place
    pg_plates.watch("MH12AB1234", reason="suspect")
    assert pg_plates.list_watch()[0]["reason"] == "suspect"
    pg_plates.add_read("MH12AB1234", 10.0, bbox=[1, 2, 3, 4])
    pg_plates.add_read("OTHER123", 11.0)
    reads = pg_plates.recent_reads(10)
    assert [r["plate"] for r in reads] == ["OTHER123", "MH12AB1234"]
    assert reads[1]["bbox"] == [1, 2, 3, 4]
    assert pg_plates.unwatch("mh12ab1234")
    assert not pg_plates.is_watched("MH12AB1234")
    assert not pg_plates.unwatch("MH12AB1234")  # already gone
