"""Unit tests for the Phase 5 user store (PBKDF2 hashing, persistence, lifecycle)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from bhairav.backend.users import MAX_USERS, UserError, UserStore


@pytest.fixture()
def store(tmp_path):
    return UserStore(tmp_path / "users.json")


def test_seed_defaults(store):
    assert store.count() == 5
    names = {u["username"] for u in store.public_view()}
    assert names == {"admin", "operator", "analyst", "viewer", "police"}
    # public view never leaks password material
    for u in store.public_view():
        assert "hash" not in u and "salt" not in u


def test_authenticate_roundtrip(store):
    rec = store.authenticate("admin", "admin123")
    assert rec == {"username": "admin", "role": "admin"}
    assert store.authenticate("admin", "wrong-password") is None
    assert store.authenticate("ghost", "admin123") is None


def test_passwords_are_salted_and_hashed(store):
    from bhairav.backend.users import PBKDF2_ITERATIONS
    raw = store.get("admin")
    assert raw["hash"] != "admin123"
    assert raw["salt"] and raw["iterations"] == PBKDF2_ITERATIONS


def test_create_validation(store):
    with pytest.raises(UserError):
        store.create("x", "short", "admin")          # weak password
    with pytest.raises(UserError):
        store.create("bob", "longenough", "root")    # bad role
    with pytest.raises(UserError):
        store.create("admin", "longenough", "viewer")  # duplicate
    store.create("bob", "secret99", "operator")
    assert store.authenticate("bob", "secret99") == {"username": "bob", "role": "operator"}


def test_delete_and_lock(store):
    store.create("bob", "secret99", "viewer")
    assert store.delete("bob") is True
    assert store.delete("bob") is False
    store.create("bob", "secret99", "viewer")
    assert store.set_locked("bob", True) is True
    assert store.authenticate("bob", "secret99") is None  # locked accounts can't log in
    assert store.set_locked("bob", False) is True
    assert store.authenticate("bob", "secret99") is not None


def test_change_password(store):
    store.change_password("viewer", "newpass99")
    assert store.authenticate("viewer", "viewer123") is None
    assert store.authenticate("viewer", "newpass99") is not None
    with pytest.raises(UserError):
        store.change_password("viewer", "x")  # too short


def test_set_role(store):
    store.set_role("viewer", "analyst")
    assert store.authenticate("viewer", "viewer123") == {"username": "viewer", "role": "analyst"}
    with pytest.raises(UserError):
        store.set_role("viewer", "root")


def test_persistence_across_reload(tmp_path):
    p = tmp_path / "users.json"
    s1 = UserStore(p)
    s1.create("bob", "secret99", "operator")
    s1.set_locked("viewer", True)
    s2 = UserStore(p, seed=False)  # reload from disk, no reseed
    assert s2.count() == 6
    assert s2.authenticate("bob", "secret99") is not None
    assert s2.authenticate("viewer", "viewer123") is None  # lock persisted
    assert s2.get("admin")["salt"]  # salt persisted


def test_corrupt_file_recovers(tmp_path):
    p = tmp_path / "users.json"
    p.write_text("{not json", encoding="utf-8")
    s = UserStore(p, seed=False)
    assert s.count() == 0  # corrupt file -> starts empty, next save repairs


def test_max_users(tmp_path):
    s = UserStore(tmp_path / "u.json", seed=False)
    for i in range(MAX_USERS):
        s.create(f"u{i}", "secret99", "viewer")
    with pytest.raises(UserError):
        s.create("one-too-many", "secret99", "viewer")


def test_create_returns_public_record(store):
    rec = store.create("bob", "secret99", "operator")
    assert "hash" not in rec and "salt" not in rec and "iterations" not in rec
    assert rec == {"username": "bob", "role": "operator",
                   "locked": False, "last_login": None, "created": rec["created"]}


def test_bruteforce_lockout(tmp_path):
    s = UserStore(tmp_path / "u.json", seed=False)
    s.create("bob", "secret99", "viewer")
    for _ in range(5):
        assert s.authenticate("bob", "wrongpw") is None
    # now locked out even with the correct password
    assert s.is_locked_out("bob") is True
    assert s.authenticate("bob", "secret99") is None


def test_lockout_expires(tmp_path):
    from bhairav.backend.users import AUTH_LOCKOUT_SEC
    s = UserStore(tmp_path / "u.json", seed=False, now=1000.0)
    s.create("bob", "secret99", "viewer")
    for _ in range(5):
        s.authenticate("bob", "wrongpw")
    assert s.is_locked_out("bob") is True
    s._now = s._now + AUTH_LOCKOUT_SEC + 1  # advance the injectable clock
    assert s.is_locked_out("bob") is False
    assert s.authenticate("bob", "secret99") is not None


def test_unknown_user_burns_pbkdf2(store, monkeypatch):
    """Unknown usernames must pay the same PBKDF2 cost as real ones, so login
    timing can't be used to enumerate accounts. Verified deterministically by
    counting `_verify_password` invocations (no flaky wall-clock asserts)."""
    calls = []
    orig = UserStore._verify_password

    def counting(*args, **kwargs):
        calls.append(args)
        return orig(*args, **kwargs)

    monkeypatch.setattr(UserStore, "_verify_password", staticmethod(counting))
    assert store.authenticate("ghost", "secret99") is None
    assert len(calls) == 1, "unknown username must still run a full PBKDF2 verify"
    # the dummy branch verifies against a *different* credential than the real
    # user would - the point is the work happens either way
    calls.clear()
    assert store.authenticate("viewer", "viewer123") is not None
    assert len(calls) == 1


def test_locked_user_burns_pbkdf2(store, monkeypatch):
    """Admin-locked accounts also pay the verify cost (no timing tell)."""
    calls = []
    orig = UserStore._verify_password

    def counting(*args, **kwargs):
        calls.append(args)
        return orig(*args, **kwargs)

    monkeypatch.setattr(UserStore, "_verify_password", staticmethod(counting))
    store.set_locked("viewer", True)
    assert store.authenticate("viewer", "viewer123") is None
    assert len(calls) == 1
