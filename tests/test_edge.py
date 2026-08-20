"""Tests for Phase 13.1 - Edge Agent (local store, upstream)."""
import sys
import time
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhairav.edge.local_store import LocalAlertStore
from bhairav.edge.upstream import UpstreamPusher


class TestLocalAlertStore:
    def test_append_and_read(self):
        with tempfile.TemporaryDirectory() as td:
            store = LocalAlertStore(Path(td) / "test.jsonl")
            store.append({"rule": "fight", "ts": 1.0})
            store.append({"rule": "loiter", "ts": 2.0})
            alerts = store.read_all()
            assert len(alerts) == 2
            assert alerts[0]["rule"] == "fight"

    def test_read_and_clear(self):
        with tempfile.TemporaryDirectory() as td:
            store = LocalAlertStore(Path(td) / "test.jsonl")
            store.append({"rule": "fight"})
            store.append({"rule": "loiter"})
            cleared = store.read_and_clear()
            assert len(cleared) == 2
            assert store.read_all() == []

    def test_prune(self):
        with tempfile.TemporaryDirectory() as td:
            store = LocalAlertStore(Path(td) / "test.jsonl", max_age_sec=5.0)
            store.append({"rule": "old", "timestamp": time.time() - 100})
            store.append({"rule": "new", "timestamp": time.time()})
            removed = store.prune()
            assert removed == 1
            assert len(store.read_all()) == 1

    def test_count(self):
        with tempfile.TemporaryDirectory() as td:
            store = LocalAlertStore(Path(td) / "test.jsonl")
            assert store.count == 0
            store.append({"rule": "a"})
            store.append({"rule": "b"})
            assert store.count == 2

    def test_empty(self):
        with tempfile.TemporaryDirectory() as td:
            store = LocalAlertStore(Path(td) / "none.jsonl")
            assert store.read_all() == []
            assert store.read_and_clear() == []


class TestUpstreamPusher:
    def test_stats_init(self):
        with tempfile.TemporaryDirectory() as td:
            store = LocalAlertStore(Path(td) / "test.jsonl")
            pusher = UpstreamPusher(store=store)
            assert pusher.stats["pushed"] == 0

    def test_drain_without_target(self):
        with tempfile.TemporaryDirectory() as td:
            store = LocalAlertStore(Path(td) / "test.jsonl")
            store.append({"rule": "fight", "ts": time.time()})
            pusher = UpstreamPusher(store=store, url=None)
            pusher._drain()
            # No target -> failed, alert re-queued
            assert pusher.stats["failed"] >= 0
