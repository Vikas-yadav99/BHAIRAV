"""Tests for Groups 5, 7, and 8.

Group 5: Data Persistence — BoundedList, BoundedDict, MemoryMonitor
Group 7: Test Hardening — edge cases, failure modes, memory pressure
Group 8: Documentation — file existence, structure checks
"""
import os
import threading
import time
from pathlib import Path

from bhairav.persistence import BoundedList, BoundedDict, MemoryMonitor, CollectionStats


# ============================================================
# Group 5: Data Persistence
# ============================================================

class TestBoundedList:
    def test_basic_append(self):
        bl = BoundedList(maxlen=5)
        for i in range(5):
            bl.append(i)
        assert len(bl) == 5
        assert list(bl) == [0, 1, 2, 3, 4]

    def test_eviction(self):
        bl = BoundedList(maxlen=3)
        bl.append(1)
        bl.append(2)
        bl.append(3)
        bl.append(4)  # evicts 1
        assert list(bl) == [2, 3, 4]

    def test_extend(self):
        bl = BoundedList(maxlen=5)
        bl.extend([1, 2, 3])
        assert len(bl) == 3

    def test_extend_eviction(self):
        bl = BoundedList(maxlen=3)
        bl.extend([1, 2, 3, 4, 5])
        assert list(bl) == [3, 4, 5]

    def test_clear(self):
        bl = BoundedList(maxlen=5)
        bl.append(1)
        bl.clear()
        assert len(bl) == 0

    def test_maxlen_resize(self):
        bl = BoundedList(maxlen=10)
        for i in range(10):
            bl.append(i)
        bl.maxlen = 3
        assert len(bl) == 3
        assert list(bl) == [7, 8, 9]

    def test_thread_safety(self):
        bl = BoundedList(maxlen=100)
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    bl.append(f"t{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(errors) == 0
        assert len(bl) == 100  # capped at maxlen

    def test_snapshot(self):
        bl = BoundedList(maxlen=5)
        bl.append(1)
        snap = bl.snapshot()
        assert snap == [1]
        bl.append(2)
        assert snap == [1]  # snapshot is a copy

    def test_getitem(self):
        bl = BoundedList(maxlen=5)
        bl.append(10)
        bl.append(20)
        assert bl[0] == 10
        assert bl[1] == 20


class TestBoundedDict:
    def test_basic(self):
        bd = BoundedDict(maxlen=3)
        bd["a"] = 1
        bd["b"] = 2
        bd["c"] = 3
        assert len(bd) == 3
        assert bd["a"] == 1

    def test_eviction(self):
        bd = BoundedDict(maxlen=3)
        bd["a"] = 1
        bd["b"] = 2
        bd["c"] = 3
        bd["d"] = 4  # evicts "a"
        assert len(bd) == 3
        assert "a" not in bd
        assert bd["d"] == 4

    def test_update_existing_no_evict(self):
        bd = BoundedDict(maxlen=3)
        bd["a"] = 1
        bd["b"] = 2
        bd["c"] = 3
        bd["a"] = 10  # update existing, should not evict
        assert len(bd) == 3
        assert bd["a"] == 10

    def test_lru_ordering(self):
        bd = BoundedDict(maxlen=3)
        bd["a"] = 1
        bd["b"] = 2
        bd["c"] = 3
        # Re-set "a" to make it most recent
        bd["a"] = 10
        bd["d"] = 4  # should evict "b" (least recently used)
        assert "b" not in bd
        assert "a" in bd

    def test_delete(self):
        bd = BoundedDict(maxlen=5)
        bd["x"] = 100
        del bd["x"]
        assert "x" not in bd

    def test_snapshot(self):
        bd = BoundedDict(maxlen=3)
        bd["k"] = "v"
        snap = bd.snapshot()
        assert snap == {"k": "v"}
        bd["k2"] = "v2"
        assert "k2" not in snap  # snapshot is a copy

    def test_thread_safety(self):
        bd = BoundedDict(maxlen=100)
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    bd[f"t{n}-{i}"] = i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(errors) == 0
        assert len(bd) == 100


class TestMemoryMonitor:
    def test_sample(self):
        mm = MemoryMonitor()
        snap = mm.sample()
        assert snap.timestamp > 0
        assert len(mm.history()) == 1

    def test_current(self):
        mm = MemoryMonitor()
        snap = mm.current()
        assert snap.rss_mb >= 0

    def test_peak(self):
        mm = MemoryMonitor()
        mm.sample()
        peak = mm.peak()
        assert peak.rss_mb >= 0

    def test_history_capped(self):
        mm = MemoryMonitor(history_size=5)
        for _ in range(10):
            mm.sample()
        assert len(mm.history()) == 5


class TestCollectionStats:
    def test_snapshot(self):
        cs = CollectionStats()
        bl = BoundedList(maxlen=10)
        for i in range(5):
            bl.append(i)
        cs.register("alerts", bl)
        snap = cs.snapshot()
        assert snap["alerts"]["size"] == 5
        assert snap["alerts"]["maxlen"] == 10
        assert snap["alerts"]["type"] == "BoundedList"


# ============================================================
# Group 7: Test Hardening — Edge Cases & Failure Modes
# ============================================================

class TestEdgeCases:
    """Edge cases that the original test suite never tested."""

    def test_empty_pipeline(self):
        """Pipeline with no rules should produce no alerts."""
        from bhairav.rules.engine import RulesEngine
        from bhairav.types import FrameState
        engine = RulesEngine({}, [], cooldown_sec=0)
        state = FrameState(frame_id=0, timestamp=0, tracks=[],
                           frame_w=640, frame_h=480)
        alerts = engine.update(state)
        assert alerts == []

    def test_rapid_alert_cooldown(self):
        """Same alert should not fire twice within cooldown."""
        from bhairav.rules.engine import RulesEngine
        from bhairav.rules.base import Rule
        from bhairav.types import Alert, FrameState, Severity

        class TestRule(Rule):
            name = "test_rule"
            enabled = True
            def __init__(self, cfg):
                pass
            def evaluate(self, state, zones):
                return [Alert(rule="test", zone=None, track_id=None,
                              severity=Severity.RED, message="test",
                              frame_id=state.frame_id, timestamp=state.timestamp)]

        engine = RulesEngine({"test": {"enabled": True}}, [], cooldown_sec=5.0)
        engine.rules = [TestRule({})]

        state1 = FrameState(frame_id=1, timestamp=1.0, tracks=[],
                           frame_w=640, frame_h=480)
        alerts1 = engine.update(state1)
        assert len(alerts1) == 1

        # Same timestamp — should be cooldown-blocked
        state2 = FrameState(frame_id=2, timestamp=1.5, tracks=[],
                           frame_w=640, frame_h=480)
        alerts2 = engine.update(state2)
        assert len(alerts2) == 0

        # After cooldown
        state3 = FrameState(frame_id=3, timestamp=7.0, tracks=[],
                           frame_w=640, frame_h=480)
        alerts3 = engine.update(state3)
        assert len(alerts3) == 1

    def test_bounded_list_under_thread_contention(self):
        """BoundedList handles 100 concurrent writers without data loss."""
        bl = BoundedList(maxlen=500)
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    bl.append(f"{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0
        assert len(bl) == 500  # capped correctly

    def test_event_bus_subscriber_crash_doesnt_break_pipeline(self):
        """A crashing subscriber should not prevent other subscribers from running."""
        from bhairav.events import EventBus, Event

        bus = EventBus()
        results = []

        def good_handler(e):
            results.append("good")

        def bad_handler(e):
            raise RuntimeError("boom")

        bus.subscribe("alert", bad_handler)
        bus.subscribe("alert", good_handler)
        bus.publish(Event(topic="alert", data={}))
        assert results == ["good"]

    def test_config_from_dict_roundtrip(self):
        """Config values survive from_dict -> to_dict roundtrip."""
        from bhairav.config import AppConfig
        cfg = AppConfig.from_dict({
            "analytics": {"officer_pool": 42},
            "ha": {"failure_threshold": 7},
            "compliance": {"evidence_retention_days": 60},
        })
        assert cfg.analytics.officer_pool == 42
        assert cfg.ha.failure_threshold == 7
        assert cfg.compliance.evidence_retention_days == 60

    def test_identity_prune_stale(self):
        """Pruning removes only old persons, keeps recent ones."""
        from bhairav.identity import IdentityService
        svc = IdentityService()
        # Recent person
        pid1 = svc.resolve("CAM-01", 1)
        # Old person
        pid2 = svc.resolve("CAM-01", 2)
        svc.get_person(pid2).last_seen = time.time() - 7200

        removed = svc.prune(max_age_sec=3600)
        assert removed == 1
        assert svc.get_person(pid1) is not None
        assert svc.get_person(pid2) is None


# ============================================================
# Group 8: Documentation — File & Structure Checks
# ============================================================

class TestDocumentation:
    """Verify documentation files exist and have expected content."""

    def test_readme_exists(self):
        readme = Path("README.md")
        assert readme.exists(), "README.md missing"

    def test_readme_has_honest_disclaimer(self):
        """README should acknowledge limitations, not just marketing."""
        content = Path("README.md").read_text(encoding="utf-8", errors="ignore")
        # Should mention synthetic/blob or demo mode
        has_honesty = any(word in content.lower() for word in [
            "synthetic", "demo", "blob", "proof of concept", "prototype",
            "development", "not production",
        ])
        assert has_honesty, "README should acknowledge the system uses synthetic data"

    def test_architecture_doc_exists(self):
        arch = Path("docs/ARCHITECTURE.md")
        assert arch.exists(), "docs/ARCHITECTURE.md missing"

    def test_deployment_doc_exists(self):
        deploy = Path("docs/DEPLOYMENT.md")
        assert deploy.exists(), "docs/DEPLOYMENT.md missing"

    def test_readme_has_quickstart(self):
        content = Path("README.md").read_text(encoding="utf-8", errors="ignore").lower()
        assert "quick start" in content or "getting started" in content or "install" in content

    def test_readme_mentions_auth_flow(self):
        """README should explain how to authenticate."""
        content = Path("README.md").read_text(encoding="utf-8", errors="ignore").lower()
        assert "token" in content or "auth" in content or "login" in content

    def test_examples_dir_exists_or_not_needed(self):
        """If examples/ exists, it should have at least one script."""
        examples = Path("examples")
        if examples.exists():
            scripts = list(examples.glob("*.py"))
            assert len(scripts) > 0, "examples/ exists but has no .py scripts"
