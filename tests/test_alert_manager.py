"""Comprehensive tests for SmartAlertManager, DataRetention, and CircuitBreaker."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from bhairav.alert_manager import SmartAlertManager, AlertStats
from bhairav.data_retention import DataRetention, RetentionStats
from bhairav.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitOpenError,
    DegradedModeManager, RetryWithBackoff
)
from bhairav.types import Alert, FrameState, Severity


# ============================================================
# Helper fixtures
# ============================================================

def _make_alert(rule="fall", zone="plaza", track_id=1,
                severity=Severity.RED, confidence=0.9, frame_id=0,
                timestamp=None) -> Alert:
    return Alert(
        rule=rule, zone=zone, track_id=track_id,
        severity=severity, message=f"{rule} detected",
        frame_id=frame_id,
        timestamp=timestamp or time.time(),
        confidence=confidence,
    )


def _make_state(frame_id=0, timestamp=None) -> FrameState:
    return FrameState(
        frame_id=frame_id,
        timestamp=timestamp or time.time(),
        tracks=[],
        frame_w=640,
        frame_h=480,
    )


# ============================================================
# SmartAlertManager tests
# ============================================================

class TestSmartAlertManager:
    """Tests for the SmartAlertManager."""

    def test_confidence_filtering(self):
        """Low-confidence alerts are dropped."""
        mgr = SmartAlertManager({"min_confidence": 0.5, "sustained_frames": 1})

        high = _make_alert(confidence=0.9)
        low = _make_alert(confidence=0.3, track_id=2)

        result = mgr.process([high, low])
        assert len(result) == 1
        assert result[0].confidence == 0.9
        assert mgr.stats.dropped_confidence == 1

    def test_sustained_detection(self):
        """Alert must appear on N consecutive frames before emission."""
        mgr = SmartAlertManager({"sustained_frames": 3, "min_confidence": 0.0})

        state1 = _make_state(frame_id=1, timestamp=1.0)
        state2 = _make_state(frame_id=2, timestamp=1.1)
        state3 = _make_state(frame_id=3, timestamp=1.2)

        # Frame 1 — first appearance, not emitted yet
        r1 = mgr.process([_make_alert(timestamp=1.0)], state1)
        assert len(r1) == 0

        # Frame 2 — still accumulating
        r2 = mgr.process([_make_alert(timestamp=1.1)], state2)
        assert len(r2) == 0

        # Frame 3 — threshold met, now emits
        r3 = mgr.process([_make_alert(timestamp=1.2)], state3)
        assert len(r3) == 1
        assert r3[0].details["sustained_frames"] == 3

    def test_sustained_detection_immediate_when_one(self):
        """sustained_frames=1 emits immediately."""
        mgr = SmartAlertManager({"sustained_frames": 1})
        result = mgr.process([_make_alert()])
        assert len(result) == 1

    def test_stale_pending_expired(self):
        """Pending alerts that don't get follow-up frames are dropped."""
        mgr = SmartAlertManager({"sustained_frames": 3})

        # First appearance
        state1 = _make_state(frame_id=1, timestamp=1.0)
        mgr.process([_make_alert(timestamp=1.0)], state1)
        assert mgr.pending_count() == 1

        # Jump forward 6 seconds — should expire
        state2 = _make_state(frame_id=100, timestamp=7.0)
        mgr.process([], state2)
        assert mgr.pending_count() == 0

    def test_cooldown_prevents_rapid_fire(self):
        """Same alert can't re-fire within cooldown window."""
        mgr = SmartAlertManager({
            "cooldown_sec": 5.0,
            "sustained_frames": 1,
            "min_confidence": 0.0,
        })

        t = time.time()
        state = _make_state(timestamp=t)

        # First emission
        r1 = mgr.process([_make_alert(timestamp=t)], state)
        assert len(r1) == 1

        # Immediate re-fire — blocked by cooldown
        state2 = _make_state(timestamp=t + 0.5)
        r2 = mgr.process([_make_alert(timestamp=t + 0.5)], state2)
        assert len(r2) == 0
        assert mgr.stats.dropped_cooldown == 1

        # After cooldown
        state3 = _make_state(timestamp=t + 6.0)
        r3 = mgr.process([_make_alert(timestamp=t + 6.0)], state3)
        assert len(r3) == 1

    def test_escalation_bypasses_cooldown(self):
        """Severity escalation gets a shorter cooldown."""
        mgr = SmartAlertManager({
            "cooldown_sec": 30.0,
            "escalate_cooldown_sec": 2.0,
            "sustained_frames": 1,
            "min_confidence": 0.0,
        })

        t = time.time()

        # Orange alert
        r1 = mgr.process([_make_alert(severity=Severity.ORANGE, timestamp=t)],
                         _make_state(timestamp=t))
        assert len(r1) == 1

        # Red alert 3s later — normally blocked (30s cooldown), but escalation allows it
        r2 = mgr.process([_make_alert(severity=Severity.RED, timestamp=t + 3.0)],
                         _make_state(timestamp=t + 3.0))
        assert len(r2) == 1

    def test_global_rate_limit(self):
        """max_alerts_per_min caps total emissions."""
        mgr = SmartAlertManager({
            "max_alerts_per_min": 5,
            "sustained_frames": 1,
            "min_confidence": 0.0,
            "cooldown_sec": 0.0,  # disable per-alert cooldown
        })

        t = time.time()
        alerts = []
        for i in range(10):
            alerts.append(_make_alert(track_id=i, timestamp=t + i * 0.1))

        # Process all at once
        result = mgr.process(alerts, _make_state(timestamp=t))
        assert len(result) == 5
        assert mgr.stats.dropped_ratelimit == 5

    def test_dedup_same_rule_same_zone(self):
        """Two alerts of same rule in same zone get merged."""
        mgr = SmartAlertManager({
            "sustained_frames": 1,
            "min_confidence": 0.0,
            "cooldown_sec": 0.0,
            "dedup_radius": 0.15,
        })

        t = time.time()
        a1 = _make_alert(rule="fall", zone="plaza", severity=Severity.ORANGE, timestamp=t)
        a2 = _make_alert(rule="fall", zone="plaza", severity=Severity.RED, timestamp=t)

        result = mgr.process([a1, a2], _make_state(timestamp=t))
        # Should keep the higher severity (RED)
        assert len(result) == 1
        assert result[0].severity == Severity.RED

    def test_different_rules_not_deduped(self):
        """Different rules are never deduped."""
        mgr = SmartAlertManager({
            "sustained_frames": 1,
            "min_confidence": 0.0,
            "cooldown_sec": 0.0,
        })

        t = time.time()
        a1 = _make_alert(rule="fall", zone="plaza", timestamp=t)
        a2 = _make_alert(rule="fight", zone="plaza", timestamp=t)

        result = mgr.process([a1, a2], _make_state(timestamp=t))
        assert len(result) == 2

    def test_stats_tracking(self):
        """Stats accurately track all filtering decisions."""
        mgr = SmartAlertManager({
            "min_confidence": 0.7,
            "sustained_frames": 1,
        })

        alerts = [
            _make_alert(confidence=0.9),    # pass
            _make_alert(confidence=0.3, track_id=2),  # dropped (confidence)
            _make_alert(confidence=0.8, track_id=3),  # pass
        ]

        result = mgr.process(alerts, _make_state())
        stats = mgr.get_stats()

        assert stats["total_received"] == 3
        assert stats["total_emitted"] == 2
        assert stats["dropped_confidence"] == 1
        assert stats["acceptance_rate"] == pytest.approx(0.667, abs=0.01)

    def test_reset_clears_state(self):
        """Reset clears all pending, cooldowns, and stats."""
        mgr = SmartAlertManager({"sustained_frames": 3})
        mgr.process([_make_alert()], _make_state())
        assert mgr.pending_count() > 0

        mgr.reset()
        assert mgr.pending_count() == 0
        stats = mgr.get_stats()
        assert stats["total_received"] == 0


# ============================================================
# DataRetention tests
# ============================================================

class TestDataRetention:
    """Tests for the DataRetention cleanup system."""

    def test_cleanup_old_alerts(self):
        """Alerts older than retention_days are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_path = Path(tmpdir) / "alerts.jsonl"
            old_ts = time.time() - (40 * 86400)  # 40 days ago
            new_ts = time.time() - (1 * 86400)   # 1 day ago

            lines = [
                json.dumps({"rule": "fall", "timestamp": old_ts}),
                json.dumps({"rule": "fight", "timestamp": new_ts}),
                json.dumps({"rule": "chase", "timestamp": old_ts}),
            ]
            alert_path.write_text("\n".join(lines) + "\n")

            ret = DataRetention({
                "retention_days": 30,
                "alert_path": str(alert_path),
            })
            stats = ret.run_once()

            assert stats["alerts_deleted"] == 2
            remaining = alert_path.read_text().strip().split("\n")
            assert len(remaining) == 1
            assert "fight" in remaining[0]

    def test_cleanup_old_trajectories(self):
        """Trajectory entries older than retention_days are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_path = Path(tmpdir) / "trajectories.jsonl"
            old_ts = time.time() - (60 * 86400)
            new_ts = time.time() - (5 * 86400)

            lines = [
                json.dumps({"person_id": "P-1", "timestamp": old_ts}),
                json.dumps({"person_id": "P-2", "timestamp": new_ts}),
            ]
            traj_path.write_text("\n".join(lines) + "\n")

            ret = DataRetention({
                "retention_days": 30,
                "trajectory_path": str(traj_path),
            })
            ret.run_once()

            remaining = traj_path.read_text().strip()
            assert "P-2" in remaining
            assert "P-1" not in remaining

    def test_cleanup_old_evidence(self):
        """Evidence directories older than retention_days are deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_dir = Path(tmpdir) / "evidence"
            evidence_dir.mkdir()

            # Old event (40 days ago)
            old_event = evidence_dir / "old_event"
            old_event.mkdir()
            (old_event / "clip.mp4").write_bytes(b"x" * 1000)

            # New event (2 days ago)
            new_event = evidence_dir / "new_event"
            new_event.mkdir()
            (new_event / "clip.mp4").write_bytes(b"y" * 1000)

            # Set old event's mtime
            import os
            old_time = time.time() - (40 * 86400)
            os.utime(old_event, (old_time, old_time))

            ret = DataRetention({
                "retention_days": 30,
                "evidence_dir": str(evidence_dir),
            })
            ret.run_once()

            assert not old_event.exists()
            assert new_event.exists()

    def test_size_limit_enforcement(self):
        """JSONL files are trimmed when over size limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_path = Path(tmpdir) / "alerts.jsonl"
            # Create 1MB of alert data
            lines = []
            for i in range(2000):
                lines.append(json.dumps({
                    "rule": "fall",
                    "timestamp": time.time() - i,
                    "message": f"Alert {i} with some padding data " + "x" * 50,
                }))
            alert_path.write_text("\n".join(lines) + "\n")
            original_size = alert_path.stat().st_size

            ret = DataRetention({
                "retention_days": 365,  # don't age out
                "max_alert_mb": 0.0001,  # 0.1 KB — force trim
                "alert_path": str(alert_path),
            })
            ret.run_once()

            new_size = alert_path.stat().st_size
            assert new_size < original_size

    def test_disk_usage_report(self):
        """get_disk_usage reports correct sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            traj = Path(tmpdir) / "trajectories.jsonl"
            traj.write_text(json.dumps({"x": 1}) + "\n")

            alerts = Path(tmpdir) / "alerts.jsonl"
            alerts.write_text(json.dumps({"y": 2}) + "\n")

            evidence = Path(tmpdir) / "evidence"
            evidence.mkdir()
            (evidence / "test.mp4").write_bytes(b"z" * 1024)

            ret = DataRetention({
                "trajectory_path": str(traj),
                "alert_path": str(alerts),
                "evidence_dir": str(evidence),
            })
            usage = ret.get_disk_usage()

            assert usage["trajectory_mb"] >= 0
            assert usage["alerts_mb"] >= 0
            assert usage["evidence_mb"] >= 0
            assert usage["total_mb"] >= 0

    def test_background_thread(self):
        """start() and stop() work without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ret = DataRetention({
                "check_interval_sec": 0.1,
                "trajectory_path": str(Path(tmpdir) / "traj.jsonl"),
                "alert_path": str(Path(tmpdir) / "alerts.jsonl"),
                "evidence_dir": str(Path(tmpdir) / "evidence"),
            })
            ret.start()
            time.sleep(0.3)
            ret.stop()
            # No errors = pass

    def test_malformed_json_preserved(self):
        """Malformed JSON lines are preserved (not dropped)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_path = Path(tmpdir) / "alerts.jsonl"
            alert_path.write_text('not json\n{"rule":"fall","timestamp":' + str(time.time()) + '}\n')

            ret = DataRetention({
                "retention_days": 30,
                "alert_path": str(alert_path),
            })
            ret.run_once()

            content = alert_path.read_text()
            assert "not json" in content


# ============================================================
# CircuitBreaker tests
# ============================================================

class TestCircuitBreaker:
    """Tests for the CircuitBreaker."""

    def test_closed_state_passes_calls(self):
        """In CLOSED state, calls pass through normally."""
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        result = cb.call(lambda: 42)
        assert result == 42

    def test_opens_after_threshold_failures(self):
        """Circuit opens after consecutive failures reach threshold."""
        cb = CircuitBreaker("test", failure_threshold=3)

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        """In OPEN state, calls raise CircuitOpenError."""
        cb = CircuitBreaker("test", failure_threshold=2)

        # Force open
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: 42)
        assert exc_info.value.circuit_name == "test"

    def test_half_open_after_recovery_timeout(self):
        """After recovery_timeout, circuit transitions to HALF_OPEN."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        # Force open
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        """Successful call in HALF_OPEN closes the circuit."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        # Force open
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        time.sleep(0.15)
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """Failed call in HALF_OPEN reopens the circuit."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        # Force open
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("still broken")))

        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """A successful call resets consecutive failure count."""
        cb = CircuitBreaker("test", failure_threshold=3)

        # 2 failures
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Success resets count
        cb.call(lambda: "ok")

        # 2 more failures — not yet open (counter was reset)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.CLOSED  # not open yet

    def test_stats_tracking(self):
        """Stats track calls, failures, rejections, and successes."""
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=0.1)

        cb.call(lambda: "ok")  # success
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = cb.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 1

    def test_decorator_wrap(self):
        """wrap() decorator works correctly."""
        cb = CircuitBreaker("test", failure_threshold=3)

        @cb.wrap
        def flaky():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            flaky()

        assert cb.get_stats()["total_failures"] == 1

    def test_on_open_callback(self):
        """on_open callback fires when circuit opens."""
        opened = []
        cb = CircuitBreaker("test", failure_threshold=2, on_open=lambda n: opened.append(n))

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert opened == ["test"]

    def test_force_reset(self):
        """reset() forces circuit back to CLOSED."""
        cb = CircuitBreaker("test", failure_threshold=2)

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


# ============================================================
# DegradedModeManager tests
# ============================================================

class TestDegradedModeManager:
    """Tests for the DegradedModeManager."""

    def test_healthy_when_all_closed(self):
        """Reports healthy when all circuits are CLOSED."""
        dm = DegradedModeManager()
        cb1 = CircuitBreaker("yolo", failure_threshold=3)
        cb2 = CircuitBreaker("db", failure_threshold=3)
        dm.register("yolo", cb1)
        dm.register("database", cb2)

        assert not dm.is_degraded
        assert dm.degraded_components == []
        assert dm.get_health()["status"] == "healthy"

    def test_degraded_when_any_open(self):
        """Reports degraded when any circuit is OPEN."""
        dm = DegradedModeManager()
        cb1 = CircuitBreaker("yolo", failure_threshold=2)
        dm.register("yolo", cb1)

        # Force open
        for _ in range(2):
            with pytest.raises(ValueError):
                cb1.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert dm.is_degraded
        assert "yolo" in dm.degraded_components
        assert dm.get_health()["status"] == "degraded"

    def test_last_known_tracks(self):
        """Last-known tracks are stored and retrievable."""
        dm = DegradedModeManager()
        dm.update_last_known_tracks("CAM-01", [{"track_id": 1}])

        tracks = dm.get_last_known_tracks("CAM-01")
        assert len(tracks) == 1
        assert tracks[0]["track_id"] == 1

    def test_reset_all(self):
        """reset_all() closes all circuits."""
        dm = DegradedModeManager()
        cb = CircuitBreaker("test", failure_threshold=1)
        dm.register("test", cb)

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert dm.is_degraded
        dm.reset_all()
        assert not dm.is_degraded

    def test_health_report(self):
        """Health report includes all circuit stats."""
        dm = DegradedModeManager()
        dm.register("yolo", CircuitBreaker("yolo"))
        dm.register("db", CircuitBreaker("db"))

        health = dm.get_health()
        assert "yolo" in health["circuits"]
        assert "db" in health["circuits"]
        assert health["status"] == "healthy"


# ============================================================
# RetryWithBackoff tests
# ============================================================

class TestRetryWithBackoff:
    """Tests for the RetryWithBackoff utility."""

    def test_succeeds_first_try(self):
        """No retries when first call succeeds."""
        retry = RetryWithBackoff(max_retries=3, base_delay=0.01)
        result = retry.call(lambda: 42)
        assert result == 42

    def test_retries_on_failure(self):
        """Retries and eventually succeeds."""
        attempt = [0]
        def flaky():
            attempt[0] += 1
            if attempt[0] < 3:
                raise ValueError("not yet")
            return "done"

        retry = RetryWithBackoff(max_retries=3, base_delay=0.01, jitter=False)
        result = retry.call(flaky)
        assert result == "done"
        assert attempt[0] == 3

    def test_raises_after_all_retries(self):
        """Raises last exception when all retries exhausted."""
        def always_fail():
            raise ValueError("always fails")

        retry = RetryWithBackoff(max_retries=2, base_delay=0.01)
        with pytest.raises(ValueError, match="always fails"):
            retry.call(always_fail)

    def test_decorator(self):
        """wrap() decorator works."""
        retry = RetryWithBackoff(max_retries=1, base_delay=0.01)

        @retry.wrap
        def succeeds():
            return "ok"

        assert succeeds() == "ok"
