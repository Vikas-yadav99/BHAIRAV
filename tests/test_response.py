"""Tests for Phase 17: PTZ, escalation, reports, tenants, integrations."""
from __future__ import annotations

import time
import pytest

from bhairav.response.ptz import (
    PTZController, PTZTracker, PTZCommand, PTZPreset, TrackingTarget,
)
from bhairav.response.escalation import (
    EscalationEngine, EscalationRule, EscalationEvent,
)
from bhairav.response.reports import ReportGenerator, IncidentReport
from bhairav.response.tenant import TenantManager, Tenant
from bhairav.response.integrations import (
    IntegrationHub, ExternalChannel, IntegrationEvent,
)


# ── PTZ ──────────────────────────────────────────────────────────────

class TestPTZController:
    def test_initial_state(self):
        ctrl = PTZController("CAM-01")
        assert ctrl.state.pan == 0.0
        assert ctrl.state.tilt == 0.0
        assert ctrl.state.zoom == 1.0
        assert not ctrl.state.moving

    def test_pan_left(self):
        ctrl = PTZController("CAM-01")
        result = ctrl.move(PTZCommand.PAN_LEFT, speed=0.5)
        assert ctrl.state.pan < 0
        assert ctrl.state.moving
        assert result["command"] == "pan_left"

    def test_pan_right(self):
        ctrl = PTZController("CAM-01")
        result = ctrl.move(PTZCommand.PAN_RIGHT, speed=0.5)
        assert ctrl.state.pan > 0

    def test_tilt_up(self):
        ctrl = PTZController("CAM-01")
        ctrl.move(PTZCommand.TILT_UP, speed=0.5)
        assert ctrl.state.tilt < 0

    def test_tilt_down(self):
        ctrl = PTZController("CAM-01")
        ctrl.move(PTZCommand.TILT_DOWN, speed=0.5)
        assert ctrl.state.tilt > 0

    def test_zoom_in(self):
        ctrl = PTZController("CAM-01")
        ctrl.move(PTZCommand.ZOOM_IN, speed=0.5)
        assert ctrl.state.zoom > 1.0

    def test_zoom_out(self):
        ctrl = PTZController("CAM-01")
        ctrl.move(PTZCommand.ZOOM_IN, speed=1.0)
        ctrl.move(PTZCommand.ZOOM_OUT, speed=0.5)
        assert ctrl.state.zoom < 30.0

    def test_stop(self):
        ctrl = PTZController("CAM-01")
        ctrl.move(PTZCommand.PAN_LEFT, speed=1.0)
        assert ctrl.state.moving
        ctrl.stop()
        assert not ctrl.state.moving

    def test_preset(self):
        presets = [PTZPreset("park", 0, 0, 1.0), PTZPreset("gate", 45, 10, 2.0)]
        ctrl = PTZController("CAM-01", presets=presets)
        ctrl.go_to_preset("gate")
        assert ctrl.state.pan == 45
        assert ctrl.state.zoom == 2.0

    def test_command_log(self):
        ctrl = PTZController("CAM-01")
        ctrl.move(PTZCommand.PAN_LEFT)
        ctrl.move(PTZCommand.ZOOM_IN)
        assert len(ctrl.command_log) == 2
        assert ctrl.command_log[0]["command"] == "pan_left"


class TestPTZTracker:
    def test_tracks_target_off_center(self):
        ctrl = PTZController("CAM-01")
        tracker = PTZTracker(ctrl, center_threshold=0.05, update_interval_ms=0)
        target = TrackingTarget(track_id=1, center_x=0.8, center_y=0.6, bbox=(0.6, 0.4, 0.9, 0.7))
        result = tracker.update_target(target)
        assert result is not None
        assert result["command"] in ("pan_right", "tilt_down")

    def test_no_command_when_centered(self):
        ctrl = PTZController("CAM-01")
        tracker = PTZTracker(ctrl, center_threshold=0.2, update_interval_ms=0)
        target = TrackingTarget(track_id=1, center_x=0.5, center_y=0.5, bbox=(0.3, 0.3, 0.7, 0.7))
        result = tracker.update_target(target)
        assert result is None

    def test_stop_tracking(self):
        ctrl = PTZController("CAM-01")
        tracker = PTZTracker(ctrl, update_interval_ms=0)
        target = TrackingTarget(track_id=1, center_x=0.9, center_y=0.9, bbox=(0.01, 0.01, 0.03, 0.03))
        tracker.update_target(target)
        assert tracker.is_tracking
        tracker.stop_tracking()
        assert not tracker.is_tracking
        assert tracker.active_target is None

    def test_update_interval(self):
        ctrl = PTZController("CAM-01")
        tracker = PTZTracker(ctrl, update_interval_ms=1000)
        target = TrackingTarget(track_id=1, center_x=0.9, center_y=0.9, bbox=(0.01, 0.01, 0.03, 0.03))
        r1 = tracker.update_target(target)
        r2 = tracker.update_target(target)  # should be throttled
        assert r2 is None


# ── Escalation ───────────────────────────────────────────────────────

class TestEscalation:
    def test_fires_on_matching_alert(self):
        rule = EscalationRule(name="test", trigger_rules=["fight"],
                              trigger_severity="red", trigger_count=1,
                              actions=["notify"])
        engine = EscalationEngine(rules=[rule])
        alert = {"rule": "fight", "severity": "red", "zone": "plaza",
                 "timestamp": time.time()}
        events = engine.process_alert(alert)
        assert len(events) == 1
        assert events[0].rule_name == "test"
        assert "notify" in events[0].actions

    def test_no_fire_on_wrong_severity(self):
        rule = EscalationRule(name="test", trigger_rules=["fight"],
                              trigger_severity="red", trigger_count=1,
                              actions=["notify"])
        engine = EscalationEngine(rules=[rule])
        alert = {"rule": "fight", "severity": "yellow", "zone": "plaza",
                 "timestamp": time.time()}
        events = engine.process_alert(alert)
        assert len(events) == 0

    def test_cooldown_prevents_rapid_fire(self):
        rule = EscalationRule(name="test", trigger_rules=["fight"],
                              trigger_severity="red", trigger_count=1,
                              cooldown_sec=60, actions=["notify"])
        engine = EscalationEngine(rules=[rule])
        alert = {"rule": "fight", "severity": "red", "timestamp": time.time()}
        events1 = engine.process_alert(alert)
        events2 = engine.process_alert(alert)
        assert len(events1) == 1
        assert len(events2) == 0

    def test_count_threshold(self):
        rule = EscalationRule(name="test", trigger_rules=["fight"],
                              trigger_severity="red", trigger_count=3,
                              trigger_window_sec=60, actions=["lockdown"])
        engine = EscalationEngine(rules=[rule])
        now = time.time()
        for i in range(2):
            engine.process_alert({"rule": "fight", "severity": "red", "timestamp": now + i})
        events = engine.process_alert({"rule": "fight", "severity": "red", "timestamp": now + 2})
        assert len(events) == 1
        assert "lockdown" in events[0].actions

    def test_callback(self):
        received = []
        rule = EscalationRule(name="test", trigger_rules=["fight"],
                              trigger_severity="red", trigger_count=1,
                              actions=["notify"])
        engine = EscalationEngine(rules=[rule],
                                  on_escalate=lambda e: received.append(e))
        engine.process_alert({"rule": "fight", "severity": "red", "timestamp": time.time()})
        assert len(received) == 1

    def test_zone_filter(self):
        rule = EscalationRule(name="test", trigger_rules=["zone_crossing"],
                              trigger_severity="red", trigger_count=1,
                              zones=["server_room"], actions=["lockdown"])
        engine = EscalationEngine(rules=[rule])
        events = engine.process_alert(
            {"rule": "zone_crossing", "severity": "red", "zone": "plaza",
             "timestamp": time.time()})
        assert len(events) == 0
        events = engine.process_alert(
            {"rule": "zone_crossing", "severity": "red", "zone": "server_room",
             "timestamp": time.time()})
        assert len(events) == 1


# ── Reports ──────────────────────────────────────────────────────────

class TestReports:
    def test_create_report(self, tmp_path):
        rg = ReportGenerator(output_dir=str(tmp_path))
        report = rg.create_report("INC-001", "Fight detected", "red", zone="plaza")
        assert report.incident_id == "INC-001"
        assert report.severity == "red"

 
