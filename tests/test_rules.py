from bhairav.rules import CrowdDensityRule, LoiteringRule, RulesEngine, ZoneCrossingRule
from bhairav.types import FrameState, Severity, Track, Zone

W, H = 1280, 720

PLAZA = Zone("plaza", "monitored", [(0.30, 0.30), (0.60, 0.30), (0.60, 0.55), (0.30, 0.55)])
SERVER = Zone("server_room", "restricted", [(0.70, 0.35), (0.92, 0.35), (0.92, 0.62), (0.70, 0.62)])

# plaza pixel bounds: x 384..768, y 216..396
PLAZA_CENTER = (576, 306)
SERVER_CENTER = (1056, 349)
OUTSIDE = (200, 500)


def make_state(t, tracks, fid=0):
    return FrameState(frame_id=fid, timestamp=t, tracks=tracks, frame_w=W, frame_h=H, frame=None)


def tr(tid, cx, cy, label="person", w=44, h=88, class_id=None):
    if class_id is None:
        class_id = 0 if label == "person" else 2
    return Track(tid, (cx - w / 2, cy - h, cx + w / 2, cy), label, 0.9, class_id)


# ---- loitering -------------------------------------------------------------
def test_loitering_fires_after_duration():
    rule = LoiteringRule({"enabled": True, "duration_sec": 5.0})
    person = tr(1, *PLAZA_CENTER)
    assert rule.evaluate(make_state(0.0, [person]), [PLAZA]) == []
    assert rule.evaluate(make_state(4.9, [person]), [PLAZA]) == []
    alerts = rule.evaluate(make_state(5.0, [person]), [PLAZA])
    assert len(alerts) == 1 and alerts[0].severity == Severity.YELLOW


def test_loitering_escalates_to_orange():
    rule = LoiteringRule({"enabled": True, "duration_sec": 5.0, "escalate": True})
    person = tr(1, *PLAZA_CENTER)
    rule.evaluate(make_state(0.0, [person]), [PLAZA])
    rule.evaluate(make_state(5.0, [person]), [PLAZA])
    alerts = rule.evaluate(make_state(10.0, [person]), [PLAZA])
    assert any(a.severity == Severity.ORANGE for a in alerts)


def test_loitering_resets_after_leaving():
    rule = LoiteringRule({"enabled": True, "duration_sec": 5.0})
    person = tr(1, *PLAZA_CENTER)
    rule.evaluate(make_state(0.0, [person]), [PLAZA])
    person2 = tr(1, *OUTSIDE)
    rule.evaluate(make_state(2.0, [person2]), [PLAZA])   # left
    rule.evaluate(make_state(4.0, [person2]), [PLAZA])   # grace over -> reset
    person3 = tr(1, *PLAZA_CENTER)
    assert rule.evaluate(make_state(4.1, [person3]), [PLAZA]) == []  # clock restarted
    alerts = rule.evaluate(make_state(9.2, [person3]), [PLAZA])
    assert len(alerts) == 1


def test_loitering_ignores_non_people():
    rule = LoiteringRule({"enabled": True, "duration_sec": 1.0})
    car = tr(1, *PLAZA_CENTER, label="vehicle", w=120, h=60)
    assert rule.evaluate(make_state(10.0, [car]), [PLAZA]) == []


# ---- zone crossing ---------------------------------------------------------
def test_crossing_fires_red_inside_restricted():
    rule = ZoneCrossingRule({"enabled": True, "severity": "red"})
    intruder = tr(2, *SERVER_CENTER)
    alerts = rule.evaluate(make_state(1.0, [intruder]), [SERVER])
    assert len(alerts) == 1 and alerts[0].severity == Severity.RED
    assert alerts[0].zone == "server_room"


def test_crossing_silent_outside():
    rule = ZoneCrossingRule({"enabled": True})
    passer = tr(2, *OUTSIDE)
    assert rule.evaluate(make_state(1.0, [passer]), [SERVER]) == []


def test_crossing_can_include_vehicles():
    rule = ZoneCrossingRule({"enabled": True, "include_vehicles": True})
    truck = tr(3, *SERVER_CENTER, label="vehicle", w=140, h=70)
    assert len(rule.evaluate(make_state(1.0, [truck]), [SERVER])) == 1
    rule_no = ZoneCrossingRule({"enabled": True, "include_vehicles": False})
    assert rule_no.evaluate(make_state(1.0, [truck]), [SERVER]) == []


# ---- crowd density ---------------------------------------------------------
def test_crowd_fires_at_threshold():
    rule = CrowdDensityRule({"enabled": True, "min_people": 4})
    crowd = [tr(i, *PLAZA_CENTER, w=20, h=40) for i in range(4)]
    alerts = rule.evaluate(make_state(1.0, crowd), [PLAZA])
    assert len(alerts) == 1 and alerts[0].severity == Severity.ORANGE


def test_crowd_silent_below_threshold():
    rule = CrowdDensityRule({"enabled": True, "min_people": 4})
    few = [tr(i, *PLAZA_CENTER, w=20, h=40) for i in range(3)]
    assert rule.evaluate(make_state(1.0, few), [PLAZA]) == []


def test_crowd_escalates_at_double_threshold():
    rule = CrowdDensityRule({"enabled": True, "min_people": 4, "escalate": True})
    big = [tr(i, *PLAZA_CENTER, w=20, h=40) for i in range(8)]
    alerts = rule.evaluate(make_state(1.0, big), [PLAZA])
    assert alerts[0].severity == Severity.RED


# ---- engine cooldown -------------------------------------------------------
def test_engine_dedupes_within_cooldown():
    eng = RulesEngine({"loitering": {"enabled": True, "duration_sec": 1.0, "escalate": False},
                       "zone_crossing": {"enabled": False}, "crowd_density": {"enabled": False}},
                      [PLAZA], cooldown_sec=10.0)
    person = tr(1, *PLAZA_CENTER)
    eng.update(make_state(0.0, [person]))                        # enters (clock starts)
    first = eng.update(make_state(1.0, [person]))                # duration 1.0 -> fires
    assert len(first) == 1
    assert eng.update(make_state(5.0, [person])) == []           # within cooldown
    refire = eng.update(make_state(11.0, [person]))              # cooldown elapsed
    assert len(refire) == 1


def test_engine_allows_severity_escalation_during_cooldown():
    eng = RulesEngine({"loitering": {"enabled": True, "duration_sec": 1.0, "escalate": True},
                       "zone_crossing": {"enabled": False}, "crowd_density": {"enabled": False}},
                      [PLAZA], cooldown_sec=10.0)
    person = tr(1, *PLAZA_CENTER)
    eng.update(make_state(0.0, [person]))      # enters
    yellow = eng.update(make_state(1.0, [person]))   # dur 1.0 -> yellow
    assert any(a.severity == Severity.YELLOW for a in yellow)
    alerts = eng.update(make_state(3.0, [person]))   # dur 3.0 >= 2x -> orange, different key
    assert any(a.severity == Severity.ORANGE for a in alerts)
