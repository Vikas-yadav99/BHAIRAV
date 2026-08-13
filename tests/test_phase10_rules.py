"""Unit tests for the Phase 10 proactive-scene rules: abandoned-object,
accident and riot detection."""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhairav.behavior.accident import AccidentRule
from bhairav.behavior.riot import RiotRule
from bhairav.rules.abandoned_object import AbandonedObjectRule
from bhairav.types import FrameState, Severity, Track, Zone

W, H = 1280, 720
DT = 1.0 / 15.0

PLAZA = Zone("plaza", "monitored", [(0.30, 0.30), (0.60, 0.30),
                                    (0.60, 0.55), (0.30, 0.55)])


def _frame(t, tracks, poses=None, fid=0):
    return FrameState(frame_id=fid, timestamp=round(t, 3), tracks=tracks,
                      frame_w=W, frame_h=H, frame=None, poses=poses or [])


def _person(tid, cx, cy, flat=False):
    if flat:
        return Track(tid, (cx - 80, cy - 26, cx + 80, cy), "person", 0.9, 0)
    return Track(tid, (cx - 22, cy - 59, cx + 22, cy), "person", 0.9, 0)


def _car(tid, cx, cy):
    return Track(tid, (cx - 83, cy - 43, cx + 83, cy), "car", 0.9, 2)


def _bag(tid, cx, cy):
    return Track(tid, (cx - 19, cy - 36, cx + 19, cy), "suitcase", 0.9, 28)


# ---- abandoned_object -----------------------------------------------------
def test_abandoned_fires_after_duration():
    rule = AbandonedObjectRule({"enabled": True, "abandon_sec": 1.0,
                                "still_speed_norm": 0.02})
    t = 0.0
    fired = []
    for _ in range(40):  # a static bag in the plaza, nobody nearby
        fired += rule.evaluate(_frame(t, [_bag(1, 640, 300)]), [PLAZA])
        t += DT
    assert any(a.rule == "abandoned_object" for a in fired)
    assert fired[0].severity == Severity.ORANGE
    assert fired[0].zone == "plaza"


def test_abandoned_escalates_to_red():
    rule = AbandonedObjectRule({"enabled": True, "abandon_sec": 1.0,
                                "escalate": True, "still_speed_norm": 0.02})
    t = 0.0
    fired = []
    for _ in range(90):  # 6 s unattended -> 2x abandon_sec -> red
        fired += rule.evaluate(_frame(t, [_bag(1, 640, 300)]), [PLAZA])
        t += DT
    assert any(a.severity == Severity.RED for a in fired)


def test_abandoned_pauses_while_attended():
    rule = AbandonedObjectRule({"enabled": True, "abandon_sec": 1.0,
                                "owner_dist_norm": 0.06,
                                "still_speed_norm": 0.02})
    t = 0.0
    fired = []
    # owner stands right next to the bag the whole time
    for _ in range(60):
        fired += rule.evaluate(_frame(t, [_bag(1, 640, 300),
                                          _person(9, 660, 320)]), [PLAZA])
        t += DT
    assert fired == []
    # owner walks away -> the clock starts and it fires
    for _ in range(20):
        fired += rule.evaluate(_frame(t, [_bag(1, 640, 300)]), [PLAZA])
        t += DT
    assert any(a.rule == "abandoned_object" for a in fired)


def test_abandoned_ignores_moving_bag_and_people():
    rule = AbandonedObjectRule({"enabled": True, "abandon_sec": 1.0,
                                "still_speed_norm": 0.02})
    t = 0.0
    # a bag carried across the zone keeps moving -> never abandoned
    for _ in range(40):
        fired = rule.evaluate(_frame(t, [_bag(1, 100 + 120 * t, 300)]), [PLAZA])
        assert fired == []
        t += DT
    # a person standing in the plaza is not an abandoned object
    for _ in range(40):
        assert rule.evaluate(_frame(t, [_person(2, 640, 300)]), [PLAZA]) == []
        t += DT
    # a bag outside the monitored zone is ignored
    for _ in range(40):
        assert rule.evaluate(_frame(t, [_bag(3, 900, 500)]), [PLAZA]) == []
        t += DT


# ---- accident -------------------------------------------------------------
def test_accident_fires_for_hard_stop_with_downed_victim():
    rule = AccidentRule({"enabled": True, "confirm_sec": 0.5,
                         "still_speed_norm": 0.02, "cruise_speed_norm": 0.06,
                         "impact_dist_norm": 0.10})
    t = 0.0
    fired = []
    cx = 900.0
    for _ in range(25):  # vehicle cruising at ~140 px/s
        fired += rule.evaluate(_frame(t, [_car(1, cx, 400),
                                          _person(2, 700, 460, flat=True)]), [])
        cx -= 9.3
        t += DT
    for _ in range(40):  # hard stop: vehicle now still, victim down beside it
        fired += rule.evaluate(_frame(t, [_car(1, cx, 400),
                                          _person(2, cx - 60, 470, flat=True)]), [])
        t += DT
    assert any(a.rule == "accident" for a in fired)
    assert fired[0].severity == Severity.RED
    assert fired[0].details["victim"] == 2


def test_accident_silent_for_moving_vehicle():
    rule = AccidentRule({"enabled": True, "confirm_sec": 0.5,
                         "still_speed_norm": 0.02, "cruise_speed_norm": 0.06})
    t = 0.0
    cx = 900.0
    for _ in range(60):  # never stops -> never an accident
        fired = rule.evaluate(_frame(t, [_car(1, cx, 400),
                                         _person(2, cx - 60, 470, flat=True)]), [])
        assert fired == []
        cx -= 9.3
        t += DT


def test_accident_silent_for_never_moving_vehicle():
    rule = AccidentRule({"enabled": True, "confirm_sec": 0.5,
                         "still_speed_norm": 0.02, "cruise_speed_norm": 0.06})
    t = 0.0
    for _ in range(60):  # parked from the start next to a fallen person
        fired = rule.evaluate(_frame(t, [_car(1, 700, 400),
                                         _person(2, 680, 470, flat=True)]), [])
        assert fired == []
        t += DT


def test_accident_silent_when_victim_standing():
    rule = AccidentRule({"enabled": True, "confirm_sec": 0.5,
                         "still_speed_norm": 0.02, "cruise_speed_norm": 0.06})
    t = 0.0
    cx = 900.0
    fired = []
    for _ in range(25):
        fired += rule.evaluate(_frame(t, [_car(1, cx, 400),
                                          _person(2, 700, 460)]), [])
        cx -= 9.3
        t += DT
    for _ in range(40):  # stopped, but the person is just standing there
        fired += rule.evaluate(_frame(t, [_car(1, cx, 400),
                                          _person(2, cx - 60, 460)]), [])
        t += DT
    assert not any(a.rule == "accident" for a in fired)


def test_accident_silent_without_victim():
    rule = AccidentRule({"enabled": True, "confirm_sec": 0.5,
                         "still_speed_norm": 0.02, "cruise_speed_norm": 0.06})
    t = 0.0
    cx = 900.0
    for _ in range(25):
        rule.evaluate(_frame(t, [_car(1, cx, 400)]), [])
        cx -= 9.3
        t += DT
    for _ in range(40):  # hard stop, nobody near
        assert rule.evaluate(_frame(t, [_car(1, cx, 400)]), []) == []
        t += DT


# ---- riot -----------------------------------------------------------------
def _milling_people(center, n, t):
    tracks = []
    for i in range(n):
        ph = i * 1.7
        x = center[0] + 10 * math.sin(2 * math.pi * 2.0 * t + ph)
        y = center[1] + 8 * math.sin(2 * math.pi * 1.6 * t + ph * 1.3)
        tracks.append(_person(i + 1, x, y))
    return tracks


def test_riot_fires_for_milling_cluster():
    rule = RiotRule({"enabled": True, "duration_sec": 1.0, "min_people": 4,
                     "cluster_radius_norm": 0.10, "speed_norm": 0.04,
                     "wobble_deg": 20.0})
    t = 0.0
    fired = []
    for _ in range(90):  # 6 s of agitated milling
        fired += rule.evaluate(_frame(t, _milling_people((500, 400), 4, t)), [])
        t += DT
    assert any(a.rule == "riot" for a in fired)
    assert fired[0].severity == Severity.RED
    assert fired[0].details["people"] == 4


def test_riot_silent_for_standing_crowd():
    rule = RiotRule({"enabled": True, "duration_sec": 1.0, "min_people": 4,
                     "cluster_radius_norm": 0.10, "speed_norm": 0.04,
                     "wobble_deg": 20.0})
    t = 0.0
    people = [_person(i + 1, 500 + 40 * i, 400 + 20 * (i % 2)) for i in range(4)]
    for _ in range(60):  # standing in a cluster: too slow to be a riot
        assert rule.evaluate(_frame(t, people), []) == []
        t += DT


def test_riot_silent_for_walking_formation():
    rule = RiotRule({"enabled": True, "duration_sec": 1.0, "min_people": 4,
                     "cluster_radius_norm": 0.10, "speed_norm": 0.04,
                     "wobble_deg": 20.0})
    t = 0.0
    for _ in range(60):  # marching in lockstep: high speed, no wobble
        tracks = [_person(i + 1, 200 + 100 * i + 120 * t, 300) for i in range(4)]
        assert rule.evaluate(_frame(t, tracks), []) == []
        t += DT


def test_riot_silent_below_min_people():
    rule = RiotRule({"enabled": True, "duration_sec": 0.5, "min_people": 4,
                     "cluster_radius_norm": 0.10, "speed_norm": 0.04,
                     "wobble_deg": 20.0})
    t = 0.0
    for _ in range(60):  # only 3 agitators
        assert rule.evaluate(_frame(t, _milling_people((500, 400), 3, t)), []) == []
        t += DT


def test_rules_registered_in_engine():
    from bhairav.rules.engine import RULES
    names = {name for name, _ in RULES}
    assert {"abandoned_object", "accident", "riot"} <= names
