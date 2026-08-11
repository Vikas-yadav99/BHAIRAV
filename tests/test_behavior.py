"""Unit tests for Phase 2 behavior intelligence: kinematics, pose, and the
fall / fight / chase / trespass / anomaly classifiers."""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from bhairav.behavior import AnomalyRule, ChaseRule, FallRule, FightRule, MotionBuffer, TrespassRule


def _IMPORTABLE(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False
from bhairav.detectors.scenario import PersonSpec, ScenePosition
from bhairav.pose import SyntheticPoseModel
from bhairav.types import FrameState, Keypoint, Pose, Severity, Track, Zone

W, H = 1280, 720
DT = 1.0 / 15.0

PLAZA = Zone("plaza", "monitored", [(0.30, 0.30), (0.60, 0.30), (0.60, 0.55), (0.30, 0.55)])
SERVER = Zone("server_room", "restricted", [(0.70, 0.35), (0.92, 0.35), (0.92, 0.62), (0.70, 0.62)])
PLAZA_CENTER = (576, 306)
SERVER_CENTER = (1056, 349)


def tr(tid, cx, cy, label="person", w=44, h=88):
    return Track(tid, (cx - w / 2, cy - h, cx + w / 2, cy), label, 0.9, 0 if label == "person" else 2)


def frame(t, tracks, poses=None, fid=0):
    return FrameState(frame_id=fid, timestamp=t, tracks=tracks, frame_w=W, frame_h=H,
                      frame=None, poses=poses or [])


# ---- MotionBuffer ----------------------------------------------------------
def test_motion_buffer_speed_and_heading():
    buf = MotionBuffer(window_sec=1.0)
    t = 0.0
    for _ in range(15):
        buf.push(1, t, 100 + 80 * t, 200)
        t += DT
    speed = buf.speed(1, t)
    assert speed is not None and 75 < speed < 85
    heading = buf.heading_deg(1, t)
    assert heading is not None and abs(heading) < 5


def test_motion_buffer_mean_speed_captures_oscillation():
    """Oscillatory motion has ~zero net displacement but real per-step speed."""
    buf = MotionBuffer(window_sec=1.0)
    t = 0.0
    for _ in range(20):
        buf.push(1, t, 500 + 15 * math.sin(2 * math.pi * 2 * t), 400)
        t += DT
    assert buf.speed(1, t) < 30  # window-averaged: near-zero net displacement
    assert buf.mean_speed(1, t) > 100  # per-step speed is substantial


def test_motion_buffer_peak_downward_vy():
    buf = MotionBuffer(window_sec=1.0)
    t = 0.0
    for _ in range(10):
        buf.push(1, t, 300, 400)
        t += DT
    for i in range(8):  # rapid drop
        buf.push(1, t, 300, 400 + 30 * i)
        t += DT
    assert buf.peak_downward_vy(1, t) > 200


# ---- Pose geometry ---------------------------------------------------------
def test_pose_horizontal_angle():
    upright = Pose(track_id=1, keypoints=[Keypoint(x, y, 0.9) for x, y in [
        (0.5, 0.20), (0.49, 0.20), (0.51, 0.20), (0.48, 0.20), (0.52, 0.20),
        (0.48, 0.25), (0.52, 0.25), (0.45, 0.30), (0.55, 0.30), (0.43, 0.34),
        (0.57, 0.34), (0.48, 0.40), (0.52, 0.40), (0.46, 0.50), (0.54, 0.50),
        (0.45, 0.60), (0.55, 0.60)]])
    assert upright.horizontal_angle_deg() < 15
    lying = Pose(track_id=1, keypoints=[Keypoint(x, y, 0.9) for x, y in [
        (0.50, 0.30), (0.49, 0.30), (0.51, 0.30), (0.48, 0.30), (0.52, 0.30),
        (0.42, 0.30), (0.52, 0.30), (0.40, 0.32), (0.54, 0.32), (0.38, 0.34),
        (0.56, 0.34), (0.47, 0.30), (0.53, 0.30), (0.49, 0.32), (0.51, 0.32),
        (0.50, 0.34), (0.50, 0.34)]])
    assert lying.horizontal_angle_deg() > 70


def test_synthetic_pose_model_returns_17_keypoints():
    spec = PersonSpec(1, "person", 0, [])
    # track centroid sits ~half a body height above the feet position
    track = tr(1, 0.4 * W, 0.6 * H)
    model = SyntheticPoseModel([ScenePosition(spec, 0.4, (0.6 * H - 44) / H)], t=1.0)
    poses = model.estimate(frame(1.0, [track]))
    assert len(poses) == 1
    assert len(poses[0].keypoints) == 17


# ---- FallRule --------------------------------------------------------------
def _run_fall(rule, down_frames=45):
    t = 0.0
    cx, cy, w, h = 300.0, 400.0, 44, 88
    fired = []
    for _ in range(15):  # idle walking (flat y)
        fired += rule.evaluate(frame(t, [tr(1, cx, cy, w=w, h=h)]), [])
        t += DT
    for k in range(10):  # fall: y drops and bbox flattens
        f = (k + 1) / 10.0
        fired += rule.evaluate(
            frame(t, [tr(1, cx, cy + 90 * f, w=w * (1 + 0.9 * f), h=h * (1 - 0.55 * f))]), [])
        t += DT
    for _ in range(down_frames):  # lying still
        fired += rule.evaluate(frame(t, [tr(1, cx, cy + 90, w=w * 1.9, h=h * 0.45)]), [])
        t += DT
    return fired


def test_fall_fires_orange_then_escalates_red():
    rule = FallRule({"enabled": True, "vy_thresh_norm": 0.10, "down_sec": 0.5})
    fired = _run_fall(rule)
    sevs = [a.severity for a in fired]
    assert any(s == Severity.ORANGE for s in sevs)
    assert any(s == Severity.RED for s in sevs)
    assert all(0.5 <= a.confidence <= 0.95 for a in fired)


def test_fall_silent_for_normal_walker():
    rule = FallRule({"enabled": True})
    t = 0.0
    for _ in range(60):
        assert rule.evaluate(frame(t, [tr(1, 100 + 60 * t, 400)]), []) == []
        t += DT


def test_fall_detected_via_pose_without_bbox_flatten():
    """A horizontal pose alone can confirm the fall."""
    rule = FallRule({"enabled": True, "vy_thresh_norm": 0.10})
    t = 0.0
    lying = [Keypoint(0.45 + 0.05 * i, 0.62) for i in range(17)]
    for k in range(10):  # downward drop
        f = (k + 1) / 10.0
        rule.evaluate(frame(t, [tr(1, 300, 400 + 60 * f, w=44, h=88)],
                            poses=[Pose(track_id=1, keypoints=lying)]), [])
        t += DT
    fired = []
    for _ in range(30):
        fired += rule.evaluate(frame(t, [tr(1, 300, 460, w=44, h=88)],
                                     poses=[Pose(track_id=1, keypoints=lying)]), [])
        t += DT
    assert any(a.rule == "fall" for a in fired)


# ---- FightRule -------------------------------------------------------------
def test_fight_fires_for_jostling_pair():
    rule = FightRule({"enabled": True, "duration_sec": 1.0})
    t = 0.0
    fired = []
    for _ in range(60):
        th = 2 * math.pi * 2.0 * t
        fired += rule.evaluate(frame(t, [tr(1, 500 + 15 * math.sin(th), 400),
                                         tr(2, 590 + 15 * math.sin(th + math.pi), 400)]), [])
        t += DT
    assert fired
    assert fired[0].rule == "fight"
    assert fired[0].severity == Severity.RED
    assert fired[0].confidence >= 0.6


def test_fight_ignores_distant_or_straight_walkers():
    rule = FightRule({"enabled": True, "duration_sec": 0.5})
    t = 0.0
    for _ in range(45):  # fast walkers passing straight: no wobble
        assert rule.evaluate(frame(t, [tr(1, 300 + 120 * t, 400),
                                       tr(2, 700 - 120 * t, 400)]), []) == []
        t += DT
    for _ in range(30):  # jostling but far apart
        assert rule.evaluate(frame(t, [tr(1, 200 + 15 * math.sin(2 * math.pi * 2 * t), 300),
                                       tr(2, 900 + 15 * math.sin(2 * math.pi * 2 * t), 600)]), []) == []
        t += DT


# ---- ChaseRule -------------------------------------------------------------
def test_chase_fires_for_pursuit():
    rule = ChaseRule({"enabled": True, "duration_sec": 1.0})
    t = 0.0
    fired = []
    for _ in range(75):
        rx = 900 - 90 * t
        fired += rule.evaluate(frame(t, [tr(1, rx, 300), tr(2, rx + 80, 300)]), [])
        t += DT
    assert any(a.rule == "chase" for a in fired)
    assert fired[0].severity == Severity.ORANGE


def test_chase_silent_for_head_on_walkers():
    rule = ChaseRule({"enabled": True, "duration_sec": 0.5})
    t = 0.0
    for _ in range(45):  # runner moves TOWARD the follower (not fleeing)
        assert rule.evaluate(frame(t, [tr(1, 500 + 90 * t, 300),
                                       tr(2, 800 - 90 * t, 300)]), []) == []
        t += DT


def test_chase_silent_for_stationary_follower():
    rule = ChaseRule({"enabled": True, "duration_sec": 0.5})
    t = 0.0
    for _ in range(45):
        assert rule.evaluate(frame(t, [tr(1, 900 - 90 * t, 300), tr(2, 980, 300)]), []) == []
        t += DT


# ---- TrespassRule ----------------------------------------------------------
def test_trespass_fires_after_dwell():
    rule = TrespassRule({"enabled": True, "dwell_sec": 2.0})
    t = 0.0
    fired = []
    for _ in range(60):
        fired += rule.evaluate(frame(t, [tr(2, *SERVER_CENTER)]), [SERVER])
        t += DT
    assert any(a.rule == "trespass" for a in fired)
    assert fired[0].severity == Severity.ORANGE


def test_trespass_resets_after_exit():
    rule = TrespassRule({"enabled": True, "dwell_sec": 2.0})
    t = 0.0
    for _ in range(10):
        rule.evaluate(frame(t, [tr(2, *SERVER_CENTER)]), [SERVER])
        t += DT
    for _ in range(30):  # left the zone
        rule.evaluate(frame(t, [tr(2, 200, 500)]), [SERVER])
        t += DT
    assert rule.evaluate(frame(t, [tr(2, *SERVER_CENTER)]), [SERVER]) == []  # clock reset


# ---- AnomalyRule -----------------------------------------------------------
def test_anomaly_fires_on_count_spike_after_warmup():
    rule = AnomalyRule({"enabled": True, "warmup_frames": 30, "z_thresh": 3.0, "min_count": 2})
    t = 0.0
    for _ in range(30):  # calm baseline: 1 person
        rule.evaluate(frame(t, [tr(1, *PLAZA_CENTER)]), [PLAZA])
        t += DT
    fired = []
    for _ in range(20):  # sudden gathering
        people = [tr(i + 1, 500 + i * 30, 306) for i in range(6)]
        fired += rule.evaluate(frame(t, people), [PLAZA])
        t += DT
    assert any(a.rule == "anomaly" for a in fired)
    assert fired[0].severity == Severity.YELLOW


def test_anomaly_silent_while_stable():
    rule = AnomalyRule({"enabled": True, "warmup_frames": 20, "z_thresh": 3.0, "min_count": 2})
    t = 0.0
    people = [tr(i + 1, 500 + i * 30, 306) for i in range(3)]
    for _ in range(60):
        assert rule.evaluate(frame(t, people), [PLAZA]) == []
        t += DT


def test_fight_ignores_stationary_bystander():
    rule = FightRule({"enabled": True, "duration_sec": 1.0})
    t = 0.0
    for _ in range(60):  # jostler + stationary person right next to it
        th = 2 * math.pi * 2.0 * t
        a = tr(1, 500 + 15 * math.sin(th), 400)
        b = tr(2, 530, 400)  # bystander, still
        assert rule.evaluate(frame(t, [a, b]), []) == []
        t += DT


# ---- escalation + serialization + error paths ------------------------------
def test_chase_escalates_to_red():
    rule = ChaseRule({"enabled": True, "duration_sec": 0.5, "escalate": True})
    t = 0.0
    fired = []
    for _ in range(120):  # 8 s of sustained pursuit
        rx = 900 - 90 * t
        fired += rule.evaluate(frame(t, [tr(1, rx, 300), tr(2, rx + 80, 300)]), [])
        t += DT
    assert any(a.severity == Severity.RED for a in fired)


def test_trespass_escalates_to_red():
    rule = TrespassRule({"enabled": True, "dwell_sec": 2.0, "escalate": True})
    t = 0.0
    fired = []
    for _ in range(180):  # 12 s inside the zone
        fired += rule.evaluate(frame(t, [tr(2, *SERVER_CENTER)]), [SERVER])
        t += DT
    assert any(a.severity == Severity.RED for a in fired)


def test_alert_to_dict_includes_confidence():
    FallRule({"enabled": True}).evaluate  # just reference the class exists
    from bhairav.types import Alert
    alert = Alert(rule="fall", zone=None, track_id=1, severity=Severity.ORANGE,
                  message="x", frame_id=1, timestamp=1.0, confidence=0.83)
    assert alert.to_dict()["confidence"] == 0.83


@pytest.mark.skipif(_IMPORTABLE("mediapipe"), reason="mediapipe installed here; error path covered on fresh envs")
def test_mediapipe_model_raises_clean_error_without_mediapipe():
    from bhairav.pose import MediaPipePoseModel
    with pytest.raises(RuntimeError, match="mediapipe"):
        MediaPipePoseModel()


def test_pose_shoulder_hip_axis_none_when_low_confidence():
    kps = [Keypoint(0.1 * i, 0.1, 0.05) for i in range(17)]  # all below 0.1 conf
    pose = Pose(track_id=1, keypoints=kps)
    assert pose.shoulder_hip_axis() is None
    assert pose.horizontal_angle_deg() is None


def test_motion_buffer_prune_removes_stale_tracks():
    buf = MotionBuffer(window_sec=1.0, max_age_sec=2.0)
    t = 0.0
    for _ in range(10):
        buf.push(1, t, 100, 200)
        buf.push(2, t, 300, 400)
        t += DT
    buf.prune({1}, now=t + 5.0)  # track 2 gone, track 1 still active but stale
    # after prune, both are stale relative to now; only explicit active ids kept
    assert buf.speed(1, t + 5.0) is None
    buf2 = MotionBuffer(window_sec=1.0, max_age_sec=2.0)
    for i in range(10):
        buf2.push(1, i * 0.1, 100 + i, 200)
    buf2.prune({1}, now=1.0)  # fresh + active -> retained
    assert buf2.speed(1, 1.0) is not None
