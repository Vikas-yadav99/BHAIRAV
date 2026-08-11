"""Tests for the Phase 9 M1 real-footage validation harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from bhairav.config import AppConfig
from bhairav.eval.harness import (MetricCollector, ValidationSummary,
                                  check_thresholds, parse_thresholds,
                                  render_html, render_markdown,
                                  run_validation)
from bhairav.pipeline import build_engine
from bhairav.types import Alert, FrameState, Keypoint, Pose, Severity, Track


class StubDetector:
    """Deterministic fake detector: two stable tracks + poses on frame < 10."""

    fps = 30.0

    def __init__(self, frames: int = 30, with_pose: bool = True):
        self.frames = frames
        self.with_pose = with_pose

    def stream(self, source=None, max_frames=None, opener=None):
        n = max_frames if max_frames is not None else self.frames
        for i in range(n):
            tracks = [
                Track(1, (10.0, 20.0, 60.0, 120.0), "person", 0.9, 0),
                Track(2, (100.0, 30.0, 160.0, 130.0), "car", 0.85, 2),
            ]
            poses = []
            if self.with_pose and i < 10:
                poses = [Pose(1, [Keypoint(0.1, 0.2, 0.9)] * 17, 0.9)]
            yield FrameState(frame_id=i, timestamp=i / self.fps, tracks=tracks,
                             frame_w=320, frame_h=240,
                             frame=np.full((240, 320, 3), 60, np.uint8),
                             poses=poses)


def _engine():
    return build_engine(AppConfig.from_dict({}))


def test_collector_counts_detections_and_tracks():
    c = MetricCollector(nominal_fps=30.0)
    for i in range(20):
        c.observe(FrameState(frame_id=i, timestamp=i / 30.0,
                             tracks=[Track(1, (0, 0, 10, 10), "person", 0.9, 0)],
                             frame_w=320, frame_h=240), [])
    s = c.summary(wall_clock_sec=0.67)
    assert s.frames == 20
    assert s.detections_total == 20
    assert s.person_frames == 20
    assert s.total_tracks == 1
    assert s.mean_track_len == 20.0
    assert s.longest_track == 20


def test_collector_pose_and_simultaneous_counts():
    c = MetricCollector(nominal_fps=30.0)
    for i in range(10):
        st = next(StubDetector(frames=10).stream())
        c.observe(st, [])
    s = c.summary(wall_clock_sec=0.33)
    assert s.max_simultaneous == 2
    assert s.pose_coverage == 1.0
    assert s.pose_person_tracks == 1
    assert s.detections_by_class.get("person") == 10


def test_dropped_frame_detection():
    c = MetricCollector(nominal_fps=30.0)
    # 10 clean 30fps frames calibrate the median period...
    for i in range(10):
        c.observe(FrameState(frame_id=i, timestamp=i / 30.0,
                             tracks=[], frame_w=320, frame_h=240), [])
    assert c.dropped_frames == 0
    # ...then two 1s gaps (instead of 1/30s) are counted as drops
    c.observe(FrameState(frame_id=10, timestamp=10.0, tracks=[],
                         frame_w=320, frame_h=240), [])
    c.observe(FrameState(frame_id=11, timestamp=11.0, tracks=[],
                         frame_w=320, frame_h=240), [])
    assert c.dropped_frames == 2


def test_alerts_are_tallied_by_rule_and_severity():
    c = MetricCollector()
    a = Alert(rule="fight", zone=None, track_id=3, severity=Severity.RED,
              message="x", frame_id=1, timestamp=0.0)
    b = Alert(rule="chase", zone=None, track_id=1, severity=Severity.ORANGE,
              message="y", frame_id=1, timestamp=0.0)
    c.observe(FrameState(frame_id=1, timestamp=0.0, tracks=[], frame_w=10,
                         frame_h=10), [a, b])
    s = c.summary()
    assert s.alerts_total == 2
    assert s.alerts_by_rule == {"fight": 1, "chase": 1}
    assert s.alerts_by_severity == {"red": 1, "orange": 1}


def test_possible_handovers_estimated():
    c = MetricCollector(nominal_fps=30.0)
    box = (10.0, 20.0, 60.0, 120.0)
    for i in range(10):
        c.observe(FrameState(frame_id=i, timestamp=i / 30.0,
                             tracks=[Track(1, box, "person", 0.9, 0)],
                             frame_w=320, frame_h=240), [])
    # track 1 disappears at frame 10; track 3 appears in the same spot at 12
    c.observe(FrameState(frame_id=11, timestamp=11 / 30.0, tracks=[],
                         frame_w=320, frame_h=240), [])
    c.observe(FrameState(frame_id=12, timestamp=12 / 30.0,
                         tracks=[Track(3, box, "person", 0.9, 0)],
                         frame_w=320, frame_h=240), [])
    assert c.possible_handovers == 1


def test_summary_json_roundtrip():
    st = next(StubDetector(frames=1).stream())
    c = MetricCollector(nominal_fps=30.0)
    c.observe(st, [])
    summary = c.summary(wall_clock_sec=0.1)
    assert ValidationSummary.from_dict(summary.to_dict()) == summary


def test_thresholds_parse_and_check():
    thr = parse_thresholds("fps>=10,mean_track_len>=5,detections_total==2")
    assert thr == {"fps": (">=", 10.0), "mean_track_len": (">=", 5.0),
                   "detections_total": ("==", 2.0)}
    c = MetricCollector(nominal_fps=30.0)
    c.observe(next(StubDetector(frames=1).stream()), [])
    rows = check_thresholds(c.summary(wall_clock_sec=0.1), thr)
    by = {r["metric"]: r for r in rows}
    assert by["detections_total"]["ok"]        # exactly 2 detections
    assert by["fps"]["ok"]                     # 1 frame / 0.1s == 10.0 fps
    assert not by["mean_track_len"]["ok"]      # 2 one-frame tracks -> mean 1.0
    # unknown metric fails loudly instead of silently passing
    assert not check_thresholds(c.summary(), {"bogus_metric": (">=", 1)})[0]["ok"]


def test_run_validation_end_to_end_on_synthetic_scene():
    cfg = AppConfig.from_dict({})
    from bhairav.pipeline import make_detector
    det = make_detector(cfg, "blob", "blob")
    summary, alerts = run_validation(det, _engine(), source="blob",
                                     max_frames=120)
    assert summary.frames == 120
    assert summary.effective_fps > 0
    assert summary.detections_total > 0
    assert summary.total_tracks > 0
    assert isinstance(alerts, list)
    # the deterministic scene fires zone crossing + trespass within 120 frames
    assert summary.alerts_total > 0


def test_run_validation_with_stub_and_threshold_gate():
    det = StubDetector(frames=20)
    summary, _ = run_validation(det, _engine(), max_frames=20)
    rows = check_thresholds(summary, {"detections_total": (">=", 40),
                                      "mean_track_len": (">=", 5)})
    assert all(r["ok"] for r in rows)


def test_reports_render():
    c = MetricCollector(nominal_fps=30.0)
    c.observe(next(StubDetector(frames=1).stream()), [])
    s = c.summary(wall_clock_sec=0.05)
    md = render_markdown(s, label="smoke")
    assert "# BHAIRAV validation report - smoke" in md
    assert "Frames: 1" in md
    html = render_html(s, label="smoke",
                       checks=check_thresholds(s, {"frames": ("==", 1)}))
    assert "VALIDATION PASSED" in html
    html_bad = render_html(s, checks=[{"metric": "fps", "ok": False,
                                       "expected": ">= 99", "actual": 1}])
    assert "VALIDATION FAILED" in html_bad
