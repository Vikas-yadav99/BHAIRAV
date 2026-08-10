"""End-to-end integration: blob detector + rules over the full scripted scene."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from bhairav.config import load_config
from bhairav.pipeline import build_engine, make_detector, run_pipeline


def test_full_pipeline_fires_all_alert_types():
    cfg = load_config("config.yaml")
    detector = make_detector(cfg, "blob", "blob")
    engine = build_engine(cfg)

    frames_seen = []
    seen_rules = set()
    seen_severities = set()

    def on_frame(state, alerts):
        frames_seen.append(state)
        seen_rules.update(a.rule for a in alerts)
        seen_severities.update(a.severity.value for a in alerts)

    run_pipeline(detector, engine, source="blob", on_frame=on_frame)

    assert len(frames_seen) > 200                       # full ~24s scene at 15fps
    # Phase 1 rules
    assert {"loitering", "zone_crossing", "crowd_density"} <= seen_rules
    # Phase 2 behavior rules
    assert {"fall", "fight", "chase", "trespass", "anomaly"} <= seen_rules
    assert "yellow" in seen_severities                  # loitering base + anomaly
    assert "orange" in seen_severities                  # escalation + crowd
    assert "red" in seen_severities                     # fight / zone / escalation


def test_detector_frames_are_valid_images_with_poses():
    cfg = load_config("config.yaml")
    detector = make_detector(cfg, "blob", "blob")
    engine = build_engine(cfg)
    out = run_pipeline(detector, engine, source="blob", max_frames=3)
    assert out == []
    # re-check frames via direct stream
    det2 = make_detector(cfg, "blob", "blob")
    state = next(det2.stream())
    assert state.frame is not None
    assert state.frame.shape == (cfg.synthetic.height, cfg.synthetic.width, 3)
    assert state.frame.dtype == np.uint8
    assert len(state.poses) == len(state.tracks)        # Phase 2: a skeleton per track
    assert all(len(p.keypoints) == 17 for p in state.poses)


def test_yolo_without_ultralytics_raises_clean_error():
    cfg = load_config("config.yaml")
    import pytest
    with pytest.raises(RuntimeError, match="ultralytics"):
        make_detector(cfg, "yolo", "clip.mp4")
