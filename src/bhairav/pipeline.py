"""Pipeline assembly: detector + rules engine wiring and the frame loop."""
from __future__ import annotations

from .config import AppConfig
from .rules import RulesEngine
from .types import Alert
from .detectors import Detector, BlobDetector, YoloDetector, default_scenario


def build_engine(app: AppConfig) -> RulesEngine:
    return RulesEngine(app.rules, app.zones, cooldown_sec=app.alert.cooldown_sec)


def make_detector(app: AppConfig, detector: str | None = None, source: str | None = None) -> Detector:
    """Choose a detector. `auto` -> blob for the synthetic source, yolo otherwise."""
    choice = (detector or app.detector).lower()
    if choice == "auto":
        choice = "blob" if source == "blob" else "yolo"
    if choice == "blob":
        return BlobDetector(default_scenario(app.synthetic), fps=app.synthetic.fps,
                            width=app.synthetic.width, height=app.synthetic.height)
    if choice == "yolo":
        return YoloDetector(app.model)
    raise ValueError(f"unknown detector: {choice} (expected blob | yolo | auto)")


def run_pipeline(detector: Detector, engine: RulesEngine, source: str | None = None,
                 max_frames: int | None = None, on_frame=None, opener=None) -> list[Alert]:
    """Run the frame loop; `on_frame(state, alerts)` may return False to stop early.

    `opener` (callable -> opened cv2.VideoCapture) is forwarded to the detector
    so the sources layer can retry live-stream opens with backoff.
    """
    all_alerts: list[Alert] = []
    for state in detector.stream(source=source, max_frames=max_frames, opener=opener):
        alerts = engine.update(state)
        all_alerts.extend(alerts)
        if on_frame is not None:
            if on_frame(state, alerts) is False:
                break
    return all_alerts
