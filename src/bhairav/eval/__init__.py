"""Real-footage validation harness (Phase 9 M1).

Feeds the live pipeline (detector -> tracker -> rules) over any source and
turns the resulting frame/track/alert stream into hard metrics: effective
FPS, detection rates per class, track continuity and fragmentation, pose
coverage, alert counts by rule/severity and dropped-frame accounting. A
threshold checker turns the metrics into a pass/fail report, so the same
harness drives a local smoke check, a nightly regression run, or a CI gate
on real CCTV footage.

Pure numpy/cv2 + stdlib: no ML dependency to import the module; the YOLO
path stays lazy inside scripts/validate_footage.py.
"""
from .harness import (MetricCollector, ValidationSummary, check_thresholds,
                      parse_thresholds, render_html, render_markdown,
                      run_validation)

__all__ = [
    "MetricCollector",
    "ValidationSummary",
    "check_thresholds",
    "parse_thresholds",
    "render_html",
    "render_markdown",
    "run_validation",
]
