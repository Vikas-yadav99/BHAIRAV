"""Detector package."""
from .base import Detector
from .blob_detector import BlobDetector
from .scenario import Scenario, default_scenario
from .yolo_detector import YoloDetector

__all__ = ["Detector", "BlobDetector", "YoloDetector", "Scenario", "default_scenario"]
