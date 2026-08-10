"""Phase 2 - pose estimation package.

A `PoseModel` turns per-frame detections into 17-keypoint skeletons:
  - `SyntheticPoseModel`  - ML-free, drives the offline demo/tests
  - `MediaPipePoseModel`  - real CCTV path (lazy `mediapipe` import)
"""
from .base import PoseModel
from .mediapipe_model import MediaPipePoseModel
from .synthetic import SyntheticPoseModel

__all__ = ["PoseModel", "SyntheticPoseModel", "MediaPipePoseModel"]
