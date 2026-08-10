"""Core data types shared across the BHAIRAV pipeline (Phases 1 & 2)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Severity(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


SEVERITY_ORDER = (Severity.GREEN, Severity.YELLOW, Severity.ORANGE, Severity.RED)

# COCO class ids used by the vision model
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Phase 2 - pose keypoints (COCO 17-keypoint order)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# Bone pairs (indices into the keypoint list) for stick-figure rendering.
POSE_BONES = [
    (0, 5), (0, 6),            # nose -> shoulders
    (5, 6),                    # shoulder line
    (5, 7), (7, 9),            # left arm
    (6, 8), (8, 10),           # right arm
    (5, 11), (6, 12),          # torso
    (11, 12),                  # hip line
    (11, 13), (13, 15),        # left leg
    (12, 14), (14, 16),        # right leg
]


@dataclass
class Keypoint:
    """A single 2D pose keypoint in normalized (0..1) frame coordinates."""

    x: float
    y: float
    confidence: float = 1.0


@dataclass
class Pose:
    """A person's 17-keypoint skeleton, normalized to the frame."""

    track_id: int
    keypoints: list[Keypoint]
    confidence: float = 1.0

    def point(self, idx: int) -> tuple[float, float] | None:
        kp = self.keypoints[idx]
        if kp.confidence < 0.1:
            return None
        return kp.x, kp.y

    def shoulder_hip_axis(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """The shoulder-midpoint -> hip-midpoint axis (pixel-ish normalized coords)."""
        ls = self.point(5)
        rs = self.point(6)
        lh = self.point(11)
        rh = self.point(12)
        if not (ls and rs and lh and rh):
            return None
        return ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2), ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    def horizontal_angle_deg(self) -> float | None:
        """Angle of the torso axis from vertical: 0 = upright, ~90 = lying flat."""
        axis = self.shoulder_hip_axis()
        if axis is None:
            return None
        (sx, sy), (hx, hy) = axis
        dx = sx - hx
        dy = sy - hy
        return math.degrees(math.atan2(abs(dx), abs(dy)))


@dataclass
class Detection:
    """A single bounding-box detection in pixel space (x1, y1, x2, y2)."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str = "person"


@dataclass
class Track:
    """A persistent object identity produced by the tracker."""

    track_id: int
    bbox: tuple[float, float, float, float]
    label: str
    confidence: float = 1.0
    class_id: int = 0  # COCO class id (0=person, 2/5/7=vehicles)

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def is_person(self) -> bool:
        return self.label == "person"


@dataclass
class Alert:
    """A rule violation event, ready to be logged / displayed / escalated."""

    rule: str
    zone: str | None
    track_id: int | None
    severity: Severity
    message: str
    frame_id: int
    timestamp: float
    details: dict = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "zone": self.zone,
            "track_id": self.track_id,
            "severity": self.severity.value,
            "message": self.message,
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "confidence": round(self.confidence, 3),
            "details": self.details,
        }


@dataclass
class FrameState:
    """Everything the rules engine + visualizer need for one frame."""

    frame_id: int
    timestamp: float
    tracks: list[Track]
    frame_w: int
    frame_h: int
    frame: np.ndarray | None = None  # raw source frame (BGR) if available
    poses: list[Pose] = field(default_factory=list)  # Phase 2: skeletons per track


@dataclass
class Zone:
    """A named polygon region, defined in normalized (0..1) coordinates."""

    name: str
    kind: str  # "monitored" (loitering/crowd) | "restricted" (crossing)
    points_norm: list[tuple[float, float]]

    def to_pixels(self, w: int, h: int) -> list[tuple[float, float]]:
        return [(x * w, y * h) for x, y in self.points_norm]
