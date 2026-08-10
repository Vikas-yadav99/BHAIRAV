"""Detector interface: yields FrameState per processed frame."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import FrameState


class Detector(ABC):
    @property
    @abstractmethod
    def fps(self) -> float: ...

    @abstractmethod
    def stream(self, source: str | None = None, max_frames: int | None = None,
               opener=None):
        """Yield a FrameState per frame (with a raw BGR frame when available).

        `opener` (callable -> opened cv2.VideoCapture) lets the sources layer
        retry live-stream opens; detectors that synthesize frames ignore it.
        """
