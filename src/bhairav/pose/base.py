"""PoseModel interface: detection tracks -> 17-keypoint skeletons."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import FrameState, Pose


class PoseModel(ABC):
    @abstractmethod
    def estimate(self, state: FrameState) -> list[Pose]:
        """Return one Pose per person track in the frame (or an empty list)."""
