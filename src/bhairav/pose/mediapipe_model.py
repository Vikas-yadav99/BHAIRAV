"""MediaPipe pose wrapper for the real CCTV path (lazy import).

Not exercised by tests or the offline demo - `mediapipe` is an optional
dependency. Falls back gracefully: constructing the model without mediapipe
raises a clear error, so callers can simply disable pose when it is missing.
"""
from __future__ import annotations

from ..types import FrameState, Keypoint, Pose


class MediaPipePoseModel:
    """Runs MediaPipe Pose on each frame and maps skeletons to tracks by
    bounding-box overlap with the detection boxes."""

    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "MediaPipePoseModel requires 'mediapipe' (pip install mediapipe). "
                "Use the synthetic path or install the dependency."
            ) from exc
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def estimate(self, state: FrameState) -> list[Pose]:
        import cv2

        if state.frame is None:
            return []
        rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        out: list[Pose] = []
        if not results.pose_landmarks:
            return out
        lm = results.pose_landmarks.landmark
        w, h = state.frame_w, state.frame_h
        kps = [Keypoint(x=p.x, y=p.y, confidence=p.visibility) for p in lm]
        # Map to the most overlapping person track.
        bbox = (
            min(p.x for p in lm) * w,
            min(p.y for p in lm) * h,
            max(p.x for p in lm) * w,
            max(p.y for p in lm) * h,
        )
        best = None
        best_iou = 0.0
        from ..geometry import bbox_iou

        for tr in state.tracks:
            if not tr.is_person:
                continue
            iou = bbox_iou(bbox, tr.bbox)
            if iou > best_iou:
                best_iou = iou
                best = tr.track_id
        if best is not None and best_iou > 0.05:
            out.append(Pose(track_id=best, keypoints=kps, confidence=best_iou))
        return out
