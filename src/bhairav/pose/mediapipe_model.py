"""MediaPipe pose wrapper for the real CCTV path (lazy import).

Not exercised by tests or the offline demo - `mediapipe` is an optional
dependency. Falls back gracefully: constructing the model without mediapipe
raises a clear error, so callers can simply disable pose when it is missing.

Uses the current Tasks API (``PoseLandmarker``); the legacy
``mp.solutions.pose`` API was removed in mediapipe 1.0. The model file
(``pose_landmarker_full.task``) is downloaded by ``scripts/fetch_models.py``
into <repo>/models/ with SHA-256 verification.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..types import FrameState, Keypoint, Pose


def pose_model_path() -> Path:
    """Resolve the pose landmarker model: $BHAIRAV_MODELS_DIR or <repo>/models."""
    env = os.environ.get("BHAIRAV_MODELS_DIR")
    if env:
        return Path(env) / "pose_landmarker_full.task"
    here = Path(__file__).resolve()
    repo_root = here.parents[3]   # .../project/src/bhairav/pose -> project
    return repo_root / "models" / "pose_landmarker_full.task"


class MediaPipePoseModel:
    """Runs MediaPipe Pose Landmarker on each frame and maps skeletons to
    tracks by bounding-box overlap with the detection boxes."""

    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "MediaPipePoseModel requires 'mediapipe' (pip install mediapipe). "
                "Use the synthetic path or install the dependency."
            ) from exc
        model = pose_model_path()
        if not model.exists():
            raise RuntimeError(
                f"pose landmarker model not found at {model}. "
                "Fetch it with:  python scripts/fetch_models.py")
        self._mp = mp
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            num_poses=1,
        )
        self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(opts)

    def estimate(self, state: FrameState) -> list[Pose]:
        import cv2

        if state.frame is None:
            return []
        rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        results = self._pose.detect(mp_img)
        out: list[Pose] = []
        if not results.pose_landmarks:
            return out
        lm = results.pose_landmarks[0]
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
