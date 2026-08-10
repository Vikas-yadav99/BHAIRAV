"""Real CCTV path: YOLO detection + ByteTrack via ultralytics.

Requires `pip install ultralytics` (pulls in PyTorch). Lazy-imported so the
rest of Phase 1 works without it. When `mediapipe` and the pose landmarker
model are present, skeletons are attached to person tracks automatically.
"""
from __future__ import annotations

import cv2

from ..config import ModelConfig
from ..types import COCO_NAMES, FrameState, Track
from .base import Detector


class YoloDetector(Detector):
    def __init__(self, model_cfg: ModelConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "ultralytics is not installed. Install it with:  pip install ultralytics"
            ) from exc
        self.cfg = model_cfg
        self.model = YOLO(model_cfg.name)
        self._fps = 30.0
        # Optional pose estimation: enabled only when mediapipe + the model
        # file are available, so the pipeline degrades gracefully.
        self._pose = None
        try:
            from ..pose.mediapipe_model import MediaPipePoseModel, pose_model_path
            if pose_model_path().exists():
                self._pose = MediaPipePoseModel(min_detection_confidence=0.3)
        except Exception:
            self._pose = None

    @property
    def fps(self) -> float:
        return self._fps

    def stream(self, source: str | None = None, max_frames: int | None = None,
               opener=None):
        """Yield FrameStates.

        `opener` (callable -> opened cv2.VideoCapture) is used when provided;
        the sources layer uses it to retry RTSP/network opens with backoff.
        """
        if source is None or source == "blob":
            raise ValueError("YoloDetector needs a video file or camera index as source")
        if opener is not None:
            cap = opener()
        else:
            source = int(source) if source.isdigit() else source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video source: {source}")
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        i = 0
        while True:
            if max_frames is not None and i >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            results = self.model.track(
                frame,
                persist=True,                  # keeps ByteTrack state across frames
                conf=self.cfg.conf,
                imgsz=self.cfg.imgsz,
                classes=list(self.cfg.classes),
                tracker=self.cfg.tracker,      # bytetrack.yaml = ByteTrack
                verbose=False,
            )
            tracks: list[Track] = []
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                for tid, box, conf, cls in zip(ids, xyxy, confs, clss):
                    label = COCO_NAMES.get(int(cls), "object")
                    tracks.append(Track(int(tid), tuple(float(v) for v in box), label,
                                          float(conf), int(cls)))
            st = FrameState(frame_id=i, timestamp=i / self._fps, tracks=tracks,
                            frame_w=frame.shape[1], frame_h=frame.shape[0], frame=frame)
            if self._pose is not None:
                st.poses = self._pose.estimate(st)
            yield st
            i += 1
        cap.release()
