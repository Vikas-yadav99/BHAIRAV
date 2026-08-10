"""Real CCTV path: YOLO detection + ByteTrack via ultralytics.

Requires `pip install ultralytics` (pulls in PyTorch). Lazy-imported so the
rest of Phase 1 works without it.
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

    @property
    def fps(self) -> float:
        return self._fps

    def stream(self, source: str | None = None, max_frames: int | None = None):
        if source is None or source == "blob":
            raise ValueError("YoloDetector needs a video file or camera index as source")
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
            yield FrameState(frame_id=i, timestamp=i / self._fps, tracks=tracks,
                             frame_w=frame.shape[1], frame_h=frame.shape[0], frame=frame)
            i += 1
        cap.release()
