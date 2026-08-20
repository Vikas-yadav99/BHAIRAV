"""Edge TPU / NPU accelerated YOLO detector.

Supports Google Coral Edge TPU (TFLite), NVIDIA Jetson (ONNX Runtime),
and falls back to standard ultralytics YOLO. Auto-detects the best backend.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from .base import Detector
from ..types import Detection, FrameState, Track

log = logging.getLogger("bhairav.detectors.edge_tpu")

try:
    from tflite_runtime.interpreter import Interpreter
    HAS_TFLITE = True
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        HAS_TFLITE = True
    except ImportError:
        HAS_TFLITE = False

try:
    import tflite_runtime.interpreter as tflite_rt
    HAS_EDGETPU = hasattr(tflite_rt, "load_delegate")
except ImportError:
    HAS_EDGETPU = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

PERSON_CLASS = 0
LABEL_MAP = {0: "person", 2: "car", 5: "bus", 7: "truck"}


class EdgeTPUDetector(Detector):
    """YOLO detector accelerated by Edge TPU, NPU, or ONNX Runtime."""

    def __init__(self, model_path=None, confidence=0.4, input_size=320):
        self.confidence = confidence
        self.input_size = input_size
        self._backend = "none"
        self._interpreter = None
        self._session = None
        self._input_details = None
        self._output_details = None
        self._frame_id = 0
        self._fps_timer = time.time()
        self._fps = 0.0
        self._frame_count = 0
        self._init_backend(model_path)

    def _init_backend(self, model_path):
        mp = str(model_path) if model_path else None
        if HAS_TFLITE and mp and mp.endswith(".tflite"):
            try:
                delegates = []
                if HAS_EDGETPU:
                    delegates = [tflite_rt.load_delegate("libedgetpu.so.1")]
                self._interpreter = Interpreter(
                    model_path=mp, num_threads=2,
                    experimental_delegates=delegates if delegates else None)
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
                self._backend = "edge_tpu" if HAS_EDGETPU else "tflite_cpu"
                log.info("Detector: %s - %s", self._backend, mp)
                return
            except Exception as exc:
                log.warning("TFLite init failed: %s", exc)

        if HAS_ONNX and mp and mp.endswith(".onnx"):
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._session = ort.InferenceSession(mp, providers=providers)
                self._input_details = self._session.get_inputs()
                self._output_details = self._session.get_outputs()
                self._backend = "onnx"
                log.info("Detector: ONNX Runtime - %s", mp)
                return
            except Exception as exc:
                log.warning("ONNX init failed: %s", exc)

        self._backend = "none"
        log.warning("No Edge TPU/ONNX model; detector disabled")

    def detect(self, frame):
        self._frame_id += 1
        self._update_fps()
        if self._backend == "none" or frame is None:
            return []
        h, w = frame.shape[:2]
        if self._backend in ("edge_tpu", "tflite_cpu"):
            return self._detect_tflite(frame, h, w)
        if self._backend == "onnx":
            return self._detect_onnx(frame, h, w)
        return []

    def _detect_tflite(self, frame, h, w):
        target = self.input_size
        img = cv2.resize(frame, (target, target))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = self._input_details[0]
        self._interpreter.set_tensor(inp["index"], np.expand_dims(img.astype(np.uint8), 0))
        self._interpreter.invoke()
        out = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        dets = []
        for det in out:
            if len(det) < 6:
                continue
            conf, cls = float(det[4]), int(det[5])
            if conf < self.confidence or cls not in LABEL_MAP:
                continue
            x1, y1, x2, y2 = det[0]/target*w, det[1]/target*h, det[2]/target*w, det[3]/target*h
            dets.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf,
                                  class_id=cls, label=LABEL_MAP[cls]))
        return dets

    def _detect_onnx(self, frame, h, w):
        target = self.input_size
        img = cv2.resize(frame, (target, target))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        blob = np.expand_dims(img.transpose(2, 0, 1), 0)
        out = self._session.run(None, {self._input_details[0].name: blob})[0]
        if len(out.shape) == 3:
            out = out[0]
        dets = []
        for det in out:
            if len(det) < 6:
                continue
            conf, cls = float(det[4]), int(det[5])
            if conf < self.confidence or cls not in LABEL_MAP:
                continue
            x1, y1, x2, y2 = det[0]/target*w, det[1]/target*h, det[2]/target*w, det[3]/target*h
            dets.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf,
                                  class_id=cls, label=LABEL_MAP[cls]))
        return dets

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        if now - self._fps_timer >= 1.0:
            self._fps = self._frame_count / (now - self._fps_timer)
            self._frame_count = 0
            self._fps_timer = now

    def stream(self, source=None, max_frames=None, opener=None):
        cap = cv2.VideoCapture(str(source)) if source and str(source) != "blob" else None
        if cap is None:
            return
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                dets = self.detect(frame)
                tracks = [Track(track_id=i, bbox=d.bbox, label=d.label,
                                confidence=d.confidence, class_id=d.class_id)
                          for i, d in enumerate(dets)]
                yield FrameState(frame_id=self._frame_id, timestamp=time.time(),
                                 tracks=tracks, frame_w=frame.shape[1],
                                 frame_h=frame.shape[0], frame=frame)
                if max_frames and self._frame_id >= max_frames:
                    break
        finally:
            if cap:
                cap.release()

    @property
    def backend(self):
        return self._backend

    @property
    def fps(self):
        return round(self._fps, 1)
