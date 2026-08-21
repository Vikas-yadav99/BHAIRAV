"""Deep-learning person re-identification embeddings (Phase 14).

Loads a lightweight ONNX model (e.g. OSNet-AIN-X1.0, MobileNetV2-ReID)
for much stronger cross-camera matching than HSV+HOG.  When no model
file is available, the class falls back transparently to the legacy
``AppearanceExtractor`` so the rest of the pipeline never needs to
know.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .._reid_impl import AppearanceExtractor

logger = logging.getLogger(__name__)

_MIN_CROP_PX = 40


class DeepAppearanceExtractor:
    """ONNX-based deep person re-ID with automatic legacy fallback."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        self._session = None
        self._input_name: str = ""
        self._input_size: tuple[int, int] = (128, 256)
        self._fallback = AppearanceExtractor()
        self._is_deep = False

        if model_path is not None:
            self._try_load(model_path, input_size)

    def _try_load(self, model_path, input_size):
        path = Path(model_path)
        if not path.exists():
            logger.warning("Deep ReID model not found at %s — using HSV+HOG", path)
            return
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(path), opts, providers=["CPUExecutionProvider"]
            )
            meta = self._session.get_inputs()[0]
            self._input_name = meta.name
            shape = meta.shape
            if len(shape) == 4:
                if shape[1] == 3:
                    self._input_size = (int(shape[3]), int(shape[2]))
                else:
                    self._input_size = (int(shape[2]), int(shape[1]))
            if input_size is not None:
                self._input_size = tuple(input_size)
            self._is_deep = True
            logger.info("Deep ReID loaded: %s  input=%s", path.name, self._input_size)
        except Exception as exc:
            logger.warning("Deep ReID load failed: %s — using HSV+HOG", exc)
            self._session = None
            self._is_deep = False

    def _preprocess(self, crop):
        w, h = self._input_size
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        meta = self._session.get_inputs()[0]
        shape = meta.shape
        is_nchw = len(shape) == 4 and shape[1] == 3
        blob = resized.astype(np.float32) / 255.0
        if is_nchw:
            blob = blob.transpose(2, 0, 1)
            blob = np.expand_dims(blob, axis=0)
        else:
            blob = np.expand_dims(blob, axis=0)
        return blob

    def embed(self, crop):
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if min(h, w) < _MIN_CROP_PX:
            return None
        if not self._is_deep:
            return self._fallback.embed(crop)
        try:
            blob = self._preprocess(crop)
            outputs = self._session.run(None, {self._input_name: blob})
            emb = outputs[0].flatten().astype(np.float64)
            norm = np.linalg.norm(emb)
            if norm < 1e-9:
                return None
            return emb / norm
        except Exception:
            return self._fallback.embed(crop)

    def extract_from_frame(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
            return None
        return self.embed(frame[y1:y2, x1:x2])

    def crop_thumbnail(self, frame, bbox, blur_head=True, size=96):
        return self._fallback.crop_thumbnail(frame, bbox, blur_head=blur_head, size=size)

    @property
    def embedding_dim(self):
        if not self._is_deep:
            return None
        try:
            shape = self._session.get_outputs()[0].shape
            if len(shape) == 2:
                return int(shape[1])
            return int(shape[-1])
        except Exception:
            return None

    @property
    def is_deep(self):
        return self._is_deep


def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def batch_cosine_matrix(embeddings):
    if not embeddings:
        return np.empty((0, 0), dtype=np.float64)
    mat = np.vstack(embeddings)
    return mat @ mat.T
