"""Batched multi-camera inference engine (Phase 15).

Accumulates frames from N camera pipelines into micro-batches and
runs a single YOLO/ONNX inference call per batch, amortising GPU
warmup and PCIe transfer costs.  Designed to sit between the camera
sources and the detector.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class BatchedInferenceEngine:
    """Collects frames from multiple cameras and runs batched detection.

    Parameters
    ----------
    detector_fn : callable
        ``detector_fn(batch: list[np.ndarray]) -> list[list[Detection]]``
        called once per accumulated batch.
    max_batch : int
        Maximum frames per batch before forcing an inference call.
    max_wait_ms : float
        Maximum milliseconds to wait before running a partial batch.
    """

    def __init__(
        self,
        detector_fn: Callable[[list[np.ndarray]], list],
        max_batch: int = 4,
        max_wait_ms: float = 33.0,
    ) -> None:
        self._detector = detector_fn
        self._max_batch = max_batch
        self._max_wait_ms = max_wait_ms
        self._lock = threading.Lock()
        self._pending: list[tuple[str, np.ndarray, Callable]] = []
        self._batch_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def submit(self, camera_id: str, frame: np.ndarray,
               callback: Callable[[str, list], None]) -> None:
        """Queue a frame for batched inference.

        Parameters
        ----------
        camera_id : str
            Camera identifier for tracking.
        frame : np.ndarray
            BGR image.
        callback : callable
            ``callback(camera_id, detections)`` called on completion.
        """
        with self._lock:
            self._pending.append((camera_id, frame, callback))
            should_launch = (
                len(self._pending) >= self._max_batch
                or self._batch_thread is None
                or not self._batch_thread.is_alive()
            )
        if should_launch:
            self._run_batch()

    def _run_batch(self) -> None:
        with self._lock:
            pending = self._pending[:self._max_batch]
            self._pending = self._pending[self._max_batch:]
            if not pending:
                return

        def _worker(batch):
            frames = [f for _, f, _ in batch]
            try:
                results = self._detector(frames)
            except Exception as exc:
                logger.error("Batched inference failed: %s", exc)
                results = [[] for _ in frames]
            for (cam_id, _, cb), dets in zip(batch, results):
                try:
                    cb(cam_id, dets)
                except Exception as exc:
                    logger.error("Batch callback error for %s: %s", cam_id, exc)

        t = threading.Thread(target=_worker, args=(pending,), daemon=True)
        self._batch_thread = t
        t.start()

    def flush(self) -> None:
        """Force-run any remaining pending frames."""
        with self._lock:
            if self._pending:
                self._run_batch()
        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_thread.join(timeout=5.0)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
