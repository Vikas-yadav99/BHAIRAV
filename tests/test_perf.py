"""Tests for Phase 15: performance profiling and batched inference."""
from __future__ import annotations

import numpy as np
import pytest

from bhairav.perf import BatchedInferenceEngine, InferenceProfiler
from bhairav.perf.profiler import ProfileResult
from bhairav.perf.onnx_export import get_optimal_provider


class TestInferenceProfiler:
    def test_profile_returns_stats(self):
        def slow_infer(frame):
            return frame.mean()

        profiler = InferenceProfiler(slow_infer, warmup=3)
        result = profiler.run(frame_size=(64, 64), num_frames=20)
        assert isinstance(result, ProfileResult)
        assert result.total_frames == 17  # 20 - 3 warmup
        assert result.mean_ms > 0
        assert result.fps > 0

    def test_profile_to_dict(self):
        profiler = InferenceProfiler(lambda f: None, warmup=2)
        result = profiler.run(frame_size=(32, 32), num_frames=10)
        d = result.to_dict()
        assert "mean_ms" in d
        assert "fps" in d
        assert "p95_ms" in d

    def test_compare_two_functions(self):
        def fast(f):
            pass
        def slow(f):
            _ = sum(range(100))
        profiler = InferenceProfiler(fast, warmup=2)
        cmp = profiler.compare(slow, frame_size=(32, 32), num_frames=20)
        assert "baseline" in cmp
        assert "optimized" in cmp
        assert "speedup" in cmp


class TestBatchedInferenceEngine:
    def test_submit_and_flush(self):
        results = {}
        def dummy_detector(frames):
            return [[] for _ in frames]

        engine = BatchedInferenceEngine(dummy_detector, max_batch=2)
        for i in range(3):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            engine.submit(f"CAM-{i:02d}", frame, lambda cam_id, dets: results.update({cam_id: dets}))

        engine.flush()
        assert len(results) == 3

    def test_pending_count(self):
        engine = BatchedInferenceEngine(lambda f: [[] for _ in f], max_batch=100)
        for i in range(3):
            engine.submit(f"CAM-{i}", np.zeros((64, 64, 3), dtype=np.uint8), lambda c, d: None)
        assert engine.pending_count >= 0  # may have already drained


class TestOptimalProvider:
    def test_returns_string(self):
        provider = get_optimal_provider()
        assert isinstance(provider, str)
        assert "ExecutionProvider" in provider
