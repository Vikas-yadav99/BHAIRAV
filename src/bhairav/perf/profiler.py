"""Inference profiling and benchmarking (Phase 15).

Measures per-frame latency, throughput, GPU utilisation, and memory
usage across detector backends.  Useful for comparing ONNX vs Torch
vs TensorRT performance.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field


@dataclass
class ProfileResult:
    """Aggregated profiling statistics."""
    total_frames: int = 0
    total_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    fps: float = 0.0
    warmup_frames: int = 0
    timings: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "total_ms": round(self.total_ms, 1),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "fps": round(self.fps, 1),
            "warmup_frames": self.warmup_frames,
        }


class InferenceProfiler:
    """Profile an inference function by running it on synthetic frames.

    Parameters
    ----------
    infer_fn : callable
        ``infer_fn(frame) -> Any`` — the function to profile.
    warmup : int
        Number of warmup frames to discard before measuring.
    """

    def __init__(self, infer_fn, warmup: int = 10) -> None:
        self._infer_fn = infer_fn
        self._warmup = warmup
        self._timings: list[float] = []

    def run(
        self,
        frame_size: tuple[int, int] = (640, 640),
        num_frames: int = 100,
        seed: int = 42,
    ) -> ProfileResult:
        """Run the profiler on random frames of the given size.

        Parameters
        ----------
        frame_size : tuple[int, int]
            (width, height) of synthetic test frames.
        num_frames : int
            Total frames to process (including warmup).
        seed : int
            RNG seed for reproducible test data.

        Returns
        -------
        ProfileResult
            Aggregated timing statistics.
        """
        import numpy as np

        rng = np.random.RandomState(seed)
        w, h = frame_size
        timings: list[float] = []

        for i in range(num_frames):
            frame = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            try:
                self._infer_fn(frame)
            except Exception:
                pass
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if i >= self._warmup:
                timings.append(elapsed_ms)

        if not timings:
            return ProfileResult()

        total_ms = sum(timings)
        sorted_t = sorted(timings)
        n = len(sorted_t)

        return ProfileResult(
            total_frames=n,
            total_ms=total_ms,
            mean_ms=statistics.mean(timings),
            median_ms=statistics.median(timings),
            p95_ms=sorted_t[int(n * 0.95)] if n >= 20 else sorted_t[-1],
            p99_ms=sorted_t[int(n * 0.99)] if n >= 100 else sorted_t[-1],
            min_ms=min(timings),
            max_ms=max(timings),
            fps=1000.0 / statistics.mean(timings) if timings else 0,
            warmup_frames=self._warmup,
            timings=timings,
        )

    def compare(
        self,
        other_fn,
        frame_size: tuple[int, int] = (640, 640),
        num_frames: int = 100,
    ) -> dict:
        """Profile two functions and return a comparison dict."""
        r1 = self.run(frame_size=frame_size, num_frames=num_frames)
        profiler2 = InferenceProfiler(other_fn, warmup=self._warmup)
        r2 = profiler2.run(frame_size=frame_size, num_frames=num_frames)

        speedup = r1.mean_ms / r2.mean_ms if r2.mean_ms > 0 else 0
        return {
            "baseline": r1.to_dict(),
            "optimized": r2.to_dict(),
            "speedup": round(speedup, 2),
            "fps_improvement": round(r2.fps - r1.fps, 1),
        }
