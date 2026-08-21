"""Re-identification package.

Phase 14 adds deep-learning ONNX embeddings alongside the original
HSV+HOG appearance extractor.
"""
from __future__ import annotations

from .._reid_impl import (  # noqa: F401
    AppearanceExtractor,
    ReidService,
    ReidStore,
    cosine,
    _new_id,
)

from .deep_embedder import DeepAppearanceExtractor  # noqa: F401

__all__ = [
    "AppearanceExtractor",
    "DeepAppearanceExtractor",
    "ReidService",
    "ReidStore",
    "cosine",
    "_new_id",
]
