"""Tests for Phases 14-16: deep Re-ID, performance profiling, and dashboard map data."""
from __future__ import annotations

import tempfile
import time
import numpy as np
import pytest

from bhairav.reid import AppearanceExtractor, DeepAppearanceExtractor, ReidStore, cosine
from bhairav.reid.deep_embedder import batch_cosine_matrix, cosine_similarity


# ── Phase 14: Deep Re-ID ────────────────────────────────────────────

class TestDeepAppearanceExtractor:
    """Tests for the ONNX deep re-ID embedder with fallback."""

    def test_no_model_falls_back_to_hsv_hog(self):
        """Without a model file, DeepAppearanceExtractor uses HSV+HOG."""
        ext = DeepAppearanceExtractor(model_path=None)
        assert not ext.is_deep
        assert ext.embedding_dim is None
        crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        emb = ext.embed(crop)
        assert emb is not None
        assert emb.ndim == 1

    def test_missing_model_file_falls_back(self):
        """Non-existent model path falls back gracefully."""
        ext = DeepAppearanceExtractor(model_path="/nonexistent/model.onnx")
        assert not ext.is_deep
        crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        emb = ext.embed(crop)
        assert emb is not None

    def test_invalid_onnx_falls_back(self):
        """A non-ONNX file treated as model falls back gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"not a real onnx model")
            f.flush()
            ext = DeepAppearanceExtractor(model_path=f.name)
            assert not ext.is_deep
            crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
            emb = ext.embed(crop)
            assert emb is not None

    def test_none_crop_returns_none(self):
        ext = DeepAppearanceExtractor()
        assert ext.embed(None) is None
        assert ext.embed(np.array([])) is None

    def test_small_crop_returns_none(self):
        ext = DeepAppearanceExtractor()
        small = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        assert ext.embed(small) is None

    def test_extract_from_frame(self):
        ext = DeepAppearanceExtractor()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        emb = ext.extract_from_frame(frame, [100, 50, 200, 250])
        assert emb is not None
        assert emb.ndim == 1

    def test_crop_thumbnail(self):
        ext = DeepAppearanceExtractor()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        thumb = ext.crop_thumbnail(frame, [100, 50, 200, 250])
        assert thumb is not None
        assert isinstance(thumb, str)
        assert len(thumb) > 100  # base64 encoded

    def test_cosine_similarity(self):
        a = np.array([1, 0, 0], dtype=np.float64)
        b = np.array([0, 1, 0], dtype=np.float64)
        c = np.array([1, 0, 0], dtype=np.float64)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)
        assert cosine_similarity(a, c) == pytest.approx(1.0, abs=1e-6)

    def test_batch_cosine_matrix(self):
        embs = [
            np.array([1, 0, 0], dtype=np.float64),
            np.array([0, 1, 0], dtype=np.float64),
            np.array([1, 0, 0], dtype=np.float64),
        ]
        mat = batch_cosine_matrix(embs)
        assert mat.shape == (3, 3)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[0, 1] == pytest.approx(0.0)
        assert mat[0, 2] == pytest.approx(1.0)

    def test_batch_cosine_matrix_empty(self):
        mat = batch_cosine_matrix([])
        assert mat.shape == (0, 0)


class TestReidStoreIntegration:
    """Verify the ReidStore works with both legacy and deep extractors."""

    def test_store_with_legacy_extractor(self, tmp_path):
        store = ReidStore(tmp_path / "gallery")
        ext = AppearanceExtractor()
        crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        emb = ext.embed(crop)
        rec = store.create_subject("Test Person", emb.tolist())
        assert rec["name"] == "Test Person"
        assert len(rec["embedding"]) > 0

    def test_store_with_deep_extractor_fallback(self, tmp_path):
        store = ReidStore(tmp_path / "gallery")
        ext = DeepAppearanceExtractor()
        crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        emb = ext.embed(crop)
        rec = store.create_subject("Deep Person", emb.tolist())
        assert rec["name"] == "Deep Person"
