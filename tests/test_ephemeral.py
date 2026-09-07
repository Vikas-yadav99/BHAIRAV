#!/usr/bin/env python3
"""Tests for the ephemeral privacy-first pipeline."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bhairav.ephemeral import (
    BoundingBox, Detection, DetectionClass, EventRecord,
    EphemeralProcessor, FrameMetadata, MetadataStore, PrivacyFirstPipeline,
)


# ── BoundingBox & Detection ─────────────────────────────────────────────────

class TestBoundingBox:
    def test_creation(self):
        bb = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8, confidence=0.95)
        assert bb.x1 == 0.1
        assert bb.confidence == 0.95

    def test_to_dict_rounds(self):
        bb = BoundingBox(x1=0.123456, y1=0.2, x2=0.5, y2=0.8, confidence=0.999)
        d = bb.to_dict()
        assert d["x1"] == 0.1235  # rounded to 4 decimals
        assert d["conf"] == 0.999

    def test_normalized_coordinates(self):
        bb = BoundingBox(x1=0, y1=0, x2=1, y2=1, confidence=1.0)
        d = bb.to_dict()
        assert 0 <= d["x1"] <= 1
        assert 0 <= d["y2"] <= 1


class TestDetection:
    def test_basic_detection(self):
        bb = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8, confidence=0.9)
        det = Detection(class_id="person", bbox=bb, track_id="T-1")
        d = det.to_dict()
        assert d["cls"] == "person"
        assert d["track"] == "T-1"
        assert "bbox" in d

    def test_no_track_id(self):
        bb = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8, confidence=0.9)
        det = Detection(class_id="vehicle", bbox=bb)
        d = det.to_dict()
        assert "track" not in d

    def test_attributes(self):
        bb = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8, confidence=0.9)
        det = Detection(class_id="face", bbox=bb, attributes={"face_vec": [0.1, 0.2, 0.3]})
        d = det.to_dict()
        assert "attr" in d
        assert d["attr"]["face_vec"] == [0.1, 0.2, 0.3]


# ── EphemeralProcessor ──────────────────────────────────────────────────────

class TestEphemeralProcessor:
    def test_process_frame_no_detections(self):
        proc = EphemeralProcessor("CAM-01")
        metadata = proc.process(None, [])
        assert metadata.camera_id == "CAM-01"
        assert metadata.frame_index == 1
        assert len(metadata.detections) == 0

    def test_process_frame_with_detections(self):
        proc = EphemeralProcessor("CAM-02")
        dets = [
            {"class": "person", "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8, "confidence": 0.95, "track_id": "T-1"},
            {"class": "vehicle", "x1": 0.6, "y1": 0.3, "x2": 0.9, "y2": 0.7, "confidence": 0.88, "track_id": "T-2"},
        ]
        metadata = proc.process(None, dets)
        assert len(metadata.detections) == 2
        assert metadata.detections[0].class_id == "person"
        assert metadata.detections[1].track_id == "T-2"

    def test_frame_index_increments(self):
        proc = EphemeralProcessor("CAM-03")
        proc.process(None, [])
        proc.process(None, [])
        m3 = proc.process(None, [])
        assert m3.frame_index == 3

    def test_context_buffer(self):
        proc = EphemeralProcessor("CAM-04", {"context_frames": 5})
        for _ in range(10):
            proc.process(None, [{"class": "person", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.9}])
        ctx = proc.get_context_frames(3)
        assert len(ctx) == 3
        # Latest frames
        assert ctx[-1]["idx"] == 10

    def test_track_history(self):
        proc = EphemeralProcessor("CAM-05")
        # Simulate person moving
        for i in range(5):
            x = 0.1 + i * 0.1
            proc.process(None, [{"class": "person", "x1": x, "y1": 0.5, "x2": x+0.1, "y2": 0.9, "confidence": 0.9, "track_id": "T-1"}])
        history = proc.get_track_history("T-1")
        assert len(history) == 5
        assert history[0]["x"] < history[-1]["x"]  # moving right

    def test_clear_stale_tracks(self):
        proc = EphemeralProcessor("CAM-06")
        proc.process(None, [{"class": "person", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.9, "track_id": "T-old"}])
        # Manually age the track
        proc._active_tracks["T-old"][-1]["ts"] = time.time() - 600
        proc.clear_tracks(stale_seconds=300)
        assert "T-old" not in proc._active_tracks

    def test_stats(self):
        proc = EphemeralProcessor("CAM-07")
        proc.process(None, [{"class": "person", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.9, "track_id": "T-1"}])
        proc.process(None, [{"class": "person", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.9, "track_id": "T-1"}])
        s = proc.stats()
        assert s["total_frames"] == 2
        assert s["total_detections"] == 2
        assert s["active_tracks"] == 1

    def test_no_video_stored(self):
        """Verify that process() returns metadata, not image data."""
        proc = EphemeralProcessor("CAM-08")
        metadata = proc.process(None, [{"class": "person", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.9}])
        d = metadata.to_dict()
        # Should contain detection metadata, not pixel data
        assert "det" in d
        assert "ts" in d
        # No image/video data
        assert "frame" not in d
        assert "image" not in d
        assert "pixels" not in d


# ── MetadataStore ────────────────────────────────────────────────────────────

class TestMetadataStore:
    def test_save_and_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(root=tmpdir, retention_days=30)
            event = EventRecord(
                id="test-001", timestamp=time.time(), camera_id="CAM-01",
                category="crime", severity="red", description="Fight detected",
                detection_count=3, track_ids=["T-1", "T-2"],
            )
            store.save_event(event)

            results = store.search(camera_id="CAM-01")
            assert len(results) == 1
            assert results[0]["category"] == "crime"

    def test_search_by_category(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(root=tmpdir)
            for i, cat in enumerate(["crime", "medical", "fire", "crime"]):
                event = EventRecord(
                    id=f"e-{cat}-{i}", timestamp=time.time() + i, camera_id="CAM-01",
                    category=cat, severity="yellow", description="test",
                )
                store.save_event(event)

            results = store.search(category="crime")
            assert len(results) == 2

    def test_delete_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(root=tmpdir)
            now = time.time()
            # Save events at different times
            for i in range(5):
                event = EventRecord(
                    id=f"e-{i}", timestamp=now - (i * 3600),
                    camera_id="CAM-01", category="test",
                    severity="yellow", description="test",
                )
                store.save_event(event)

            # Delete events from last 2 hours
            deleted = store.delete_range(now - 7200, now)
            assert deleted >= 1

    def test_enforce_retention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(root=tmpdir, retention_days=1)
            # Save an event from 10 days ago
            event = EventRecord(
                id="old-event", timestamp=time.time() - (10 * 86400),
                camera_id="CAM-01", category="test",
                severity="yellow", description="old",
            )
            store.save_event(event)
            deleted = store.enforce_retention()
            assert deleted >= 1

    def test_storage_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(root=tmpdir)
            event = EventRecord(
                id="stat-test", timestamp=time.time(), camera_id="CAM-01",
                category="test", severity="yellow", description="test",
            )
            store.save_event(event)
            stats = store.stats()
            assert stats["total_files"] >= 1
            assert stats["total_size_bytes"] > 0
            assert stats["total_size_mb"] >= 0

    def test_no_video_files(self):
        """Verify store never creates .mp4, .jpg, or .avi files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(root=tmpdir)
            for i in range(10):
                event = EventRecord(
                    id=f"vid-test-{i}", timestamp=time.time(), camera_id="CAM-01",
                    category="test", severity="yellow", description="test",
                )
                store.save_event(event)

            all_files = [f for f in Path(tmpdir).rglob("*") if f.is_file()]
            for f in all_files:
                assert f.suffix in (".json", ".jsonl"), f"Unexpected file type: {f}"


# ── EventRecord ─────────────────────────────────────────────────────────────

class TestEventRecord:
    def test_to_dict(self):
        event = EventRecord(
            id="evt-001", timestamp=time.time(), camera_id="CAM-01",
            category="crime", severity="red", description="Fight",
            detection_count=5, track_ids=["T-1", "T-2"],
        )
        d = event.to_dict()
        assert d["id"] == "evt-001"
        assert d["category"] == "crime"
        assert d["detection_count"] == 5
        assert len(d["track_ids"]) == 2

    def test_privacy_no_images(self):
        """Event record should never contain image data."""
        event = EventRecord(
            id="priv-001", timestamp=time.time(), camera_id="CAM-01",
            category="test", severity="yellow", description="test",
        )
        d = event.to_dict()
        serialized = json.dumps(d)
        assert "base64" not in serialized.lower()
        assert ".jpg" not in serialized
        assert ".mp4" not in serialized


# ── PrivacyFirstPipeline ─────────────────────────────────────────────────────

class TestPrivacyFirstPipeline:
    def test_register_camera(self):
        pipeline = PrivacyFirstPipeline()
        proc = pipeline.register_camera("CAM-01")
        assert isinstance(proc, EphemeralProcessor)
        assert pipeline.total_cameras == 1

    def test_process_frame(self):
        pipeline = PrivacyFirstPipeline()
        metadata = pipeline.process_frame("CAM-01", None, [
            {"class": "person", "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8, "confidence": 0.9}
        ])
        assert metadata.camera_id == "CAM-01"
        assert pipeline.total_frames == 1

    def test_on_alert_saves_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = PrivacyFirstPipeline({"metadata_dir": tmpdir})
            pipeline.register_camera("CAM-01")

            # Process some frames first
            for _ in range(5):
                pipeline.process_frame("CAM-01", None, [
                    {"class": "person", "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8, "confidence": 0.9, "track_id": "T-1"}
                ])

            # Fire alert
            event = pipeline.on_detection_alert("CAM-01", {
                "category": "crime",
                "severity": "red",
                "description": "Fight detected",
                "detection_count": 3,
                "track_ids": ["T-1"],
            })

            assert event is not None
            assert event.category == "crime"
            assert pipeline.total_alerts == 1

            # Verify metadata was saved (no video)
            results = pipeline.store.search()
            assert len(results) == 1
            assert results[0]["category"] == "crime"

    def test_alert_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = PrivacyFirstPipeline({"metadata_dir": tmpdir})
            alerts_received = []
            pipeline.on_alert = lambda e: alerts_received.append(e)

            pipeline.register_camera("CAM-01")
            pipeline.on_detection_alert("CAM-01", {
                "category": "fire", "severity": "orange", "description": "Smoke",
            })

            assert len(alerts_received) == 1
            assert alerts_received[0]["category"] == "fire"

    def test_retention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = PrivacyFirstPipeline({
                "metadata_dir": tmpdir,
                "retention_days": 1,
            })
            pipeline.register_camera("CAM-01")

            # Save an old event
            from bhairav.ephemeral import EventRecord
            old = EventRecord(
                id="old-1", timestamp=time.time() - (5 * 86400),
                camera_id="CAM-01", category="test",
                severity="yellow", description="old",
            )
            pipeline.store.save_event(old)

            result = pipeline.run_retention()
            assert result["deleted_events"] >= 1

    def test_stats(self):
        pipeline = PrivacyFirstPipeline()
        pipeline.register_camera("CAM-01")
        pipeline.process_frame("CAM-01", None, [])
        s = pipeline.stats()
        assert s["pipeline"]["total_cameras"] == 1
        assert s["pipeline"]["total_frames"] == 1
        assert "storage" in s

    def test_multi_camera(self):
        pipeline = PrivacyFirstPipeline()
        for i in range(5):
            pipeline.process_frame(f"CAM-{i:02d}", None, [
                {"class": "person", "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8, "confidence": 0.9}
            ])
        assert pipeline.total_cameras == 5
        assert pipeline.total_frames == 5

    def test_zero_video_guarantee(self):
        """End-to-end: process frames, fire alerts, verify no video stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = PrivacyFirstPipeline({"metadata_dir": tmpdir})
            pipeline.register_camera("CAM-01")

            # Process 100 frames with detections
            for i in range(100):
                pipeline.process_frame("CAM-01", None, [
                    {"class": "person", "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8,
                     "confidence": 0.9, "track_id": f"T-{i%3}"}
                ])

            # Fire 10 alerts
            for i in range(10):
                pipeline.on_detection_alert("CAM-01", {
                    "category": "test", "severity": "yellow",
                    "description": f"Alert {i}", "track_ids": [f"T-{i%3}"],
                })

            # Verify: ONLY .json and .jsonl files exist
            all_files = list(Path(tmpdir).rglob("*"))
            for f in all_files:
                if f.is_file():
                    assert f.suffix in (".json", ".jsonl"), f"VIDEO FOUND: {f}"

            # Verify: metadata is small
            total_size = sum(f.stat().st_size for f in all_files if f.is_file())
            assert total_size < 100_000  # < 100KB for 100 frames + 10 alerts
            # (vs ~50MB if we stored video)
