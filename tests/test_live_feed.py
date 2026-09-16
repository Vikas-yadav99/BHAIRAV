"""Tests for live feed ingestion pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestStreamResolver:
    """Test stream URL resolution."""

    def test_resolve_youtube(self):
        from bhairav.live_feed import StreamResolver
        url = StreamResolver.resolve("https://www.youtube.com/watch?v=8aINZD7m_s4")
        assert url is not None
        assert "googlevideo.com" in url or "youtube.com" in url

    def test_resolve_hls_passthrough(self):
        from bhairav.live_feed import StreamResolver
        url = StreamResolver.resolve("https://example.com/stream.m3u8")
        assert url == "https://example.com/stream.m3u8"

    def test_resolve_rtsp_passthrough(self):
        from bhairav.live_feed import StreamResolver
        url = StreamResolver.resolve("rtsp://192.168.1.100:554/stream")
        assert url == "rtsp://192.168.1.100:554/stream"


class TestLiveFeedProcessor:
    """Test feed processor configuration and status."""

    def test_create_processor(self):
        from bhairav.live_feed import LiveFeedProcessor, StreamConfig
        cfg = StreamConfig(id="test", url="rtsp://fake", name="Test", fps=1)
        proc = LiveFeedProcessor(cfg)
        status = proc.get_status()
        assert status["id"] == "test"
        assert status["running"] is False
        assert status["frames"] == 0

    def test_manager_add_remove(self):
        from bhairav.live_feed import LiveFeedManager, StreamConfig, LiveFeedProcessor
        manager = LiveFeedManager()
        cfg = StreamConfig(id="test-mgr", url="rtsp://fake", name="Test Mgr")
        # Don't actually start — just test the manager logic
        processor = LiveFeedProcessor(cfg)
        manager._feeds["test-mgr"] = processor
        assert len(manager.list_streams()) == 1
        manager._feeds.clear()
        assert len(manager.list_streams()) == 0

    def test_config_loading(self):
        from bhairav.live_feed import LiveFeedManager
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"id": "s1", "url": "rtsp://cam1", "name": "Camera 1", "fps": 2},
                {"id": "s2", "url": "rtsp://cam2", "name": "Camera 2"},
            ], f)
            f.flush()
            manager = LiveFeedManager()
            manager.load_config(f.name)
            assert len(manager._feeds) == 2


class TestLiveFeedDetection:
    """Test detection output format."""

    def test_detection_log_format(self):
        """Detection logs should be valid JSONL."""
        log_path = Path("output/live/custom/detections.jsonl")
        if not log_path.exists():
            pytest.skip("No live feed data yet")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) > 0
        for line in lines[:5]:
            d = json.loads(line)
            assert "timestamp" in d
            assert "persons" in d
            assert "vehicles" in d
            assert "detections" in d

    def test_alert_log_format(self):
        """Alert logs should be valid JSONL."""
        log_path = Path("output/live/custom/alerts.jsonl")
        if not log_path.exists():
            pytest.skip("No live feed data yet")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) > 0
        for line in lines[:5]:
            d = json.loads(line)
            assert "timestamp" in d
            assert "stream" in d
            assert "persons" in d
            assert d["type"] == "crowd_detection"


class TestDefaultStreams:
    """Test default stream configs."""

    def test_default_streams_exist(self):
        from bhairav.live_feed import DEFAULT_STREAMS
        assert len(DEFAULT_STREAMS) >= 2

    def test_default_streams_have_urls(self):
        from bhairav.live_feed import DEFAULT_STREAMS
        for s in DEFAULT_STREAMS:
            assert s.url.startswith("http")
            assert s.id
