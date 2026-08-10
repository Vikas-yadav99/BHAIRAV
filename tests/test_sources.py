"""Tests for the camera/video source layer (classification, health, retry)."""
import sys
import time

import pytest

sys.path.insert(0, "src")  # noqa: E402

from bhairav.sources import (  # noqa: E402
    SourceKind,
    SourceMonitor,
    classify_source,
    open_capture,
)


class FakeCap:
    def __init__(self, opened=True):
        self._opened = opened

    def isOpened(self):
        return self._opened

    def release(self):
        self._opened = False

    def set(self, *a):
        return True


# ---- classification ------------------------------------------------------

def test_classify_blob():
    assert classify_source(None)[0] is SourceKind.BLOB
    assert classify_source("blob")[0] is SourceKind.BLOB


def test_classify_camera_index():
    kind, desc = classify_source("0")
    assert kind is SourceKind.CAMERA
    assert "0" in desc
    assert classify_source(3)[0] is SourceKind.CAMERA


@pytest.mark.parametrize("path", [
    "clip.mp4", "C:/vids/cam1.avi", r"C:\vids\cam1.mov", "data/x.mkv",
])
def test_classify_file(path):
    assert classify_source(path)[0] is SourceKind.FILE


@pytest.mark.parametrize("url", [
    "rtsp://user:pass@10.0.0.5:554/stream1",
    "rtmp://cam.local/live",
    "http://cdn.example/live.m3u8",
    "https://cam.example/hls/index.mpd",
])
def test_classify_network(url):
    assert classify_source(url)[0] is SourceKind.NETWORK


# ---- monitor -------------------------------------------------------------

def test_monitor_counts_and_snapshot():
    m = SourceMonitor(SourceKind.NETWORK, "rtsp://cam")
    m.connecting()
    m.connecting()
    m.connected()
    m.dropped("boom")
    snap = m.snapshot()
    assert snap["attempts"] == 2
    assert snap["connects"] == 1
    assert snap["drops"] == 1
    assert snap["last_error"] == "boom"
    assert snap["kind"] == "network"
    assert snap["connected"] is True


def test_monitor_connected_false_when_stale():
    m = SourceMonitor(SourceKind.NETWORK, "rtsp://cam")
    m.connected()
    m._lock.acquire()
    m.last_connect_ts = time.time() - 120
    m._lock.release()
    assert m.snapshot()["connected"] is False


# ---- open_capture retry/backoff -----------------------------------------

def test_open_capture_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def fake_videocapture(src):
        calls["n"] += 1
        return FakeCap(opened=False)

    monkeypatch.setattr("bhairav.sources.cv2.VideoCapture", fake_videocapture)
    m = SourceMonitor(SourceKind.NETWORK, "rtsp://down")
    with pytest.raises(RuntimeError, match="cannot open"):
        open_capture("rtsp://down", monitor=m, retries=3, base_delay=0.01)
    assert calls["n"] == 3
    assert m.attempts == 3
    assert m.connects == 0


def test_open_capture_success_records_connect(monkeypatch):
    monkeypatch.setattr("bhairav.sources.cv2.VideoCapture",
                        lambda src: FakeCap(opened=True))
    m = SourceMonitor(SourceKind.FILE, "clip.mp4")
    cap = open_capture("clip.mp4", monitor=m, retries=2, base_delay=0.01)
    assert cap.isOpened()
    assert m.connects == 1
    assert m.snapshot()["connected"] is True


def test_open_capture_sets_low_latency_opts_for_network(monkeypatch):
    captured = {}

    def fake_videocapture(src):
        captured["src"] = src
        return FakeCap(opened=True)

    monkeypatch.setattr("bhairav.sources.cv2.VideoCapture", fake_videocapture)
    open_capture("rtsp://cam/live", retries=1)
    assert captured["src"] == "rtsp://cam/live"
    assert "rtsp_transport=tcp" in __import__("os").environ.get(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS", "")


def test_open_capture_camera_index_converted(monkeypatch):
    captured = {}

    def fake_videocapture(src):
        captured["src"] = src
        return FakeCap(opened=True)

    monkeypatch.setattr("bhairav.sources.cv2.VideoCapture", fake_videocapture)
    open_capture("2", retries=1)
    assert captured["src"] == 2
