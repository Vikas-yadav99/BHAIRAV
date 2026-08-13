"""Tests for Phase 10 M4 - field-officer alert dispatch (AlertNotifier)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bhairav.backend.notify import AlertNotifier, SEVERITY_RANK, _Channel, channels_from_config


def _alert(rule="riot", severity="red", **kw):
    base = {
        "rule": rule, "severity": severity, "message": "test alert",
        "zone": "plaza", "track_id": 1, "frame_id": 120,
        "timestamp": 1.0, "confidence": 0.95, "details": {},
        "camera": "CAM-01",
    }
    base.update(kw)
    return base


class _Collector(BaseHTTPRequestHandler):
    """Stub endpoint that records POSTed payloads for assertions."""

    received = []
    fail_until = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.fail_until > time.time():
            self.send_response(503)
            self.end_headers()
            return
        _Collector.received.append(json.loads(body))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def collector():
    _Collector.received = []
    _Collector.fail_until = 0
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


# ------------------------------------------------------------------ matching
def test_severity_rank_ordering():
    assert SEVERITY_RANK["yellow"] < SEVERITY_RANK["orange"] < SEVERITY_RANK["red"]


def test_channel_min_severity_filter():
    ch = _Channel({"url": "http://x", "min_severity": "orange"})
    assert not ch.matches(_alert(severity="yellow"))
    assert ch.matches(_alert(severity="orange"))
    assert ch.matches(_alert(severity="red"))


def test_channel_rules_allowlist():
    ch = _Channel({"url": "http://x", "rules": ["riot", "accident"]})
    assert ch.matches(_alert(rule="riot"))
    assert ch.matches(_alert(rule="accident"))
    assert not ch.matches(_alert(rule="fight"))


def test_channel_without_url_never_matches():
    ch = _Channel({"name": "off", "url": ""})
    assert not ch.matches(_alert())


# ---------------------------------------------------------------- delivery
def test_notify_posts_payload(collector):
    notifier = AlertNotifier([{"name": "slack", "url": collector,
                               "min_severity": "yellow"}])
    assert notifier.notify(_alert()) == 1
    deadline = time.time() + 5
    while not _Collector.received and time.time() < deadline:
        time.sleep(0.02)
    assert len(_Collector.received) == 1
    payload = _Collector.received[0]
    assert payload["type"] == "bhairav_alert"
    assert payload["alert"]["rule"] == "riot"
    assert payload["alert"]["severity"] == "red"


def test_filtered_alert_not_delivered(collector):
    notifier = AlertNotifier([{"url": collector, "min_severity": "red"}])
    assert notifier.notify(_alert(severity="yellow")) == 0
    time.sleep(0.2)
    assert _Collector.received == []


def test_retry_then_success(collector):
    _Collector.fail_until = time.time() + 0.5
    notifier = AlertNotifier([{"url": collector, "retries": 5,
                               "backoff_sec": 0.05}])
    assert notifier.notify(_alert()) == 1
    deadline = time.time() + 6
    while not _Collector.received and time.time() < deadline:
        time.sleep(0.02)
    assert len(_Collector.received) == 1
    stats = notifier.stats()[0]
    assert stats["failed"] == 0
    assert stats["sent"] == 1


def test_exhausted_retries_tracked(collector):
    _Collector.fail_until = time.time() + 5
    notifier = AlertNotifier([{"url": collector, "retries": 1,
                               "backoff_sec": 0.05}])
    assert notifier.notify(_alert()) == 1
    deadline = time.time() + 6
    while True:
        stats = notifier.stats()[0]
        if stats["failed"] >= 1 or time.time() > deadline:
            break
        time.sleep(0.02)
    assert stats["failed"] == 1
    assert stats["sent"] == 0
    assert stats["last_error"]


def test_queue_full_drops_oldest():
    ch = _Channel({"url": "http://x", "min_severity": "yellow", "retries": 0,
                   "backoff_sec": 0.0})
    ch._q = _FullQueue()  # monkeypatch a queue that is always full
    assert ch.enqueue(_alert()) is False
    assert ch.stats()["dropped"] == 1


class _FullQueue:
    """Stand-in queue whose put_nowait always raises queue.Full."""

    def put_nowait(self, _item):
        import queue
        raise queue.Full


# ---------------------------------------------------------------- lifecycle
def test_channel_stats_and_bool():
    notifier = AlertNotifier([{"url": "http://x", "name": "a"},
                              {"name": "b", "url": ""}])
    assert bool(notifier)
    assert [c["name"] for c in notifier.stats()] == ["a"]
    assert AlertNotifier([]).__bool__() is False


def test_channels_from_config_prefers_structured():
    n = channels_from_config("http://legacy", [{"url": "http://new",
                                                "min_severity": "red"}])
    assert [c.url for c in n.channels] == ["http://new"]


def test_channels_from_config_falls_back_to_webhook():
    n = channels_from_config("http://legacy", None)
    assert len(n.channels) == 1
    assert n.channels[0].url == "http://legacy"
    assert n.channels[0].min_severity == "red"
    assert channels_from_config(None, None).__bool__() is False


def test_test_ping_enqueues_synthetic(collector):
    notifier = AlertNotifier([{"url": collector}])
    notifier.test()
    deadline = time.time() + 5
    while not _Collector.received and time.time() < deadline:
        time.sleep(0.02)
    assert _Collector.received[0]["alert"]["rule"] == "test"
