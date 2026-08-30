"""Tests for the event bus and publish helpers."""
from bhairav.events import EventBus, Event, publish_alert, publish_frame


class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("alert", lambda e: received.append(e))
        bus.publish(Event(topic="alert", data={"rule": "test"}))
        assert len(received) == 1
        assert received[0].data["rule"] == "test"

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = {"a": 0, "b": 0}
        bus.subscribe("alert", lambda e: results.update({"a": results["a"] + 1}))
        bus.subscribe("alert", lambda e: results.update({"b": results["b"] + 1}))
        bus.publish(Event(topic="alert", data={}))
        assert results["a"] == 1
        assert results["b"] == 1

    def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e.topic))
        bus.publish(Event(topic="alert", data={}))
        bus.publish(Event(topic="frame", data={}))
        assert received == ["alert", "frame"]

    def test_unsubscribe(self):
        bus = EventBus()
        counter = [0]
        handler = lambda e: counter.__setitem__(0, counter[0] + 1)
        bus.subscribe("alert", handler)
        bus.publish(Event(topic="alert", data={}))
        assert counter[0] == 1
        bus.unsubscribe("alert", handler)
        bus.publish(Event(topic="alert", data={}))
        assert counter[0] == 1

    def test_subscriber_error_doesnt_break_pipeline(self):
        bus = EventBus()
        def bad_handler(e):
            raise ValueError("boom")
        good_received = []
        bus.subscribe("alert", bad_handler)
        bus.subscribe("alert", lambda e: good_received.append(True))
        bus.publish(Event(topic="alert", data={}))
        assert len(good_received) == 1

    def test_stats(self):
        bus = EventBus()
        bus.publish(Event(topic="alert", data={}))
        bus.publish(Event(topic="alert", data={}))
        bus.publish(Event(topic="frame", data={}))
        assert bus.stats() == {"alert": 2, "frame": 1}

    def test_subscriber_count(self):
        bus = EventBus()
        bus.subscribe("alert", lambda e: None)
        bus.subscribe("alert", lambda e: None)
        bus.subscribe("frame", lambda e: None)
        assert bus.subscriber_count("alert") == 2
        assert bus.subscriber_count("frame") == 1
        assert bus.subscriber_count() == 3

    def test_source_field(self):
        bus = EventBus()
        received = []
        bus.subscribe("alert", lambda e: received.append(e.source))
        bus.publish(Event(topic="alert", data={}, source="CAM-01"))
        assert received[0] == "CAM-01"


class TestPublishHelpers:
    def test_publish_alert(self):
        bus = EventBus()
        received = []
        bus.subscribe("alert", lambda e: received.append(e))
        publish_alert(bus, {"rule": "intrusion", "severity": "red"}, camera="CAM-01")
        assert len(received) == 1
        assert received[0].data["rule"] == "intrusion"
        assert received[0].source == "CAM-01"

    def test_publish_frame(self):
        bus = EventBus()
        received = []
        bus.subscribe("frame", lambda e: received.append(e))
        publish_frame(bus, frame_id=42, timestamp=1.0, tracks=[], camera="CAM-02")
        assert received[0].data["frame_id"] == 42
        assert received[0].source == "CAM-02"
