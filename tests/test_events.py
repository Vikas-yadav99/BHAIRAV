"""Tests for the event bus and subscriber wiring."""
from bhairav.events import EventBus, Event, publish_alert, publish_frame
from bhairav.subscribers import (
    EscalationSubscriber, PTZSubscriber, IntegrationSubscriber,
    FederationSubscriber, AuditSubscriber, wire_subscribers,
)


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
        assert counter[0] == 1  # not incremented

    def test_subscriber_error_doesnt_break_pipeline(self):
        bus = EventBus()
        def bad_handler(e):
            raise ValueError("boom")
        good_received = []
        bus.subscribe("alert", bad_handler)
        bus.subscribe("alert", lambda e: good_received.append(True))
        bus.publish(Event(topic="alert", data={}))
        assert len(good_received) == 1  # good handler still ran

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


class TestEscalationSubscriber:
    def test_red_alert_triggers_escalation(self):
        class FakeEngine:
            def __init__(self):
                self.calls = []
            def evaluate(self, alert):
                self.calls.append(alert)
                return [{"escalated": True}]
        class FakeHub:
            def __init__(self):
                self.calls = []
            def publish_field_alert(self, a):
                self.calls.append(a)

        engine = FakeEngine()
        hub = FakeHub()
        sub = EscalationSubscriber(engine, hub=hub)
        sub(Event(topic="alert", data={"severity": "red", "rule": "intrusion"}))
        assert len(engine.calls) == 1
        assert len(hub.calls) == 1

    def test_yellow_alert_skipped(self):
        class FakeEngine:
            calls = []
            def evaluate(self, alert):
                self.calls.append(alert)
                return []
        engine = FakeEngine()
        sub = EscalationSubscriber(engine)
        sub(Event(topic="alert", data={"severity": "yellow", "rule": "loitering"}))
        assert len(engine.calls) == 0


class TestPTZSubscriber:
    def test_person_track_updates_tracker(self):
        class FakeTracker:
            def __init__(self):
                self.calls = []
            def update(self, tracks, ts):
                self.calls.append((len(tracks), ts))

        tracker = FakeTracker()
        sub = PTZSubscriber(tracker)
        sub(Event(topic="frame", data={
            "tracks": [{"id": 1, "label": "person", "bbox": [100, 100, 200, 300]}],
            "timestamp": 123.4,
        }))
        assert len(tracker.calls) == 1
        assert tracker.calls[0] == (1, 123.4)

    def test_no_persons_skipped(self):
        class FakeTracker:
            calls = []
            def update(self, tracks, ts):
                self.calls.append(tracks)
        tracker = FakeTracker()
        sub = PTZSubscriber(tracker)
        sub(Event(topic="frame", data={
            "tracks": [{"id": 1, "label": "car", "bbox": [0, 0, 100, 100]}],
            "timestamp": 0,
        }))
        assert len(tracker.calls) == 0


class TestAuditSubscriber:
    def test_logs_alert(self):
        class FakeAuditLog:
            def __init__(self):
                self.events = []
            def log(self, **kwargs):
                self.events.append(kwargs)

        log = FakeAuditLog()
        sub = AuditSubscriber(log)
        sub(Event(topic="alert", data={
            "rule": "fight", "severity": "red", "zone": "plaza"
        }, source="CAM-01"))
        assert len(log.events) == 1
        assert "fight" in log.events[0]["details"]
        assert log.events[0]["severity"] == "warning"


class TestWireSubscribers:
    def test_wire_all(self):
        bus = EventBus()

        class FakeEngine:
            def evaluate(self, a): return []
        class FakeTracker:
            def update(self, t, ts): pass
        class FakeHub:
            def dispatch(self, a): pass
            def publish_field_alert(self, a): pass
        class FakeClient:
            def send_alert(self, a): pass
        class FakeAudit:
            def log(self, **kwargs): pass

        active = wire_subscribers(
            bus,
            escalation_engine=FakeEngine(),
            ptz_tracker=FakeTracker(),
            integration_hub=FakeHub(),
            federation_client=FakeClient(),
            audit_log=FakeAudit(),
            live_hub=FakeHub(),
        )
        assert "escalation" in active
        assert "ptz" in active
        assert "integrations" in active
        assert "federation" in active
        assert "audit" in active
        assert bus.subscriber_count() >= 5

    def test_skip_none_modules(self):
        bus = EventBus()
        active = wire_subscribers(bus)
        assert active == {}
        assert bus.subscriber_count() == 0
