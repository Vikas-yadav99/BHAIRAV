"""Event Bus: publish/subscribe system for decoupling pipeline modules.

Instead of hard-wiring every module into on_frame(), modules subscribe to
event types and receive callbacks when events fire. This replaces the
ad-hoc wiring in serve.py's 150-line on_frame callback.

Event types:
    alert       - A rule violation was detected
    frame       - A new frame was processed (for PTZ tracking)
    person      - A person was detected/re-identified (for identity)
    audio       - An audio event was detected
    evidence    - An evidence clip was created
    analytics   - Analytics snapshot ready for broadcast
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("bhairav.events")


@dataclass
class Event:
    """A single event on the bus."""
    topic: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # camera_id or module name


# Subscriber callback type: fn(event: Event) -> None
Subscriber = Callable[[Event], None]


class EventBus:
    """Thread-safe publish/subscribe event bus.

    Usage::

        bus = EventBus()
        bus.subscribe("alert", my_handler)
        bus.publish(Event(topic="alert", data={...}))

    Subscribers are called synchronously in the publishing thread.
    If a subscriber raises, the error is logged but does not prevent
    other subscribers from running.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stats: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        """Register a handler for a topic. Thread-safe."""
        with self._lock:
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Subscriber) -> None:
        """Remove a handler. Thread-safe."""
        with self._lock:
            subs = self._subscribers.get(topic, [])
            if handler in subs:
                subs.remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers of its topic.

        Handlers are called synchronously. Errors are logged but swallowed
        so one broken subscriber cannot crash the pipeline.
        """
        self._stats[event.topic] += 1
        with self._lock:
            handlers = list(self._subscribers.get(event.topic, []))
            # Also notify wildcard subscribers
            if event.topic != "*":
                handlers.extend(self._subscribers.get("*", []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                log.error("EventBus: subscriber %s for topic '%s' failed: %s",
                          getattr(handler, "__name__", str(handler)),
                          event.topic, exc)

    def stats(self) -> dict[str, int]:
        """Return publish counts per topic."""
        return dict(self._stats)

    def subscriber_count(self, topic: str = "") -> int:
        """Number of subscribers for a topic (or total if empty)."""
        with self._lock:
            if topic:
                return len(self._subscribers.get(topic, []))
            return sum(len(v) for v in self._subscribers.values())


# --- Convenience publishers for common event types ---

def publish_alert(bus: EventBus, alert_dict: dict, camera: str = "") -> None:
    """Publish an alert event."""
    bus.publish(Event(topic="alert", data=alert_dict, source=camera))


def publish_frame(bus: EventBus, frame_id: int, timestamp: float,
                  tracks: list, camera: str = "") -> None:
    """Publish a frame event (for PTZ tracking, analytics)."""
    bus.publish(Event(topic="frame", data={
        "frame_id": frame_id,
        "timestamp": timestamp,
        "tracks": tracks,
    }, source=camera))


def publish_person(bus: EventBus, person_id: str, track_id: int,
                   camera: str = "", embedding=None) -> None:
    """Publish a person detection/re-id event."""
    bus.publish(Event(topic="person", data={
        "person_id": person_id,
        "track_id": track_id,
        "has_embedding": embedding is not None,
    }, source=camera))


def publish_audio(bus: EventBus, audio_event: dict, camera: str = "") -> None:
    """Publish an audio detection event."""
    bus.publish(Event(topic="audio", data=audio_event, source=camera))


def publish_evidence(bus: EventBus, evidence_dict: dict, camera: str = "") -> None:
    """Publish an evidence creation event."""
    bus.publish(Event(topic="evidence", data=evidence_dict, source=camera))
