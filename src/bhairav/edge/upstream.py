"""Upstream pusher: sends alerts from local store to a remote server.

Supports HTTPS webhook (POST JSON array) and MQTT publish (QoS 1).
Runs on a background thread, periodically draining the local store.
"""
from __future__ import annotations

import json
import logging
import threading
from urllib.request import Request, urlopen
from urllib.error import URLError

log = logging.getLogger("bhairav.edge.upstream")

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False


class UpstreamPusher:
    """Periodically drains LocalAlertStore and pushes upstream."""

    def __init__(self, store, url=None, mqtt_broker=None, mqtt_port=1883,
                 mqtt_topic="bhairav/alerts", interval_sec=5.0, max_batch=100):
        self.store = store
        self.url = url
        self.mqtt_topic = mqtt_topic
        self.interval_sec = interval_sec
        self.max_batch = max_batch
        self._stop = threading.Event()
        self._thread = None
        self._mqtt_client = None
        self.stats = {"pushed": 0, "failed": 0, "retried": 0}
        if mqtt_broker and HAS_MQTT:
            try:
                self._mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                self._mqtt_client.connect(mqtt_broker, mqtt_port, 60)
                self._mqtt_client.loop_start()
            except Exception:
                self._mqtt_client = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._mqtt_client:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                self._drain()
            except Exception as exc:
                log.error("Push cycle error: %s", exc)
            self._stop.wait(self.interval_sec)

    def _drain(self):
        alerts = self.store.read_and_clear()
        if not alerts:
            return
        batch = alerts[:self.max_batch]
        leftover = alerts[self.max_batch:]
        success = False
        if self.url:
            success = self._push_https(batch)
        if self._mqtt_client:
            self._push_mqtt(batch)
            success = True
        if success:
            self.stats["pushed"] += len(batch)
        else:
            self.stats["failed"] += len(batch)
            self.stats["retried"] += len(batch)
            for a in reversed(leftover + batch):
                self.store.append(a)
        for a in leftover:
            self.store.append(a)

    def _push_https(self, alerts):
        try:
            data = json.dumps(alerts, default=str).encode("utf-8")
            req = Request(self.url, data=data,
                          headers={"Content-Type": "application/json"},
                          method="POST")
            with urlopen(req, timeout=10) as resp:
                return resp.status < 400
        except (URLError, OSError, TimeoutError):
            return False

    def _push_mqtt(self, alerts):
        for alert in alerts:
            try:
                self._mqtt_client.publish(
                    self.mqtt_topic,
                    json.dumps(alert, default=str), qos=1)
            except Exception:
                pass

    def push_now(self):
        before = self.stats["pushed"]
        self._drain()
        return self.stats["pushed"] - before
