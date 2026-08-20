"""Tests for Phase 13.3 - Multi-Site Federation."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhairav.federation.protocol import FederationMessage, MessageType
from bhairav.federation.client import FederationClient


class TestFederationMessage:
    def test_create_and_serialize(self):
        msg = FederationMessage(
            msg_type=MessageType.ALERT,
            source_site="site-west",
            payload={"rule": "fight", "severity": "red"})
        d = msg.to_dict()
        assert d["type"] == "alert"
        assert d["source_site"] == "site-west"
        assert d["payload"]["rule"] == "fight"

    def test_roundtrip(self):
        msg = FederationMessage(
            msg_type=MessageType.SIGHTING,
            source_site="site-east",
            payload={"person_id": "p123", "camera": "cam-1"})
        d = msg.to_dict()
        msg2 = FederationMessage.from_dict(d)
        assert msg2.msg_type == MessageType.SIGHTING
        assert msg2.source_site == "site-east"
        assert msg2.payload["person_id"] == "p123"

    def test_analytics_type(self):
        msg = FederationMessage(
            msg_type=MessageType.ANALYTICS,
            source_site="site-north",
            payload={"forecast": {"trend": "rising"}})
        assert msg.to_dict()["type"] == "analytics"

    def test_heartbeat_type(self):
        msg = FederationMessage(
            msg_type=MessageType.HEARTBEAT,
            source_site="site-south")
        assert msg.to_dict()["type"] == "heartbeat"


class TestFederationClient:
    def test_send_alert_queues(self):
        client = FederationClient(site_id="site-1", peers=[])
        client.send_alert({"rule": "fight", "ts": 1.0})
        assert client.pending == 1

    def test_send_sighting_queues(self):
        client = FederationClient(site_id="site-1", peers=[])
        client.send_sighting({"person_id": "p1"})
        assert client.pending == 1

    def test_send_analytics_queues(self):
        client = FederationClient(site_id="site-1", peers=[])
        client.send_analytics({"heatmap": {}})
        assert client.pending == 1

    def test_receive_processes(self):
        client = FederationClient(site_id="site-1")
        messages = [
            {"type": "alert", "source_site": "site-2",
             "payload": {"rule": "fight"}, "timestamp": time.time()},
            {"type": "sighting", "source_site": "site-2",
             "payload": {"person_id": "p1"}, "timestamp": time.time()},
        ]
        received = client.receive(messages)
        assert len(received) == 2
        assert received[0].msg_type == MessageType.ALERT
        assert received[1].msg_type == MessageType.SIGHTING
        assert client.stats["received"] == 2

    def test_stats(self):
        client = FederationClient(site_id="site-1")
        assert client.stats["sent"] == 0
        assert client.stats["failed"] == 0
        assert client.stats["received"] == 0

    def test_pending_empty(self):
        client = FederationClient(site_id="site-1")
        assert client.pending == 0
