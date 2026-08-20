"""Federation client: shares alerts, re-ID sightings, and analytics with peer servers.

Each BHAIRAV server can federate with N peers. Outbound: alerts and analytics
are pushed via HTTP POST. Inbound: a /api/federation/ingest endpoint accepts
messages from peers and replays them into the local alert log + analytics.
"""
from __future__ import annotations

import json
import logging
import threading
from urllib.request import Request, urlopen
from urllib.error import URLError

from .protocol import FederationMessage, MessageType

log = logging.getLogger("bhairav.federation")


class FederationClient:
    """Push alerts/sightings/analytics to peer BHAIRAV servers.

    Parameters
    ----------
    site_id : str
        Unique identifier for this site (e.g., "site-west").
    peers : list[str]
        List of peer server URLs (e.g., ["https://peer1:8000", "..."]).
    secret : str
        Shared federation secret for HMAC auth.
    push_interval : float
        Seconds between outbound push cycles (default 10).
    """

    def __init__(self, site_id, peers=None, secret="", push_interval=10.0):
        self.site_id = site_id
        self.peers = peers or []
        self.secret = secret
        self.push_interval = push_interval
        self._stop = threading.Event()
        self._thread = None
        self._outbox: list[dict] = []
        self._lock = threading.Lock()
        self.stats = {"sent": 0, "failed": 0, "received": 0}

    def send_alert(self, alert: dict) -> None:
        """Queue an alert for federation to all peers."""
        msg = FederationMessage(
            msg_type=MessageType.ALERT,
            source_site=self.site_id,
            payload=alert)
        with self._lock:
            self._outbox.append(msg.to_dict())

    def send_sighting(self, sighting: dict) -> None:
        """Queue a re-ID sighting for federation."""
        msg = FederationMessage(
            msg_type=MessageType.SIGHTING,
            source_site=self.site_id,
            payload=sighting)
        with self._lock:
            self._outbox.append(msg.to_dict())

    def send_analytics(self, snapshot: dict) -> None:
        """Queue an analytics snapshot for federation."""
        msg = FederationMessage(
            msg_type=MessageType.ANALYTICS,
            source_site=self.site_id,
            payload=snapshot)
        with self._lock:
            self._outbox.append(msg.to_dict())

    def start(self):
        """Start background push thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Federation client started: site=%s peers=%d",
                 self.site_id, len(self.peers))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._push_cycle()
            except Exception as exc:
                log.error("Federation push error: %s", exc)
            self._stop.wait(self.push_interval)

    def _push_cycle(self):
        with self._lock:
            batch = self._outbox[:]
            self._outbox.clear()
        if not batch:
            return
        data = json.dumps(batch, default=str).encode("utf-8")
        for peer in self.peers:
            try:
                url = peer.rstrip("/") + "/api/federation/ingest"
                req = Request(url, data=data,
                              headers={"Content-Type": "application/json",
                                       "X-Federation-Site": self.site_id,
                                       "X-Federation-Secret": self.secret},
                              method="POST")
                with urlopen(req, timeout=10) as resp:
                    if resp.status < 400:
                        self.stats["sent"] += len(batch)
                    else:
                        self.stats["failed"] += len(batch)
            except (URLError, OSError) as exc:
                log.warning("Federation push to %s failed: %s", peer, exc)
                self.stats["failed"] += len(batch)
                # Re-queue
                with self._lock:
                    self._outbox.extend(batch)

    def receive(self, messages: list[dict]) -> list[FederationMessage]:
        """Process inbound federation messages from /api/federation/ingest."""
        received = []
        for d in messages:
            try:
                msg = FederationMessage.from_dict(d)
                received.append(msg)
                self.stats["received"] += 1
            except Exception as exc:
                log.warning("Federation message parse error: %s", exc)
        return received

    @property
    def pending(self):
        with self._lock:
            return len(self._outbox)
