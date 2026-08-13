"""Phase 10 M4 - field-officer alert dispatch.

`AlertNotifier` fans alerts out to one or more outbound channels (webhook
endpoints such as Slack / Telegram / an SMS gateway). Each channel:
  - filters by `min_severity` and an optional `rules` allow-list
  - delivers on its own daemon worker thread through a bounded queue, so a
    slow/unreachable endpoint can never stall the pipeline or other channels
  - retries with exponential backoff before giving up (best-effort; the alert
    is already in the feed + audit trail)

`notify()` is synchronous and non-blocking (queue put, drop-oldest on full).
Pure stdlib - no new dependencies.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request

SEVERITY_RANK = {"yellow": 1, "orange": 2, "red": 3}


class _Channel:
    """One outbound destination with its own queue + worker thread."""

    def __init__(self, cfg: dict):
        self.name = str(cfg.get("name") or "channel")
        self.url = str(cfg.get("url") or "").strip()
        self.min_severity = str(cfg.get("min_severity") or "orange").lower()
        self.rules = set(str(r) for r in (cfg.get("rules") or []))
        self.retries = max(0, int(cfg.get("retries", 2)))
        self.backoff_sec = max(0.0, float(cfg.get("backoff_sec", 1.0)))
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._lock = threading.Lock()
        self._delivered = 0      # alerts accepted into this channel's queue
        self._sent = 0           # alerts POSTed successfully
        self._failed = 0         # alerts that exhausted their retries
        self._dropped = 0        # alerts dropped because the queue was full
        self._last_error: str | None = None
        self._last_sent_at: float | None = None
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name=f"bhairav-notify-{self.name}")
        self._worker.start()

    # ---- matching ---------------------------------------------------------
    def matches(self, alert: dict) -> bool:
        if not self.url:
            return False
        sev = str(alert.get("severity", "")).lower()
        if SEVERITY_RANK.get(sev, 0) < SEVERITY_RANK.get(self.min_severity, 3):
            return False
        if self.rules and str(alert.get("rule", "")) not in self.rules:
            return False
        return True

    # ---- pipeline side (never blocks) ------------------------------------
    def enqueue(self, alert: dict) -> bool:
        try:
            self._q.put_nowait(dict(alert))
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False
        with self._lock:
            self._delivered += 1
        return True

    def test(self) -> None:
        """Enqueue a synthetic alert so an operator can verify delivery."""
        self.enqueue({
            "rule": "test", "severity": "red", "message": "BHAIRAV dispatch test",
            "zone": None, "track_id": None, "frame_id": 0,
            "timestamp": round(time.time(), 3), "confidence": 1.0,
            "details": {}, "camera": "CAM-01",
        })

    # ---- worker thread ----------------------------------------------------
    def _run(self) -> None:
        while True:
            alert = self._q.get()
            try:
                self._deliver(alert)
            finally:
                self._q.task_done()

    def _deliver(self, alert: dict) -> None:
        payload = json.dumps({"type": "bhairav_alert", "alert": alert},
                             separators=(",", ":")).encode("utf-8")
        delay = self.backoff_sec
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, data=payload,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5):
                    pass
                with self._lock:
                    self._sent += 1
                    self._last_sent_at = time.time()
                    self._last_error = None
                return
            except Exception as exc:  # best-effort: never crash the notifier
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                with self._lock:
                    self._failed += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"

    def stats(self) -> dict:
        with self._lock:
            return {"name": self.name, "url": self.url,
                    "min_severity": self.min_severity,
                    "rules": sorted(self.rules), "retries": self.retries,
                    "delivered": self._delivered, "sent": self._sent,
                    "failed": self._failed, "dropped": self._dropped,
                    "last_error": self._last_error,
                    "last_sent_at": self._last_sent_at}


class AlertNotifier:
    """Fan-out dispatcher over a list of channel configs."""

    def __init__(self, channels: list[dict]):
        self.channels: list[_Channel] = []
        for cfg in channels or []:
            if cfg and str(cfg.get("url") or "").strip():
                self.channels.append(_Channel(cfg))

    def __bool__(self) -> bool:
        return bool(self.channels)

    def notify(self, alert: dict) -> int:
        """Queue an alert to every matching channel.

        Returns how many channels accepted it. Never blocks, never raises.
        """
        n = 0
        for ch in self.channels:
            if ch.matches(alert) and ch.enqueue(alert):
                n += 1
        return n

    def test(self) -> None:
        for ch in self.channels:
            ch.test()

    def stats(self) -> list[dict]:
        return [ch.stats() for ch in self.channels]


def channels_from_config(webhook_url: str | None,
                         alert_channels: list[dict] | None) -> AlertNotifier:
    """Build the notifier from config.

    `alert_channels` (Phase 10 M4) wins when present; otherwise a single
    channel is derived from the legacy `backend.webhook_url` (red-only, to
    match the historical behaviour) so existing setups keep working.
    """
    if alert_channels:
        return AlertNotifier(alert_channels)
    if webhook_url:
        return AlertNotifier([{"name": "webhook", "url": webhook_url,
                               "min_severity": "red"}])
    return AlertNotifier([])
