"""Edge Agent: lightweight single-camera surveillance agent.

Runs a minimal pipeline (detector + rules engine + optional audio) on a
single camera, stores alerts locally, and pushes upstream via HTTPS/MQTT.
Designed for resource-constrained devices (Raspberry Pi, Jetson Nano, etc.).

Usage:
    python -m bhairav.edge.agent --source rtsp://cam1 --upstream https://server/api/edge/alerts
    python -m bhairav.edge.agent --source 0 --mqtt-broker 192.168.1.100
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bhairav.config import load_config
from bhairav.pipeline import build_engine, make_detector, run_pipeline
from bhairav.types import Severity
from bhairav.edge.local_store import LocalAlertStore
from bhairav.edge.upstream import UpstreamPusher

log = logging.getLogger("bhairav.edge.agent")


class EdgeAgent:
    """Lightweight single-camera agent.

    Parameters
    ----------
    source : str
        Camera source (rtsp://, file path, webcam index, or "blob").
    upstream_url : str | None
        HTTPS webhook URL to push alerts to.
    mqtt_broker : str | None
        MQTT broker hostname.
    mqtt_port : int
        MQTT broker port.
    store_path : str | Path
        Local JSONL alert store path.
    config_path : str | None
        Path to a bhairav config.yaml for rules/zones.
    fps_cap : int
        Max frames per second to process (default 10 for edge devices).
    """

    def __init__(self, source="blob", upstream_url=None, mqtt_broker=None,
                 mqtt_port=1883, store_path="output/edge_alerts.jsonl",
                 config_path=None, fps_cap=10):
        self.source = source
        self.fps_cap = fps_cap
        self._stop = threading.Event()
        self._frame_count = 0
        self._alert_count = 0

        # Local store
        self.store = LocalAlertStore(store_path)
        self.store.prune()

        # Upstream pusher
        self.pusher = UpstreamPusher(
            store=self.store,
            url=upstream_url,
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            interval_sec=5.0)

        # Pipeline components
        if config_path:
            self.cfg = load_config(config_path)
        else:
            self.cfg = load_config("config.yaml")
        self.engine = build_engine(self.cfg)
        self.detector = make_detector(self.cfg, source=source)

    def _on_frame(self, state, alerts):
        if self._stop.is_set():
            return False
        if state.frame is None:
            return None
        self._frame_count += 1
        # Store alerts
        for a in alerts:
            ad = a.to_dict()
            ad["edge_agent"] = True
            self.store.append(ad)
            self._alert_count += 1
            # Log red alerts immediately
            if a.severity == Severity.RED:
                log.warning("RED ALERT: %s | %s", a.rule, a.message)
        return None

    def run(self):
        """Run the edge agent until interrupted."""
        self.pusher.start()
        log.info("Edge Agent starting: source=%s", self.source)
        delay = 1.0
        while not self._stop.is_set():
            try:
                run_pipeline(self.detector, self.engine,
                             source=self.source, on_frame=self._on_frame)
                # Source ended (file replay / stream EOF)
                log.info("Source ended, reconnecting...")
            except RuntimeError as exc:
                log.warning("Pipeline error: %s; retrying in %.0fs", exc, delay)
                self._stop.wait(delay)
                delay = min(delay * 2, 30.0)
                continue
            self._stop.wait(0.5)
            delay = 1.0
        self.pusher.stop()

    def stop(self):
        self._stop.set()
        self.pusher.stop()

    def snapshot(self):
        return {
            "source": self.source,
            "frames": self._frame_count,
            "alerts": self._alert_count,
            "pending": self.store.count,
            "pushed": self.pusher.stats["pushed"],
            "failed": self.pusher.stats["failed"],
        }


def main():
    ap = argparse.ArgumentParser(description="BHAIRAV Edge Agent (Phase 13)")
    ap.add_argument("--source", default="blob", help="camera source")
    ap.add_argument("--config", default="config.yaml", help="config file")
    ap.add_argument("--upstream", default=None, help="HTTPS webhook URL")
    ap.add_argument("--mqtt-broker", default=None, help="MQTT broker host")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--store", default="output/edge_alerts.jsonl")
    ap.add_argument("--fps-cap", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    agent = EdgeAgent(
        source=args.source,
        upstream_url=args.upstream,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        store_path=args.store,
        config_path=args.config,
        fps_cap=args.fps_cap)

    def _sigterm(sig, frame):
        log.info("Shutting down...")
        agent.stop()

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)

    print(f"Edge Agent: source={args.source}")
    if args.upstream:
        print(f"  upstream: {args.upstream}")
    if args.mqtt_broker:
        print(f"  mqtt: {args.mqtt_broker}:{args.mqtt_port}")
    print(f"  local store: {args.store}")

    agent.run()
    print(f"Stopped. {agent.snapshot()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
