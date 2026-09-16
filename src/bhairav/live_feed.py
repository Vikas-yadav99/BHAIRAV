"""Live Feed Ingestion — pull from YouTube, HLS, or RTSP streams and run BHAIRAV detection.

Usage:
    # Single stream
    python -m bhairav.live_feed --url "https://www.youtube.com/watch?v=VIDEO_ID"

    # Multiple streams
    python -m bhairav.live_feed --config feeds.json

    # With custom settings
    python -m bhairav.live_feed --url "rtsp://..." --fps 5 --min-persons 3 --output output/live

feeds.json format:
    [
        {"id": "delhi-traffic", "url": "https://youtube.com/watch?v=...", "name": "Delhi Traffic"},
        {"id": "mumbai-crowd", "url": "https://youtube.com/watch?v=...", "name": "Mumbai Crowd"},
        {"id": "hls-cam1", "url": "https://example.com/stream.m3u8", "name": "Traffic Cam"}
    ]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("bhairav.live_feed")


@dataclass
class StreamConfig:
    """Configuration for a single live stream."""
    id: str
    url: str
    name: str = ""
    fps: int = 2  # Frames per second to process (lower = less CPU)
    min_persons: int = 1  # Minimum persons to trigger alert
    alert_threshold: float = 0.5  # YOLO confidence threshold


@dataclass
class StreamStats:
    """Runtime statistics for a stream."""
    frames_processed: int = 0
    persons_detected: int = 0
    vehicles_detected: int = 0
    alerts_fired: int = 0
    errors: int = 0
    last_frame_time: float = 0.0
    fps_actual: float = 0.0
    _fps_window_start: float = 0.0
    _fps_counter: int = 0


class StreamResolver:
    """Resolve stream URLs from various sources (YouTube, HLS, RTSP)."""

    @staticmethod
    def resolve_youtube(url: str) -> str | None:
        """Extract direct video stream URL from YouTube using yt-dlp."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "--no-download", "-g", url],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                # First line is video URL
                video_url = result.stdout.strip().split("\n")[0]
                log.info("Resolved YouTube URL: %s -> stream", url[:50])
                return video_url
            log.warning("yt-dlp failed for %s: %s", url[:50], result.stderr[:200])
            return None
        except Exception as exc:
            log.error("YouTube resolution failed: %s", exc)
            return None

    @staticmethod
    def resolve(url: str) -> str:
        """Resolve any URL to a direct stream URL."""
        if "youtube.com" in url or "youtu.be" in url:
            resolved = StreamResolver.resolve_youtube(url)
            if resolved:
                return resolved
            log.warning("Could not resolve YouTube URL, trying direct: %s", url)
        return url


class LiveFeedProcessor:
    """Process a single live stream with YOLO detection."""

    def __init__(self, config: StreamConfig, output_dir: str = "output/live"):
        self.config = config
        self.stats = StreamStats()
        self._running = False
        self._thread: threading.Thread | None = None
        self._output_dir = Path(output_dir) / config.id
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._detections_log = self._output_dir / "detections.jsonl"
        self._alert_log = self._output_dir / "alerts.jsonl"

    def start(self):
        """Start processing the stream in a background thread."""
        if self._running:
            log.warning("Stream %s already running", self.config.id)
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name=f"feed-{self.config.id}",
        )
        self._thread.start()
        log.info("Started feed processor: %s (%s)", self.config.name, self.config.id)

    def stop(self):
        """Stop processing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        log.info("Stopped feed: %s (processed %d frames, %d alerts)",
                 self.config.id, self.stats.frames_processed, self.stats.alerts_fired)

    def _process_loop(self):
        """Main processing loop — reads frames and runs detection."""
        try:
            import cv2
        except ImportError:
            log.error("cv2 not available — cannot process stream %s", self.config.id)
            self._running = False
            return

        # Resolve the stream URL
        stream_url = StreamResolver.resolve(self.config.url)
        log.info("Opening stream: %s", stream_url[:80])

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            log.error("Cannot open stream: %s", stream_url[:80])
            self.stats.errors += 1
            self._running = False
            return

        # Get video properties
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1.0 / self.config.fps
        self.stats._fps_window_start = time.time()

        log.info("Stream opened: %.1f FPS, processing every %.1fs",
                 orig_fps, frame_interval)

        # Load YOLO model
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            log.info("YOLO model loaded for %s", self.config.id)
        except Exception as exc:
            log.error("Cannot load YOLO: %s", exc)
            self._running = False
            return

        last_process_time = 0
        frame_count = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame read failed for %s, reconnecting...", self.config.id)
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    log.error("Reconnect failed for %s", self.config.id)
                    self.stats.errors += 1
                    break
                continue

            frame_count += 1
            now = time.time()

            # Process at configured FPS
            if now - last_process_time < frame_interval:
                continue

            last_process_time = now
            self.stats.frames_processed += 1
            self.stats.last_frame_time = now

            # Run YOLO detection
            try:
                results = model(frame, verbose=False, conf=self.config.alert_threshold)
            except Exception as exc:
                log.warning("Detection error: %s", exc)
                self.stats.errors += 1
                continue

            # Count detections
            persons = 0
            vehicles = 0
            detections = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # COCO class names: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
                    if cls == 0:
                        persons += 1
                    elif cls in (2, 3, 5, 7):
                        vehicles += 1

                    detections.append({
                        "class": cls,
                        "confidence": round(conf, 3),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    })

            self.stats.persons_detected += persons
            self.stats.vehicles_detected += vehicles

            # Log detection
            if detections:
                det_record = {
                    "timestamp": now,
                    "frame": frame_count,
                    "persons": persons,
                    "vehicles": vehicles,
                    "total": len(detections),
                    "detections": detections[:20],  # Cap for storage
                }
                with open(self._detections_log, "a") as f:
                    f.write(json.dumps(det_record) + "\n")

            # Fire alert if threshold exceeded
            if persons >= self.config.min_persons:
                self.stats.alerts_fired += 1
                alert = {
                    "timestamp": now,
                    "stream": self.config.id,
                    "type": "crowd_detection",
                    "persons": persons,
                    "vehicles": vehicles,
                    "confidence": max((d["confidence"] for d in detections), default=0),
                }
                with open(self._alert_log, "a") as f:
                    f.write(json.dumps(alert) + "\n")
                log.info("ALERT [%s]: %d persons, %d vehicles",
                         self.config.id, persons, vehicles)

            # FPS calculation
            self.stats._fps_counter += 1
            elapsed = now - self.stats._fps_window_start
            if elapsed >= 10.0:
                self.stats.fps_actual = self.stats._fps_counter / elapsed
                self.stats._fps_counter = 0
                self.stats._fps_window_start = now

            # Periodic status
            if self.stats.frames_processed % 50 == 0:
                log.info("Feed %s: %d frames, %.1f FPS, %d persons, %d vehicles, %d alerts",
                         self.config.id, self.stats.frames_processed,
                         self.stats.fps_actual, self.stats.persons_detected,
                         self.stats.vehicles_detected, self.stats.alerts_fired)

        cap.release()
        log.info("Feed %s processing loop ended", self.config.id)

    def get_status(self) -> dict:
        return {
            "id": self.config.id,
            "name": self.config.name,
            "url": self.config.url[:60],
            "running": self._running,
            "frames": self.stats.frames_processed,
            "persons": self.stats.persons_detected,
            "vehicles": self.stats.vehicles_detected,
            "alerts": self.stats.alerts_fired,
            "fps": round(self.stats.fps_actual, 1),
            "errors": self.stats.errors,
        }


class LiveFeedManager:
    """Manage multiple live feed processors."""

    def __init__(self, output_dir: str = "output/live"):
        self._feeds: dict[str, LiveFeedProcessor] = {}
        self._output_dir = output_dir

    def add_stream(self, config: StreamConfig) -> LiveFeedProcessor:
        """Add and start processing a stream."""
        processor = LiveFeedProcessor(config, self._output_dir)
        self._feeds[config.id] = processor
        processor.start()
        return processor

    def remove_stream(self, stream_id: str):
        """Stop and remove a stream."""
        if stream_id in self._feeds:
            self._feeds[stream_id].stop()
            del self._feeds[stream_id]

    def list_streams(self) -> list[dict]:
        return [p.get_status() for p in self._feeds.values()]

    def stop_all(self):
        for p in self._feeds.values():
            p.stop()
        self._feeds.clear()

    def load_config(self, path: str):
        """Load stream configs from a JSON file."""
        with open(path) as f:
            configs = json.load(f)
        for cfg in configs:
            self.add_stream(StreamConfig(
                id=cfg["id"],
                url=cfg["url"],
                name=cfg.get("name", cfg["id"]),
                fps=cfg.get("fps", 2),
                min_persons=cfg.get("min_persons", 1),
            ))
        log.info("Loaded %d streams from %s", len(configs), path)


# -- Default test streams ------------------------------------------------------

DEFAULT_STREAMS = [
    StreamConfig(
        id="yt-delhi-traffic",
        url="https://www.youtube.com/watch?v=8aINZD7m_s4",
        name="Delhi Traffic CCTV (YouTube)",
        fps=1,
        min_persons=2,
    ),
    StreamConfig(
        id="yt-indian-street",
        url="https://www.youtube.com/watch?v=KQX1x-rGVVk",
        name="Indian Street Live (YouTube)",
        fps=1,
        min_persons=2,
    ),
]


def main():
    parser = argparse.ArgumentParser(description="BHAIRAV Live Feed Ingestion")
    parser.add_argument("--url", help="Single stream URL to process")
    parser.add_argument("--config", help="JSON config file with stream list")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to process")
    parser.add_argument("--min-persons", type=int, default=2, help="Min persons for alert")
    parser.add_argument("--output", default="output/live", help="Output directory")
    parser.add_argument("--list", action="store_true", help="Use default test streams")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    manager = LiveFeedManager(args.output)

    if args.config:
        manager.load_config(args.config)
    elif args.url:
        manager.add_stream(StreamConfig(
            id="custom",
            url=args.url,
            name="Custom Stream",
            fps=args.fps,
            min_persons=args.min_persons,
        ))
    elif args.list:
        for cfg in DEFAULT_STREAMS:
            cfg.fps = args.fps
            cfg.min_persons = args.min_persons
            manager.add_stream(cfg)
    else:
        print("No streams specified. Use --url, --config, or --list")
        sys.exit(1)

    print(f"\nBHAIRAV Live Feed — {len(manager._feeds)} streams active")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(5)
            status = manager.list_streams()
            for s in status:
                print(f"  [{s['id']}] {s['frames']} frames, "
                      f"{s['persons']} persons, {s['vehicles']} vehicles, "
                      f"{s['alerts']} alerts, {s['fps']} FPS")
    except KeyboardInterrupt:
        print("\nStopping all feeds...")
        manager.stop_all()
        print("Done.")


if __name__ == "__main__":
    main()
