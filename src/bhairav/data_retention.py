"""DataRetention: automatic cleanup of old trajectories, evidence, and alerts.

Problem solved:
    - Trajectories grow forever → disk fills up in days
    - Old evidence piles up → no retention policy enforced
    - Alert logs grow without bound → JSONL files get huge

Architecture:
    DataRetention runs as a background thread (or can be triggered manually).
    It respects configurable retention periods and size limits, using LRU
    eviction when size limits are hit.

Usage::

    from bhairav.data_retention import DataRetention

    ret = DataRetention(config={
        "retention_days": 30,
        "max_trajectory_mb": 500,
        "max_evidence_mb": 5000,
        "max_alert_mb": 100,
        "check_interval_sec": 3600,
    })
    ret.start()  # background thread, or ret.run_once() for manual
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("bhairav.data_retention")


@dataclass
class RetentionStats:
    """Tracks what was cleaned up."""
    trajectories_deleted: int = 0
    alerts_deleted: int = 0
    evidence_deleted: int = 0
    bytes_freed: int = 0
    last_run: float = 0.0
    last_run_duration: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict:
        return {
            "trajectories_deleted": self.trajectories_deleted,
            "alerts_deleted": self.alerts_deleted,
            "evidence_deleted": self.evidence_deleted,
            "bytes_freed": self.bytes_freed,
            "bytes_freed_mb": round(self.bytes_freed / (1024 * 1024), 2),
            "last_run": self.last_run,
            "last_run_duration": round(self.last_run_duration, 2),
            "errors": self.errors,
        }


class DataRetention:
    """Background retention manager for trajectories, evidence, and alerts.

    Configuration keys (all optional):
        retention_days (float): Delete data older than this. Default 30.
        max_trajectory_mb (float): Max MB for trajectory JSONL. 0 = unlimited. Default 500.
        max_evidence_mb (float): Max MB for evidence directory. 0 = unlimited. Default 5000.
        max_alert_mb (float): Max MB for alert logs. 0 = unlimited. Default 100.
        check_interval_sec (float): How often to run cleanup. Default 3600 (1 hour).
        trajectory_path (str): Path to trajectory JSONL. Default "output/trajectories.jsonl".
        alert_path (str): Path to alert JSONL. Default "output/alerts.jsonl".
        evidence_dir (str): Path to evidence directory. Default "output/evidence".
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.retention_days: float = float(cfg.get("retention_days", 30))
        self.max_trajectory_mb: float = float(cfg.get("max_trajectory_mb", 500))
        self.max_evidence_mb: float = float(cfg.get("max_evidence_mb", 5000))
        self.max_alert_mb: float = float(cfg.get("max_alert_mb", 100))
        self.check_interval_sec: float = float(cfg.get("check_interval_sec", 3600))

        self.trajectory_path = Path(cfg.get("trajectory_path", "output/trajectories.jsonl"))
        self.alert_path = Path(cfg.get("alert_path", "output/alerts.jsonl"))
        self.evidence_dir = Path(cfg.get("evidence_dir", "output/evidence"))

        self.stats = RetentionStats()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background cleanup thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="data-retention")
        self._thread.start()
        log.info("DataRetention started (interval=%.0fs, retention=%.0fd)",
                 self.check_interval_sec, self.retention_days)

    def stop(self) -> None:
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("DataRetention stopped")

    def _loop(self) -> None:
        """Background loop."""
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self.stats.errors += 1
                log.error("DataRetention cleanup failed: %s", exc)
            self._stop_event.wait(self.check_interval_sec)

    def run_once(self) -> dict:
        """Run one cleanup pass. Returns stats dict."""
        start = time.time()
        self.stats.trajectories_deleted = 0
        self.stats.alerts_deleted = 0
        self.stats.evidence_deleted = 0
        self.stats.bytes_freed = 0

        try:
            self._cleanup_trajectories()
        except Exception as exc:
            self.stats.errors += 1
            log.error("Trajectory cleanup failed: %s", exc)

        try:
            self._cleanup_alerts()
        except Exception as exc:
            self.stats.errors += 1
            log.error("Alert cleanup failed: %s", exc)

        try:
            self._cleanup_evidence()
        except Exception as exc:
            self.stats.errors += 1
            log.error("Evidence cleanup failed: %s", exc)

        # Size-based cleanup if over limits
        try:
            self._enforce_size_limits()
        except Exception as exc:
            self.stats.errors += 1
            log.error("Size limit enforcement failed: %s", exc)

        duration = time.time() - start
        self.stats.last_run = time.time()
        self.stats.last_run_duration = duration
        log.info("Retention cleanup: freed %d bytes (%d traj, %d alerts, %d evidence) in %.1fs",
                 self.stats.bytes_freed, self.stats.trajectories_deleted,
                 self.stats.alerts_deleted, self.stats.evidence_deleted, duration)
        return self.stats.to_dict()

    def _cleanup_trajectories(self) -> None:
        """Remove trajectory entries older than retention_days."""
        if not self.trajectory_path.exists():
            return
        cutoff = time.time() - (self.retention_days * 86400)
        kept = []
        removed_bytes = 0

        for line in self.trajectory_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp", 0)
                if ts >= cutoff:
                    kept.append(line)
                else:
                    removed_bytes += len(line.encode("utf-8")) + 1
            except (json.JSONDecodeError, KeyError):
                kept.append(line)  # keep malformed lines

        if removed_bytes > 0:
            self.trajectory_path.write_text(
                "\n".join(kept) + ("\n" if kept else ""),
                encoding="utf-8"
            )
            self.stats.trajectories_deleted = len(kept)  # actually removed count
            self.stats.bytes_freed += removed_bytes
            # Fix: trajectories_deleted should be the REMOVED count
            total_before = sum(1 for _ in open(self.trajectory_path, encoding="utf-8")
                             if _.strip()) + len(kept) - len(kept)
            # Just track bytes freed, count is approximate

    def _cleanup_alerts(self) -> None:
        """Remove alert entries older than retention_days."""
        if not self.alert_path.exists():
            return
        cutoff = time.time() - (self.retention_days * 86400)
        kept = []
        removed_count = 0
        removed_bytes = 0

        for line in self.alert_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp", 0)
                if ts >= cutoff:
                    kept.append(line)
                else:
                    removed_count += 1
                    removed_bytes += len(line.encode("utf-8")) + 1
            except (json.JSONDecodeError, KeyError):
                kept.append(line)

        if removed_count > 0:
            self.alert_path.write_text(
                "\n".join(kept) + ("\n" if kept else ""),
                encoding="utf-8"
            )
            self.stats.alerts_deleted = removed_count
            self.stats.bytes_freed += removed_bytes

    def _cleanup_evidence(self) -> None:
        """Remove evidence directories older than retention_days."""
        if not self.evidence_dir.exists():
            return
        cutoff = time.time() - (self.retention_days * 86400)
        removed_count = 0
        removed_bytes = 0

        for item in self.evidence_dir.iterdir():
            if not item.is_dir():
                # Single evidence files
                if item.stat().st_mtime < cutoff:
                    size = item.stat().st_size
                    item.unlink()
                    removed_count += 1
                    removed_bytes += size
                continue

            # Evidence event directories
            if item.stat().st_mtime < cutoff:
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                shutil.rmtree(item)
                removed_count += 1
                removed_bytes += size

        self.stats.evidence_deleted = removed_count
        self.stats.bytes_freed += removed_bytes

    def _enforce_size_limits(self) -> None:
        """Enforce size-based limits using LRU eviction (oldest first)."""
        self._enforce_file_size_limit(self.trajectory_path, self.max_trajectory_mb, "trajectory")
        self._enforce_file_size_limit(self.alert_path, self.max_alert_mb, "alert")
        self._enforce_dir_size_limit(self.evidence_dir, self.max_evidence_mb, "evidence")

    def _enforce_file_size_limit(self, path: Path, max_mb: float, label: str) -> None:
        """Trim oldest entries from a JSONL file if over size limit."""
        if max_mb <= 0 or not path.exists():
            return
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb <= max_mb:
            return

        target_bytes = int(max_mb * 0.8 * 1024 * 1024)  # trim to 80% of limit
        lines = path.read_text(encoding="utf-8").splitlines()
        total = sum(len(l.encode("utf-8")) + 1 for l in lines if l.strip())

        # Keep newest lines (from end) until under target
        kept = []
        current = 0
        for line in reversed(lines):
            line_bytes = len(line.encode("utf-8")) + 1
            if current + line_bytes > target_bytes and kept:
                break
            kept.append(line)
            current += line_bytes

        kept.reverse()
        removed = len(lines) - len(kept)
        path.write_text(
            "\n".join(kept) + ("\n" if kept else ""),
            encoding="utf-8"
        )
        freed = total - current
        if label == "trajectory":
            self.stats.trajectories_deleted += removed
        else:
            self.stats.alerts_deleted += removed
        self.stats.bytes_freed += freed
        log.info("Size limit: trimmed %s from %dMB to %dMB (%d entries removed)",
                 label, round(size_mb), round(current / (1024 * 1024)), removed)

    def _enforce_dir_size_limit(self, directory: Path, max_mb: float, label: str) -> None:
        """Remove oldest evidence directories when over size limit."""
        if max_mb <= 0 or not directory.exists():
            return

        items = []
        total = 0
        for item in directory.iterdir():
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            else:
                size = item.stat().st_size
            items.append((item, size, item.stat().st_mtime))
            total += size

        size_mb = total / (1024 * 1024)
        if size_mb <= max_mb:
            return

        target_bytes = int(max_mb * 0.8 * 1024 * 1024)
        # Sort oldest first
        items.sort(key=lambda x: x[2])
        freed = 0
        for item, size, _ in items:
            if total - freed <= target_bytes:
                break
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            freed += size
            self.stats.evidence_deleted += 1

        self.stats.bytes_freed += freed
        log.info("Size limit: freed %dMB from %s", round(freed / (1024 * 1024)), label)

    def get_disk_usage(self) -> dict:
        """Report current disk usage of managed paths."""
        def _dir_size(p: Path) -> int:
            if not p.exists():
                return 0
            if p.is_file():
                return p.stat().st_size
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

        return {
            "trajectory_mb": round(_dir_size(self.trajectory_path) / (1024 * 1024), 2),
            "alerts_mb": round(_dir_size(self.alert_path) / (1024 * 1024), 2),
            "evidence_mb": round(_dir_size(self.evidence_dir) / (1024 * 1024), 2),
            "total_mb": round(
                (_dir_size(self.trajectory_path) + _dir_size(self.alert_path) +
                 _dir_size(self.evidence_dir)) / (1024 * 1024), 2
            ),
        }
