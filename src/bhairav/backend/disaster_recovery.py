"""Disaster Recovery & Business Continuity (Phase 27).

Automated backup verification, point-in-time recovery planning,
failover drills, health monitoring, and DR runbook generation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


@dataclass
class DRConfig:
    backup_dir: str = "./backups"
    backup_interval_hours: int = 6
    retention_days: int = 30
    verify_after_backup: bool = True
    health_check_interval_sec: int = 30
    failover_threshold: int = 3  # consecutive failures before failover
    rto_target_hours: float = 4.0  # Recovery Time Objective
    rpo_target_hours: float = 1.0  # Recovery Point Objective
    db_url: str = ""
    redis_url: str = ""
    upstream_url: str = ""


@dataclass
class BackupManifest:
    id: str = ""
    timestamp: float = 0
    type: str = "full"  # full, incremental, wal
    size_bytes: int = 0
    checksum: str = ""
    tables: list[str] = field(default_factory=list)
    verified: bool = False
    verification_ts: float = 0
    status: str = "pending"  # pending, completed, verified, failed


@dataclass
class FailoverState:
    primary_healthy: bool = True
    consecutive_failures: int = 0
    last_health_check: float = 0
    failover_active: bool = False
    failover_ts: float = 0
    recovery_ts: float = 0
    events: list[dict] = field(default_factory=list)


class BackupVerifier:
    """Automated backup verification and integrity checking."""

    def __init__(self, backup_dir: str, db_url: str = ""):
        self.backup_dir = Path(backup_dir)
        self.db_url = db_url

    def verify_latest(self) -> dict:
        """Verify the most recent backup file."""
        backups = sorted(self.backup_dir.glob("bhairav_*.backup.json.gz"), reverse=True)
        if not backups:
            return {"ok": False, "error": "No backups found"}
        return self.verify_file(backups[0])

    def verify_file(self, path: Path) -> dict:
        """Verify a single backup file."""
        import gzip
        try:
            raw = gzip.decompress(path.read_bytes())
            data = json.loads(raw.decode("utf-8"))

            checks = {
                "format_valid": data.get("format") == "bhairav-logical-backup",
                "has_tables": isinstance(data.get("tables"), list),
                "tables_populated": all(
                    isinstance(t.get("rows"), list) and isinstance(t.get("columns"), list)
                    for t in data.get("tables", [])
                ),
                "size_bytes": path.stat().st_size,
                "uncompressed_bytes": len(raw),
                "tables": [
                    {"name": t["name"], "columns": len(t["columns"]), "rows": len(t["rows"])}
                    for t in data.get("tables", [])
                ],
                "created_at": data.get("created_at"),
            }
            checks["ok"] = all([checks["format_valid"], checks["has_tables"], checks["tables_populated"]])
            return checks
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def schedule_verification(self, interval_hours: float = 24) -> dict:
        """Return schedule info for automated verification."""
        return {
            "interval_hours": interval_hours,
            "next_run": time.time() + interval_hours * 3600,
            "verify_after_backup": True,
        }


class FailoverManager:
    """Automated failover with health monitoring and recovery."""

    def __init__(self, config: DRConfig):
        self.config = config
        self.state = FailoverState()
        self._health_checkers: list[Callable] = []

    def register_health_check(self, fn: Callable):
        self._health_checkers.append(fn)

    def check_health(self) -> dict:
        """Run all health checks and update state."""
        results = {}
        all_healthy = True
        for checker in self._health_checkers:
            name = getattr(checker, "__name__", "checker")
            try:
                ok = checker()
                results[name] = {"healthy": ok}
                if not ok:
                    all_healthy = False
            except Exception as exc:
                results[name] = {"healthy": False, "error": str(exc)}
                all_healthy = False

        self.state.last_health_check = time.time()

        if all_healthy:
            self.state.consecutive_failures = 0
            if self.state.failover_active:
                self.state.failover_active = False
                self.state.recovery_ts = time.time()
                self._log_event("recovery", "Primary recovered, failover deactivated")
        else:
            self.state.consecutive_failures += 1
            if (self.state.consecutive_failures >= self.config.failover_threshold
                    and not self.state.failover_active):
                self.state.failover_active = True
                self.state.failover_ts = time.time()
                self._log_event("failover", f"Failover activated after {self.state.consecutive_failures} failures")

        return {
            "healthy": all_healthy,
            "failover_active": self.state.failover_active,
            "consecutive_failures": self.state.consecutive_failures,
            "checks": results,
            "last_check": self.state.last_health_check,
        }

    def _log_event(self, event_type: str, details: str):
        self.state.events.append({
            "ts": time.time(), "type": event_type, "details": details,
        })

    def get_rto_status(self) -> dict:
        """Recovery Time Objective status."""
        if not self.state.failover_active:
            return {"status": "ok", "rto_met": True}
        elapsed = time.time() - self.state.failover_ts
        target = self.config.rto_target_hours * 3600
        return {
            "status": "failover_active",
            "elapsed_sec": round(elapsed, 1),
            "rto_target_sec": target,
            "rto_met": elapsed <= target,
            "breach_risk": "high" if elapsed > target * 0.8 else "low",
        }

    def get_rpo_status(self, last_backup_ts: float) -> dict:
        """Recovery Point Objective status."""
        if not last_backup_ts:
            return {"status": "unknown", "rpo_met": False}
        gap = time.time() - last_backup_ts
        target = self.config.rpo_target_hours * 3600
        return {
            "status": "ok" if gap <= target else "breach",
            "data_loss_sec": round(gap, 1),
            "rpo_target_sec": target,
            "rpo_met": gap <= target,
        }


class DRRunbook:
    """Automated DR runbook generator and executor."""

    def __init__(self, config: DRConfig):
        self.config = config

    def generate_runbook(self) -> dict:
        """Generate a complete DR runbook."""
        return {
            "title": "BHAIRAV Disaster Recovery Runbook",
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "objectives": {
                "rto": f"{self.config.rto_target_hours} hours",
                "rpo": f"{self.config.rpo_target_hours} hours",
            },
            "scenarios": [
                {
                    "name": "Database Failure",
                    "severity": "critical",
                    "detection": [
                        "Health check fails for PostgreSQL connection",
                        "Alert: 'Database unreachable' in monitoring",
                    ],
                    "immediate_actions": [
                        "1. Verify failure is not network-related (ping, traceroute)",
                        "2. Check PostgreSQL service status: systemctl status postgresql",
                        "3. Review PostgreSQL logs: journalctl -u postgresql --since '1 hour ago'",
                    ],
                    "recovery_steps": [
                        "1. If service is down, restart: systemctl restart postgresql",
                        "2. If data corruption suspected, restore from latest backup:",
                        "2. Restore from latest backup using bhairav.backend.backups.restore()",
                        "3. Verify data integrity with pg_metrics()",
                        "4. Restart BHAIRAV service",
                        "5. Monitor for 1 hour post-recovery",
                    ],
                    "rollback": "Restore from the backup taken immediately before the failure",
                    "escalation": "If not resolved in 30 minutes, contact DBA team",
                },
                {
                    "name": "Redis Failure (HA Cluster)",
                    "severity": "high",
                    "detection": [
                        "Cluster leader election fails",
                        "Heartbeat timeouts on all nodes",
                    ],
                    "immediate_actions": [
                        "1. Verify Redis service: redis-cli ping",
                        "2. Check Redis logs for OOM or persistence errors",
                        "3. Verify memory usage: redis-cli info memory",
                    ],
                    "recovery_steps": [
                        "1. Restart Redis: systemctl restart redis",
                        "2. If data loss, BHAIRAV falls back to singleton mode (automatic)",
                        "3. Rejoin cluster: redis-cli cluster meet <node-ip> <port>",
                        "4. Verify cluster state: redis-cli cluster info",
                    ],
                    "rollback": "Redis failure should not affect BHAIRAV core — it degrades to single-node",
                },
                {
                    "name": "Camera Source Failure",
                    "severity": "medium",
                    "detection": [
                        "Camera shows 'offline' in dashboard",
                        "FPS drops to 0 for affected camera",
                    ],
                    "immediate_actions": [
                        "1. Check camera power and network connectivity",
                        "2. Verify RTSP/RTMP stream: ffprobe <stream-url>",
                        "3. Check if other cameras on same switch are affected",
                    ],
                    "recovery_steps": [
                        "1. Restart camera PoE if power issue",
                        "2. BHAIRAV auto-reconnects (exponential backoff up to 60s)",
                        "3. If persistent, replace camera or check network switch",
                        "4. Verify reconnection in dashboard Status tab",
                    ],
                    "rollback": "N/A — camera failure is isolated per-stream",
                },
                {
                    "name": "Full System Failure",
                    "severity": "critical",
                    "detection": [
                        "All health checks failing",
                        "No WebSocket connections possible",
                        "Dashboard unreachable",
                    ],
                    "immediate_actions": [
                        "1. Check host machine: SSH access, disk space, memory",
                        "2. Check Docker: docker ps, docker logs bhairav-app",
                        "3. Check system resources: top, df -h, free -m",
                    ],
                    "recovery_steps": [
                        "1. If OOM: kill excess processes, increase swap",
                        "2. If disk full: clear old logs/backups, increase volume",
                        "3. Restart full stack: docker compose -f deploy/docker-compose.yml up -d",
                        "4. Verify DB connectivity",
                        "5. Verify all camera reconnections",
                        "6. Restore from latest backup if data integrity suspect",
                        "7. Monitor for 4 hours post-recovery",
                    ],
                    "rollback": "Restore entire system from latest verified backup",
                    "escalation": "Immediate escalation to infrastructure team",
                },
            ],
            "backup_schedule": {
                "full_backup": f"Every {self.config.backup_interval_hours} hours",
                "retention": f"{self.config.retention_days} days",
                "verification": "Automated after each backup",
                "offsite": "Copy to separate storage location weekly",
            },
            "contact_chain": [
                "Level 1: On-call engineer (response: 15 min)",
                "Level 2: Senior engineer (response: 30 min)",
                "Level 3: Engineering lead (response: 1 hour)",
                "Level 4: CTO (response: 2 hours, for critical data loss)",
            ],
            "post_recovery_checklist": [
                "Verify all camera feeds are live",
                "Confirm alert pipeline is processing",
                "Check re-ID embeddings are populating",
                "Verify backup schedule resumed",
                "Review security audit log for anomalies",
                "Update incident report with timeline",
                "Schedule post-mortem meeting",
            ],
        }

    def generate_runbook_file(self, output_dir: str) -> str:
        """Write the runbook as a JSON file."""
        rb = self.generate_runbook()
        path = Path(output_dir) / "dr_runbook.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rb, indent=2))
        return str(path)


class PointInTimeRecovery:
    """Plan and execute point-in-time recovery using WAL archives."""

    def __init__(self, config: DRConfig):
        self.config = config

    def plan_recovery(self, target_time: float) -> dict:
        """Plan recovery to a specific point in time."""
        return {
            "target_time": target_time,
            "target_datetime": datetime.fromtimestamp(target_time, timezone.utc).isoformat(),
            "steps": [
                "1. Stop the BHAIRAV service to prevent new writes",
                "2. Identify the last full backup before target time",
                f"3. Restore full backup to {self.config.db_url}",
                "4. Apply WAL segments from backup time to target time",
                "5. Verify data integrity at target time",
                "6. Start BHAIRAV service",
                "7. Verify all subsystems operational",
            ],
            "estimated_duration": "30-60 minutes depending on WAL volume",
            "data_loss_window": f"Max {self.config.rpo_target_hours} hours of data",
        }

    def get_recovery_window(self) -> dict:
        """Get the available recovery time range."""
        return {
            "earliest_recovery": time.time() - (self.config.retention_days * 86400),
            "latest_recovery": time.time(),
            "backup_interval": f"{self.config.backup_interval_hours} hours",
            "retention_days": self.config.retention_days,
        }


class DRDashboard:
    """Unified DR status dashboard data provider."""

    def __init__(self, config: DRConfig):
        self.config = config
        self.verifier = BackupVerifier(config.backup_dir, config.db_url)
        self.failover = FailoverManager(config)
        self.runbook = DRRunbook(config)
        self.pitr = PointInTimeRecovery(config)

    def get_status(self) -> dict:
        """Complete DR status for the dashboard."""
        latest_backup = self.verifier.verify_latest()
        failover = self.failover.check_health()

        return {
            "timestamp": time.time(),
            "backup_status": latest_backup,
            "failover_status": failover,
            "rto_status": self.failover.get_rto_status(),
            "rpo_status": self.failover.get_rpo_status(
                latest_backup.get("created_at", 0) if latest_backup.get("ok") else 0
            ),
            "recovery_window": self.pitr.get_recovery_window(),
            "config": {
                "rto_target": f"{self.config.rto_target_hours}h",
                "rpo_target": f"{self.config.rpo_target_hours}h",
                "backup_interval": f"{self.config.backup_interval_hours}h",
                "retention_days": self.config.retention_days,
            },
        }

    def get_runbook(self) -> dict:
        return self.runbook.generate_runbook()
