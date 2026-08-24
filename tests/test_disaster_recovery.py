"""Disaster Recovery module tests."""
import json
import time
import gzip
from pathlib import Path
from bhairav.backend.disaster_recovery import (
    DRConfig, BackupVerifier, FailoverManager, DRRunbook,
    PointInTimeRecovery, DRDashboard,
)


class TestBackupVerifier:
    def test_verify_latest_empty(self, tmp_path):
        v = BackupVerifier(str(tmp_path))
        r = v.verify_latest()
        assert r["ok"] is False

    def test_verify_valid_backup(self, tmp_path):
        # Create a minimal backup
        payload = {
            "format": "bhairav-logical-backup", "version": 1,
            "created_at": time.time(),
            "tables": [{"name": "users", "columns": [{"name": "id", "type": "integer"}], "rows": [[1]]}],
        }
        import datetime
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = tmp_path / f"bhairav_{stamp}.backup.json.gz"
        path.write_bytes(gzip.compress(json.dumps(payload).encode()))

        v = BackupVerifier(str(tmp_path))
        r = v.verify_latest()
        assert r["ok"] is True
        assert len(r["tables"]) == 1
        assert r["tables"][0]["rows"] == 1

    def test_verify_corrupt_backup(self, tmp_path):
        path = tmp_path / "bhairav_20260101_120000.backup.json.gz"
        path.write_bytes(b"not-gzip-data")
        v = BackupVerifier(str(tmp_path))
        r = v.verify_latest()
        assert r["ok"] is False

    def test_verify_wrong_format(self, tmp_path):
        payload = {"format": "wrong", "version": 1, "tables": []}
        import datetime
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = tmp_path / f"bhairav_{stamp}.backup.json.gz"
        path.write_bytes(gzip.compress(json.dumps(payload).encode()))
        v = BackupVerifier(str(tmp_path))
        r = v.verify_latest()
        assert r["ok"] is False


class TestFailoverManager:
    def test_healthy_state(self):
        fm = FailoverManager(DRConfig())
        fm.register_health_check(lambda: True)
        r = fm.check_health()
        assert r["healthy"] is True
        assert r["failover_active"] is False

    def test_failover_triggers(self):
        cfg = DRConfig(failover_threshold=3)
        fm = FailoverManager(cfg)
        fm.register_health_check(lambda: False)
        for _ in range(3):
            fm.check_health()
        assert fm.state.failover_active is True
        assert len(fm.state.events) == 1
        assert fm.state.events[0]["type"] == "failover"

    def test_recovery(self):
        cfg = DRConfig(failover_threshold=2)
        fm = FailoverManager(cfg)
        # Trigger failover
        fm.register_health_check(lambda: False)
        for _ in range(2):
            fm.check_health()
        assert fm.state.failover_active is True
        # Recover
        fm._health_checkers = [lambda: True]
        fm.check_health()
        assert fm.state.failover_active is False
        assert fm.state.recovery_ts > 0

    def test_rto_status(self):
        fm = FailoverManager(DRConfig(rto_target_hours=1))
        r = fm.get_rto_status()
        assert r["status"] == "ok"

    def test_rpo_status(self):
        fm = FailoverManager(DRConfig(rpo_target_hours=1))
        r = fm.get_rpo_status(time.time() - 1800)  # 30 min ago
        assert r["rpo_met"] is True
        r = fm.get_rpo_status(time.time() - 7200)  # 2 hours ago
        assert r["rpo_met"] is False


class TestDRRunbook:
    def test_generate(self):
        rb = DRRunbook(DRConfig())
        r = rb.generate_runbook()
        assert "scenarios" in r
        assert len(r["scenarios"]) >= 3
        assert "backup_schedule" in r
        assert "contact_chain" in r
        assert "post_recovery_checklist" in r

    def test_write_file(self, tmp_path):
        rb = DRRunbook(DRConfig())
        path = rb.generate_runbook_file(str(tmp_path))
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert "scenarios" in data


class TestPITR:
    def test_plan_recovery(self):
        p = PointInTimeRecovery(DRConfig())
        r = p.plan_recovery(time.time() - 3600)
        assert "steps" in r
        assert len(r["steps"]) >= 5

    def test_recovery_window(self):
        p = PointInTimeRecovery(DRConfig(retention_days=30))
        w = p.get_recovery_window()
        assert "earliest_recovery" in w
        assert "latest_recovery" in w


class TestDRDashboard:
    def test_get_status(self):
        cfg = DRConfig(backup_dir="/nonexistent")
        dash = DRDashboard(cfg)
        s = dash.get_status()
        assert "backup_status" in s
        assert "failover_status" in s
        assert "rto_status" in s
        assert "rpo_status" in s
        assert "config" in s

    def test_get_runbook(self):
        dash = DRDashboard(DRConfig())
        r = dash.get_runbook()
        assert "scenarios" in r
