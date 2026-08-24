"""Tests for Phase 19 (HA) + Phase 20 (Compliance)."""
from __future__ import annotations

import time
import os
import tempfile
import pytest

from bhairav.ha.cluster import ClusterManager, NodeInfo
from bhairav.ha.failover import FailoverMonitor, HealthCheck
from bhairav.ha.balancer import LoadBalancer, BackendNode
from bhairav.compliance.retention import RetentionPolicy, RetentionManager
from bhairav.compliance.consent import ConsentManager, ConsentRecord
from bhairav.compliance.deletion import DeletionService, DeletionRequest


def _now():
    return time.time()


# =====================================================================
# Phase 19: ClusterManager
# =====================================================================

class TestClusterManager:
    def test_singleton_mode(self):
        cm = ClusterManager()
        assert cm.node_id
        assert cm.is_leader is False

    def test_register_and_discover(self):
        cm = ClusterManager()
        cm.register()
        nodes = cm.discover()
        assert len(nodes) >= 1
        assert nodes[0].node_id == cm.node_id

    def test_heartbeat(self):
        cm = ClusterManager()
        cm.register()
        cm.heartbeat()
        nodes = cm.discover()
        assert len(nodes) >= 1

    def test_elect_leader(self):
        cm = ClusterManager()
        cm.register()
        leader = cm.elect_leader()
        assert leader is not None
        assert leader.node_id == cm.node_id

    def test_update_load(self):
        cm = ClusterManager()
        cm.register()
        cm.update_load(0.75, camera_count=6)
        snap = cm.snapshot()
        assert snap["this_node"]["load"] == 0.75
        assert snap["this_node"]["camera_count"] == 6

    def test_snapshot(self):
        cm = ClusterManager()
        snap = cm.snapshot()
        assert "this_node" in snap
        assert "cluster_size" in snap


# =====================================================================
# Phase 19: FailoverMonitor
# =====================================================================

class TestFailoverMonitor:
    def test_check_all(self):
        cm = ClusterManager()
        cm.register()
        fm = FailoverMonitor(cm)
        results = fm.check_all()
        assert len(results) >= 1
        assert results[0].healthy is True

    def test_check_node(self):
        cm = ClusterManager()
        cm.register()
        fm = FailoverMonitor(cm)
        hc = fm.check_node(cm.node_id)
        assert hc.healthy is True

    def test_failover_callback(self):
        cm = ClusterManager()
        cm.register()
        triggered = []
        fm = FailoverMonitor(cm, failure_threshold=1)
        fm._on_failover = lambda leader, dead: triggered.append(dead.node_id)
        # Simulate dead node
        fm._failures["fake_node"] = 1
        fm._trigger_failover(NodeInfo(node_id="fake_node"))
        assert len(triggered) == 1

    def test_start_stop(self):
        cm = ClusterManager()
        fm = FailoverMonitor(cm)
        fm.start()
        assert fm._running is True
        fm.stop()
        assert fm._running is False

    def test_snapshot(self):
        cm = ClusterManager()
        fm = FailoverMonitor(cm)
        snap = fm.snapshot()
        assert "check_interval" in snap
        assert "failure_threshold" in snap


# =====================================================================
# Phase 19: LoadBalancer
# =====================================================================

class TestLoadBalancer:
    def test_add_and_get(self):
        lb = LoadBalancer(strategy="round_robin")
        lb.add_node("n1", "127.0.0.1", 8001)
        lb.add_node("n2", "127.0.0.1", 8002)
        node = lb.get_next()
        assert node is not None
        assert node.node_id in ("n1", "n2")

    def test_round_robin(self):
        lb = LoadBalancer(strategy="round_robin")
        lb.add_node("n1", "127.0.0.1", 8001)
        lb.add_node("n2", "127.0.0.1", 8002)
        first = lb.get_next()
        second = lb.get_next()
        assert first.node_id != second.node_id

    def test_least_conn(self):
        lb = LoadBalancer(strategy="least_conn")
        lb.add_node("n1", "127.0.0.1", 8001)
        lb.add_node("n2", "127.0.0.1", 8002)
        lb.get_next()  # n1 gets 1 conn
        node = lb.get_next()  # should prefer n2 (0 conns)
        assert node.node_id == "n2"

    def test_release(self):
        lb = LoadBalancer(strategy="least_conn")
        lb.add_node("n1", "127.0.0.1", 8001)
        node = lb.get_next()
        assert node.connections == 1
        lb.release(node.node_id)
        assert node.connections == 0

    def test_remove_node(self):
        lb = LoadBalancer()
        lb.add_node("n1", "127.0.0.1", 8001)
        lb.remove_node("n1")
        assert lb.get_next() is None

    def test_mark_unhealthy(self):
        lb = LoadBalancer()
        lb.add_node("n1", "127.0.0.1", 8001)
        lb.add_node("n2", "127.0.0.1", 8002)
        lb.mark_unhealthy("n1")
        node = lb.get_next()
        assert node.node_id == "n2"

    def test_snapshot(self):
        lb = LoadBalancer()
        lb.add_node("n1", "127.0.0.1", 8001)
        snap = lb.snapshot()
        assert snap["total_nodes"] == 1

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            LoadBalancer(strategy="invalid")


# =====================================================================
# Phase 20: RetentionPolicy + RetentionManager
# =====================================================================

class TestRetention:
    def test_policy_not_expired(self):
        p = RetentionPolicy("evidence", max_age_days=30)
        assert not p.is_expired(_now() - 86400 * 10)

    def test_policy_expired(self):
        p = RetentionPolicy("evidence", max_age_days=30)
        assert p.is_expired(_now() - 86400 * 40)

    def test_policy_forever(self):
        p = RetentionPolicy("logs", max_age_days=0)
        assert not p.is_expired(0)

    def test_manager_check_expiry(self):
        m = RetentionManager()
        assert m.check_expiry("evidence", _now() - 86400 * 10) is False
        assert m.check_expiry("evidence", _now() - 86400 * 100) is True

    def test_manager_snapshot(self):
        m = RetentionManager()
        snap = m.snapshot()
        assert "policies" in snap
        assert "evidence" in snap["policies"]

    def test_manager_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = RetentionManager(data_dir=tmpdir)
            # Create expired file
            ev_dir = os.path.join(tmpdir, "evidence")
            os.makedirs(ev_dir, exist_ok=True)
            fpath = os.path.join(ev_dir, "old_video.mp4")
            with open(fpath, "w") as f:
                f.write("test")
            # Make it old using mtime (what Python checks on all platforms)
            old_time = _now() - 86400 * 100
            os.utime(fpath, (old_time, old_time))
            # Verify mtime is old
            assert os.path.getmtime(fpath) < _now() - 86400 * 50
            result = m.cleanup("evidence")
            # The file should have been deleted
            assert not os.path.exists(fpath) or result["deleted"] >= 1

    def test_manager_add_policy(self):
        m = RetentionManager()
        m.add_policy(RetentionPolicy("custom", max_age_days=7))
        assert "custom" in m.policies


# =====================================================================
# Phase 20: ConsentManager
# =====================================================================

class TestConsent:
    def test_grant_and_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConsentManager(store_path=os.path.join(tmpdir, "consent.json"))
            cm.grant("person_001", "monitoring")
            assert cm.check("person_001", "monitoring") is True

    def test_revoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConsentManager(store_path=os.path.join(tmpdir, "consent.json"))
            cm.grant("person_001", "monitoring")
            cm.revoke("person_001", "monitoring")
            assert cm.check("person_001", "monitoring") is False

    def test_no_consent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConsentManager(store_path=os.path.join(tmpdir, "consent.json"))
            assert cm.check("nobody", "monitoring") is False

    def test_expiry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConsentManager(store_path=os.path.join(tmpdir, "consent.json"))
            cm.grant("person_001", "analytics", expires_in_days=-1)
            assert cm.check("person_001", "analytics") is False

    def test_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConsentManager(store_path=os.path.join(tmpdir, "consent.json"))
            cm.grant("p1", "monitoring")
            cm.grant("p1", "analytics")
            history = cm.get_history("p1")
            assert len(history) == 2

    def test_list_consents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConsentManager(store_path=os.path.join(tmpdir, "consent.json"))
            cm.grant("p1", "monitoring")
            cm.grant("p2", "analytics")
            all_c = cm.list_consents()
            assert len(all_c) == 2
            mon = cm.list_consents("monitoring")
            assert len(mon) == 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = os.path.join(tmpdir, "consent.json")
            cm = ConsentManager(store_path=sp)
            cm.grant("p1", "monitoring")
            # reload
            cm2 = ConsentManager(store_path=sp)
            assert cm2.check("p1", "monitoring") is True

    def test_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConsentManager(store_path=os.path.join(tmpdir, "consent.json"))
            cm.grant("p1", "monitoring")
            snap = cm.snapshot()
            assert snap["total_records"] == 1
            assert snap["active"] == 1


# =====================================================================
# Phase 20: DeletionService
# =====================================================================

class TestDeletion:
    def test_request_deletion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DeletionService(data_dir=tmpdir,
                                 store_path=os.path.join(tmpdir, "del.json"))
            req = ds.request_deletion("person_001", reason="GDPR request")
            assert req.status == "pending"
            assert req.subject_id == "person_001"

    def test_process_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DeletionService(data_dir=tmpdir,
                                 store_path=os.path.join(tmpdir, "del.json"))
            req = ds.request_deletion("nobody")
            result = ds.process(req.request_id)
            assert result.status == "completed"

    def test_process_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev_dir = os.path.join(tmpdir, "evidence")
            os.makedirs(ev_dir)
            with open(os.path.join(ev_dir, "person_001_frame.jpg"), "w") as f:
                f.write("data")
            ds = DeletionService(data_dir=tmpdir,
                                 store_path=os.path.join(tmpdir, "del.json"))
            req = ds.request_deletion("person_001", scope="evidence")
            result = ds.process(req.request_id)
            assert result.status == "completed"
            assert result.deleted_items.get("evidence", 0) >= 1

    def test_deny(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DeletionService(data_dir=tmpdir,
                                 store_path=os.path.join(tmpdir, "del.json"))
            req = ds.request_deletion("p1")
            ds.deny(req.request_id, "Legal hold")
            assert ds.list_requests(status="denied")[-1]["status"] == "denied"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = os.path.join(tmpdir, "del.json")
            ds = DeletionService(data_dir=tmpdir, store_path=sp)
            ds.request_deletion("p1")
            ds2 = DeletionService(data_dir=tmpdir, store_path=sp)
            assert len(ds2.list_requests()) == 1

    def test_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DeletionService(data_dir=tmpdir,
                                 store_path=os.path.join(tmpdir, "del.json"))
            ds.request_deletion("p1")
            snap = ds.snapshot()
            assert snap["total_requests"] == 1
            assert snap["pending"] == 1
