"""Health checks and automatic failover monitoring.

Periodically pings nodes, detects failures, and triggers
failover callbacks (e.g., redistribute cameras).
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class HealthCheck:
    """Result of a single health check."""
    node_id: str
    healthy: bool
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


class FailoverMonitor:
    """Monitors node health and triggers failover on failure.

    Parameters
    ----------
    cluster : ClusterManager
        The cluster manager to monitor.
    check_interval : float
        Seconds between health checks (default 5).
    failure_threshold : int
        Consecutive failures before declaring node dead (default 3).
    on_failover : callable | None
        Callback(leader_node, dead_node) when failover triggers.
    """

    def __init__(self, cluster, check_interval: float = 5.0,
                 failure_threshold: int = 3,
                 on_failover: Callable | None = None):
        self._cluster = cluster
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self._on_failover = on_failover
        self._failures: dict[str, int] = {}
        self._history: list[HealthCheck] = []
        self._dead_nodes: set[str] = set()
        self._running = False
        self._thread: threading.Thread | None = None

    def check_node(self, node_id: str) -> HealthCheck:
        """Check if a single node is alive."""
        nodes = self._cluster.discover()
        node = next((n for n in nodes if n.node_id == node_id), None)
        if node is None:
            return HealthCheck(node_id=node_id, healthy=False,
                               error="node not found")
        now = time.time()
        age = now - node.last_heartbeat
        healthy = age < self._cluster.expire_after
        return HealthCheck(
            node_id=node_id,
            healthy=healthy,
            latency_ms=age * 1000,
            error=None if healthy else f"heartbeat stale ({age:.1f}s ago)",
        )

    def check_all(self) -> list[HealthCheck]:
        """Check all nodes."""
        results = []
        for node in self._cluster.discover():
            hc = self.check_node(node.node_id)
            results.append(hc)
            if not hc.healthy:
                self._failures[node.node_id] = (
                    self._failures.get(node.node_id, 0) + 1
                )
                if self._failures[node.node_id] >= self.failure_threshold:
                    self._trigger_failover(node)
            else:
                self._failures[node.node_id] = 0
                self._dead_nodes.discard(node.node_id)
        self._history.extend(results)
        self._history = self._history[-500:]
        return results

    def _trigger_failover(self, dead_node) -> None:
        if dead_node.node_id in self._dead_nodes:
            return  # already handled
        self._dead_nodes.add(dead_node.node_id)
        leader = self._cluster.elect_leader()
        if self._on_failover and leader:
            try:
                self._on_failover(leader, dead_node)
            except Exception:
                pass

    def start(self) -> None:
        """Start background health-check loop."""
        if self._running:
            return
        self._running = True

        def _loop():
            while self._running:
                try:
                    self.check_all()
                except Exception:
                    pass
                time.sleep(self.check_interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def recovery(self, node_id: str) -> bool:
        """Mark a dead node as recovered."""
        if node_id in self._dead_nodes:
            self._dead_nodes.discard(node_id)
            self._failures.pop(node_id, None)
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "running": self._running,
            "check_interval": self.check_interval,
            "failure_threshold": self.failure_threshold,
            "failures": dict(self._failures),
            "dead_nodes": list(self._dead_nodes),
            "recent_checks": len(self._history),
        }
