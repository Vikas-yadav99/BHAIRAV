"""Load balancer with round-robin and least-connections strategies.

Distributes incoming camera streams across available nodes
based on current load and connection count.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass


@dataclass
class BackendNode:
    """A backend node for load balancing."""
    node_id: str
    host: str
    port: int
    weight: int = 1
    connections: int = 0
    max_connections: int = 100
    last_used: float = 0.0
    healthy: bool = True

    @property
    def load_ratio(self) -> float:
        if self.max_connections <= 0:
            return 1.0
        return self.connections / self.max_connections

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "connections": self.connections,
            "max_connections": self.max_connections,
            "load_ratio": round(self.load_ratio, 3),
            "healthy": self.healthy,
        }


class LoadBalancer:
    """Distributes work across cluster nodes.

    Strategies:
      - round_robin: cycle through nodes sequentially
      - least_conn: pick node with fewest active connections
      - weighted: weighted round-robin based on node weight

    Parameters
    ----------
    strategy : str
        Balancing strategy (default "least_conn").
    """

    STRATEGIES = ("round_robin", "least_conn", "weighted")

    def __init__(self, strategy: str = "least_conn"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self._backends: dict[str, BackendNode] = {}
        self._rr_index = 0
        self._lock = threading.Lock()
        self._history: list[dict] = []

    def add_node(self, node_id: str, host: str, port: int,
                 weight: int = 1, max_connections: int = 100) -> BackendNode:
        with self._lock:
            node = BackendNode(
                node_id=node_id, host=host, port=port,
                weight=weight, max_connections=max_connections,
            )
            self._backends[node_id] = node
            return node

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            self._backends.pop(node_id, None)

    def get_next(self) -> BackendNode | None:
        """Select the next backend node using the configured strategy."""
        with self._lock:
            healthy = [n for n in self._backends.values() if n.healthy]
            if not healthy:
                return None

            if self.strategy == "round_robin":
                node = healthy[self._rr_index % len(healthy)]
                self._rr_index += 1
            elif self.strategy == "least_conn":
                node = min(healthy, key=lambda n: n.connections)
            elif self.strategy == "weighted":
                # weighted random
                total = sum(n.weight for n in healthy)
                import random
                r = random.uniform(0, total)
                cumulative = 0
                node = healthy[0]
                for n in healthy:
                    cumulative += n.weight
                    if r <= cumulative:
                        node = n
                        break
            else:
                node = healthy[0]

            node.connections += 1
            node.last_used = time.time()
            self._history.append({
                "node_id": node.node_id,
                "timestamp": time.time(),
            })
            self._history = self._history[-500:]
            return node

    def release(self, node_id: str) -> None:
        """Release a connection from a node."""
        with self._lock:
            node = self._backends.get(node_id)
            if node:
                node.connections = max(0, node.connections - 1)

    def mark_unhealthy(self, node_id: str) -> None:
        with self._lock:
            node = self._backends.get(node_id)
            if node:
                node.healthy = False

    def mark_healthy(self, node_id: str) -> None:
        with self._lock:
            node = self._backends.get(node_id)
            if node:
                node.healthy = True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "strategy": self.strategy,
                "total_nodes": len(self._backends),
                "healthy_nodes": sum(
                    1 for n in self._backends.values() if n.healthy
                ),
                "total_connections": sum(
                    n.connections for n in self._backends.values()
                ),
                "backends": [n.to_dict() for n in self._backends.values()],
            }
