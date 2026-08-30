"""Redis-backed cluster management and node discovery.

Uses Redis pub/sub for node heartbeats, a sorted set for node
ranking, and JSON-serialised shared state for leader election.
Works with or without Redis (falls back to in-process singleton).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class NodeInfo:
    """Metadata for a cluster node."""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    host: str = "127.0.0.1"
    port: int = 8000
    role: str = "follower"       # leader / follower
    started_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    load: float = 0.0            # 0-1 normalised CPU/connection load
    camera_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "role": self.role,
            "started_at": self.started_at,
            "last_heartbeat": self.last_heartbeat,
            "load": round(self.load, 3),
            "camera_count": self.camera_count,
            "metadata": self.metadata,
        }


class ClusterManager:
    """Manages node discovery, heartbeats, and leader election.

    Falls back to in-process singleton mode when Redis is unavailable.

    Parameters
    ----------
    redis_url : str | None
        Redis connection string.  None = singleton mode.
    node : NodeInfo
        This node's identity.
    heartbeat_interval : float
        Seconds between heartbeats (default 5).
    expire_after : float
        Seconds before a silent node is declared dead (default 15).
    """

    def __init__(self, redis_url: str | None = None,
                 node: NodeInfo | None = None,
                 heartbeat_interval: float = 5.0,
                 expire_after: float = 15.0):
        self.heartbeat_interval = heartbeat_interval
        self.expire_after = expire_after
        self._node = node or NodeInfo()
        self._redis = None
        self._nodes: dict[str, NodeInfo] = {self._node.node_id: self._node}
        self._running = False

        if redis_url:
            try:
                import redis as _redis
                self._redis = _redis.Redis.from_url(
                    redis_url, decode_responses=True
                )
                self._redis.ping()
            except Exception:
                self._redis = None

    @property
    def node_id(self) -> str:
        return self._node.node_id

    @property
    def is_leader(self) -> bool:
        return self._node.role == "leader"

    def register(self) -> None:
        """Register this node in the cluster."""
        self._node.last_heartbeat = time.time()
        if self._redis:
            key = f"bhairav:node:{self._node.node_id}"
            self._redis.setex(key, int(self.expire_after * 2),
                              json.dumps(self._node.to_dict()))
            self._redis.sadd("bhairav:nodes", self._node.node_id)
            # publish heartbeat
            self._redis.publish("bhairav:heartbeat",
                                json.dumps(self._node.to_dict()))
        else:
            self._nodes[self._node.node_id] = self._node

    def heartbeat(self) -> None:
        """Send a heartbeat."""
        self._node.last_heartbeat = time.time()
        if self._redis:
            key = f"bhairav:node:{self._node.node_id}"
            self._redis.setex(key, int(self.expire_after * 2),
                              json.dumps(self._node.to_dict()))
            self._redis.publish("bhairav:heartbeat",
                                json.dumps(self._node.to_dict()))
        else:
            self._nodes[self._node.node_id] = self._node

    def discover(self) -> list[NodeInfo]:
        """Return all live nodes."""
        if self._redis:
            members = self._redis.smembers("bhairav:nodes")
            now = time.time()
            nodes = []
            for nid in members:
                raw = self._redis.get(f"bhairav:node:{nid}")
                if raw:
                    data = json.loads(raw)
                    if now - data.get("last_heartbeat", 0) < self.expire_after:
                        nodes.append(NodeInfo(**{
                            k: v for k, v in data.items()
                            if k in NodeInfo.__dataclass_fields__
                        }))
            self._prune_dead(nodes)
            return nodes
        else:
            now = time.time()
            return [n for n in self._nodes.values()
                    if now - n.last_heartbeat < self.expire_after]

    def _prune_dead(self, live: list[NodeInfo]) -> None:
        if self._redis:
            live_ids = {n.node_id for n in live}
            for nid in list(self._nodes.keys()):
                if nid not in live_ids:
                    self._nodes.pop(nid, None)

    def elect_leader(self) -> NodeInfo | None:
        """Elect leader as node with lowest load + earliest start."""
        nodes = self.discover()
        if not nodes:
            return None
        # sort by load (asc), then started_at (asc = earliest first)
        nodes.sort(key=lambda n: (n.load, n.started_at))
        leader = nodes[0]
        # update roles
        for n in nodes:
            n.role = "leader" if n.node_id == leader.node_id else "follower"
            if self._redis:
                self._redis.setex(
                    f"bhairav:node:{n.node_id}",
                    int(self.expire_after * 2),
                    json.dumps(n.to_dict()),
                )
        self._nodes.update({n.node_id: n for n in nodes})
        self._node.role = leader.role
        return leader

    def update_load(self, load: float, camera_count: int = 0) -> None:
        """Update this node's load metric."""
        self._node.load = max(0.0, min(1.0, load))
        self._node.camera_count = camera_count
        self.heartbeat()

    def node_count(self) -> int:
        return len(self.discover())

    def snapshot(self) -> dict:
        nodes = self.discover()
        return {
            "this_node": self._node.to_dict(),
            "cluster_size": len(nodes),
            "nodes": [n.to_dict() for n in nodes],
            "redis_connected": self._redis is not None,
        }
