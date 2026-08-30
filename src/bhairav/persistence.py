"""Bounded data collections and memory monitoring (Group 5 of audit fix).

Replaces unbounded lists/dicts that grow forever with bounded versions
that evict old entries. Prevents memory leaks in long-running deployments.
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


class BoundedList:
    """A list that caps its size and evicts the oldest entries.

    Thread-safe. Use as a drop-in replacement for list when you need
    a rolling window (e.g., recent alerts, analytics history).

    Parameters
    ----------
    maxlen : int
        Maximum number of items. Oldest items are evicted when full.
    """

    def __init__(self, maxlen: int = 200):
        self._maxlen = maxlen
        self._items: list = []
        self._lock = threading.Lock()

    def append(self, item: Any) -> None:
        with self._lock:
            self._items.append(item)
            if len(self._items) > self._maxlen:
                self._items = self._items[-self._maxlen:]

    def extend(self, items: list) -> None:
        with self._lock:
            self._items.extend(items)
            if len(self._items) > self._maxlen:
                self._items = self._items[-self._maxlen:]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __iter__(self):
        with self._lock:
            return iter(list(self._items))

    def __getitem__(self, index):
        with self._lock:
            return self._items[index]

    def __repr__(self) -> str:
        return f"BoundedList(maxlen={self._maxlen}, len={len(self._items)})"

    def snapshot(self) -> list:
        """Return a copy of current items (thread-safe)."""
        with self._lock:
            return list(self._items)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @maxlen.setter
    def maxlen(self, value: int) -> None:
        with self._lock:
            self._maxlen = value
            if len(self._items) > value:
                self._items = self._items[-value:]


class BoundedDict:
    """A dict that caps its size and evicts the oldest entries (LRU).

    Thread-safe. Use for bounded caches (e.g., re-ID gallery, analytics).
    """

    def __init__(self, maxlen: int = 10000):
        self._maxlen = maxlen
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
            self._items[key] = value
            while len(self._items) > self._maxlen:
                self._items.popitem(last=False)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._items[key]

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._items.get(key, default)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._items[key]

    def keys(self) -> list:
        with self._lock:
            return list(self._items.keys())

    def values(self) -> list:
        with self._lock:
            return list(self._items.values())

    def items(self) -> list:
        with self._lock:
            return list(self._items.items())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._items)


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot."""
    rss_bytes: int = 0
    vms_bytes: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 * 1024)

    @property
    def vms_mb(self) -> float:
        return self.vms_bytes / (1024 * 1024)

    def to_dict(self) -> dict:
        return {
            "rss_bytes": self.rss_bytes,
            "vms_bytes": self.vms_bytes,
            "rss_mb": round(self.rss_mb, 2),
            "vms_mb": round(self.vms_mb, 2),
            "timestamp": self.timestamp,
        }


class MemoryMonitor:
    """Tracks process memory usage over time.

    Use snapshot() to get current memory, or history() for the
    time series. Memory is sampled periodically by calling sample().
    """

    def __init__(self, history_size: int = 60):
        self._history: list[MemorySnapshot] = []
        self._max = history_size
        self._lock = threading.Lock()

    def sample(self) -> MemorySnapshot:
        """Take a memory snapshot and add to history."""
        snap = self._get_memory()
        with self._lock:
            self._history.append(snap)
            if len(self._history) > self._max:
                self._history = self._history[-self._max:]
        return snap

    def current(self) -> MemorySnapshot:
        """Get current memory without storing to history."""
        return self._get_memory()

    def history(self) -> list[dict]:
        """Get memory history as list of dicts."""
        with self._lock:
            return [s.to_dict() for s in self._history]

    def peak(self) -> MemorySnapshot:
        """Get the peak memory usage from history."""
        with self._lock:
            if not self._history:
                return self._get_memory()
            return max(self._history, key=lambda s: s.rss_bytes)

    @staticmethod
    def _get_memory() -> MemorySnapshot:
        """Get current process memory usage."""
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            return MemorySnapshot(rss_bytes=mem.rss, vms_bytes=mem.vms)
        except ImportError:
            pass
        # Fallback: try /proc on Linux
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1]) * 1024
                        return MemorySnapshot(rss_bytes=rss_kb, vms_bytes=rss_kb * 2)
                    elif line.startswith("VmSize:"):
                        vms_kb = int(line.split()[1]) * 1024
                        return MemorySnapshot(rss_bytes=0, vms_bytes=vms_kb)
        except (OSError, ValueError):
            pass
        # Last resort: sys.getsizeof estimate (very rough)
        return MemorySnapshot(rss_bytes=0, vms_bytes=0)


class CollectionStats:
    """Reports the size of all bounded collections for /api/status.

    Register collections with register() and call snapshot() to get
    a summary dict suitable for the status endpoint.
    """

    def __init__(self):
        self._collections: dict[str, Any] = {}

    def register(self, name: str, collection) -> None:
        """Register a BoundedList or BoundedDict for tracking."""
        self._collections[name] = collection

    def snapshot(self) -> dict:
        """Get sizes of all registered collections."""
        result = {}
        for name, coll in self._collections.items():
            try:
                result[name] = {
                    "size": len(coll),
                    "maxlen": getattr(coll, "maxlen", None),
                    "type": type(coll).__name__,
                }
            except Exception:
                result[name] = {"size": -1, "error": "unreadable"}
        return result
