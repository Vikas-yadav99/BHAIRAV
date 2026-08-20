"""Federation protocol: message types for cross-server sharing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class MessageType(str, Enum):
    ALERT = "alert"
    SIGHTING = "sighting"       # re-ID sighting
    ANALYTICS = "analytics"     # periodic analytics snapshot
    HEARTBEAT = "heartbeat"


@dataclass
class FederationMessage:
    """A message exchanged between federated BHAIRAV servers."""
    msg_type: MessageType
    source_site: str             # unique site identifier
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    msg_id: str = ""

    def to_dict(self):
        return {
            "type": self.msg_type.value,
            "source_site": self.source_site,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "msg_id": self.msg_id,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            msg_type=MessageType(d["type"]),
            source_site=d.get("source_site", "unknown"),
            payload=d.get("payload", {}),
            timestamp=d.get("timestamp", time.time()),
            msg_id=d.get("msg_id", ""),
        )
