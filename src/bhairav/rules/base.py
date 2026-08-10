"""Rule base class."""
from __future__ import annotations

from ..types import Alert, FrameState, Zone


class Rule:
    name: str = "base"

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        # Optional filter: only consider zones with these names.
        self.zone_names: list[str] | None = config.get("zones")

    def _zone_selected(self, zone: Zone) -> bool:
        return self.zone_names is None or zone.name in self.zone_names

    def evaluate(self, state: FrameState, zones: list[Zone]) -> list[Alert]:
        raise NotImplementedError
