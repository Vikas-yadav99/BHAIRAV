"""Rule base class.

A Rule inspects one FrameState (detector output for a frame) against the
configured Zones and returns Alert objects. Rules are pure and stateless:
evaluate() must not keep history between frames - anything that needs a
memory of previous frames (loitering, crowd density) keeps it on the
FrameState, which the pipeline passes through unchanged for a track. This
keeps every rule deterministic and safe to run on any camera's thread
(Phase 8 M2 runs one independent rules engine per camera).

Subclasses set `name` and implement evaluate(); the engine picks rules by
name from config, so adding a rule is: subclass + register in the engine.
"""
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
