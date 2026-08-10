"""Rule engine package (Phase 1 geometric + Phase 2 behavior rules)."""
from ..behavior import AnomalyRule, ChaseRule, FallRule, FightRule, TrespassRule
from .base import Rule
from .crowd_density import CrowdDensityRule
from .engine import RULES, RulesEngine
from .loitering import LoiteringRule
from .zone_crossing import ZoneCrossingRule

__all__ = [
    "Rule", "RulesEngine", "RULES",
    "LoiteringRule", "ZoneCrossingRule", "CrowdDensityRule",
    "FallRule", "FightRule", "ChaseRule", "TrespassRule", "AnomalyRule",
]
