"""RulesEngine: runs every enabled rule per frame and de-duplicates via cooldown."""
from __future__ import annotations

from ..backend.anpr import StolenVehicleRule
from ..behavior.accident import AccidentRule
from ..behavior.anomaly import AnomalyRule
from ..behavior.chase import ChaseRule
from ..behavior.fall import FallRule
from ..behavior.fight import FightRule
from ..behavior.riot import RiotRule
from ..behavior.trespass import TrespassRule
from ..types import Alert, FrameState, Zone
from .abandoned_object import AbandonedObjectRule
from .base import Rule
from .crowd_density import CrowdDensityRule
from .loitering import LoiteringRule
from .zone_crossing import ZoneCrossingRule

# Behavior rules duck-type the Rule contract (name / enabled / evaluate) rather
# than subclassing rules.base.Rule, to keep the behavior <-> rules import graph
# acyclic (rules.base would otherwise pull rules into behavior's imports).
RULES: list[tuple[str, type]] = [
    # Phase 1 - geometric / statistical rules
    ("loitering", LoiteringRule),
    ("zone_crossing", ZoneCrossingRule),
    ("crowd_density", CrowdDensityRule),
    # Phase 2 - behavior intelligence
    ("fall", FallRule),
    ("fight", FightRule),
    ("chase", ChaseRule),
    ("trespass", TrespassRule),
    ("anomaly", AnomalyRule),
    # Phase 6 - vehicle watchlist (ANPR)
    ("stolen_vehicle", StolenVehicleRule),
    # Phase 10 - proactive scene intelligence
    ("abandoned_object", AbandonedObjectRule),
    ("accident", AccidentRule),
    ("riot", RiotRule),
]


class RulesEngine:
    def __init__(self, rules_config: dict, zones: list[Zone], cooldown_sec: float = 10.0):
        self.zones = zones
        self.cooldown_sec = cooldown_sec
        self.rules: list[Rule] = []
        for name, cls in RULES:
            rule = cls(rules_config.get(name, {}))
            if rule.enabled:
                self.rules.append(rule)
        self._last_fired: dict[tuple, float] = {}

    def _key(self, alert: Alert) -> tuple:
        # (rule, zone, track, severity) - lets escalations fire while base is cooling down.
        return (alert.rule, alert.zone, alert.track_id, alert.severity.value)

    def update(self, state: FrameState) -> list[Alert]:
        alerts: list[Alert] = []
        for rule in self.rules:
            for alert in rule.evaluate(state, self.zones):
                key = self._key(alert)
                last = self._last_fired.get(key)
                if last is not None and state.timestamp - last < self.cooldown_sec:
                    continue
                self._last_fired[key] = state.timestamp
                alerts.append(alert)
        if len(self._last_fired) > 1024:
            cutoff = state.timestamp - (self.cooldown_sec * 3.0 + 5.0)
            self._last_fired = {k: t for k, t in self._last_fired.items() if t >= cutoff}
        return alerts
