"""Phase 2 - Behavior Intelligence package.

Behavior classifiers (rule-based, dependency-free) plus a lightweight
anomaly layer. Each rule exposes the same `evaluate(state, zones)` interface
as the Phase 1 rules so the engine treats them uniformly.
"""
from .anomaly import AnomalyRule
from .chase import ChaseRule
from .fall import FallRule
from .fight import FightRule
from .kinematics import MotionBuffer
from .trespass import TrespassRule

__all__ = [
    "MotionBuffer", "FallRule", "FightRule", "ChaseRule", "TrespassRule", "AnomalyRule",
]
