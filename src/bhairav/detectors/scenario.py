"""A deterministic, scripted synthetic scene used for offline demo + tests.

The scene guarantees every Phase 1 + Phase 2 alert type fires within
`duration_sec`:
  - a loiterer who hangs around the monitored plaza
  - an intruder who enters the restricted server_room and stays (trespass)
  - four people who gather in the plaza (crowd density)
  - a pedestrian who stumbles and falls, then lies on the ground (fall)
  - two people who jostle in a scuffle (fight)
  - a fast runner pursued by a follower along the top walkway (chase)
  - two passers-by who never trigger anything

Roles (driven by `PersonSpec.role`) shape both the motion and the synthetic
skeleton emitted by `pose.synthetic.SyntheticPoseModel`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import SyntheticConfig


@dataclass
class PersonSpec:
    pid: int
    label: str
    class_id: int
    waypoints: list  # [(x_norm, y_norm, speed_norm/s to NEXT point, hold_s before moving), ...]
    size: str = "person"
    jitter_phase: float = 0.0
    role: str = "walk"  # walk | stand | fall | fight | chase
    special: dict = field(default_factory=dict)


@dataclass
class ScenePosition:
    """A person's position at a moment in time, plus pose hints."""

    person: PersonSpec
    x: float  # normalized
    y: float
    progress: float = 0.0  # 0..1 animation progress (e.g. fall rotation)


def _waypoint_pos(waypoints, t: float) -> tuple[float, float]:
    """Piecewise-linear waypoint walk with per-waypoint hold times."""
    t_left = t
    for i in range(len(waypoints) - 1):
        x0, y0, sp, hold = waypoints[i]
        x1, y1, _, _ = waypoints[i + 1]
        t_left -= hold
        if t_left < 0:
            return x0, y0
        seg = math.hypot(x1 - x0, y1 - y0) / sp if sp > 0 else 0.0
        if t_left < seg:
            f = t_left / seg
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        t_left -= seg
    x, y, _, _ = waypoints[-1]
    return x, y


class Scenario:
    def __init__(self, persons: list[PersonSpec], duration_sec: float, jitter_amp: float = 0.004):
        self.persons = persons
        self.duration_sec = float(duration_sec)
        self.jitter_amp = jitter_amp
        for i, p in enumerate(persons):
            p.jitter_phase = i * 1.7

    def positions_at(self, t: float) -> list[ScenePosition]:
        out: list[ScenePosition] = []
        for p in self.persons:
            sp = self._position_for(p, t)
            jx = self.jitter_amp * math.sin(2 * math.pi * 0.7 * t + p.jitter_phase)
            jy = self.jitter_amp * math.sin(2 * math.pi * 0.5 * t + p.jitter_phase * 1.3)
            out.append(ScenePosition(p, sp[0] + jx, sp[1] + jy, progress=sp[2]))
        return out

    def _position_for(self, p: PersonSpec, t: float) -> tuple[float, float, float]:
        if p.role == "fall":
            return self._fall(p, t)
        if p.role == "fight":
            return self._fight(p, t)
        x, y = _waypoint_pos(p.waypoints, t)
        return x, y, 0.0

    def _fall(self, p: PersonSpec, t: float) -> tuple[float, float, float]:
        """Walk along the waypoint path, then collapse to the ground."""
        fa = float(p.special.get("fall_at", 6.0))
        fd = float(p.special.get("fall_dur", 0.5))
        ground = float(p.special.get("ground", 0.72))
        if t < fa:
            x, y = _waypoint_pos(p.waypoints, t)
            return x, y, 0.0
        x_end, y_end = _waypoint_pos(p.waypoints, fa)
        f = min((t - fa) / fd, 1.0)
        y = y_end + (ground - y_end) * f
        return x_end, y, f

    def _fight(self, p: PersonSpec, t: float) -> tuple[float, float, float]:
        """Approach the anchor waypoint, then jostle erratically around it."""
        anchor = p.special["anchor"]
        arrive = float(p.special.get("arrive", 4.0))
        if t < arrive:
            x, y = _waypoint_pos(p.waypoints, t)
            return x, y, 0.0
        dt = t - arrive
        amp = float(p.special.get("amp", 0.010))
        freq = float(p.special.get("freq", 2.0))
        phase = p.jitter_phase
        x = anchor[0] + amp * math.sin(2 * math.pi * freq * dt + phase)
        y = anchor[1] + 0.008 * math.sin(2 * math.pi * freq * 0.8 * dt + phase * 1.7)
        return x, y, 0.0


def default_scenario(cfg: SyntheticConfig) -> Scenario:
    """The scripted demo scene. Timings chosen so every alert fires on schedule."""
    persons = [
        # --- Phase 1 actors ---
        # Loiterer: enters the plaza and stays.
        PersonSpec(1, "person", 0, [(0.20, 0.60, 0.05, 0.0), (0.38, 0.44, 0.05, 14.0)]),
        # Intruder: walks into the restricted server room and remains.
        PersonSpec(2, "person", 0, [(0.55, 0.85, 0.06, 0.0), (0.80, 0.48, 0.06, 6.0)]),
        # Crowd: four people gather inside the plaza.
        PersonSpec(3, "person", 0, [(0.30, 0.80, 0.08, 0.0), (0.45, 0.44, 0.08, 10.0)]),
        PersonSpec(4, "person", 0, [(0.62, 0.85, 0.08, 0.0), (0.47, 0.46, 0.08, 10.0)]),
        PersonSpec(5, "person", 0, [(0.38, 0.85, 0.08, 0.0), (0.44, 0.47, 0.08, 10.0)]),
        PersonSpec(6, "person", 0, [(0.55, 0.82, 0.08, 0.0), (0.48, 0.45, 0.08, 10.0)]),
        # Passers-by: never enter a zone. pid 8 walks the LOW lane (y~0.79)
        # so it never overlaps the fight corner at y~0.68 (greedy-IoU swaps
        # happen when boxes touch during the pass).
        PersonSpec(7, "person", 0, [(0.05, 0.15, 0.08, 0.0), (0.95, 0.15, 0.08, 0.0)]),
        PersonSpec(8, "person", 0, [(0.95, 0.80, 0.07, 0.0), (0.03, 0.78, 0.07, 0.0)]),
        # --- Phase 2 actors ---
        # Faller: waits for the crowd to clear, strolls the street band,
        # collapses at t=10.5 and stays down.
        PersonSpec(9, "person", 0, [(0.25, 0.62, 0.06, 5.0), (0.55, 0.62, 0.06, 60.0)],
                   role="fall", special={"fall_at": 10.5, "fall_dur": 0.5, "ground": 0.74}),
        # Fighters: converge at the lower-left street corner, then scuffle.
        PersonSpec(10, "person", 0, [(0.05, 0.80, 0.07, 0.0), (0.12, 0.68, 0.07, 0.0)],
                   role="fight", special={"anchor": (0.12, 0.68), "arrive": 4.0, "amp": 0.010}),
        PersonSpec(11, "person", 0, [(0.28, 0.78, 0.07, 0.0), (0.19, 0.68, 0.07, 0.0)],
                   role="fight", special={"anchor": (0.19, 0.68), "arrive": 4.0, "amp": 0.010}),
        # Chasers: a fast runner on the top walkway with a follower in pursuit.
        PersonSpec(12, "person", 0, [(0.80, 0.20, 0.12, 0.0), (0.20, 0.20, 0.12, 1.0)], role="chase"),
        PersonSpec(13, "person", 0, [(0.95, 0.20, 0.11, 0.0), (0.20, 0.20, 0.11, 1.0)], role="chase"),
        # Phase 6: a car with a license plate drives the top lane (~0-9.6 s),
        # giving the ANPR / stolen-vehicle watchlist something to read.
        PersonSpec(14, "car", 2, [(0.02, 0.25, 0.10, 0.0), (0.98, 0.25, 0.10, 0.0)],
                   size="vehicle", special={"plate": "MH12AB1234"}),
    ]
    return Scenario(persons, cfg.duration_sec)
