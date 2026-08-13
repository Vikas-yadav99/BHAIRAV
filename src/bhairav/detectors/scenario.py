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
    role: str = "walk"  # walk | stand | fall | fight | chase | mob
    special: dict = field(default_factory=dict)
    # Phase 10: scene time (s) the actor first appears; None = from t=0.
    appears_at: float | None = None


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
            if p.appears_at is not None and t < p.appears_at:
                continue  # Phase 10: actor has not entered the scene yet
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
        if p.role == "mob":
            return self._mob(p, t)
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

    def _mob(self, p: PersonSpec, t: float) -> tuple[float, float, float]:
        """Mob member: walk to the anchor, then mill in a tight circle.

        Constant-speed circular milling (radius/omega in `special`) reads as
        disorderly agitation - high heading wobble, moderate speed - without
        the brawling signature (no erratic high-speed jostle) and without
        boxes ever touching, so the greedy-IoU tracker cannot swap ids.
        """
        anchor = p.special["anchor"]
        arrive = float(p.special.get("arrive", 4.0))
        if t < arrive:
            x, y = _waypoint_pos(p.waypoints, t)
            return x, y, 0.0
        dt = t - arrive
        r = float(p.special.get("radius", 0.025))
        w = float(p.special.get("omega", 1.8))
        x = anchor[0] + r * math.cos(w * dt + p.jitter_phase)
        y = anchor[1] + r * math.sin(w * dt + p.jitter_phase * 1.3)
        return x, y, 0.0

    def _fight(self, p: PersonSpec, t: float) -> tuple[float, float, float]:
        """Approach the anchor waypoint, then jostle erratically around it."""
        anchor = p.special["anchor"]
        arrive = float(p.special.get("arrive", 4.0))
        if t < arrive:
            x, y = _waypoint_pos(p.waypoints, t)
            return x, y, 0.0
        dt = t - arrive
        amp = float(p.special.get("amp", 0.010))
        amp_y = float(p.special.get("amp_y", 0.008))
        freq = float(p.special.get("freq", 2.0))
        phase = p.jitter_phase
        x = anchor[0] + amp * math.sin(2 * math.pi * freq * dt + phase)
        y = anchor[1] + amp_y * math.sin(2 * math.pi * freq * 0.8 * dt + phase * 1.7)
        return x, y, 0.0


def variant_scenario(cfg: SyntheticConfig, offset_x: float = 0.14,
                     offset_y: float = 0.05,
                     jitter_amp: float | None = None) -> Scenario:
    """The SAME people in a different layout - a second 'camera view'.

    Every actor keeps its pid, colors and behavior role, but the waypoint
    paths are translated (and clamped to the frame) so positions differ
    from the default scene. Used by the Phase 9 re-id validation to score
    cross-camera identity matching with known ground truth.
    """
    base = default_scenario(cfg)
    shifted = []
    for p in base.persons:
        wps = [(min(max(x + offset_x, 0.03), 0.97),
                min(max(y + offset_y, 0.10), 0.90), sp, hold)
               for (x, y, sp, hold) in p.waypoints]
        special = dict(p.special)
        if "anchor" in special:  # keep fight clusters coherent
            ax, ay = special["anchor"]
            special["anchor"] = (min(max(ax + offset_x, 0.03), 0.97),
                                  min(max(ay + offset_y, 0.10), 0.90))
        shifted.append(PersonSpec(
            p.pid, p.label, p.class_id, wps, size=p.size,
            jitter_phase=p.jitter_phase, role=p.role, special=special,
            appears_at=p.appears_at))
    return Scenario(shifted, base.duration_sec,
                    jitter_amp=base.jitter_amp if jitter_amp is None
                    else jitter_amp)


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
        # --- Phase 10 actors ---
        # Abandoned bag: a suitcase appears in the plaza at t=6 and stays put;
        # nobody ever comes within owner range, so it fires the
        # abandoned_object alert (~t=14 orange, ~t=22 red).
        PersonSpec(15, "suitcase", 28, [(0.55, 0.33, 0.0, 100.0)],
                   size="baggage", appears_at=6.0),
        # Accident: a plate-less car drives the lower street band and brakes
        # hard to a stop at (0.66, 0.64) at ~t=7. A pedestrian (pid 17) steps
        # into the road and collapses next to it at t=8.3 -> accident alert.
        PersonSpec(16, "car", 2, [(0.95, 0.66, 0.14, 5.0), (0.66, 0.66, 0.14, 100.0)],
                   size="vehicle", appears_at=5.0),
        PersonSpec(17, "person", 0, [(0.68, 0.85, 0.072, 5.8), (0.67, 0.69, 0.072, 0.0)],
                   role="fall", special={"fall_at": 8.3, "fall_dur": 0.4, "ground": 0.75}),
        # Riot: four people file in at t=14 and mill (orbit) agitatedly in the
        # lower street band (outside every zone) from ~t=20 -> riot alert ~t=24.5.
        # Anchors sit ~100-150 px apart so the orbiting boxes never touch ->
        # no greedy-IoU tracker swaps -> no fight false alarms, and the
        # constant-speed milling stays under the fight speed gate.
        PersonSpec(18, "person", 0, [(0.35, 0.95, 0.06, 14.0), (0.35, 0.62, 0.06, 0.0)],
                   role="mob", appears_at=14.0,
                   special={"anchor": (0.35, 0.62), "arrive": 20.0,
                            "radius": 0.025, "omega": 1.8}),
        PersonSpec(19, "person", 0, [(0.43, 0.95, 0.06, 14.0), (0.43, 0.62, 0.06, 0.0)],
                   role="mob", appears_at=14.0,
                   special={"anchor": (0.43, 0.62), "arrive": 20.0,
                            "radius": 0.025, "omega": 1.8}),
        PersonSpec(20, "person", 0, [(0.51, 0.95, 0.06, 14.0), (0.51, 0.62, 0.06, 0.0)],
                   role="mob", appears_at=14.0,
                   special={"anchor": (0.51, 0.62), "arrive": 20.0,
                            "radius": 0.025, "omega": 1.8}),
        PersonSpec(21, "person", 0, [(0.43, 0.95, 0.06, 14.0), (0.43, 0.64, 0.06, 0.0)],
                   role="mob", appears_at=14.0,
                   special={"anchor": (0.43, 0.64), "arrive": 20.0,
                            "radius": 0.025, "omega": 1.8}),
    ]
    return Scenario(persons, cfg.duration_sec)
