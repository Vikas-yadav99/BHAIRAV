"""Traffic flow analysis: vehicle counting, speed estimation, congestion.

Analyses vehicle tracks to produce:
  - Per-lane / per-zone vehicle counts
  - Average speed estimation from track displacement
  - Congestion level (free_flow / light / moderate / heavy / gridlock)
  - Intersection throughput
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class CongestionLevel(str, Enum):
    FREE_FLOW = "free_flow"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    GRIDLOCK = "gridlock"


@dataclass
class VehicleCount:
    """Vehicle count for a zone/lane."""
    zone: str
    count: int
    avg_speed_kmh: float = 0.0
    congestion: CongestionLevel = CongestionLevel.FREE_FLOW
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "zone": self.zone, "count": self.count,
            "avg_speed_kmh": round(self.avg_speed_kmh, 1),
            "congestion": self.congestion.value,
            "timestamp": self.timestamp,
        }


@dataclass
class VehicleTrack:
    """Internal tracking record for a vehicle."""
    track_id: int
    zone: str
    positions: list = field(default_factory=list)  # [(timestamp, x, y)]
    vehicle_type: str = "car"
    camera_id: str = ""

    @property
    def speed_kmh(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        # positions are in insertion order; first is newest, last is oldest
        dt = self.positions[0][0] - self.positions[-1][0]
        if dt <= 0:
            return 0.0
        dx = self.positions[0][1] - self.positions[-1][1]
        dy = self.positions[0][2] - self.positions[-1][2]
        dist_px = math.sqrt(dx * dx + dy * dy)
        # rough: 1px ~ 0.1m at typical 640px frame
        dist_m = dist_px * 0.1
        return (dist_m / dt) * 3.6  # m/s -> km/h


class TrafficAnalyzer:
    """Analyses traffic flow from vehicle tracks.

    Parameters
    ----------
    speed_thresholds : dict
        Congestion thresholds in km/h (free > 40, light > 25, moderate > 15, heavy > 5).
    window_sec : float
        Rolling window for counts (default 300 = 5 min).
    min_speed_samples : int
        Min positions for speed estimate (default 3).
    """

    DEFAULT_THRESHOLDS = {
        CongestionLevel.FREE_FLOW: 40.0,
        CongestionLevel.LIGHT: 25.0,
        CongestionLevel.MODERATE: 15.0,
        CongestionLevel.HEAVY: 5.0,
    }

    def __init__(self, speed_thresholds: dict | None = None,
                 window_sec: float = 300.0, min_speed_samples: int = 3):
        self.thresholds = speed_thresholds or self.DEFAULT_THRESHOLDS
        self.window_sec = window_sec
        self.min_speed_samples = min_speed_samples
        self._vehicles: dict[int, VehicleTrack] = {}
        self._counts: defaultdict[str, list] = defaultdict(list)
        self._total_count = 0

    def observe(self, timestamp: float, track_id: int, zone: str,
                x: float, y: float, vehicle_type: str = "car",
                camera_id: str = "") -> None:
        """Record a vehicle track observation."""
        if track_id not in self._vehicles:
            self._vehicles[track_id] = VehicleTrack(
                track_id=track_id, zone=zone,
                vehicle_type=vehicle_type, camera_id=camera_id,
            )
            self._total_count += 1
        v = self._vehicles[track_id]
        v.positions.append((timestamp, x, y))
        # keep only recent positions
        cutoff = timestamp - self.window_sec
        v.positions = [(t, px, py) for t, px, py in v.positions if t >= cutoff]

    def get_zone_counts(self) -> list[VehicleCount]:
        """Get vehicle counts per zone."""
        zone_vehicles: defaultdict[str, set] = defaultdict(set)
        zone_speeds: defaultdict[str, list] = defaultdict(list)
        now = time.time()
        cutoff = now - self.window_sec

        for v in self._vehicles.values():
            if v.positions and v.positions[-1][0] >= cutoff:
                zone_vehicles[v.zone].add(v.track_id)
                if len(v.positions) >= self.min_speed_samples:
                    zone_speeds[v.zone].append(v.speed_kmh)

        results = []
        for zone, vids in zone_vehicles.items():
            speeds = zone_speeds.get(zone, [])
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
            congestion = self._classify_congestion(avg_speed)
            results.append(VehicleCount(
                zone=zone, count=len(vids),
                avg_speed_kmh=avg_speed, congestion=congestion,
            ))
        return sorted(results, key=lambda c: c.count, reverse=True)

    def _classify_congestion(self, speed_kmh: float) -> CongestionLevel:
        for level, threshold in sorted(self.thresholds.items(),
                                        key=lambda x: x[1], reverse=True):
            if speed_kmh >= threshold:
                return level
        return CongestionLevel.GRIDLOCK

    def intersection_throughput(self, zone_a: str, zone_b: str,
                                window_sec: float = 60.0) -> int:
        """Count vehicles that moved from zone_a to zone_b."""
        now = time.time()
        count = 0
        for v in self._vehicles.values():
            zones_seen = [(t, z) for t, _, _ in v.positions
                          if t >= now - window_sec]
            # approximate zone from position
            zone_seq = list(dict.fromkeys(z for _, z in zones_seen))
            if zone_a in zone_seq and zone_b in zone_seq:
                if zone_seq.index(zone_a) < zone_seq.index(zone_b):
                    count += 1
        return count

    def snapshot(self) -> dict:
        counts = self.get_zone_counts()
        return {
            "total_vehicles_tracked": self._total_count,
            "active_vehicles": len([v for v in self._vehicles.values()
                                    if v.positions and v.positions[-1][0] >= time.time() - self.window_sec]),
            "zones": [c.to_dict() for c in counts],
        }

    def reset(self) -> None:
        self._vehicles.clear()
        self._counts.clear()
        self._total_count = 0
