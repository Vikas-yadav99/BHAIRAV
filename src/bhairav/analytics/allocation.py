"""Automated resource allocation recommendations.

Uses hotspot predictions and alert trends to suggest where
to deploy officers, which cameras to prioritize, and when
to escalate to backup units.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ResourceRecommendation:
    """A single resource allocation recommendation."""
    action: str               # deploy_officer / reassign_camera / call_backup / ...
    priority: str             # critical / high / medium / low
    zone: str
    detail: str
    confidence: float         # 0-1
    expires_at: float         # when this recommendation is stale
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "priority": self.priority,
            "zone": self.zone,
            "detail": self.detail,
            "confidence": round(self.confidence, 2),
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }


class ResourceAllocator:
    """Generates resource allocation recommendations from hotspot data.

    Parameters
    ----------
    officer_pool : int
        Total available officers (default 10).
    cameras : list[str]
        Managed camera IDs.
    recommendation_ttl : float
        Seconds before a recommendation expires (default 600).
    """

    def __init__(self, officer_pool: int = 10,
                 cameras: list[str] | None = None,
                 recommendation_ttl: float = 600.0):
        self.officer_pool = officer_pool
        self.cameras = cameras or []
        self.ttl = recommendation_ttl
        self._active_deployments: dict[str, float] = {}  # zone -> officer count
        self._recommendations: list[ResourceRecommendation] = []
        self._history: list[dict] = []

    def analyze(self, hotspots: list[dict], alerts_by_zone: dict[str, int],
                trend_data: dict | None = None) -> list[ResourceRecommendation]:
        """Generate recommendations from current hotspot + trend data.

        Parameters
        ----------
        hotspots : list[dict]
            Ranked hotspot dicts from PredictiveHotspot.snapshot()["hotspots"].
        alerts_by_zone : dict[str, int]
            Current alert count per zone.
        trend_data : dict | None
            Trend snapshot from TrendAnalyzer.snapshot().

        Returns
        -------
        list[ResourceRecommendation]
        """
        now = time.time()
        self._recommendations.clear()

        # --- critical zones: deploy officers ---
        for h in hotspots[:5]:
            risk = h.get("risk_score", 0)
            zone = h.get("zone", "")
            trend = h.get("trend", "stable")
            predicted = h.get("predicted_next_hour", 0)

            if risk >= 0.7 or (trend == "rising" and predicted > 5):
                priority = "critical" if risk >= 0.85 else "high"
                self._recommendations.append(ResourceRecommendation(
                    action="deploy_officer",
                    priority=priority,
                    zone=zone,
                    detail=(
                        f"Risk score {risk:.0%} — "
                        f"{predicted:.0f} alerts predicted next hour"
                    ),
                    confidence=risk,
                    expires_at=now + self.ttl,
                    metadata={"trend": trend},
                ))
            elif risk >= 0.4:
                self._recommendations.append(ResourceRecommendation(
                    action="reassign_camera",
                    priority="medium",
                    zone=zone,
                    detail=(
                        f"Moderate risk ({risk:.0%}), consider "
                        f"increasing camera coverage"
                    ),
                    confidence=risk * 0.8,
                    expires_at=now + self.ttl,
                    metadata={"trend": trend},
                ))

        # --- check officer deployment balance ---
        total_deployed = sum(self._active_deployments.values())
        remaining = self.officer_pool - total_deployed

        if remaining <= 2 and self._recommendations:
            self._recommendations.append(ResourceRecommendation(
                action="call_backup",
                priority="high" if remaining == 0 else "medium",
                zone="(all)",
                detail=(
                    f"Only {remaining} officers available "
                    f"({total_deployed}/{self.officer_pool} deployed)"
                ),
                confidence=0.9,
                expires_at=now + self.ttl,
            ))

        # --- camera prioritisation ---
        if self.cameras and alerts_by_zone:
            top_zone = max(alerts_by_zone, key=alerts_by_zone.get)
            top_count = alerts_by_zone[top_zone]
            if top_count > 10:
                self._recommendations.append(ResourceRecommendation(
                    action="prioritize_camera",
                    priority="medium",
                    zone=top_zone,
                    detail=(
                        f"Zone {top_zone} has {top_count} alerts — "
                        f"prioritise live monitoring"
                    ),
                    confidence=0.7,
                    expires_at=now + self.ttl,
                    metadata={"alert_count": top_count},
                ))

        # --- burst response ---
        if trend_data:
            bursts = trend_data.get("bursts", [])
            for b in bursts:
                self._recommendations.append(ResourceRecommendation(
                    action="escalate_response",
                    priority="critical",
                    zone=b.get("zone", "(unknown)") if "zone" in b else "(all)",
                    detail=(
                        f"Burst detected: {b.get('count', 0)} "
                        f"{b.get('rule', '')} alerts in "
                        f"{b.get('window_sec', 10)}s"
                    ),
                    confidence=0.95,
                    expires_at=now + 300,
                    metadata={"burst_rule": b.get("rule", "")},
                ))

        # sort by priority
        prio_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self._recommendations.sort(
            key=lambda r: prio_order.get(r.priority, 9)
        )

        # record in history
        self._history.append({
            "timestamp": now,
            "count": len(self._recommendations),
            "actions": [r.action for r in self._recommendations],
        })
        self._history = self._history[-100:]

        return self._recommendations

    def deploy_officer(self, zone: str, count: int = 1) -> None:
        self._active_deployments[zone] = (
            self._active_deployments.get(zone, 0) + count
        )

    def recall_officer(self, zone: str, count: int = 1) -> None:
        current = self._active_deployments.get(zone, 0)
        new = max(0, current - count)
        if new:
            self._active_deployments[zone] = new
        else:
            self._active_deployments.pop(zone, None)

    def snapshot(self) -> dict:
        return {
            "officer_pool": self.officer_pool,
            "deployed": dict(self._active_deployments),
            "available": self.officer_pool - sum(
                self._active_deployments.values()
            ),
            "active_recommendations": [
                r.to_dict() for r in self._recommendations
            ],
            "history_count": len(self._history),
        }

    def reset(self) -> None:
        self._active_deployments.clear()
        self._recommendations.clear()
        self._history.clear()
