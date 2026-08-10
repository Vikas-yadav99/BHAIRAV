"""Fall detection (Phase 2).

Signals:
  - downward vertical velocity spike (centroid drops faster than a run)
  - torso rotates toward horizontal (pose) or bbox turns wide/flat
  - the person stays low and still afterwards (no immediate recovery)

Orange on confirmation; escalates to Red when the person remains down
(2x the confirm window).
"""
from __future__ import annotations

from ..types import Alert, FrameState, Severity
from .kinematics import MotionBuffer


class FallRule:
    name = "fall"
    enabled = True

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.severity = Severity(config.get("severity", "orange"))
        self.escalate = bool(config.get("escalate", True))
        self.vy_thresh_norm = float(config.get("vy_thresh_norm", 0.10))  # x frame height / s
        self.flat_aspect = float(config.get("flat_aspect", 1.0))
        self.down_sec = float(config.get("down_sec", 0.5))
        self.confirm_grace = float(config.get("confirm_grace", 0.8))
        self.buf = MotionBuffer(window_sec=1.0)
        self._state: dict[int, dict] = {}

    def evaluate(self, state: FrameState, zones: list) -> list[Alert]:
        if not self.enabled:
            return []
        alerts: list[Alert] = []
        h = float(state.frame_h)
        poses = {p.track_id: p for p in state.poses}
        for tr in state.tracks:
            if not tr.is_person:
                continue
            cx, cy = tr.centroid
            self.buf.push(tr.track_id, state.timestamp, cx, cy)
            # Peak per-step downward velocity captures the fall spike even
            # though net displacement over the window is small.
            vy = self.buf.peak_downward_vy(tr.track_id, state.timestamp)
            w = tr.bbox[2] - tr.bbox[0]
            bh = tr.bbox[3] - tr.bbox[1]
            aspect = bh / w if w > 0 else 0.0
            pose = poses.get(tr.track_id)
            pose_flat = bool(pose and (pose.horizontal_angle_deg() or 0) > 60)
            flat = aspect < self.flat_aspect or pose_flat
            st = self._state.setdefault(tr.track_id, {"phase": "idle", "t_fall": 0.0,
                                                       "t_flat": 0.0, "last_flat": 0.0,
                                                       "peak_vy": 0.0})
            t = state.timestamp

            if vy > self.vy_thresh_norm * h:
                st["peak_vy"] = max(st["peak_vy"], vy)
                if st["phase"] != "falling":
                    st["phase"] = "falling"
                    st["t_fall"] = t
                if flat:
                    st["t_flat"] = t
            else:
                if st["phase"] == "falling":
                    # fall ended: confirmed if it went flat within the grace window
                    if flat or (t - st["last_flat"]) <= self.confirm_grace:
                        st["phase"] = "down"
                        st["t_flat"] = max(st["t_flat"], st["last_flat"])
                    else:
                        st["phase"] = "idle"
                        st["peak_vy"] = 0.0
            if flat:
                st["last_flat"] = t

            if st["phase"] == "down":
                if flat or (t - st["last_flat"]) <= self.confirm_grace:
                    down_for = t - max(st["t_flat"], st["t_fall"])
                    if down_for >= self.down_sec:
                        sev = self.severity
                        if self.escalate and down_for >= self.down_sec * 2.0:
                            sev = Severity.RED
                        peak = st["peak_vy"]
                        conf = min(0.95, 0.55 + 0.25 * min(peak / (2 * self.vy_thresh_norm * h), 1.0) + 0.2)
                        alerts.append(Alert(
                            rule=self.name, zone=None, track_id=tr.track_id,
                            severity=sev,
                            message=f"FALL detected: track #{tr.track_id} down {down_for:.1f}s "
                                    f"(drop vy {peak:.0f}px/s)",
                            frame_id=state.frame_id, timestamp=t,
                            details={"down_sec": round(down_for, 2),
                                     "peak_vy_px_s": round(peak, 1),
                                     "aspect": round(aspect, 2),
                                     "confidence": round(conf, 3)},
                            confidence=conf,
                        ))
                else:
                    st["phase"] = "idle"
                    st["peak_vy"] = 0.0

        self.buf.prune({tr.track_id for tr in state.tracks}, state.timestamp)
        # Drop long-idle state so long-running feeds do not grow unbounded.
        self._state = {k: v for k, v in self._state.items()
                       if v["phase"] != "idle"
                       or state.timestamp - v["last_flat"] <= 120.0}
        return alerts
