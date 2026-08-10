"""OpenCV overlay rendering: zones, tracks, HUD, alert banners.

Colors are BGR (OpenCV convention).
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import POSE_BONES, FrameState, Severity, Zone

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colors
SEVERITY_BGR = {
    Severity.GREEN: (94, 197, 34),
    Severity.YELLOW: (8, 179, 234),
    Severity.ORANGE: (22, 115, 249),
    Severity.RED: (68, 68, 239),
}
PERSON_BGR = (238, 211, 34)      # cyan
VEHICLE_BGR = (11, 158, 245)     # amber
TEXT_BGR = (249, 245, 241)       # near-white
MUTED_BGR = (163, 184, 148)      # slate
PANEL_BGR = (15, 23, 42)         # dark navy
JOINT_BGR = (251, 146, 60)       # orange-ish joints for skeletons


def _blend(img: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)


def _label(img: np.ndarray, text: str, org, scale: float, color, thickness: int = 1, bg=None):
    if bg is not None:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base + 2), bg, -1)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_zones(img: np.ndarray, zones: list[Zone]) -> np.ndarray:
    overlay = img.copy()
    for z in zones:
        pts = np.array(z.to_pixels(img.shape[1], img.shape[0]), dtype=np.int32).reshape(-1, 1, 2)
        if z.kind == "restricted":
            cv2.fillPoly(overlay, [pts], SEVERITY_BGR[Severity.RED])
        else:
            cv2.fillPoly(overlay, [pts], SEVERITY_BGR[Severity.YELLOW])
    img = _blend(img, overlay, 0.14)
    for z in zones:
        pts = np.array(z.to_pixels(img.shape[1], img.shape[0]), dtype=np.int32).reshape(-1, 1, 2)
        color = SEVERITY_BGR[Severity.RED] if z.kind == "restricted" else SEVERITY_BGR[Severity.YELLOW]
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        cx = int(sum(p[0] for p in z.points_norm) / len(z.points_norm) * img.shape[1])
        cy = int(sum(p[1] for p in z.points_norm) / len(z.points_norm) * img.shape[0])
        _label(img, f"{z.name} [{z.kind}]", (cx - 70, cy), 0.45, color, 1, (15, 23, 42))
    return img


def draw_tracks(img: np.ndarray, state: FrameState,
                active_severity: dict[int, Severity] | None = None) -> np.ndarray:
    active_severity = active_severity or {}
    for tr in state.tracks:
        sev = active_severity.get(tr.track_id)
        if sev is not None:
            color = SEVERITY_BGR[sev]
            thickness = 3
        else:
            color = PERSON_BGR if tr.is_person else VEHICLE_BGR
            thickness = 2
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cx, cy = tr.centroid
        cv2.circle(img, (int(cx), int(cy)), 3, color, -1)
        tag = f"#{tr.track_id} {tr.label} {tr.confidence:.2f}"
        _label(img, tag, (x1, max(y1 - 6, 14)), 0.45, color, 1, (15, 23, 42))
    return img


def draw_poses(img: np.ndarray, state: FrameState) -> np.ndarray:
    """Stick-figure skeletons over each tracked person (Phase 2)."""
    w, h = img.shape[1], img.shape[0]
    for pose in state.poses:
        pts: dict[int, tuple[int, int]] = {}
        for idx, kp in enumerate(pose.keypoints):
            if kp.confidence < 0.1:
                continue
            pts[idx] = (int(kp.x * w), int(kp.y * h))
        for a, b in POSE_BONES:
            if a in pts and b in pts:
                cv2.line(img, pts[a], pts[b], JOINT_BGR, 1, cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(img, p, 2, JOINT_BGR, -1, cv2.LINE_AA)
    return img


def draw_behavior_tags(img: np.ndarray, state: FrameState, alerts: list) -> np.ndarray:
    """Overlay labels/links for active behavior alerts (fight/chase/fall/trespass)."""
    pos = {tr.track_id: tr.centroid for tr in state.tracks}
    for a in alerts:
        if a.rule == "fight" and a.details.get("tracks"):
            ids = a.details["tracks"]
            if ids[0] in pos and ids[1] in pos:
                p1 = tuple(int(v) for v in pos[ids[0]])
                p2 = tuple(int(v) for v in pos[ids[1]])
                cv2.line(img, p1, p2, SEVERITY_BGR[Severity.RED], 2, cv2.LINE_AA)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 16)
                _label(img, "FIGHT", mid, 0.5, SEVERITY_BGR[Severity.RED], 2, PANEL_BGR)
        elif a.rule == "chase" and a.details.get("follower") is not None:
            f = a.details["follower"]
            r = a.details["runner"]
            if f in pos and r in pos:
                p1 = tuple(int(v) for v in pos[f])
                p2 = tuple(int(v) for v in pos[r])
                cv2.arrowedLine(img, p1, p2, SEVERITY_BGR[Severity.ORANGE], 2,
                                cv2.LINE_AA, tipLength=0.15)
                _label(img, "CHASE", ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 14),
                       0.5, SEVERITY_BGR[Severity.ORANGE], 2, PANEL_BGR)
        elif a.rule == "fall" and a.track_id in pos:
            p = tuple(int(v) for v in pos[a.track_id])
            _label(img, "FALL", (p[0] + 8, p[1] - 12), 0.5, SEVERITY_BGR[Severity.ORANGE], 2, PANEL_BGR)
        elif a.rule == "trespass" and a.track_id in pos:
            p = tuple(int(v) for v in pos[a.track_id])
            _label(img, "TRESPASS", (p[0] + 8, p[1] - 12), 0.5, SEVERITY_BGR[Severity.RED], 2, PANEL_BGR)
    return img


def draw_zone_stats(img: np.ndarray, zone_counts: dict[str, int]) -> np.ndarray:
    # Top-left, below the HUD + alert banner - avoids colliding with the
    # recent-alert list anchored bottom-left.
    for i, (name, count) in enumerate(zone_counts.items()):
        text = f"{name}: {count} people"
        _label(img, text, (16, 124 + i * 22), 0.45, TEXT_BGR, 1, PANEL_BGR)
    return img


def draw_hud(img: np.ndarray, state: FrameState, alerts_total: int, fps: float) -> np.ndarray:
    w = img.shape[1]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), PANEL_BGR, -1)
    img = _blend(img, overlay, 0.78)
    _label(img, "BHAIRAV", (14, 24), 0.8, TEXT_BGR, 2)
    _label(img, "BEHAVIOR INTELLIGENCE - PHASE 2", (14, 42), 0.42, MUTED_BGR, 1)

    n_p = sum(1 for t in state.tracks if t.is_person)
    n_v = len(state.tracks) - n_p
    stats = f"PERSONS {n_p}   VEHICLES {n_v}   TRACKS {len(state.tracks)}   ALERTS {alerts_total}   FPS {fps:.0f}"
    (tw, th), _ = cv2.getTextSize(stats, FONT, 0.5, 1)
    _label(img, stats, (w - tw - 16, 32), 0.5, TEXT_BGR, 1)
    return img


def draw_alert_banner(img: np.ndarray, alerts: list) -> np.ndarray:
    if not alerts:
        return img
    latest = alerts[-1]
    color = SEVERITY_BGR[latest.severity]
    w = img.shape[1]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 58), (w, 102), color, -1)
    img = _blend(img, overlay, 0.32)
    cv2.rectangle(img, (0, 58), (10, 102), color, -1)
    text = f"{latest.severity.value.upper()} ALERT :: {latest.message}"
    _label(img, text, (24, 88), 0.62, TEXT_BGR, 2)
    return img


def draw_recent_alerts(img: np.ndarray, alerts: list, max_n: int = 4) -> np.ndarray:
    h = img.shape[0]
    recent = alerts[-max_n:]
    for i, a in enumerate(reversed(recent)):
        y = h - 12 - i * 24
        color = SEVERITY_BGR[a.severity]
        cv2.rectangle(img, (12, y - 15), (30, y + 1), color, -1)
        text = f"[{a.severity.value.upper()}] {a.message}"
        _label(img, text, (38, y), 0.45, TEXT_BGR, 1, None)
    return img


def render(img: np.ndarray, state: FrameState, zones: list[Zone], alerts: list,
           fps: float, active_severity: dict[int, Severity] | None = None,
           zone_counts: dict[str, int] | None = None,
           alerts_total: int | None = None) -> np.ndarray:
    """Compose the full Phase 2 overlay onto a raw frame."""
    out = img.copy()
    out = draw_zones(out, zones)
    out = draw_poses(out, state)
    out = draw_tracks(out, state, active_severity)
    out = draw_behavior_tags(out, state, alerts)
    out = draw_alert_banner(out, alerts)
    out = draw_hud(out, state, alerts_total if alerts_total is not None else len(alerts), fps)
    out = draw_zone_stats(out, zone_counts or {})
    out = draw_recent_alerts(out, alerts)
    return out
