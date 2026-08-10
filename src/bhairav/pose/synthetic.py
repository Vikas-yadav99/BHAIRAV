"""Synthetic pose model: generates plausible 17-keypoint skeletons for the
scripted demo scene, so Phase 2 behavior detection runs end-to-end with zero
ML dependencies. Keypoints are derived deterministically from each actor's
role + animation clock, so the same scene always produces the same skeletons.

Roles understood (mirrors `scenario.PersonSpec.role`):
  walk / stand - upright torso, walking stride or sway
  fall         - torso rotates from upright to horizontal (uses pose.progress)
  fight        - crouched, flailing arms (high-frequency limb oscillation)
  chase        - forward lean, pumping arms, long stride
"""
from __future__ import annotations

import math

from ..types import FrameState, Keypoint, Pose, Track

# Body height in normalized frame units (person bbox is ~0.082 of frame height).
BH = 0.082
# Keypoint indices
NOSE, LE, RE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LEL, REL, LWR, RWR = 5, 6, 7, 8, 9, 10
LHP, RHP, LKN, RKN, LAN, RAN = 11, 12, 13, 14, 15, 16


def _kp(x: float, y: float, conf: float = 0.95) -> Keypoint:
    return Keypoint(x=round(x, 5), y=round(y, 5), confidence=round(conf, 3))


def _upright(x: float, y: float, t: float, phase: float, lean: float = 0.0,
             crouch: float = 0.0) -> list[Keypoint]:
    """Standard upright skeleton. x,y = feet center, y increasing downward."""
    stride = math.sin(2 * math.pi * 1.1 * t + phase)          # walk cycle
    bob = 0.004 * abs(math.cos(2 * math.pi * 1.1 * t + phase))
    hip_y = y - (0.47 - crouch) * BH + bob
    sh_y = y - 0.82 * BH + bob
    sh_sway = 0.015 * BH * stride
    lx, rx = x + 0.11 * BH * stride, x - 0.11 * BH * stride
    l_sw = -0.12 * BH * stride
    r_sw = 0.12 * BH * stride
    return [
        _kp(x + lean, y - 0.97 * BH + bob),                    # 0 nose
        _kp(x + lean - 0.02 * BH, y - 0.95 * BH + bob),        # 1 left eye
        _kp(x + lean + 0.02 * BH, y - 0.95 * BH + bob),        # 2 right eye
        _kp(x + lean - 0.045 * BH, y - 0.94 * BH + bob, 0.3),  # 3 left ear
        _kp(x + lean + 0.045 * BH, y - 0.94 * BH + bob, 0.3),  # 4 right ear
        _kp(x - 0.05 * BH + lean + sh_sway, sh_y),             # 5 left shoulder
        _kp(x + 0.05 * BH + lean - sh_sway, sh_y),             # 6 right shoulder
        _kp(x - 0.09 * BH + lean + l_sw, y - 0.68 * BH + bob), # 7 left elbow
        _kp(x + 0.09 * BH + lean + r_sw, y - 0.68 * BH + bob), # 8 right elbow
        _kp(x - 0.10 * BH + lean + 1.6 * l_sw, y - 0.50 * BH + bob),  # 9 left wrist
        _kp(x + 0.10 * BH + lean + 1.6 * r_sw, y - 0.50 * BH + bob),  # 10 right wrist
        _kp(x - 0.04 * BH + lean, hip_y),                      # 11 left hip
        _kp(x + 0.04 * BH + lean, hip_y),                      # 12 right hip
        _kp(lx, y - 0.26 * BH + bob),                          # 13 left knee
        _kp(rx, y - 0.26 * BH + bob),                          # 14 right knee
        _kp(lx, y - 0.03 * BH),                                # 15 left ankle
        _kp(rx, y - 0.03 * BH),                                # 16 right ankle
    ]


def _fallen(x: float, y: float, progress: float) -> list[Keypoint]:
    """Body rotated by `progress * 90deg` around the feet (forward fall)."""
    theta = math.radians(90.0 * progress)
    cos, sin = math.cos(theta), math.sin(theta)
    base = _upright(x, y, 0.0, 0.0)
    out: list[Keypoint] = []
    for kp in base:
        dx = kp.x - x
        dy = kp.y - y
        rx = dx * cos - dy * sin
        ry = dx * sin + dy * cos
        out.append(_kp(x + rx, y + ry, kp.confidence * 0.9))
    return out


def _fighter(x: float, y: float, t: float, phase: float) -> list[Keypoint]:
    """Crouched boxer: fast flailing arms, slight bob, torso forward."""
    kps = _upright(x, y, t, phase, lean=0.03 * BH, crouch=0.10)
    for idx, base_idx, f, ph in [
        (LEL, LSH, 4.5, phase),
        (REL, RSH, 4.5, phase + math.pi),
    ]:
        bx = kps[base_idx].x
        by = kps[base_idx].y
        ang = 2 * math.pi * f * t + ph
        kps[idx] = _kp(bx + 0.09 * BH * math.cos(ang), by - 0.10 * BH * abs(math.sin(ang)))
    for idx, base_idx, f, ph in [
        (LWR, LEL, 5.5, phase),
        (RWR, REL, 5.5, phase + math.pi),
    ]:
        bx = kps[base_idx].x
        by = kps[base_idx].y
        ang = 2 * math.pi * f * t + ph
        kps[idx] = _kp(bx + 0.13 * BH * math.cos(ang), by - 0.08 * BH)
    return kps


def _chaser(x: float, y: float, t: float, phase: float) -> list[Keypoint]:
    """Sprinting: forward lean, driving knees, pumping arms."""
    kps = _upright(x, y, t, phase, lean=0.06 * BH, crouch=0.03)
    stride = math.sin(2 * math.pi * 1.5 * t + phase)
    kps[LKN] = _kp(x - 0.06 * BH, y - (0.30 + 0.12 * max(0.0, stride)) * BH)
    kps[RKN] = _kp(x + 0.06 * BH, y - (0.30 + 0.12 * max(0.0, -stride)) * BH)
    kps[LWR] = _kp(x - 0.10 * BH, y - (0.52 + 0.14 * max(0.0, stride)) * BH)
    kps[RWR] = _kp(x + 0.10 * BH, y - (0.52 + 0.14 * max(0.0, -stride)) * BH)
    kps[LEL] = _kp(x - 0.08 * BH, y - 0.66 * BH)
    kps[REL] = _kp(x + 0.08 * BH, y - 0.66 * BH)
    return kps


class SyntheticPoseModel:
    """Generates skeletons for the scripted scene.

    `positions` is the list of `scenario.ScenePosition` for the current frame;
    `t` is the current scene time. Skeletons are matched to tracks by nearest
    centroid.
    """

    def __init__(self, positions: list, t: float):
        self._positions = positions
        self._t = t

    def estimate(self, state: FrameState) -> list[Pose]:
        """Greedy one-to-one matching: each scene position is used at most
        once, so two tracks near each other cannot both claim the same actor
        (and one steal the other's role/progress)."""
        # Anchor on the bbox bottom-center (feet), not the centroid: scene
        # positions are feet-level, and matching centroid-to-feet biases the
        # distances by half a body height, letting a neighbor steal a match.
        candidates: list[tuple[float, Track, object]] = []
        for tr in state.tracks:
            x1, y1, x2, y2 = tr.bbox
            fx, fy = (x1 + x2) / 2 / state.frame_w, y2 / state.frame_h
            for sp in self._positions:
                d = math.hypot(sp.x - fx, sp.y - fy)
                candidates.append((d, tr, sp))
        candidates.sort(key=lambda c: c[0])
        used_positions: set[int] = set()
        used_tracks: set[int] = set()
        out: list[Pose] = []
        for d, tr, sp in candidates:
            if tr.track_id in used_tracks or id(sp) in used_positions:
                continue
            if d > 0.07:
                break
            used_tracks.add(tr.track_id)
            used_positions.add(id(sp))
            kps = self._skeleton(sp)
            if kps is not None:
                out.append(Pose(track_id=tr.track_id, keypoints=kps))
        return out

    def _skeleton(self, sp) -> list[Keypoint] | None:
        role = sp.person.role
        t = self._t
        phase = sp.person.jitter_phase
        x = sp.x
        y = sp.y
        if role == "fall":
            return _fallen(x, y, getattr(sp, "progress", 0.0))
        if role == "fight":
            return _fighter(x, y, t, phase)
        if role == "chase":
            return _chaser(x, y, t, phase)
        if role == "stand":
            return _upright(x, y, t * 0.4, phase)
        return _upright(x, y, t, phase)
