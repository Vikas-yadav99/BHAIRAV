"""Synthetic blob detector: renders a scripted scene and tracks its actors.

Runs the FULL pipeline (detection -> tracking -> pose -> rules -> viz)
offline with zero ML dependencies, so every phase is demoable before
torch/ultralytics/mediapipe land.
"""
from __future__ import annotations

import numpy as np
import cv2

from ..pose import SyntheticPoseModel
from ..trackers import IoUTracker
from ..types import Detection, FrameState
from .base import Detector
from .scenario import ScenePosition, Scenario


class BlobDetector(Detector):
    def __init__(self, scenario: Scenario, fps: float = 15.0, width: int = 1280, height: int = 720,
                 tracker: IoUTracker | None = None):
        self.scenario = scenario
        self._fps = float(fps)
        self.width = width
        self.height = height
        self.tracker = tracker or IoUTracker(iou_threshold=0.25, max_age=45, min_hits=1)
        self._bg: np.ndarray | None = None

    @property
    def fps(self) -> float:
        return self._fps

    def stream(self, source: str | None = None, max_frames: int | None = None):
        dt = 1.0 / self._fps
        t = 0.0
        i = 0
        while t < self.scenario.duration_sec - 1e-9:
            if max_frames is not None and i >= max_frames:
                break
            positions = self.scenario.positions_at(t)
            dets = [Detection(bbox=self._bbox_for(p), confidence=0.92,
                              class_id=p.person.class_id, label=p.person.label)
                    for p in positions]
            tracks = self.tracker.update(dets)
            frame = self._render(positions)
            state = FrameState(frame_id=i, timestamp=round(t, 3), tracks=tracks,
                               frame_w=self.width, frame_h=self.height, frame=frame)
            state.poses = SyntheticPoseModel(positions, t).estimate(state)
            yield state
            t += dt
            i += 1

    # ---- geometry ---------------------------------------------------------
    def _bbox_for(self, pose: ScenePosition) -> tuple[float, float, float, float]:
        px = pose.x * self.width
        py = pose.y * self.height
        if pose.person.size == "vehicle":
            # big enough that the 10-char plate text fits at the renderer's
            # font scale without glyphs touching (backend/anpr templates match)
            bw, bh = 0.13 * self.width, 0.060 * self.height
        else:
            bw, bh = 0.034 * self.width, 0.082 * self.height
            if pose.person.role == "fall" and pose.progress > 0:
                # Collapsing: the bbox flattens as the body rotates down.
                f = pose.progress
                bh = bh * (1.0 - 0.55 * f)
                bw = bw * (1.0 + 0.9 * f)
        return (px - bw / 2, py - bh, px + bw / 2, py)

    # ---- rendering --------------------------------------------------------
    def base_frame(self) -> np.ndarray:
        if self._bg is None:
            self._bg = self._render_background()
        return self._bg.copy()

    def _render(self, poses: list[ScenePosition]) -> np.ndarray:
        img = self.base_frame()
        for p in poses:
            self._draw_blob(img, p)
        return img

    def _render_background(self) -> np.ndarray:
        w, h = self.width, self.height
        # Vertical gradient, surveillance-camera tint.
        top = np.array([46, 52, 58], dtype=np.uint8)   # dark slate
        bottom = np.array([18, 21, 24], dtype=np.uint8)
        grad = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        img = (top[None, None, :] * (1 - grad) + bottom[None, None, :] * grad).astype(np.uint8)
        img = np.repeat(img, w, axis=1)

        rng = np.random.default_rng(3)
        # Ground band.
        cv2.rectangle(img, (0, int(h * 0.72)), (w, h), (14, 16, 19), -1)
        # Two building blocks with windows (left + right background).
        for (bx, by, bw, bh) in [(int(w * 0.04), int(h * 0.10), int(w * 0.16), int(h * 0.30)),
                                 (int(w * 0.82), int(h * 0.08), int(w * 0.14), int(h * 0.34))]:
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (30, 34, 40), -1)
            for wy in range(by + 14, by + bh, 26):
                for wx in range(bx + 10, bx + bw - 8, 30):
                    cv2.rectangle(img, (wx, wy), (wx + 8, wy + 12), (70, 78, 90), -1)
        # Fence line + poles along the ground band.
        fence_y = int(h * 0.72)
        cv2.line(img, (0, fence_y), (w, fence_y), (24, 28, 32), 3)
        for x in range(0, w, 90):
            cv2.line(img, (x, fence_y), (x, fence_y + 8), (24, 28, 32), 2)
        # Sparse static noise dots for texture.
        xs = rng.integers(0, w, 900)
        ys = rng.integers(0, h, 900)
        for x, y in zip(xs, ys):
            img[y, x] = np.clip(img[y, x].astype(int) + int(rng.integers(-12, 12)), 0, 255).astype(np.uint8)
        return img

    def _draw_blob(self, img: np.ndarray, pose: ScenePosition) -> None:
        px = int(pose.x * self.width)
        py = int(pose.y * self.height)
        shade = 60 + (pose.person.pid * 13) % 40
        if pose.person.size == "vehicle":
            bw, bh = int(0.13 * self.width), int(0.060 * self.height)
            cv2.rectangle(img, (px - bw // 2, py - bh), (px + bw // 2, py), (shade, shade, shade + 12), -1)
            cv2.rectangle(img, (px - bw // 2, py - bh), (px + bw // 2, py), (40, 44, 50), 2)
            plate = pose.person.special.get("plate")
            if plate:
                # white plate (no border - keeps OCR segmentation clean) with
                # chars drawn one-by-one at a fixed advance so they never
                # touch; region matches backend/anpr.plate_region()
                rx1 = px - int(bw * 0.425)
                ry1 = py - int(bh * 0.70)
                rw, rh = int(bw * 0.85), int(bh * 0.65)
                cv2.rectangle(img, (rx1, ry1), (rx1 + rw, ry1 + rh), (228, 228, 232), -1)
                # thin strokes (1px) keep glyph interiors open at this size;
                # must mirror backend/anpr.PlateReader._build_templates
                scale, thickness = 0.5, 1
                gap = 1
                ty = ry1 + int(rh * 0.82)
                widths = [cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
                          for ch in plate]
                tw = sum(widths) + gap * (len(plate) - 1)
                tx = rx1 + max(2, (rw - tw) // 2)
                for i, ch in enumerate(plate):
                    cv2.putText(img, ch, (tx + sum(widths[:i]) + gap * i, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, scale,
                                (20, 20, 24), thickness, cv2.LINE_AA)
        else:
            bw, bh = int(0.034 * self.width), int(0.082 * self.height)
            f = pose.progress if pose.person.role == "fall" else 0.0
            bw = int(bw * (1.0 + 0.9 * f))
            bh = int(bh * (1.0 - 0.55 * f))
            cv2.ellipse(img, (px, py - 2), (max(bw // 2, 4), 4), 0, 0, 360, (30, 34, 38), -1)  # shadow
            cv2.ellipse(img, (px, py - bh // 2 - 2), (int(bw * 0.42), int(bh * 0.52)),
                        0, 0, 360, (shade, shade, shade + 6), -1)  # body
            cv2.circle(img, (px, py - bh - 4), max(int(bw * 0.30), 3), (shade + 18, shade + 18, shade + 22), -1)  # head
