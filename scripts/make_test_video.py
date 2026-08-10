"""Render the scripted Phase 1 scene to an MP4 for later YOLO testing.

Usage: python scripts/make_test_video.py [--out output/sample_scene.mp4] [--fps 15]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from bhairav.config import load_config
from bhairav.detectors import BlobDetector
from bhairav.detectors.scenario import default_scenario


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/sample_scene.mp4")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    scenario = default_scenario(cfg.synthetic)
    det = BlobDetector(scenario, fps=args.fps, width=cfg.synthetic.width, height=cfg.synthetic.height)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (cfg.synthetic.width, cfg.synthetic.height))
    n = 0
    for state in det.stream():
        writer.write(state.frame)
        n += 1
    writer.release()
    print(f"Wrote {n} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
