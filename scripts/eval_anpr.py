"""Evaluate the ANPR backends on REAL license-plate images.

Compares the deterministic 'template' backend against the deep-learning
'easyocr' backend on real plate crops (fetch with
scripts/fetch_real_plate_samples.py first). Also evaluates the synthetic
scene plate so you can see both backends' accuracy where ground truth is
known exactly.

Usage:  python scripts/eval_anpr.py [--synthetic]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from bhairav.backend.anpr import PlateReader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="also evaluate the synthetic scene plate (known GT)")
    args = ap.parse_args()

    crops = sorted((ROOT / "output/real_plates/crops").glob("crop_*.png"))
    if not crops:
        print("no real plate crops found - run: "
              "python scripts/fetch_real_plate_samples.py")
        return 2

    tpl = PlateReader(backend="template")
    try:
        deep = PlateReader(backend="easyocr")
    except Exception as exc:  # pragma: no cover
        print(f"easyocr backend unavailable ({exc}); install: pip install easyocr")
        deep = None

    print(f"{'image':14s} {'template':16s} {'easyocr':16s} conf")
    print("-" * 56)
    for p in crops:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        bbox = (0, 0, w, h)
        t_txt, t_conf = tpl.read(img, bbox)
        if deep is not None:
            d_txt, d_conf = deep.read(img, bbox)
        else:
            d_txt, d_conf = None, 0.0
        print(f"{p.stem:14s} {str(t_txt):16s} {str(d_txt):16s} {d_conf:.2f}")

    if args.synthetic:
        # known ground truth: the demo vehicle's plate
        from bhairav.config import load_config
        from bhairav.pipeline import make_detector, build_engine, run_pipeline
        cfg = load_config(str(ROOT / "config.yaml"))
        det = make_detector(cfg, "blob", "blob")
        eng = build_engine(cfg)
        seen = set()
        for st in det.stream(source="blob", max_frames=400):
            for tr in st.tracks:
                if tr.class_id in (2, 5, 7):
                    img = st.frame
                    for backend in ("template", "easyocr"):
                        r = PlateReader(backend=backend)
                        txt, conf = r.read(img, tr.bbox)
                        if txt and (backend, txt) not in seen:
                            seen.add((backend, txt))
                            print(f"synthetic GT=MH12AB1234  {backend}: {txt!r} conf={conf:.2f}")
                    break
            if seen:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
