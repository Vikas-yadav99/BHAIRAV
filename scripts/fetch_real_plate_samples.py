"""Fetch real license-plate sample images for ANPR evaluation.

Downloads the UFPR-ALPR dataset's sample contact sheet (public GitHub repo,
research dataset) and extracts a handful of plate regions into
output/real_plates/crops/. These are REAL plates, so the template OCR backend
fails on them and the easyocr backend reads them - the honest way to see what
each backend can do.

Usage:  python scripts/fetch_real_plate_samples.py
Then:   python scripts/eval_anpr.py
"""
from __future__ import annotations

import pathlib
import urllib.request

import cv2

SAMPLES_URL = ("https://raw.githubusercontent.com/raysonlaroca/ufpr-alpr-dataset/"
               "master/media/samples.png")
# (x0, y0, x1, y1) regions of the contact sheet that contain a plate
CLUSTERS = [
    (480, 500, 700, 640),    # ENE 545 ESCOLAR (school bus)
    (100, 800, 300, 920),    # Mercosur plate (L04Z1-style)
    (900, 0, 1100, 100),     # BOMBEIROS (fire dept)
    (1380, 240, 1560, 340),  # angled plate
]
OUT = pathlib.Path("output/real_plates/crops")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SAMPLES_URL, headers={"User-Agent": "bhairav-eval"})
    data = urllib.request.urlopen(req, timeout=120).read()
    img = cv2.imdecode(__import__("numpy").frombuffer(data, __import__("numpy").uint8),
                       cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    for i, (x0, y0, x1, y1) in enumerate(CLUSTERS):
        crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(OUT / f"crop_{i}.png"), crop)
        print(f"wrote {OUT / f'crop_{i}.png'}")
    print("Samples: UFPR-ALPR dataset (research use); see "
          "https://github.com/raysonlaroca/ufpr-alpr-dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
