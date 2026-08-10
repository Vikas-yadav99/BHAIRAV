"""Download the ML models into <repo>/models/ with SHA-256 integrity
verification (supply-chain check before anything runs).

Usage:
    python scripts/fetch_models.py          # -> models/*
Sources (official releases):
    - YuNet face detection: opencv_zoo/models/face_detection_yunet
    - SFace face recognition: opencv_zoo/models/face_recognition_sface
    - MediaPipe Pose Landmarker: storage.googleapis.com/mediapipe-models
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (filename, url, sha256) - hashes pinned for supply-chain verification
MODELS = [
    ("face_detection_yunet.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/"
     "face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    ("face_recognition_sface.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/"
     "face_recognition_sface/face_recognition_sface_2021dec.onnx",
     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    ("pose_landmarker_full.task",
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
     "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
     "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out_dir = ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url, expected in MODELS:
        dest = out_dir / name
        if dest.exists():
            digest = sha256(dest)
            if digest == expected:
                print(f"  {name}: verified ({dest.stat().st_size} bytes)")
                continue
            print(f"  {name}: hash mismatch ({digest[:16]}...), re-downloading")
        print(f"  downloading {name} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        digest = sha256(tmp)
        if digest != expected:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"integrity check FAILED for {name}: {digest} != {expected}")
        tmp.rename(dest)
        print(f"  {name}: {len(data)} bytes, sha256 verified")
    print("Models ready in", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
