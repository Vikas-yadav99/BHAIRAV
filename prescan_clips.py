#!/usr/bin/env python3
"""Fast pre-scan: check each clip for people using YOLO on sampled frames."""
import sys, json
from pathlib import Path

CLIPS_DIR = Path("src/bhairav/test_footage/heavy_crowd")

def prescan(clip_path, sample_frames=5):
    from ultralytics import YOLO
    import cv2
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"file": clip_path.name, "error": "cannot open"}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    frame_indices = [int(i * total_frames / max(sample_frames, 1)) for i in range(sample_frames)]
    max_persons = 0
    total_persons = 0
    cap = cv2.VideoCapture(str(clip_path))
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        results = model(frame, imgsz=480, conf=0.3, verbose=False)
        persons = sum(1 for r in results for box in r.boxes if int(box.cls[0]) == 0)
        max_persons = max(max_persons, persons)
        total_persons += persons
    cap.release()
    return {
        "file": clip_path.name, "path": str(clip_path),
        "width": width, "height": height, "fps": round(fps, 1),
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 1) if fps > 0 else 0,
        "size_mb": round(clip_path.stat().st_size / 1_048_576, 1),
        "max_persons_per_frame": max_persons,
        "avg_persons_per_sample": round(total_persons / max(sample_frames, 1), 1),
    }

if __name__ == "__main__":
    clips = sorted(CLIPS_DIR.glob("clip_*.mp4"))
    print(f"Scanning {len(clips)} clips...")
    results = []
    for i, clip in enumerate(clips):
        r = prescan(clip, sample_frames=5)
        results.append(r)
        persons = r.get("max_persons_per_frame", 0)
        marker = "HOT" if persons >= 5 else "OK" if persons >= 1 else "---"
        print(f"  [{i+1}/{len(clips)}] {marker} {r['file']}: max={persons} persons, {r.get('duration_sec',0):.1f}s")
    results.sort(key=lambda x: x.get("max_persons_per_frame", 0), reverse=True)
    with open(CLIPS_DIR / "prescan_results.json", "w") as f:
        json.dump(results, f, indent=2)
    crowd_clips = [r for r in results if r.get("max_persons_per_frame", 0) >= 2]
    empty_clips = [r for r in results if r.get("max_persons_per_frame", 0) == 0]
    print(f"\n{'='*60}")
    print(f"PRESCAN: {len(results)} clips, {len(crowd_clips)} with crowd, {len(empty_clips)} empty")
    print(f"\nTop crowd clips:")
    for r in crowd_clips[:20]:
        print(f"  {r['file']}: {r['max_persons_per_frame']} persons, {r['duration_sec']}s, {r['size_mb']}MB")
    with open(CLIPS_DIR / "crowd_clips.json", "w") as f:
        json.dump(crowd_clips, f, indent=2)
