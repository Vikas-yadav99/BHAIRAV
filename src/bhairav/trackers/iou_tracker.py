"""Greedy IoU tracker used by the synthetic blob path.

Deterministic, dependency-free, and unit-testable. The real CCTV path
uses ByteTrack via ultralytics (see bhairav/detectors/yolo_detector.py).
"""
from __future__ import annotations

from ..geometry import bbox_iou
from ..types import Detection, Track


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.25, max_age: int = 45, min_hits: int = 1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        track_ids = list(self._tracks.keys())

        # Score all (track, detection) pairs and greedily assign best matches.
        pairs: list[tuple[float, int, int]] = []
        for i, det in enumerate(detections):
            for tid in track_ids:
                iou = bbox_iou(det.bbox, self._tracks[tid]["bbox"])
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, i))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _, tid, i in pairs:
            if tid in matched_tracks or i in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(i)
            st = self._tracks[tid]
            st.update(bbox=detections[i].bbox, label=detections[i].label,
                      conf=detections[i].confidence, class_id=detections[i].class_id,
                      hits=st["hits"] + 1, missed=0)

        for tid, st in self._tracks.items():
            if tid not in matched_tracks:
                st["missed"] += 1

        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"bbox": det.bbox, "label": det.label,
                                 "conf": det.confidence, "class_id": det.class_id,
                                 "hits": 1, "missed": 0}
            matched_tracks.add(tid)

        for tid in [t for t, st in self._tracks.items() if st["missed"] > self.max_age]:
            del self._tracks[tid]

        out: list[Track] = []
        for tid in sorted(matched_tracks):
            st = self._tracks[tid]
            if st["hits"] >= self.min_hits:
                out.append(Track(track_id=tid, bbox=st["bbox"], label=st["label"],
                                  confidence=st["conf"], class_id=st["class_id"]))
        return out
