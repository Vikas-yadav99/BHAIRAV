"""BHAIRAV Phase 2 demo: video in -> detection + tracking + pose -> behavior alerts.

Usage:
  python scripts/run_demo.py                          # synthetic scene, watch live
  python scripts/run_demo.py --source blob --headless # offline, no window
  python scripts/run_demo.py --source clip.mp4        # real video (needs ultralytics)
  python scripts/run_demo.py --source 0               # webcam (needs ultralytics)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from bhairav.alert_log import AlertLog
from bhairav.config import load_config
from bhairav.pipeline import build_engine, make_detector, run_pipeline
from bhairav.rules.crowd_density import count_people_in_zone
from bhairav.viz import render


def main() -> int:
    ap = argparse.ArgumentParser(description="BHAIRAV Phase 2 demo")
    ap.add_argument("--source", default="blob", help="blob | video path | camera index")
    ap.add_argument("--detector", default="auto", help="auto | blob | yolo")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="output/demo.mp4", help="annotated video ('' to skip)")
    ap.add_argument("--alerts", default="output/alerts.jsonl")
    ap.add_argument("--frames", default="output/frames", help="snapshot dir ('' to skip)")
    ap.add_argument("--snapshot-every", type=int, default=40, help="save a frame PNG every N frames")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--headless", action="store_true", help="no GUI window")
    ap.add_argument("--evidence", default=None, metavar="DIR",
                    help="record Phase 3 evidence events (pre/during/post clips) to DIR")
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = build_engine(cfg)
    detector = make_detector(cfg, args.detector, args.source)

    evidence_recorder = None
    evidence_store = None
    if args.evidence:
        from bhairav.backend.evidence import EvidenceStore, EventRecorder
        ev_cfg = cfg.evidence
        evidence_store = EvidenceStore(args.evidence, camera=ev_cfg.camera,
                                       fps=detector.fps, blur_faces=ev_cfg.blur_faces,
                                       encrypt=ev_cfg.encrypt)
        evidence_recorder = EventRecorder(evidence_store, pre_sec=ev_cfg.pre_sec,
                                          post_sec=ev_cfg.post_sec,
                                          min_gap_sec=ev_cfg.min_gap_sec,
                                          blur_faces=ev_cfg.blur_faces)

    out_path = Path(args.out) if args.out else None
    frames_dir = Path(args.frames) if args.frames else None
    # Created lazily from the first frame, so real (YOLO) sources with
    # arbitrary resolution and fps write correctly.
    writer = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    log = AlertLog(args.alerts)
    log.clear()  # fresh run, fresh log

    frame_idx = 0
    alerts_total = 0
    start = time.perf_counter()

    def on_frame(state, alerts):
        nonlocal frame_idx, alerts_total, writer
        alerts_total += len(alerts)
        for a in alerts:
            log.write(a)
            print(f"[{a.timestamp:7.2f}s] {a.severity.value.upper():6s} {a.rule:10s} "
                  f"conf={a.confidence:.2f}  {a.message}")

        # Phase 3: keep the pre-event buffer warm, open/extend evidence events
        if evidence_recorder is not None:
            evidence_recorder.observe(state, state.frame)
            for a in alerts:
                evidence_recorder.on_alert(a, state.frame, state=state)
            # close events that have gone quiet (respects post_sec)
            evidence_recorder.finalize_due(state.timestamp)

        if state.frame is None:
            return None
        if writer is None and out_path:
            h, w = state.frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     detector.fps, (w, h))

        active = {}
        for a in alerts:
            if a.track_id is not None:
                active[a.track_id] = a.severity
        zone_counts = {z.name: count_people_in_zone(state, z)
                       for z in cfg.zones if z.kind == "monitored"}
        img = render(state.frame.copy(), state, cfg.zones, alerts, detector.fps,
                     active_severity=active, zone_counts=zone_counts, alerts_total=alerts_total)
        if writer is not None:
            writer.write(img)
        if frames_dir is not None and frame_idx % args.snapshot_every == 0:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}.png"), img)
        if not args.headless:
            cv2.imshow("BHAIRAV - Phase 2", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        frame_idx += 1

    try:
        run_pipeline(detector, engine, source=args.source, max_frames=args.max_frames,
                     on_frame=on_frame)
    finally:
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        # Phase 3: finalize any open evidence events
        if evidence_recorder is not None:
            finalized = evidence_recorder.flush()
            print(f"Evidence events captured: {len(finalized)}  ->  {args.evidence}")
            if evidence_store is not None:
                for rec in evidence_store.list_all():
                    print(f"  {rec.event_id}  {rec.rule:10s} {rec.severity:6s} "
                          f"{rec.frame_count:3d} frames  start={rec.start_ts:.1f}s")

    elapsed = time.perf_counter() - start
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s")
    print(f"Alerts fired: {alerts_total}  ->  {args.alerts}")
    print(f"Summary: {log.summary()}")
    if out_path:
        print(f"Annotated video: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
