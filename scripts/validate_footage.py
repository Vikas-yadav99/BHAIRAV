"""BHAIRAV Phase 9 M1 - real-footage validation harness CLI.

Runs the real pipeline (detector -> tracker -> rules) over any source and
writes a metrics report you can gate on. Works with the blob detector on
the synthetic scene (zero ML deps) or with YOLO + ByteTrack on a real
video file / camera stream (requires `pip install ultralytics`).

Usage:
  python scripts/validate_footage.py --source output/real/vtest.avi --detector yolo
  python scripts/validate_footage.py --source blob --max-frames 90
  python scripts/validate_footage.py --source cam.mp4 \
      --threshold "fps>=8,mean_track_len>=5,fragmentation_rate<=0.3" \
      --report output/validation.html --json output/validation.json
  python scripts/validate_footage.py --list-metrics

Exit code 1 when any configured threshold fails (usable as a CI gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_thresholds(inline: str | None, path: str | None) -> dict:
    """Merge --threshold (metric>=value,...) and --thresholds <json file>."""
    from bhairav.eval.harness import parse_thresholds

    merged: dict = {}
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        merged.update(parse_thresholds(raw) if isinstance(raw, str) else raw)
    if inline:
        merged.update(parse_thresholds(inline))
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate the BHAIRAV pipeline over footage (Phase 9 M1)")
    ap.add_argument("--source", default="blob",
                    help="video file / rtsp:// / camera index / 'blob' "
                         "(synthetic scene, no ML deps)")
    ap.add_argument("--detector", default="auto", choices=["auto", "blob", "yolo"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap the run (default: the whole source)")
    ap.add_argument("--label", default=None, help="report label, e.g. 'vtest run 3'")
    ap.add_argument("--threshold", default=None,
                    help="inline pass/fail thresholds, e.g. "
                         "'fps>=8,mean_track_len>=5'")
    ap.add_argument("--thresholds", default=None,
                    help="JSON file with {metric: [op, value]} or a string")
    ap.add_argument("--json", default=None, dest="json_out",
                    help="write the full metrics as JSON to this path")
    ap.add_argument("--report", default=None,
                    help="write a self-contained HTML report to this path")
    ap.add_argument("--list-metrics", action="store_true",
                    help="print every metric the harness collects, then exit")
    ap.add_argument("--reid", action="store_true",
                    help="also run re-id validation: real-footage track "
                         "self-consistency + deterministic cross-camera "
                         "rank-1 accuracy (Phase 9)")
    ap.add_argument("--reid-threshold", default=None,
                    help="inline pass/fail thresholds for re-id metrics, "
                         "e.g. 'reid_separation>=0.15,reid_rank1>=0.8'")
    args = ap.parse_args()

    from bhairav.config import load_config
    from bhairav.eval.harness import (check_thresholds, render_html,
                                      render_markdown, run_validation)
    from bhairav.pipeline import build_engine, make_detector

    if args.list_metrics:
        from bhairav.eval.harness import ValidationSummary
        print("\n".join(sorted(ValidationSummary().to_dict())))
        return 0

    cfg = load_config(args.config)
    detector = make_detector(cfg, args.detector, args.source)
    engine = build_engine(cfg)

    print(f"[validate] source={args.source} detector={args.detector} "
          f"max_frames={args.max_frames}", flush=True)
    summary, alerts = run_validation(detector, engine, source=args.source,
                                     max_frames=args.max_frames)

    thresholds = _load_thresholds(args.threshold, args.thresholds)
    checks = check_thresholds(summary, thresholds) if thresholds else None

    # Phase 9 M5: optional re-id validation (real footage + cross camera)
    reid, reid_checks, extra_md = None, None, None
    if args.reid:
        from bhairav.eval.reid_eval import (check_reid_thresholds,
                                            render_reid_markdown,
                                            run_reid_validation)
        print(f"[validate] running re-id validation (source={args.source})",
              flush=True)
        reid = run_reid_validation(detector, engine, cfg,
                                   source=args.source,
                                   max_frames=args.max_frames)
        extra_md = render_reid_markdown(reid)
        reid_checks = check_reid_thresholds(reid, args.reid_threshold)

    print(render_markdown(summary, args.label, checks, extra_md), flush=True)

    if args.json_out:
        payload = {
            "label": args.label,
            "source": args.source,
            "detector": args.detector,
            "summary": summary.to_dict(),
            "thresholds": {k: list(v) for k, v in thresholds.items()},
            "checks": checks or [],
            "reid": reid,
            "reid_checks": reid_checks or [],
            "alerts": [a.to_dict() for a in alerts],
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[validate] wrote {out}")
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(summary, args.label, checks, extra_md),
                       encoding="utf-8")
        print(f"[validate] wrote {out}")

    all_checks = (checks or []) + (reid_checks or [])
    if all_checks:
        failed = [c for c in all_checks if not c["ok"]]
        if failed:
            print(f"[validate] FAILED {len(failed)} of {len(all_checks)} "
                  f"thresholds", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
