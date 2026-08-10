"""BHAIRAV Phase 3-5 - live server: pipeline -> LiveHub -> FastAPI/WebSocket.

Usage:
  python scripts/serve.py                    # synthetic scene + API on :8000
  python scripts/serve.py --port 9000        # custom port
  python scripts/serve.py --source clip.mp4  # real video (needs ultralytics)

Endpoints (see src/bhairav/backend/server.py):
  POST /auth/login            {"username": "alice", "password": "..."}
  GET  /health | /api/status
  GET  /api/evidence?rule=&severity=&q=   | /api/evidence/export (analyst+)
  GET  /api/evidence/{id}/snapshot | /clip
  POST /api/evidence/{id}/status | /notes
  DELETE /api/evidence/{id}
  GET  /api/audit | /api/users (admin)
  GET  /api/alerts/recent
  WS   /ws/stream?token=<login token>

Seeded accounts on first run (change the admin password via the API or the
BHAIRAV_ADMIN_PW env var): admin/admin123, operator/operator123,
analyst/analyst123, viewer/viewer123.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from bhairav.config import load_config
from bhairav.pipeline import build_engine, make_detector, run_pipeline


def _jpeg_b64(frame: np.ndarray, scale: float = 0.5) -> str:
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else ""


def run_stream(cfg, source, detector, engine, hub, store, evidence_dir, stop,
               stats, webhook_url):
    """Pipeline loop on a background thread; pushes frames+alerts to the hub.

    `store` is the SAME EvidenceStore the API reads from, so the recorder's
    writes stay visible to search/status (the store keeps an in-memory index
    that only its own mutations update).
    """
    from bhairav.alert_log import AlertLog
    from bhairav.backend.evidence import EventRecorder
    from bhairav.backend.server import webhook_notify

    ev_cfg = cfg.evidence
    recorder = EventRecorder(store, pre_sec=ev_cfg.pre_sec, post_sec=ev_cfg.post_sec,
                             min_gap_sec=ev_cfg.min_gap_sec,
                             blur_faces=ev_cfg.blur_faces)
    log = AlertLog(Path(evidence_dir).parent / "alerts_live.jsonl")
    log.clear()

    def on_frame(state, alerts):
        if stop.is_set():
            return False
        if state.frame is None:
            return None
        # evidence capture + alert log
        recorder.observe(state, state.frame)
        for a in alerts:
            recorder.on_alert(a, state.frame, state=state)
            log.write(a)
            if a.severity.value == "red":
                webhook_notify(webhook_url, a.to_dict())
        # close events that have gone quiet (respects post_sec)
        recorder.finalize_due(state.timestamp)
        # ops telemetry
        stats.bump(frames=1, alerts=len(alerts))
        # live feed (downscaled jpeg to keep the WS light)
        hub.publish_frame(
            frame_id=state.frame_id, timestamp=state.timestamp,
            jpeg_b64=_jpeg_b64(state.frame),
            tracks=[{"id": t.track_id, "label": t.label, "bbox": list(t.bbox),
                     "conf": round(t.confidence, 3)} for t in state.tracks],
            poses=[{"track_id": p.track_id,
                    "kps": [[round(k.x, 4), round(k.y, 4), round(k.confidence, 3)]
                            for k in p.keypoints]} for p in state.poses],
            alerts=[a.to_dict() for a in alerts])
        return None

    # loop the source until the server shuts down ("live" feel)
    while not stop.is_set():
        run_pipeline(detector, engine, source=source, on_frame=on_frame)
        recorder.flush()
        # scene clock restarts at 0 next pass: reset cooldown/open state so
        # the second pass isn't blocked by replay-1 timestamps
        recorder.reset()
        # brief pause between replays so the stream doesn't hot-loop
        import time as _time
        stop.wait(0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description="BHAIRAV Phase 3 live server")
    ap.add_argument("--source", default="blob")
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--evidence", default=None, help="evidence dir (default: cfg)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = build_engine(cfg)
    detector = make_detector(cfg, args.detector, args.source)
    host = args.host or cfg.backend.host
    port = args.port or cfg.backend.port
    evidence_dir = args.evidence or cfg.evidence.dir
    secret = os.environ.get("BHAIRAV_SECRET", cfg.backend.secret)

    # assemble the app
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.server import LiveHub, PipelineStats, create_app
    from bhairav.backend.users import UserStore

    audit = AuditLog(Path(evidence_dir) / "audit.jsonl")
    store = EvidenceStore(evidence_dir, camera=cfg.evidence.camera,
                          fps=detector.fps, blur_faces=cfg.evidence.blur_faces,
                          encrypt=cfg.evidence.encrypt,
                          max_events=cfg.evidence.max_events)
    users = UserStore(cfg.backend.users_file)
    stats = PipelineStats()
    webhook_url = cfg.backend.webhook_url
    hub = LiveHub()
    app = create_app(store, audit, secret=secret, hub=hub, users=users,
                     stats=stats, webhook_url=webhook_url)

    # run the pipeline on a background thread
    stop = threading.Event()
    t = threading.Thread(target=run_stream, args=(cfg, args.source, detector,
                                                  engine, hub, store, evidence_dir,
                                                  stop, stats, webhook_url),
                         daemon=True)
    t.start()

    import uvicorn
    print(f"BHAIRAV Phase 5 server -> http://{host}:{port}  (evidence: {evidence_dir})")
    print(f"Dashboard: http://{host}:{port}/dashboard/")
    print("Login (POST /auth/login with username + password):")
    for u in users.public_view():
        pw = "admin123" if u["username"] == "admin" else f"{u['username']}123"
        print(f"  {u['username']:9s} / {pw:<12s} role={u['role']}")
    if webhook_url:
        print(f"Webhook: red alerts -> {webhook_url}")
    print(f"Live stream: ws://{host}:{port}/ws/stream?token=<token>")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
