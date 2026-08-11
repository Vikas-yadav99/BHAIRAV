"""BHAIRAV Phase 3-7 - live server: pipeline -> LiveHub -> FastAPI/WebSocket.

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
               stats, webhook_url, recent_alerts):
    """Pipeline loop on a background thread; pushes frames+alerts to the hub.

    `store` is the SAME EvidenceStore the API reads from, so the recorder's
    writes stay visible to search/status (the store keeps an in-memory index
    that only its own mutations update).

    Live sources (RTSP/RTMP/webcam) are re-opened with exponential backoff
    when they drop, and connect/drop health is surfaced via /api/status.
    """
    from bhairav.sources import SourceMonitor, SourceKind, classify_source, open_capture

    kind, desc = classify_source(source)
    monitor = SourceMonitor(kind, desc)
    stats.set_source(monitor)
    print(f"[source] {desc} (kind={kind.value})", flush=True)

    def opener():
        if kind is SourceKind.BLOB:
            raise RuntimeError("blob source has no capture to open")
        return open_capture(source, monitor=monitor, retries=3, base_delay=2.0)
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
            recent_alerts.append(a.to_dict())
            del recent_alerts[:-cfg.backend.max_recent_alerts]  # bounded rolling feed (backend.max_recent_alerts)
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
    import time as _time
    delay = 1.0
    while not stop.is_set():
        try:
            run_pipeline(detector, engine, source=source, on_frame=on_frame,
                         opener=opener if kind is not SourceKind.BLOB else None)
            # a clean pass ended (file replay done / stream EOF) -> reconnect
            if kind is not SourceKind.BLOB:
                monitor.dropped("stream ended, reconnecting")
        except RuntimeError as exc:
            monitor.dropped(str(exc))
            print(f"[source] reconnect failed ({exc}); retrying in {delay:.0f}s",
                  flush=True)
            stop.wait(delay)
            delay = min(delay * 2, 30.0)
            continue
        recorder.flush()
        # scene clock restarts at 0 next pass: reset cooldown/open state so
        # the second pass isn't blocked by replay-1 timestamps
        recorder.reset()
        # brief pause between replays so the stream doesn't hot-loop
        stop.wait(0.5)
        delay = 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="BHAIRAV live server (Phases 1-7)")
    ap.add_argument("--source", default="blob")
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--evidence", default=None, help="evidence dir (default: cfg)")
    ap.add_argument("--db-url", default=None,
                    help="PostgreSQL URL (or $BHAIRAV_DB_URL / backend.db in "
                         "config.yaml); enables the PG evidence store")
    ap.add_argument("--evidence-key", default=None,
                    help="base64 32-byte AES-256 key for evidence at rest "
                         "(default: $BHAIRAV_EVIDENCE_KEY)")
    ap.add_argument("--tls-cert", default=None, help="TLS certificate file (PEM)")
    ap.add_argument("--tls-key", default=None, help="TLS private key file (PEM)")
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
    from bhairav.backend.hardening import is_loopback, load_evidence_key
    from bhairav.backend.server import LiveHub, PipelineStats, create_app
    from bhairav.backend.users import DEFAULT_USERS, UserStore

    # ---- startup security posture ---------------------------------------
    loopback = is_loopback(host)
    if secret in ("", "dev-secret-change-me") and not loopback and not os.environ.get("BHAIRAV_ALLOW_DEFAULT_SECRET"):
        raise SystemExit(
            "REFUSING TO START: default BHAIRAV secret on a non-loopback interface.\n"
            "  Set BHAIRAV_SECRET to a long random value (or BHAIRAV_ALLOW_DEFAULT_SECRET=1 for local dev only).")

    users = UserStore(cfg.backend.users_file)
    weak = [d["username"] for d in DEFAULT_USERS
            if users.get(d["username"]) and users._verify_password(
                d["password"], users.get(d["username"])["salt"],
                users.get(d["username"])["iterations"], users.get(d["username"])["hash"])]
    if weak:
        msg = (f"demo accounts still use default passwords: {', '.join(weak)}. "
               f"Change them via POST /api/users/{{user}}/password.")
        if not loopback and not os.environ.get("BHAIRAV_ALLOW_DEFAULT_PASSWORDS"):
            raise SystemExit("REFUSING TO START: " + msg)
        print("[security] WARNING: " + msg)

    # ---- evidence encryption at rest (AES-256-GCM) -----------------------
    evidence_key = load_evidence_key(args.evidence_key or os.environ.get("BHAIRAV_EVIDENCE_KEY"))
    if cfg.evidence.encrypt and evidence_key is None:
        raise SystemExit(
            "evidence.encrypt is true but no key is set. Set BHAIRAV_EVIDENCE_KEY "
            "(base64 32-byte key, e.g. from: python -c \"import os,base64;"
            "print(base64.b64encode(os.urandom(32)).decode())\") or pass --evidence-key.")

    db_url = args.db_url or os.environ.get("BHAIRAV_DB_URL") or cfg.backend.db
    if db_url:
        from bhairav.backend.pg_audit import PostgresAuditLog
        from bhairav.backend.pg_store import PostgresEvidenceStore
        try:
            audit = PostgresAuditLog(db_url)
            store = PostgresEvidenceStore(
                db_url, camera=cfg.evidence.camera, fps=detector.fps,
                blur_faces=cfg.evidence.blur_faces,
                encrypt=cfg.evidence.encrypt, key=evidence_key,
                max_events=cfg.evidence.max_events, root=evidence_dir)
        except RuntimeError as exc:
            raise SystemExit(
                "REFUSING TO START: PostgreSQL backend unavailable.\n"
                f"  {exc}")
        print(f"[db] evidence + audit: PostgreSQL ({db_url.split('@')[-1]})")
    else:
        audit = AuditLog(Path(evidence_dir) / "audit.jsonl")
        store = EvidenceStore(evidence_dir, camera=cfg.evidence.camera,
                              fps=detector.fps,
                              blur_faces=cfg.evidence.blur_faces,
                              encrypt=cfg.evidence.encrypt, key=evidence_key,
                              max_events=cfg.evidence.max_events)
        print(f"[db] evidence store: file-based ({evidence_dir})")
    stats = PipelineStats()
    webhook_url = cfg.backend.webhook_url
    hub = LiveHub()
    recent_alerts: list[dict] = []

    # ---- face search (Phase 6): find a person in evidence by photo --------
    from bhairav.backend.face_search import build_face_service
    try:
        face = build_face_service(store)
        print("[security] face search: ENABLED (gallery + evidence face index)")
    except RuntimeError as exc:
        print("[warn] face search disabled:", exc)
        face = None

    # ---- vehicle watchlist (Phase 6: ANPR) --------------------------------
    from bhairav.backend.anpr import PlateRegistry
    plates = PlateRegistry(Path(evidence_dir).parent / "plates.json")
    for rule in engine.rules:
        if getattr(rule, "name", None) == "stolen_vehicle":
            rule.registry = plates   # one registry shared with the REST API
    print("[security] vehicle watchlist: ENABLED (ANPR on synthetic plates)")

    app = create_app(store, audit, secret=secret, hub=hub, users=users,
                     stats=stats, webhook_url=webhook_url, face=face,
                     plates=plates, recent_alerts=recent_alerts)

    # run the pipeline on a background thread
    stop = threading.Event()
    t = threading.Thread(target=run_stream, args=(cfg, args.source, detector,
                                                  engine, hub, store, evidence_dir,
                                                  stop, stats, webhook_url,
                                                  recent_alerts),
                         daemon=True)
    t.start()

    import uvicorn
    scheme = "https" if (args.tls_cert and args.tls_key) else "http"
    print(f"BHAIRAV Phase 7 server -> {scheme}://{host}:{port}  (evidence: {evidence_dir})")
    print(f"Dashboard: {scheme}://{host}:{port}/dashboard/")
    if cfg.evidence.encrypt:
        print("[security] evidence encryption at rest: ENABLED (AES-256-GCM)")
    print("Login (POST /auth/login with username + password):")
    for u in users.public_view():
        pw = "admin123" if u["username"] == "admin" else f"{u['username']}123"
        print(f"  {u['username']:9s} / {pw:<12s} role={u['role']}")
    if webhook_url:
        print(f"Webhook: red alerts -> {webhook_url}")
    print(f"Live stream: {scheme.replace('https', 'wss').replace('http', 'ws')}://{host}:{port}/ws/stream?token=<token>")
    print("[security] login rate limit: 10/min per IP; change defaults via config")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning",
                    ssl_certfile=args.tls_cert, ssl_keyfile=args.tls_key)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
