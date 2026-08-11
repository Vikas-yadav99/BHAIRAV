"""BHAIRAV Phase 7-8 - live server: pipelines -> LiveHub -> FastAPI/WebSocket.

Usage:
  python scripts/serve.py                    # cameras from config.yaml (default: 2)
  python scripts/serve.py --port 9000        # custom port
  python scripts/serve.py --source clip.mp4  # single camera (used when cfg.cameras is empty)

Endpoints (see src/bhairav/backend/server.py):
  POST /auth/login            {"username": "alice", "password": "..."}
  GET  /health | /api/status
  GET  /api/evidence?rule=&severity=&camera=&q=   | /api/evidence/export (analyst+)
  GET  /api/evidence/{id}/snapshot | /clip
  POST /api/evidence/{id}/status | /notes
  DELETE /api/evidence/{id}
  GET  /api/audit | /api/users (admin)
  GET  /api/alerts/recent
  WS   /ws/stream?token=<token>&camera=CAM-01   (camera optional: all cameras)

Phase 8 M2 (multi-camera): each entry in the `cameras:` config list runs its
own pipeline thread (own detector, rules engine, evidence recorder). Frames
are scoped to a WS channel per camera; alerts are global. Evidence is tagged
with the camera id, and /api/evidence?camera= filters on it.

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

from bhairav.config import CameraConfig, load_config


def _jpeg_b64(frame: np.ndarray, scale: float = 0.5) -> str:
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else ""


class CameraStatsGroup:
    """Aggregates one PipelineStats per camera into the /api/status snapshot.

    The aggregate keeps the same top-level keys as a single PipelineStats
    (frames / alerts / fps / uptime_sec) so the dashboard Status tab works
    unchanged, plus a `cameras` list with per-camera detail.
    """

    def __init__(self):
        self._items: list[tuple[str, str, object]] = []  # (camera_id, name, stats)

    def add(self, camera_id: str, name: str, stats) -> None:
        self._items.append((camera_id, name, stats))

    def snapshot(self) -> dict:
        snaps = []
        for cid, name, st in self._items:
            snap = st.snapshot()
            snap["camera"] = cid
            snap["name"] = name
            snaps.append(snap)
        return {
            "cameras": snaps,
            "frames": sum(s["frames"] for s in snaps),
            "alerts": sum(s["alerts"] for s in snaps),
            "fps": round(sum(s["fps"] for s in snaps), 1),
            "uptime_sec": round(max((s["uptime_sec"] for s in snaps),
                                    default=0.0), 1),
        }


def run_stream(cfg, cam, hub, store, evidence_dir, stop, stats,
               webhook_url, recent_alerts, plates):
    """One camera pipeline loop on a background thread.

    Each camera gets an independent detector + rules engine (track ids from
    two pipelines would otherwise collide in one shared engine) and its own
    evidence recorder stamped with `cam.id`. Live sources are re-opened with
    exponential backoff when they drop; connect/drop health is surfaced per
    camera via /api/status.
    """
    from bhairav.alert_log import AlertLog
    from bhairav.backend.evidence import EventRecorder
    from bhairav.backend.server import webhook_notify
    from bhairav.pipeline import build_engine, make_detector, run_pipeline
    from bhairav.sources import (SourceKind, SourceMonitor, classify_source,
                                 open_capture)

    source = cam.source
    kind, desc = classify_source(source)
    engine = build_engine(cfg)
    detector = make_detector(cfg, cam.detector, source)
    monitor = SourceMonitor(kind, f"{cam.name} ({desc})")
    stats.set_source(monitor)
    print(f"[{cam.id}] pipeline: {cam.name} <- {desc} (kind={kind.value})", flush=True)

    def opener():
        if kind is SourceKind.BLOB:
            raise RuntimeError("blob source has no capture to open")
        return open_capture(source, monitor=monitor, retries=3, base_delay=2.0)

    # share the plate watchlist with the REST API (one registry for all cameras)
    for rule in engine.rules:
        if getattr(rule, "name", None) == "stolen_vehicle":
            rule.registry = plates

    ev_cfg = cfg.evidence
    recorder = EventRecorder(store, pre_sec=ev_cfg.pre_sec, post_sec=ev_cfg.post_sec,
                             min_gap_sec=ev_cfg.min_gap_sec,
                             blur_faces=ev_cfg.blur_faces, camera=cam.id)
    log = AlertLog(Path(evidence_dir).parent / f"alerts_live_{cam.id}.jsonl")
    log.clear()

    def on_frame(state, alerts):
        if stop.is_set():
            return False
        if state.frame is None:
            return None
        # evidence capture + alert log (camera-stamped)
        recorder.observe(state, state.frame)
        for a in alerts:
            recorder.on_alert(a, state.frame, state=state)
            log.write(a)
            ad = a.to_dict()
            ad["camera"] = cam.id
            recent_alerts.append(ad)
            del recent_alerts[:-cfg.backend.max_recent_alerts]  # bounded rolling feed
            if a.severity.value == "red":
                webhook_notify(webhook_url, ad)
        # close events that have gone quiet (respects post_sec)
        recorder.finalize_due(state.timestamp)
        # ops telemetry
        stats.bump(frames=1, alerts=len(alerts))
        # live feed (downscaled jpeg to keep the WS light), camera-scoped
        hub.publish_frame(
            frame_id=state.frame_id, timestamp=state.timestamp,
            jpeg_b64=_jpeg_b64(state.frame),
            tracks=[{"id": t.track_id, "label": t.label, "bbox": list(t.bbox),
                     "conf": round(t.confidence, 3)} for t in state.tracks],
            poses=[{"track_id": p.track_id,
                    "kps": [[round(k.x, 4), round(k.y, 4), round(k.confidence, 3)]
                            for k in p.keypoints]} for p in state.poses],
            alerts=[a.to_dict() for a in alerts],
            camera=cam.id)
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
            print(f"[{cam.id}] reconnect failed ({exc}); retrying in {delay:.0f}s",
                  flush=True)
            stop.wait(delay)
            delay = min(delay * 2, 30.0)
            continue
        recorder.flush()
        # scene clock restarts at 0 next pass: reset cooldown/open state so
        # the second pass is not blocked by replay-1 timestamps
        recorder.reset()
        # brief pause between replays so the stream does not hot-loop
        stop.wait(0.5)
        delay = 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="BHAIRAV live server (Phases 1-8)")
    ap.add_argument("--source", default="blob",
                    help="single-camera source (used only when config cameras is empty)")
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
    # cameras: config list wins; otherwise one camera from --source
    if cfg.cameras:
        cams = cfg.cameras
    else:
        cams = [CameraConfig(id=cfg.evidence.camera, name=cfg.evidence.camera,
                             source=args.source, detector=args.detector)]
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

    db_url = args.db_url or os.environ.get("BHAIRAV_DB_URL") or cfg.backend.db

    # ---- startup security posture ---------------------------------------
    loopback = is_loopback(host)
    if secret in ("", "dev-secret-change-me") and not loopback and not os.environ.get("BHAIRAV_ALLOW_DEFAULT_SECRET"):
        raise SystemExit(
            "REFUSING TO START: default BHAIRAV secret on a non-loopback interface."
            + chr(10)
            + "  Set BHAIRAV_SECRET to a long random value (or BHAIRAV_ALLOW_DEFAULT_SECRET=1 for local dev only).")

    if db_url:
        from bhairav.backend.pg_users import PostgresUserStore
        users = PostgresUserStore(db_url)
        print(f"[db] users: PostgreSQL ({db_url.split('@')[-1]})")
    else:
        users = UserStore(cfg.backend.users_file)
    weak = [d["username"] for d in DEFAULT_USERS
            if users.get(d["username"]) and UserStore._verify_password(
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
            "(base64 32-byte key, e.g. from: python -c " + chr(34) + "import os,base64;"
            "print(base64.b64encode(os.urandom(32)).decode())" + chr(34) + ") or pass --evidence-key.")

    if db_url:
        from bhairav.backend.pg_audit import PostgresAuditLog
        from bhairav.backend.pg_store import PostgresEvidenceStore
        try:
            audit = PostgresAuditLog(db_url)
            store = PostgresEvidenceStore(
                db_url, camera=cfg.evidence.camera, fps=cfg.evidence.fps,
                blur_faces=cfg.evidence.blur_faces,
                encrypt=cfg.evidence.encrypt, key=evidence_key,
                max_events=cfg.evidence.max_events, root=evidence_dir)
        except RuntimeError as exc:
            raise SystemExit(
                "REFUSING TO START: PostgreSQL backend unavailable."
                + chr(10) + f"  {exc}")
        print(f"[db] evidence + audit: PostgreSQL ({db_url.split('@')[-1]})")
    else:
        audit = AuditLog(Path(evidence_dir) / "audit.jsonl")
        store = EvidenceStore(evidence_dir, camera=cfg.evidence.camera,
                              fps=cfg.evidence.fps,
                              blur_faces=cfg.evidence.blur_faces,
                              encrypt=cfg.evidence.encrypt, key=evidence_key,
                              max_events=cfg.evidence.max_events)
        print(f"[db] evidence store: file-based ({evidence_dir})")

    # per-camera pipeline stats (aggregated in /api/status)
    per_cam: list[tuple] = [(cam, PipelineStats()) for cam in cams]
    stats = CameraStatsGroup()
    for cam, st in per_cam:
        stats.add(cam.id, cam.name, st)
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
    if db_url:
        from bhairav.backend.pg_plates import PostgresPlateRegistry
        plates = PostgresPlateRegistry(db_url)
        print("[security] vehicle watchlist: ENABLED (ANPR watchlist in PostgreSQL)")
    else:
        from bhairav.backend.anpr import PlateRegistry
        plates = PlateRegistry(Path(evidence_dir).parent / "plates.json")
        print("[security] vehicle watchlist: ENABLED (ANPR on synthetic plates)")

    cam_registry = [{"id": c.id, "name": c.name, "source": c.source} for c in cams]
    app = create_app(store, audit, secret=secret, hub=hub, users=users,
                     stats=stats, webhook_url=webhook_url, face=face,
                     plates=plates, recent_alerts=recent_alerts,
                     cameras=cam_registry)

    # run one pipeline thread per camera
    stop = threading.Event()
    for cam, st in per_cam:
        t = threading.Thread(target=run_stream,
                             args=(cfg, cam, hub, store, evidence_dir, stop, st,
                                   webhook_url, recent_alerts, plates),
                             daemon=True)
        t.start()

    import uvicorn
    scheme = "https" if (args.tls_cert and args.tls_key) else "http"
    print(f"BHAIRAV Phase 8 server -> {scheme}://{host}:{port}  (evidence: {evidence_dir})")
    print(f"Cameras: {', '.join(f'{c.id} ({c.name})' for c in cams)}")
    print(f"Dashboard: {scheme}://{host}:{port}/dashboard/")
    if cfg.evidence.encrypt:
        print("[security] evidence encryption at rest: ENABLED (AES-256-GCM)")
    print("Login (POST /auth/login with username + password):")
    for u in users.public_view():
        pw = "admin123" if u["username"] == "admin" else f"{u['username']}123"
        print(f"  {u['username']:9s} / {pw:<12s} role={u['role']}")
    if webhook_url:
        print(f"Webhook: red alerts -> {webhook_url}")
    print(f"Live stream: {scheme.replace('https', 'wss').replace('http', 'ws')}://{host}:{port}/ws/stream?token=<token>&camera=<CAM-ID>")
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
