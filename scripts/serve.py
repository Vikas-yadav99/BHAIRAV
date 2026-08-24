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
from bhairav.events import EventBus, Event, publish_alert, publish_frame
from bhairav.subscribers import wire_subscribers


def _jpeg_b64(frame: np.ndarray, scale: float = 0.5) -> str:
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else ""


def _public_sanitize(frame: np.ndarray, tracks: list, scale: float = 0.35) -> str:
    """Privacy-stripped JPEG for the Phase 9 M5 public monitor.

    Downscales the frame and heavily blurs every person's head region, so
    identities are unreadable while activity remains visible. Returns b64
    (empty string if encoding fails). No tracks/poses/alerts are included -
    the /api/public/stream endpoint forwards only this payload.
    """
    img = cv2.resize(frame, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA)
    short = min(img.shape[1], img.shape[0])
    k = max(21, (int(0.22 * short) | 1))  # strong odd blur kernel
    for t in tracks:
        if getattr(t, "label", "") != "person":
            continue
        x1, y1, x2, y2 = (int(v * scale) for v in t.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        hy2 = y1 + max(1, int((y2 - y1) * 0.25))  # head zone = top 25%
        head = img[y1:hy2, x1:x2]
        img[y1:hy2, x1:x2] = cv2.GaussianBlur(head, (k, k), 0)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
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


def _point_in_zone(point: tuple[float, float], zone_points: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test for normalised (0..1) coordinates."""
    x, y = point
    n = len(zone_points)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = zone_points[i]
        xj, yj = zone_points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def run_stream(cfg, cam, hub, store, evidence_dir, stop, stats,               webhook_url, recent_alerts, plates, reid=None,               public_token: str | None = None, notifier=None, mic: bool = False,               analytics_engine=None, event_bus=None):
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
    from bhairav.audio import AudioAnalyzer, AudioFusionProcessor, SyntheticAudioTrack, MicSource
    from bhairav.sources import (SourceKind, SourceMonitor, classify_source,
                                 open_capture)

    source = cam.source
    kind, desc = classify_source(source)
    engine = build_engine(cfg)
    detector = make_detector(cfg, cam.detector, source)
    monitor = SourceMonitor(kind, f"{cam.name} ({desc})")
    stats.set_source(monitor)
    print(f"[{cam.id}] pipeline: {cam.name} <- {desc} (kind={kind.value})", flush=True)

    # Phase 11: audio analytics (synthetic track or live microphone)
    audio_cfg = getattr(cfg, 'audio', None)
    audio_fusion = None
    mic_source = None
    if audio_cfg and getattr(audio_cfg, 'enabled', True):
        _sr = getattr(audio_cfg, 'sample_rate', 16000)
        _sens = getattr(audio_cfg, 'sensitivity', 1.0)
        _cd = getattr(audio_cfg, 'cooldown_sec', 15.0)
        _smd = getattr(audio_cfg, 'scream_min_dur_sec', 0.4)
        _analyzer = AudioAnalyzer(frame_rate=_sr, sensitivity=_sens,
                                  cooldown_sec=_cd, scream_min_dur_sec=_smd)
        audio_fusion = AudioFusionProcessor(analyzer=_analyzer, sample_rate=_sr)
        if mic:
            # Live microphone: sounddevice feeds the analyzer directly
            mic_source = MicSource(analyzer=_analyzer, sample_rate=_sr)
            mic_source.start()
            print(f"[{cam.id}] audio analytics: LIVE MICROPHONE (sample_rate={_sr})", flush=True)
        else:
            # Synthetic track (deterministic demo)
            _synth = SyntheticAudioTrack(sample_rate=_sr)
            _audio_track = _synth.generate(duration_sec=cfg.synthetic.duration_sec)
            audio_fusion.load_track(_audio_track)
            print(f"[{cam.id}] audio analytics: ENABLED (sample_rate={_sr})", flush=True)

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
            # Phase 10 M4: field-officer dispatch (channels filter internally).
            # An alert that a channel accepted also goes to the /ws/field
            # push feed (police+), so officers see exactly what is dispatched.
            # Publish to event bus for subscribers (escalation, PTZ, federation, audit)
            if event_bus is not None:
                publish_alert(event_bus, ad, camera=cam.id)
            try:
                if notifier is not None and notifier.notify(ad):
                    hub.publish_field_alert(ad)
            except Exception as exc:
                print(f"[{cam.id}] dispatch error: {exc}", flush=True)
        # Phase 11: feed audio up to current timestamp, merge audio alerts
        if audio_fusion is not None or mic_source is not None:
            try:
                if mic_source is not None:
                    # Live mic: drain events buffered by the mic callback
                    from bhairav.audio.fusion import audio_events_to_alerts
                    audio_alerts = audio_events_to_alerts(
                        mic_source.drain_events(), frame_id=state.frame_id)
                else:
                    audio_alerts = audio_fusion.process_video_frame(
                        state.frame_id, state.timestamp)
                for aa in audio_alerts:
                    recorder.on_alert(aa, state.frame, state=state)
                    log.write(aa)
                    ad_a = aa.to_dict()
                    ad_a["camera"] = cam.id
                    recent_alerts.append(ad_a)
                    del recent_alerts[:-cfg.backend.max_recent_alerts]
                    if aa.severity.value == "red":
                        webhook_notify(webhook_url, ad_a)
                    try:
                        if notifier is not None and notifier.notify(ad_a):
                            hub.publish_field_alert(ad_a)
                    except Exception as exc:
                        print(f"[{cam.id}] audio dispatch error: {exc}", flush=True)
            except Exception as exc:
                print(f"[{cam.id}] audio analytics error: {exc}", flush=True)
            # Phase 11: push audio level to the dashboard volume meter
            try:
                if mic_source is not None:
                    hub.publish_audio_level(mic_source.level)
                else:
                    hub.publish_audio_level(audio_fusion.analyzer.level)
            except Exception:
                pass
        # close events that have gone quiet (respects post_sec)
                # Phase 12: feed analytics engine
        if analytics_engine is not None:
            try:
                # compute per-zone person counts from rules engine
                _zone_counts = {}
                for z in cfg.zones:
                    _zone_counts[z.name] = sum(
                        1 for t in state.tracks if t.is_person
                        and _point_in_zone(t.centroid, z.points_norm))
                # Phase 18: feed hotspot + summarizer
                for _af_a in alerts:
                    _hotspot.observe(
                        state.timestamp,
                        zone=getattr(_af_a, 'zone', None),
                        severity=_af_a.severity.value if hasattr(_af_a.severity, 'value') else str(_af_a.severity),
                        rule=_af_a.rule,
                    )
                    _summarizer.observe(_af_a.to_dict() if hasattr(_af_a, 'to_dict') else {"rule": _af_a.rule, "severity": _af_a.severity.value if hasattr(_af_a.severity, 'value') else str(_af_a.severity), "zone": getattr(_af_a, 'zone', None), "camera": cam.id, "timestamp": frame_ts})
                # Phase 18: generate recommendations every 30 frames
                if analytics_engine._frame_count % 30 == 0:
                    hs_snap = _hotspot.snapshot()
                    _allocator.analyze(
                        hs_snap.get("hotspots", []),
                        _hotspot.zone_alerts if hasattr(_hotspot, 'zone_alerts') else {},
                    )
                analytics_engine.observe_frame(
                    timestamp=state.timestamp,
                    person_count=sum(1 for t in state.tracks if t.is_person),
                    tracks=state.tracks,
                    alerts=alerts + audio_alerts if 'audio_alerts' in dir() else alerts,
                    zone_counts=_zone_counts or None,
                    camera=cam.id)
                # push analytics snapshot every 30 frames (~1s)
                if state.frame_id % 30 == 0:
                    analytics_engine.update_heatmap()
                    _snap = analytics_engine.snapshot()
                    _snap["summarizer"] = _summarizer.snapshot()
                    _snap["hotspot"] = _hotspot.snapshot()
                    _snap["resource"] = _allocator.snapshot()
                    hub.publish_analytics(_snap)
            except Exception as exc:
                print(f"[{cam.id}] analytics error: {exc}", flush=True)
        # person re-id: fold this frame's person tracks into the shared gallery
        try:
            if reid is not None:
                reid.observe(state.frame, state, cam.id)
        except Exception as exc:
            print(f"[{cam.id}] reid error: {exc}", flush=True)
        recorder.finalize_due(state.timestamp)
        # ops telemetry
        stats.bump(frames=1, alerts=len(alerts))
        # live feed (downscaled jpeg to keep the WS light), camera-scoped
        # Publish frame to event bus (for PTZ tracking, analytics)
        if event_bus is not None:
            publish_frame(event_bus, state.frame_id, state.timestamp,
                         [{"id": t.track_id, "label": t.label, "bbox": list(t.bbox)} for t in state.tracks],
                         camera=cam.id)
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
        # public monitor (Phase 9 M5): sanitized, head-blurred copy,
        # delivered only to the "__public__" channel (never the main UI)
        if public_token:
            hub.publish_public_frame(
                frame_id=state.frame_id, timestamp=state.timestamp,
                jpeg_b64=_public_sanitize(state.frame, state.tracks))
        return None

    # loop the source until the server shuts down ("live" feel)
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
    ap.add_argument("--mic", action="store_true",
                    help="capture live audio from microphone instead of synthetic track (Phase 11)")
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
    public_token = (os.environ.get("BHAIRAV_PUBLIC_TOKEN")
                    or cfg.backend.public_token or "")

    # assemble the app
    from bhairav.backend.audit import AuditLog
    from bhairav.backend.evidence import EvidenceStore
    from bhairav.backend.hardening import is_loopback, load_evidence_key
    from bhairav.backend.server import LiveHub, PipelineStats, create_app
    from bhairav.backend.users import DEFAULT_USERS, UserStore

    # compose sets DATABASE_URL (12-factor convention); BHAIRAV_DB_URL and
    # config backend.db are the other accepted ways to pick the store
    db_url = (args.db_url or os.environ.get("BHAIRAV_DB_URL")
              or os.environ.get("DATABASE_URL") or cfg.backend.db)

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
    # Phase 12: predictive analytics engine
    from bhairav.analytics import AnalyticsEngine
    from bhairav.analytics import NLAlertSummarizer, PredictiveHotspot, ResourceAllocator
    analytics_cfg = getattr(cfg, 'analytics', None)
    analytics_engine = None
    if analytics_cfg and getattr(analytics_cfg, 'enabled', True):
        _ag = analytics_cfg
        analytics_engine = AnalyticsEngine(
            forecast_horizon_sec=getattr(_ag, 'forecast_horizon_sec', 10.0),
            heatmap_grid=(getattr(_ag, 'heatmap_grid_w', 32),
                          getattr(_ag, 'heatmap_grid_h', 24)),
            heatmap_decay_sec=getattr(_ag, 'heatmap_decay_sec', 30.0),
            trend_window_sec=getattr(_ag, 'trend_window_sec', 900.0),
        )
        print(f"[analytics] predictive analytics: ENABLED (forecast={getattr(_ag, 'forecast_horizon_sec', 10.0)}s ahead, heatmap={getattr(_ag, 'heatmap_grid_w', 32)}x{getattr(_ag, 'heatmap_grid_h', 24)})")
    # Phase 18: NL summaries + predictive hotspot + resource allocation
    _summarizer = NLAlertSummarizer(window_sec=getattr(analytics_cfg, 'summarizer_window_sec', 300.0) if analytics_cfg else 300.0)
    _hotspot = PredictiveHotspot(
        window_sec=getattr(analytics_cfg, 'hotspot_window_sec', 3600.0) if analytics_cfg else 3600.0,
        decay_sec=getattr(analytics_cfg, 'hotspot_decay_sec', 600.0) if analytics_cfg else 600.0,
        min_alerts=getattr(analytics_cfg, 'hotspot_min_alerts', 2) if analytics_cfg else 2,
    )
    _allocator = ResourceAllocator(
        officer_pool=getattr(analytics_cfg, 'officer_pool', 10) if analytics_cfg else 10,
        cameras=[c.id for c in cams],
        recommendation_ttl=getattr(analytics_cfg, 'recommendation_ttl', 600.0) if analytics_cfg else 600.0,
    )
    print(f"[analytics] Phase 18: NL summaries + hotspot + resource allocation: ENABLED")
    from bhairav.backend.notify import channels_from_config
    notifier = channels_from_config(webhook_url, cfg.backend.alert_channels)
    if notifier:
        print(f"[dispatch] field-officer alert channels: "
              f"{', '.join(ch['name'] for ch in notifier.stats())}")
    hub = LiveHub()
    event_bus = EventBus()
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

    # ---- person re-identification across cameras (Phase 9 M4) --------------
    from bhairav.reid import ReidService, ReidStore
    reid_store = ReidStore(Path(evidence_dir).parent / "reid")
    if db_url:
        from bhairav.backend.pg_reid import PostgresReidStore
        reid_store = PostgresReidStore(db_url)
        print("[reid] gallery: PostgreSQL (reid_subjects / reid_sightings)")
    else:
        print("[reid] gallery: files (reid/gallery.json + sightings.jsonl)")
    # Phase 14: deep ONNX embeddings when a model is configured
    from bhairav.reid import DeepAppearanceExtractor
    reid_extractor = DeepAppearanceExtractor(
        model_path=cfg.reid.deep_model,
        input_size=tuple(cfg.reid.deep_size) if cfg.reid.deep_size else None,
    )
    if reid_extractor.is_deep:
        print("[reid] deep ONNX embedding model loaded")
    else:
        print("[reid] using HSV+HOG appearance (no deep model configured)")
    reid_svc = ReidService(reid_store, extractor=reid_extractor,
                           assign_threshold=cfg.reid.assign_threshold,
                           sighting_gap_sec=cfg.reid.sighting_gap_sec)

    cam_registry = [{"id": c.id, "name": c.name, "source": c.source} for c in cams]

    stop = threading.Event()  # shared by the metrics sampler + pipeline threads

    # ---- Phase 9 M3: metrics + backups + readiness ------------------------
    from bhairav.backend.metrics import MetricsRegistry
    metrics = MetricsRegistry()
    for hname in ("bhairav_frames", "bhairav_fps", "bhairav_alerts",
                  "bhairav_clients"):
        metrics.history(hname, maxlen=600)

    backup_mgr = None
    db_metrics_cache: dict = {"reachable": False}

    def _ready_file_store() -> bool:
        return Path(evidence_dir).is_dir()

    def _ready_pg_store() -> bool:
        return bool(db_metrics_cache.get("reachable"))

    ready_check = _ready_file_store
    if db_url:
        from bhairav.backend.backups import BackupService, pg_metrics
        backup_mgr = BackupService(
            db_url, Path(evidence_dir).parent / "backups", retention=14)
        ready_check = _ready_pg_store
        print(f"[ops] backups -> {backup_mgr.out_dir} (retention "
              f"{backup_mgr.retention})")

    def _db_metrics() -> dict:
        return dict(db_metrics_cache)

    def _sample_metrics() -> None:
        """Background sampler: gauges + histories for /metrics and charts."""
        snap = stats.snapshot()
        frames = 0
        alerts_n = 0
        fps = 0.0
        for cam in snap.get("cameras", []):
            metrics.set("bhairav_frames", cam["frames"], {"camera": cam["camera"]})
            metrics.set("bhairav_fps", cam["fps"], {"camera": cam["camera"]})
            metrics.set("bhairav_alerts", cam["alerts"], {"camera": cam["camera"]})
            frames += cam["frames"]
            alerts_n += cam["alerts"]
            fps += cam["fps"]
        metrics.set("bhairav_frames", frames)
        metrics.set("bhairav_fps", fps)
        metrics.set("bhairav_alerts", alerts_n)
        metrics.set("bhairav_clients", hub.client_count)
        metrics.set("bhairav_uptime_seconds", snap.get("uptime_sec", 0.0))
        counts = store.counts()
        for sev, n in (counts.get("by_severity") or {}).items():
            metrics.set("bhairav_evidence_total", n, {"severity": sev})
        ok, problems = audit.verify()
        metrics.set("bhairav_audit_ok", 1.0 if ok else 0.0,
                    {"problems": len(problems)})
        if db_url:
            nonlocal_db = db_metrics_cache
            try:
                nonlocal_db.clear()
                nonlocal_db.update(pg_metrics(db_url))
            except Exception as exc:  # DB down -> mark unreachable
                nonlocal_db.clear()
                nonlocal_db.update({"reachable": False, "error": str(exc)})
            metrics.set("bhairav_db_size_bytes",
                        db_metrics_cache.get("db_size_bytes", 0.0))
        if backup_mgr is not None:
            latest = backup_mgr.latest()
            metrics.set("bhairav_backup_age_seconds",
                        latest["age_sec"] if latest else -1.0)
            metrics.set("bhairav_backups_total", len(backup_mgr.list()))

    def _sampler_loop() -> None:
        while not stop.is_set():
            try:
                _sample_metrics()
            except Exception as exc:  # never kill the process over telemetry
                print(f"[ops] metrics sampler error: {exc}", flush=True)
            stop.wait(5.0)

    threading.Thread(target=_sampler_loop, daemon=True).start()

    # Wire event bus subscribers
    from bhairav.response.escalation import EscalationEngine
    from bhairav.response.integrations import IntegrationHub
    from bhairav.response.ptz import PTZTracker
    from bhairav.federation import FederationClient
    from bhairav.backend.security import SecurityAuditLog

    _escalation = EscalationEngine(getattr(cfg.response, 'escalation_rules', []) if hasattr(cfg, 'response') else [])
    _integration_hub = IntegrationHub()
    _ptz_tracker = PTZTracker()
    _federation = FederationClient(
        site_id=getattr(cfg.federation, 'site_id', 'site-1') if hasattr(cfg, 'federation') else 'site-1',
        peers=getattr(cfg.federation, 'peers', []) if hasattr(cfg, 'federation') else [],
        secret=getattr(cfg.federation, 'secret', '') if hasattr(cfg, 'federation') else '',
    ) if hasattr(cfg, 'federation') and getattr(cfg.federation, 'peers', []) else None
    _sec_audit = SecurityAuditLog()

    wire_subscribers(
        event_bus,
        escalation_engine=_escalation,
        ptz_tracker=_ptz_tracker,
        integration_hub=_integration_hub,
        federation_client=_federation,
        audit_log=_sec_audit,
        live_hub=hub,
    )

    app = create_app(store, audit, secret=secret, hub=hub, users=users,
                     stats=stats, webhook_url=webhook_url, face=face,
                     plates=plates, recent_alerts=recent_alerts,
                     cameras=cam_registry,
                     assistant_ctx={"zones": [z.name for z in cfg.zones],
                                    "rules": list(cfg.rules.keys())},
                     metrics=metrics, backup_mgr=backup_mgr,
                     ready_check=ready_check, db_metrics_provider=_db_metrics,
                     metrics_token=os.environ.get("BHAIRAV_METRICS_TOKEN"),
                     reid=reid_svc,
                     public_token=public_token or None,
                     notifier=notifier or None)

    # run one pipeline thread per camera
    for cam, st in per_cam:
        t = threading.Thread(target=run_stream,
                             args=(cfg, cam, hub, store, evidence_dir, stop, st,
                                   webhook_url, recent_alerts, plates,
                                   reid_svc, public_token, notifier or None,
                                   getattr(args, "mic", False), analytics_engine,
                                   event_bus),
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
    if os.environ.get("BHAIRAV_METRICS_TOKEN"):
        print("[ops] /metrics scrape token: ENABLED (BHAIRAV_METRICS_TOKEN)")
    print(f"Live stream: {scheme.replace('https', 'wss').replace('http', 'ws')}://{host}:{port}/ws/stream?token=<token>&camera=<CAM-ID>")
    print(f"Field dispatch: {scheme.replace('https', 'wss').replace('http', 'ws')}://{host}:{port}/ws/field?token=<token> (police+)")
    if public_token:
        print(f"[public] read-only blurred monitor: {scheme}://{host}:{port}/?public={public_token}")
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
