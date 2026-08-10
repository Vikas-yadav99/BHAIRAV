"""FastAPI services + WebSocket live stream (Phase 3-5).

REST endpoints:
    POST /auth/login                  -> capability token (username, password)
    GET  /health                      -> service status
    GET  /api/status                  -> ops: evidence counts, audit, pipeline stats
    GET  /api/evidence                -> search evidence (?rule=&severity=&q=&t0=&t1=)
    GET  /api/evidence/export         -> analyst+: zip bundle of a search
    GET  /api/evidence/{id}           -> one evidence record (metadata)
    GET  /api/evidence/{id}/snapshot  -> snapshot jpeg
    GET  /api/evidence/{id}/clip      -> mp4 clip (download perm required)
    POST /api/evidence/{id}/status    -> operator+: new|acknowledged|resolved
    POST /api/evidence/{id}/notes     -> analyst+: append investigation note
    DELETE /api/evidence/{id}         -> delete (admin)
    POST /api/evidence/expire         -> run retention policy (admin)
    GET  /api/audit                   -> audit log (analyst+)
    GET  /api/alerts/recent           -> recent alert feed
    GET  /api/users                   -> admin: list users
    POST /api/users                   -> admin: create user
    DELETE /api/users/{username}      -> admin: delete user
    POST /api/users/{username}/lock   -> admin: lock/unlock account
    POST /api/users/{username}/password -> admin: reset password
    WS   /ws/stream                   -> live frame + alert JSON stream

Auth: every /api endpoint and the WS handshake require
`Authorization: Bearer <token>`. Roles gate permissions (rbac.py).

NOTE: no `from __future__ import annotations` here - FastAPI resolves
WebSocket/Depends annotations eagerly against create_app()'s closure locals;
stringified annotations would make it treat `websocket` as a query param.
"""

import asyncio
import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from .audit import AuditLog
from .evidence import EvidenceRecord, EvidenceStore
from .hardening import RateLimiter
from .rbac import (PERM_ALERTS, PERM_AUDIT, PERM_EVIDENCE_DELETE,
                   PERM_EVIDENCE_DOWNLOAD, PERM_EVIDENCE_EXPORT,
                   PERM_EVIDENCE_READ, PERM_STREAM, PERM_USERS,
                   ROLE_PERMISSIONS, issue_token, validate_token)
from .users import UserError, UserStore

# Reject JSON bodies above this size at the API edge (login/user/note
# endpoints): a 2 MB cap stops oversized-payload resource exhaustion while
# leaving normal evidence metadata traffic untouched.
MAX_JSON_BODY_BYTES = 2 * 1024 * 1024
LOGIN_RATE_LIMIT = 10      # attempts per IP per window
LOGIN_RATE_WINDOW_SEC = 60.0

# ---------------------------------------------------------------------------
# Live stream hub (works without FastAPI; importable on minimal install)
# ---------------------------------------------------------------------------
class LiveHub:
    """Pub/sub hub bridging the sync pipeline to async WebSocket clients.

    The pipeline thread calls ``publish_frame`` / ``publish_alert`` (thread-safe);
    every connected WebSocket receives the same messages.
    """

    def __init__(self, max_clients: int = 64):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._max_clients = max_clients

    # ---- sync side (pipeline thread) --------------------------------------
    def publish_frame(self, frame_id: int, timestamp: float, jpeg_b64: str,
                      tracks: list[dict], poses: list[dict], alerts: list[dict]) -> None:
        msg = {"type": "frame", "frame_id": frame_id, "ts": timestamp,
               "jpeg": jpeg_b64, "tracks": tracks, "poses": poses, "alerts": alerts}
        self._publish(msg)

    def publish_alert(self, alert: dict) -> None:
        self._publish({"type": "alert", "alert": alert})

    def _publish(self, msg: dict) -> None:
        if self._loop is None or not self._subscribers:
            return
        asyncio.run_coroutine_threadsafe(self._fanout(msg), self._loop)

    async def _fanout(self, msg: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    # ---- async side (server) ----------------------------------------------
    async def subscribe(self) -> asyncio.Queue:
        if self._loop is None:
            # first subscriber runs inside the server's event loop
            self._loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    @property
    def client_count(self) -> int:
        return len(self._subscribers)


# ---------------------------------------------------------------------------
# Pipeline stats + webhook notifications
# ---------------------------------------------------------------------------
class PipelineStats:
    """Thread-safe counters the pipeline thread bumps; the /api/status endpoint
    reads a snapshot. No hard dependencies, so the tests stay lightweight."""

    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.frames = 0
        self.alerts = 0
        self.fps_ema = 0.0
        self._last_t = time.time()
        self._last_n = 0
        self._source = None

    def bump(self, frames: int = 1, alerts: int = 0) -> None:
        with self._lock:
            self.frames += frames
            self.alerts += alerts
            now = time.time()
            dt = now - self._last_t
            if dt >= 1.0:
                inst = (self.frames - self._last_n) / dt
                self.fps_ema = 0.7 * self.fps_ema + 0.3 * inst
                self._last_t, self._last_n = now, self.frames

    def set_source(self, monitor) -> None:
        """Attach a sources.SourceMonitor so /api/status reports feed health."""
        with self._lock:
            self._source = monitor

    def snapshot(self) -> dict:
        with self._lock:
            snap = {"uptime_sec": round(time.time() - self.started, 1),
                    "frames": self.frames, "alerts": self.alerts,
                    "fps": round(self.fps_ema, 1)}
            if self._source is not None:
                snap["source"] = self._source.snapshot()
            return snap


def webhook_notify(url: str | None, alert: dict) -> None:
    """Fire-and-forget POST of a red/urgent alert to a webhook (Slack-style).

    Runs in a daemon thread so a slow/unreachable endpoint can never stall the
    pipeline; all failures are swallowed (the alert is already in the feed and
    the audit trail, so the webhook is best-effort delivery).
    """
    if not url:
        return
    payload = json.dumps({"type": "bhairav_alert", "alert": alert},
                         separators=(",", ":")).encode("utf-8")

    def _send():
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            pass  # best-effort: never crash the pipeline over a webhook

    threading.Thread(target=_send, daemon=True).start()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(store: EvidenceStore, audit: AuditLog, secret: str,
               hub: LiveHub | None = None,
               recent_alerts: list[dict] | None = None,
               users: UserStore | None = None,
               stats: PipelineStats | None = None,
               webhook_url: str | None = None,
               login_limiter: RateLimiter | None = None,
               face: dict | None = None,
               plates: "PlateRegistry | None" = None) -> Any:
    """Build the FastAPI application. Imports fastapi lazily."""
    try:
        from fastapi import (Body, Depends, FastAPI, HTTPException, Query,
                             Request, WebSocket, WebSocketDisconnect)
        from fastapi.responses import Response
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("the API server requires fastapi; "
                           "install it with: pip install fastapi uvicorn") from exc

    hub = hub or LiveHub()
    recent = recent_alerts if recent_alerts is not None else []
    bearer = HTTPBearer(auto_error=False)
    users = users or UserStore(Path(store.root).parent / "users.json")
    stats = stats or PipelineStats()
    login_limiter = login_limiter or RateLimiter(LOGIN_RATE_LIMIT, LOGIN_RATE_WINDOW_SEC)

    def _reject_large_body(request) -> None:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_JSON_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")

    app = FastAPI(title="BHAIRAV - Evidence & Live API", version="5.0.0")

    # Phase 4: serve the React dashboard (dashboard/index.html) from the repo.
    # os.path.dirname(__file__) is the backend/ dir, so parents[2] is the repo root.
    import os
    dashboard_dir = Path(os.path.dirname(__file__)).resolve().parents[2] / "dashboard"
    if dashboard_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

        @app.get("/")
        def root():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/dashboard/")
    else:
        @app.get("/")
        def root():
            return {"status": "ok", "hint": "dashboard/ not found; API only"}

    # ---- auth dependency --------------------------------------------------
    def _user_active(sub: str) -> bool:
        """Token revocation check: a locked or deleted account's outstanding
        tokens must stop working immediately, not at 12h expiry."""
        u = users.get(sub)
        return u is not None and not u.get("locked")

    def current_claims(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
        if creds is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        # some FastAPI versions return a masked SecretStr; unwrap to the real value
        raw = creds.credentials
        token = getattr(raw, "get_secret_value", lambda: raw)()
        claims = validate_token(secret, token) if token else None
        if claims is None or not _user_active(claims.get("sub", "")):
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return claims

    def require(permission: str):
        def dep(claims: dict = Depends(current_claims)) -> dict:
            if permission not in ROLE_PERMISSIONS.get(claims["role"], frozenset()):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return claims
        return dep

    # ---- auth -------------------------------------------------------------
    @app.post("/auth/login")
    def login(request: Request, payload: dict = Body(...)):
        _reject_large_body(request)
        # Per-IP throttle on top of the per-account lockout (users.py): stops
        # credential stuffing / distributed brute force at the edge. Behind a
        # reverse proxy this is the proxy's IP; scale the window if needed.
        client = request.client.host if request.client else "unknown"
        if not login_limiter.allow(f"login:{client}"):
            audit.append(client, "login_rate_limited", "too many attempts")
            raise HTTPException(status_code=429,
                                detail="too many login attempts, slow down")
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        user = users.authenticate(username, password)
        if user is None:
            audit.append(username or "?", "login_failed", "bad credentials")
            raise HTTPException(status_code=401, detail="invalid username or password")
        token = issue_token(secret, user["username"], user["role"])
        audit.append(user["username"], "login", f"role={user['role']}")
        return {"token": token, "role": user["role"], "username": user["username"]}

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "bhairav-phase5",
                "time": round(time.time(), 3), "clients": hub.client_count}

    # ---- ops status -------------------------------------------------------
    @app.get("/api/status")
    def api_status(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        counts = store.counts()
        audit_ok, problems = audit.verify()
        return {
            "service": "bhairav", "version": "5.0.0",
            "time": round(time.time(), 3),
            "pipeline": stats.snapshot(),
            "clients": hub.client_count,
            "evidence": counts,
            "audit": {"ok": audit_ok, "problems": problems,
                       "entries": len(audit.read())},
            "users": users.count(),
            "webhook": bool(webhook_url),
        }

    # ---- users (admin) ----------------------------------------------------
    @app.get("/api/users")
    def list_users(claims: dict = Depends(require(PERM_USERS))):
        return {"users": users.public_view()}

    @app.post("/api/users")
    def create_user(payload: dict = Body(...),
                    claims: dict = Depends(require(PERM_USERS))):
        username, password, role = (payload.get("username"), payload.get("password"),
                                    payload.get("role"))
        if not all(isinstance(v, str) for v in (username, password, role)):
            raise HTTPException(status_code=400, detail="username/password/role must be strings")
        try:
            rec = users.create(username, password, role)
        except UserError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        audit.append(claims["sub"], "create_user", rec["username"])
        return rec

    @app.delete("/api/users/{username}")
    def delete_user(username: str, claims: dict = Depends(require(PERM_USERS))):
        if username == claims["sub"]:
            raise HTTPException(status_code=400, detail="cannot delete your own account")
        if not users.delete(username):
            raise HTTPException(status_code=404, detail="user not found")
        audit.append(claims["sub"], "delete_user", username)
        return {"deleted": username}

    @app.post("/api/users/{username}/lock")
    def lock_user(username: str, payload: dict = Body(...),
                  claims: dict = Depends(require(PERM_USERS))):
        if username == claims["sub"]:
            raise HTTPException(status_code=400, detail="cannot lock your own account")
        if not users.set_locked(username, bool(payload.get("locked", True))):
            raise HTTPException(status_code=404, detail="user not found")
        audit.append(claims["sub"], "lock_user" if payload.get("locked") else "unlock_user", username)
        return {"username": username, "locked": bool(payload.get("locked", True))}

    @app.post("/api/users/{username}/password")
    def reset_password(username: str, payload: dict = Body(...),
                       claims: dict = Depends(require(PERM_USERS))):
        try:
            ok = users.change_password(username, str(payload.get("password", "")))
        except UserError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not ok:
            raise HTTPException(status_code=404, detail="user not found")
        audit.append(claims["sub"], "reset_password", username)
        return {"updated": username}

    # ---- evidence ---------------------------------------------------------
    @app.get("/api/evidence")
    def search_evidence(claims: dict = Depends(require(PERM_EVIDENCE_READ)),
                        rule: str | None = None, severity: str | None = None,
                        q: str | None = None, t0: float | None = None,
                        t1: float | None = None, limit: int = Query(50, le=200)):
        rows = store.search(rule=rule, severity=severity, q=q, t0=t0, t1=t1, limit=limit)
        return {"total": len(rows), "events": [r.to_dict() for r in rows]}

    @app.get("/api/evidence/export")
    def export_evidence(claims: dict = Depends(require(PERM_EVIDENCE_EXPORT)),
                        rule: str | None = None, severity: str | None = None,
                        q: str | None = None, t0: float | None = None,
                        t1: float | None = None, limit: int = Query(200, le=500)):
        rows = store.search(rule=rule, severity=severity, q=q, t0=t0, t1=t1, limit=limit)
        zip_bytes = store.export_zip(rows)
        audit.append(claims["sub"], "export_evidence", f"count={len(rows)}")
        return Response(content=zip_bytes, media_type="application/zip",
                        headers={"Content-Disposition":
                                 'attachment; filename="bhairav_export.zip"'})

    @app.get("/api/evidence/{event_id}")
    def get_evidence(event_id: str, claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        rec = store.get(event_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        return rec.to_dict()

    @app.post("/api/evidence/{event_id}/status")
    def set_evidence_status(event_id: str, payload: dict = Body(...),
                            claims: dict = Depends(require(PERM_EVIDENCE_DOWNLOAD))):
        status = str(payload.get("status", "")).strip()
        if status == "resolved" and claims["role"] not in ("analyst", "admin"):
            raise HTTPException(status_code=403, detail="resolving requires analyst+")
        try:
            rec = store.update_status(event_id, status, claims["sub"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if rec is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        audit.append(claims["sub"], "set_evidence_status", f"{event_id}->{status}")
        return rec.to_dict()

    @app.post("/api/evidence/{event_id}/notes")
    def add_evidence_note(event_id: str, payload: dict = Body(...),
                          claims: dict = Depends(require(PERM_AUDIT))):
        rec = store.add_note(event_id, str(payload.get("text", "")), claims["sub"])
        if rec is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        audit.append(claims["sub"], "add_evidence_note", event_id)
        return rec.to_dict()

    @app.get("/api/evidence/{event_id}/snapshot")
    def evidence_snapshot(event_id: str,
                          claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        data = store.snapshot_bytes(event_id)
        if data is None:
            raise HTTPException(status_code=404, detail="snapshot not found")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/evidence/{event_id}/clip")
    def evidence_clip(event_id: str,
                      claims: dict = Depends(require(PERM_EVIDENCE_DOWNLOAD))):
        data = store.clip_bytes(event_id)
        if data is None:
            raise HTTPException(status_code=404, detail="clip not found")
        audit.append(claims["sub"], "download_evidence", event_id)
        return Response(content=data, media_type="video/mp4",
                        headers={"Content-Disposition": f'attachment; filename="{event_id}.mp4"'})

    @app.delete("/api/evidence/{event_id}")
    def delete_evidence(event_id: str,
                        claims: dict = Depends(require(PERM_EVIDENCE_DELETE))):
        if not store.delete(event_id):
            raise HTTPException(status_code=404, detail="evidence not found")
        audit.append(claims["sub"], "delete_evidence", event_id)
        return {"deleted": event_id}

    @app.post("/api/evidence/expire")
    def expire_evidence(max_age_days: dict = Body(...),
                        claims: dict = Depends(require(PERM_EVIDENCE_DELETE))):
        days = float(max_age_days.get("max_age_days", 30))
        removed = store.expire(days)
        audit.append(claims["sub"], "expire_evidence", f"days={days} removed={removed}")
        return {"expired": removed}

    # ---- audit ------------------------------------------------------------
    @app.get("/api/audit")
    def read_audit(claims: dict = Depends(require(PERM_AUDIT)),
                   actor: str | None = None, action: str | None = None,
                   limit: int = Query(100, le=500)):
        rows = audit.query(actor=actor, action=action, limit=limit)
        ok, problems = audit.verify()
        return {"ok": ok, "problems": problems, "entries": rows}

    # ---- vehicle watchlist (Phase 6: stolen-vehicle ANPR) -----------------
    def _plates_or_503():
        if plates is None:
            raise HTTPException(status_code=503,
                                detail="vehicle watchlist unavailable")
        return plates

    @app.get("/api/vehicles/watch")
    def vehicle_watch_list(claims: dict = Depends(require(PERM_EVIDENCE_EXPORT))):
        return {"watch": _plates_or_503().list_watch()}

    @app.post("/api/vehicles/watch")
    def vehicle_watch_add(request: Request, payload: dict = Body(...),
                          claims: dict = Depends(require(PERM_EVIDENCE_DOWNLOAD))):
        _reject_large_body(request)
        try:
            rec = _plates_or_503().watch(str(payload.get("plate", "")),
                                         reason=str(payload.get("reason", "")),
                                         actor=claims["sub"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        audit.append(claims["sub"], "watch_plate", rec["plate"])
        return rec

    @app.delete("/api/vehicles/watch/{plate}")
    def vehicle_watch_remove(plate: str,
                             claims: dict = Depends(require(PERM_EVIDENCE_DOWNLOAD))):
        if not _plates_or_503().unwatch(plate):
            raise HTTPException(status_code=404, detail="plate not on watchlist")
        audit.append(claims["sub"], "unwatch_plate", plate.upper())
        return {"removed": plate.upper()}

    @app.get("/api/vehicles/reads")
    def vehicle_reads(claims: dict = Depends(require(PERM_EVIDENCE_EXPORT)),
                      limit: int = Query(50, le=200)):
        return {"reads": _plates_or_503().recent_reads(limit)}

    # ---- face search (Phase 6: find a person in evidence by photo) --------
    def _face_or_503():
        if face is None:
            raise HTTPException(
                status_code=503,
                detail="face search unavailable: models not installed. "
                       "Run: python scripts/fetch_models.py")
        return face

    def _decode_image_b64(image_b64: str):
        import base64 as _b64
        try:
            raw = _b64.b64decode(image_b64 or "", validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="image_b64 must be valid base64")
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="could not decode image")
        return img

    @app.get("/api/search/status")
    def search_status(claims: dict = Depends(require(PERM_EVIDENCE_EXPORT))):
        svc = _face_or_503()
        return {"enabled": True, "gallery_subjects": svc["gallery"].count(),
                "index": svc["index"].stats()}

    @app.post("/api/search/register")
    def search_register(request: Request, payload: dict = Body(...),
                        claims: dict = Depends(require(PERM_USERS))):
        _reject_large_body(request)
        svc = _face_or_503()
        name = str(payload.get("name", "")).strip()
        img = _decode_image_b64(str(payload.get("image_b64", "")))
        emb = svc["recognizer"].embed(img)
        if emb is None:
            raise HTTPException(status_code=400, detail="no face detected in the photo")
        rec = svc["gallery"].add(name, emb, notes=str(payload.get("notes", "")))
        audit.append(claims["sub"], "register_subject", name)
        return rec

    @app.get("/api/search/subjects")
    def search_subjects(claims: dict = Depends(require(PERM_EVIDENCE_EXPORT))):
        svc = _face_or_503()
        return {"subjects": svc["gallery"].list()}

    @app.delete("/api/search/subjects/{name}")
    def search_subject_delete(name: str,
                              claims: dict = Depends(require(PERM_USERS))):
        svc = _face_or_503()
        if not svc["gallery"].remove(name):
            raise HTTPException(status_code=404, detail="subject not found")
        audit.append(claims["sub"], "remove_subject", name)
        return {"removed": name}

    @app.post("/api/search/query")
    def search_query(request: Request, payload: dict = Body(...),
                     claims: dict = Depends(require(PERM_EVIDENCE_EXPORT))):
        _reject_large_body(request)
        svc = _face_or_503()
        img = _decode_image_b64(str(payload.get("image_b64", "")))
        emb = svc["recognizer"].embed(img)
        if emb is None:
            raise HTTPException(status_code=400, detail="no face detected in the photo")
        top_k = int(payload.get("top_k", 5))
        threshold = float(payload.get("threshold", 0.55))
        audit.append(claims["sub"], "face_search", f"top_k={top_k} threshold={threshold}")
        return {"subjects": svc["gallery"].search(emb, top_k, threshold),
                "evidence": svc["index"].search(emb, top_k, threshold)}

    @app.post("/api/search/index")
    def search_index(claims: dict = Depends(require(PERM_USERS))):
        svc = _face_or_503()
        stats = svc["index"].index()
        audit.append(claims["sub"], "index_evidence_faces",
                     f"indexed={stats['indexed_events']}")
        return stats

    # ---- alert feed -------------------------------------------------------
    @app.get("/api/alerts/recent")
    def recent_alerts(claims: dict = Depends(require(PERM_ALERTS)),
                      limit: int = Query(50, le=200)):
        return {"alerts": recent[-limit:]}

    # ---- live stream ------------------------------------------------------
    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket):
        # token via ?token= query param (browsers can't set WS headers); the
        # param holds the bare token, so skip the "Bearer " prefix handling
        token = (websocket.query_params.get("token") or "").strip()
        claims = validate_token(secret, token) if token else None
        if (claims is None or not _user_active(claims.get("sub", ""))
                or PERM_STREAM not in ROLE_PERMISSIONS.get(claims["role"], frozenset())):
            await websocket.close(code=4401, reason="unauthorized")
            return
        await websocket.accept()
        q = await hub.subscribe()
        audit.append(claims["sub"], "ws_connect", "live-stream")
        try:
            while True:
                # drain queued messages; heartbeat if idle >5s
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=5.0)
                    await websocket.send_text(json.dumps(msg))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "ping"}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(q)
            audit.append(claims["sub"], "ws_disconnect", "live-stream")

    return app
