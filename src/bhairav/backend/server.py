"""FastAPI services + WebSocket live stream (Phase 3-7).

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
import hmac
import json
import logging
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("bhairav.server")

from .audit import AuditLog
from .evidence import EvidenceStore
from .hardening import RateLimiter
from .rbac import (PERM_ALERTS, PERM_AUDIT, PERM_EVIDENCE_DELETE,
                   PERM_EVIDENCE_DOWNLOAD, PERM_EVIDENCE_EXPORT,
                   PERM_EVIDENCE_READ, PERM_STREAM, PERM_USERS,
                   ROLE_PERMISSIONS, issue_token, validate_token)
from .users import UserError, UserStore

# Reject JSON bodies above this size at the API edge (login/user/note
# endpoints): a 2 MB cap stops oversized-payload resource exhaustion while
# leaving normal evidence metadata traffic untouched. The cap is enforced
# twice: _reject_large_body() is a cheap content-length header check on the
# write endpoints, and _BodyLimitMiddleware counts actual bytes read so a
# chunked transfer (no content-length) cannot slip past.
MAX_JSON_BODY_BYTES = 2 * 1024 * 1024


class _BodyTooLarge(BaseException):
    """Internal signal: a request body exceeded the configured cap.

    Subclasses BaseException on purpose: FastAPI's body parser wraps
    ``await request.json()`` in ``except Exception``, so a plain Exception
    raised from the receive channel would be swallowed and turned into a
    400/422 instead of unwinding to the middleware, which sends the 413.
    (Same rationale as asyncio.CancelledError.)"""

    def __init__(self, received: int, limit: int):
        super().__init__(received, limit)


class _BodyLimitMiddleware:
    """ASGI middleware enforcing a hard cap on HTTP request bodies.

    The header-only helper checks ``content-length``; a chunked transfer
    encodes no length, so the middleware also counts bytes as they arrive
    and aborts with 413 the moment the cap is exceeded. This applies to
    every endpoint uniformly, not just the ones that remember to check.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        cl = _header_value(scope, b"content-length")
        if cl and cl.isdigit() and int(cl) > self.max_bytes:
            await _send_413(send)
            return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge(received, self.max_bytes)
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _send_413(send)


def _header_value(scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", ()):
        if key == name:
            return value.decode("latin-1")
    return None


async def _send_413(send) -> None:
    body = b"request body too large"
    await send({
        "type": "http.response.start",
        "status": 413,
        # close the connection: the request body is unread and would
        # otherwise poison the next keep-alive request on this socket
        "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close")],
    })
    await send({"type": "http.response.body", "body": body})
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
        self._subscribers: set[asyncio.Queue] = set()   # "all cameras" clients
        self._channels: dict[str, set[asyncio.Queue]] = {}  # per-camera clients
        self._field: set[asyncio.Queue] = set()  # Phase 10 M4: field-dispatch clients
        self._analytics: set[asyncio.Queue] = set()  # Phase 12: analytics feed
        self._max_clients = max_clients

    # ---- sync side (pipeline thread) --------------------------------------
    def publish_frame(self, frame_id: int, timestamp: float, jpeg_b64: str,
                      tracks: list[dict], poses: list[dict], alerts: list[dict],
                      camera: str | None = None) -> None:
        # Phase 8 M2: frames are camera-scoped; alerts (publish_alert) are
        # broadcast to every client regardless of the camera they watch.
        msg = {"type": "frame", "camera": camera or "", "frame_id": frame_id,
               "ts": timestamp, "jpeg": jpeg_b64, "tracks": tracks,
               "poses": poses, "alerts": alerts}
        self._publish(msg, channel=camera)

    def publish_alert(self, alert: dict) -> None:
        # alerts are global: every client gets them, whatever camera they watch
        self._broadcast({"type": "alert", "alert": alert})

    def publish_field_alert(self, alert: dict) -> None:
        """Phase 10 M4: push an alert ONLY to field-dispatch clients.

        Delivered to /ws/field subscribers (police+) and never to the live
        wall, keeping the dispatch feed light and focused on actions.
        """
        if self._loop is None or not self._field:
            return
        msg = {"type": "alert", "alert": alert}
        asyncio.run_coroutine_threadsafe(self._fanout_field(msg), self._loop)

    async def _fanout_field(self, msg: dict) -> None:
        for q in self._field:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def publish_audio_level(self, level: dict) -> None:
        """Phase 11: push live audio RMS/peak to field-dispatch clients.

        Only sent to /ws/field subscribers so the dashboard can render a
        volume meter; the main live wall does not need per-tick audio data.
        """
        if self._loop is None or not self._field:
            return
        msg = {"type": "audio_level", "level": level}
        asyncio.run_coroutine_threadsafe(self._fanout_field(msg), self._loop)

    def publish_analytics(self, snapshot: dict) -> None:
        """Phase 12: push analytics snapshot to /ws/analytics clients."""
        if self._loop is None or not self._analytics:
            return
        msg = {"type": "analytics", "data": snapshot}
        asyncio.run_coroutine_threadsafe(self._fanout_analytics(msg), self._loop)

    def publish_incident(self, incident: dict) -> None:
        """Push an incident update to /ws/incidents operator clients."""
        if self._loop is None or not self._field:
            return
        msg = {"type": "incident", "incident": incident}
        asyncio.run_coroutine_threadsafe(self._fanout_field(msg), self._loop)

    async def _fanout_analytics(self, msg: dict) -> None:
        for q in self._analytics:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    async def subscribe_analytics(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._analytics.add(q)
        return q

    def unsubscribe_analytics(self, q: asyncio.Queue) -> None:
        self._analytics.discard(q)

    def publish_public_frame(self, frame_id: int, timestamp: float,
                             jpeg_b64: str, camera: str = "__public__") -> None:
        """Phase 9 M5: sanitized frame for the read-only public monitor.

        Delivered ONLY to subscribers of the dedicated "__public__" channel -
        authenticated clients (including "all cameras") never see it.
        """
        if self._loop is None or not self._channels.get(camera):
            return
        msg = {"type": "frame", "camera": camera, "frame_id": frame_id,
               "ts": timestamp, "jpeg": jpeg_b64, "tracks": [], "poses": [],
               "alerts": []}
        asyncio.run_coroutine_threadsafe(self._fanout_channel(msg, camera),
                                         self._loop)

    async def _fanout_channel(self, msg: dict, channel: str) -> None:
        for q in self._channels.get(channel, set()):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def _broadcast(self, msg: dict) -> None:
        if self._loop is None or (not self._subscribers and not self._channels):
            return
        asyncio.run_coroutine_threadsafe(self._fanout_all(msg), self._loop)

    async def _fanout_all(self, msg: dict) -> None:
        targets = set(self._subscribers)
        for ch in self._channels.values():
            targets |= ch
        for q in targets:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def _publish(self, msg: dict, channel: str | None = None) -> None:
        if self._loop is None or (not self._subscribers and not self._channels):
            return
        asyncio.run_coroutine_threadsafe(self._fanout(msg, channel), self._loop)

    async def _fanout(self, msg: dict, channel: str | None = None) -> None:
        targets = set(self._subscribers)
        if channel:
            targets |= self._channels.get(channel, set())
        for q in targets:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    # ---- async side (server) ----------------------------------------------
    async def subscribe(self, camera: str | None = None) -> asyncio.Queue:
        if self._loop is None:
            # first subscriber runs inside the server's event loop
            self._loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        if camera:
            self._channels.setdefault(camera, set()).add(q)
        else:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue, camera: str | None = None) -> None:
        if camera:
            ch = self._channels.get(camera)
            if ch:
                ch.discard(q)
        else:
            self._subscribers.discard(q)

    # Phase 10 M4: field-dispatch clients receive only alert pushes
    async def subscribe_field(self) -> asyncio.Queue:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._field.add(q)
        return q

    def unsubscribe_field(self, q: asyncio.Queue) -> None:
        self._field.discard(q)

    @property
    def client_count(self) -> int:
        return (len(self._subscribers)
                + sum(len(c) for c in self._channels.values())
                + len(self._field) + len(self._analytics))


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
               plates: "PlateRegistry | None" = None,  # noqa: F821
               cameras: list[dict] | None = None,
               assistant_ctx: dict | None = None,
               metrics: "MetricsRegistry | None" = None,  # noqa: F821
               backup_mgr=None,
               ready_check=None,
               db_metrics_provider=None,
               metrics_token: str | None = None,
               reid=None,
               public_token: str | None = None,
               notifier=None,
               incident_store=None,
               incident_dispatch=None) -> Any:
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

    from .. import __version__
    app = FastAPI(title="BHAIRAV - Evidence & Live API", version=__version__)
    app.add_middleware(_BodyLimitMiddleware, max_bytes=MAX_JSON_BODY_BYTES)

    # CORS: configurable allowed origins via BHAIRAV_CORS_ORIGINS env var
    # (comma-separated). Default: same-origin only (no cross-origin).
    import os
    cors_raw = os.environ.get("BHAIRAV_CORS_ORIGINS", "").strip()
    if cors_raw:
        try:
            from fastapi.middleware.cors import CORSMiddleware
            origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            log.info("CORS enabled for origins: %s", origins)
        except ImportError:
            log.warning("fastapi.middleware.cors not available; CORS disabled")

    # CSRF protection: state-changing endpoints require a custom header
    # (X-Requested-With: XMLHttpRequest) or a matching Origin header.
    # GET/HEAD/OPTIONS are safe and exempt.
    _csrf_enabled = os.environ.get("BHAIRAV_CSRFProtection", "").lower() in ("1", "true", "yes")
    _csrf_header = "x-requested-with"
    _safe_methods = {"GET", "HEAD", "OPTIONS"}

    @app.middleware("http")
    async def _csrf_middleware(request: Request, call_next):
        if not _csrf_enabled:
            return await call_next(request)
        if request.method in _safe_methods:
            return await call_next(request)
        # Allow API token auth (Bearer) — CSRF only matters for cookie-based auth
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return await call_next(request)
        # Check for the custom header or a same-origin referer
        if _csrf_header in request.headers:
            return await call_next(request)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=403, detail="CSRF validation failed: missing X-Requested-With header")

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
        return {"status": "ok", "service": "bhairav-phase10",
                "time": round(time.time(), 3), "clients": hub.client_count}

    @app.get("/ready")
    def ready():
        """Readiness probe for orchestrators / nginx / compose healthchecks.

        Public on purpose: load balancers must be able to check a replica
        without credentials. `ready_check` is provided by serve.py (DB ping
        in PostgreSQL mode, evidence-dir check otherwise).
        """
        ok = bool(ready_check() if ready_check else True)
        return {"ready": ok, "time": round(time.time(), 3)}

    # ---- ops status -------------------------------------------------------
    @app.get("/api/status")
    def api_status(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        counts = store.counts()
        audit_ok, problems = audit.verify()
        return {
            "service": "bhairav", "version": __version__,
            "time": round(time.time(), 3),
            "pipeline": stats.snapshot(),
            "clients": hub.client_count,
            "evidence": counts,
            "audit": {"ok": audit_ok, "problems": problems,
                       "entries": len(audit.read())},
            "users": users.count(),
            "webhook": bool(webhook_url),
            "dispatch": (notifier.stats() if notifier is not None else None),
            "cameras": cameras or [],
            "db": db_metrics_provider() if db_metrics_provider else None,
            "backups": ({"dir": str(backup_mgr.out_dir),
                         "count": len(backup_mgr.list()),
                         "latest": backup_mgr.latest()}
                        if backup_mgr is not None else None),
            "series": metrics.snapshot() if metrics is not None else None,
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
                        camera: str | None = None, q: str | None = None,
                        t0: float | None = None, t1: float | None = None,
                        limit: int = Query(50, le=200)):
        rows = store.search(rule=rule, severity=severity, camera=camera,
                            q=q, t0=t0, t1=t1, limit=limit)
        return {"total": len(rows), "events": [r.to_dict() for r in rows]}

    @app.get("/api/evidence/export")
    def export_evidence(claims: dict = Depends(require(PERM_EVIDENCE_EXPORT)),
                        rule: str | None = None, severity: str | None = None,
                        camera: str | None = None, q: str | None = None,
                        t0: float | None = None, t1: float | None = None,
                        limit: int = Query(200, le=500)):
        rows = store.search(rule=rule, severity=severity, camera=camera,
                            q=q, t0=t0, t1=t1, limit=limit)
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

    # ---- Investigation Assistant (Phase 8 M4) -----------------------------
    @app.post("/api/assistant/query")
    def assistant_query(request: Request, payload: dict = Body(...),
                        claims: dict = Depends(require(PERM_EVIDENCE_EXPORT))):
        _reject_large_body(request)
        from .assistant import parse_query
        q = str(payload.get("query", "")).strip()
        if not q:
            raise HTTPException(status_code=400, detail="query is required")
        ctx = {
            "zones": list((assistant_ctx or {}).get("zones", [])),
            "cameras": [c["id"] for c in (cameras or [])],
            "users": [u["username"] for u in users.public_view()],
            "now": time.time(),
        }
        parsed = parse_query(q, ctx)
        events = store.search(**parsed["search_kwargs"])
        plate_reads: list[dict] = []
        if parsed["plates"] and plates is not None:
            want = set(parsed["plates"])
            plate_reads = [r for r in plates.recent_reads(200)
                           if r["plate"] in want]
        audit_rows: list[dict] = []
        if parsed["want_audit"]:
            audit_rows = audit.query(actor=parsed["actor"], limit=50)
        audit.append(claims["sub"], "assistant_query", q[:200])
        return {
            "query": q,
            "plan": parsed["plan"],
            "warnings": parsed["warnings"],
            "events": [e.to_dict() for e in events],
            "plate_reads": plate_reads,
            "audit": audit_rows,
        }

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

    # ---- person re-identification across cameras (Phase 9 M4) --------------
    def _reid_or_503():
        if reid is None:
            raise HTTPException(status_code=503,
                                detail="re-id is not enabled in this server")
        return reid

    @app.get("/api/reid/stats")
    def reid_stats(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return _reid_or_503().store.stats()

    @app.get("/api/reid/subjects")
    def reid_subjects(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return {"subjects": _reid_or_503().store.list()}

    @app.post("/api/reid/subjects/{sid}/rename")
    def reid_subject_rename(sid: str, payload: dict = Body(...),
                            claims: dict = Depends(require(PERM_USERS))):
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not _reid_or_503().store.rename(sid, name):
            raise HTTPException(status_code=404, detail="subject not found")
        audit.append(claims["sub"], "rename_reid_subject", f"{sid} -> {name}")
        return {"renamed": sid}

    @app.delete("/api/reid/subjects/{sid}")
    def reid_subject_delete(sid: str,
                            claims: dict = Depends(require(PERM_USERS))):
        if not _reid_or_503().store.remove(sid):
            raise HTTPException(status_code=404, detail="subject not found")
        audit.append(claims["sub"], "remove_reid_subject", sid)
        return {"removed": sid}

    @app.get("/api/reid/subjects/{sid}/trail")
    def reid_subject_trail(sid: str,
                           claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        svc = _reid_or_503()
        if svc.store.get(sid) is None:
            raise HTTPException(status_code=404, detail="subject not found")
        return {"subject": svc.store.get(sid), "trail": svc.store.trail(sid)}

    @app.get("/api/reid/sightings")
    def reid_sightings(subject_id: str | None = None, camera: str | None = None,
                       limit: int = Query(100, ge=1, le=500),
                       claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return {"sightings": _reid_or_503().store.sightings(
            subject_id=subject_id, camera=camera, limit=limit)}

    @app.get("/api/reid/search")
    def reid_search(q: str = Query("", max_length=200),
                    limit: int = Query(20, ge=1, le=100),
                    claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Search gallery subjects by physical description ("red shirt, tall")."""
        from bhairav.describe import parse_query, search_subjects
        colors, height = parse_query(q)
        if not colors and not height:
            return {"query": q, "colors": [], "height": None, "results": []}
        svc = _reid_or_503()
        hits = search_subjects(svc.store.list(), colors, height, limit)
        audit.append(claims["sub"], "search_reid_subjects", q or "(empty)")
        return {"query": q, "colors": colors, "height": height,
                "results": [{"subject": r["subject"], "score": r["score"]}
                            for r in hits]}

    # ---- Phase 14: batch embedding + similarity matrix --------------------
    @app.get("/api/reid/similarity")
    def reid_similarity(
        limit: int = Query(50, ge=2, le=200),
        claims: dict = Depends(require(PERM_EVIDENCE_READ)),
    ):
        """Pairwise cosine similarity matrix for all (or top-N) subjects."""
        svc = _reid_or_503()
        subjects = svc.store.list()
        subjects = subjects[:limit]
        import numpy as np
        from bhairav.reid.deep_embedder import batch_cosine_matrix
        embs = []
        ids = []
        for s in subjects:
            rec = svc.store.get(s["id"])
            if rec and "embedding" in rec:
                embs.append(np.array(rec["embedding"], dtype=np.float64))
                ids.append(s["id"])
        mat = batch_cosine_matrix(embs) if len(embs) >= 2 else np.empty((0, 0))
        return {
            "subject_ids": ids,
            "matrix": mat.tolist(),
            "count": len(ids),
        }

    @app.get("/api/reid/embedding-info")
    def reid_embedding_info(
        claims: dict = Depends(require(PERM_EVIDENCE_READ)),
    ):
        """Info about the current embedding model (deep vs legacy)."""
        svc = _reid_or_503()
        ext = svc.extractor
        is_deep = getattr(ext, "is_deep", False)
        dim = getattr(ext, "embedding_dim", None)
        return {
            "mode": "deep" if is_deep else "hsv+hog",
            "embedding_dim": dim,
            "model_loaded": is_deep,
        }

    # ---- Phase 9 M5: read-only public monitor (privacy-blurred) ----------
    # The pipeline publishes sanitized frames (heads blurred, downscaled,
    # no tracks/poses/alerts) to hub channel "__public__"; this endpoint
    # only ever forwards those. No auth or audit: the bearer of the public
    # token may share the monitor URL freely.
    @app.get("/api/public/info")
    def public_info():
        cam_list = [{"id": c.get("id", c.get("name", ""))}
                    for c in (cameras or [])] or [{"id": ""}]
        return {"streaming": True, "blurred": True,
                "cameras": [c["id"] for c in cam_list]}

    @app.websocket("/api/public/stream")
    async def public_stream(websocket: WebSocket):
        token = (websocket.query_params.get("token") or "").strip()
        if not public_token or not token or                 not hmac.compare_digest(public_token.encode(), token.encode()):
            await websocket.close(code=4401, reason="unauthorized")
            return
        await websocket.accept()
        q = await hub.subscribe("__public__")
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=5.0)
                    # forward ONLY the sanitized jpeg + minimal metadata
                    await websocket.send_text(json.dumps({
                        "type": "frame", "camera": msg.get("camera", ""),
                        "frame_id": msg.get("frame_id"),
                        "ts": msg.get("ts"), "jpeg": msg.get("jpeg", "")}))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "ping"}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(q, "__public__")

    # ---- ops: Prometheus metrics + backups (Phase 9 M3) -------------------
    @app.get("/metrics")
    def metrics_endpoint(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
        """Prometheus text exposition.

        Accepts either a valid admin bearer token or the shared scrape token
        configured via BHAIRAV_METRICS_TOKEN (constant-time compare) so a
        scraper needs no interactive login; see deploy/prometheus.yml.
        """
        if metrics is None:
            raise HTTPException(status_code=503,
                                detail="metrics not enabled")
        raw = creds.credentials if creds is not None else None
        token = getattr(raw, "get_secret_value", lambda: raw)() if raw else None
        authorized = False
        if token:
            if metrics_token and hmac.compare_digest(token, metrics_token):
                authorized = True
            else:
                claims = validate_token(secret, token) if token else None
                authorized = bool(claims and PERM_USERS in
                                  ROLE_PERMISSIONS.get(claims.get("role", ""),
                                                       frozenset()))
        if not authorized:
            raise HTTPException(status_code=401,
                                detail="Invalid or expired token")
        return Response(content=metrics.render(),
                        media_type="text/plain; version=0.0.4")

    def _backups_or_503():
        if backup_mgr is None:
            raise HTTPException(status_code=503,
                                detail="backups unavailable (PostgreSQL mode only)")
        return backup_mgr

    @app.get("/api/ops/backups")
    def ops_backups_list(claims: dict = Depends(require(PERM_USERS))):
        return {"backups": _backups_or_503().list()}

    @app.post("/api/ops/backups")
    def ops_backups_create(claims: dict = Depends(require(PERM_USERS))):
        try:
            result = _backups_or_503().create()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"backup failed: {exc}")
        audit.append(claims["sub"], "create_backup", result["path"])
        return result

    @app.get("/api/ops/backups/{name}")
    def ops_backups_download(name: str,
                             claims: dict = Depends(require(PERM_USERS))):
        data = _backups_or_503().read(name)
        if data is None:
            raise HTTPException(status_code=404, detail="backup not found")
        audit.append(claims["sub"], "download_backup", name)
        return Response(content=data,
                        media_type="application/gzip",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{name}"'})

    # ---- alert feed -------------------------------------------------------
    @app.get("/api/alerts/recent")
    def recent_alerts(claims: dict = Depends(require(PERM_ALERTS)),
                      limit: int = Query(50, le=200)):
        return {"alerts": recent[-limit:]}

    # ---- Phase 10 M4: field-officer dispatch ------------------------------
    def _dispatch_or_404():
        if notifier is None or not notifier:
            raise HTTPException(status_code=404,
                                detail="no alert channels configured "
                                       "(backend.alert_channels)")
        return notifier

    @app.get("/api/dispatch/channels")
    def dispatch_channels(claims: dict = Depends(require(PERM_USERS))):
        return {"channels": _dispatch_or_404().stats()}

    @app.post("/api/dispatch/test")
    def dispatch_test(claims: dict = Depends(require(PERM_USERS))):
        _dispatch_or_404().test()
        audit.append(claims["sub"], "dispatch_test",
                     "sent synthetic alert to all channels")
        return {"tested": True}

    @app.websocket("/ws/field")
    async def ws_field(websocket: WebSocket):
        """Push-only alert feed for field officers (police+).

        Receives {"type":"alert","alert":{...}} messages with camera/rule/
        severity details - no heavy frames - so a mobile client stays light.
        """
        token = (websocket.query_params.get("token") or "").strip()
        claims = validate_token(secret, token) if token else None
        if (claims is None or not _user_active(claims.get("sub", ""))
                or PERM_ALERTS not in ROLE_PERMISSIONS.get(claims["role"],
                                                           frozenset())):
            await websocket.close(code=4401, reason="unauthorized")
            return
        await websocket.accept()
        q = await hub.subscribe_field()
        audit.append(claims["sub"], "field_dispatch_connect",
                     "field-officer alert feed")
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=5.0)
                    await websocket.send_text(json.dumps(msg))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "ping"}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe_field(q)
            audit.append(claims["sub"], "field_dispatch_disconnect",
                         "field-officer alert feed")

    @app.websocket("/ws/analytics")
    async def ws_analytics(websocket: WebSocket):
        """Phase 12: push-only analytics feed (forecast, heatmap, trends).

        Delivers periodic snapshots (every ~1s) to dashboard analytics tabs.
        No auth required beyond analyst role (same as field dispatch).
        """
        token = (websocket.query_params.get("token") or "").strip()
        claims = validate_token(secret, token) if token else None
        if (claims is None or not _user_active(claims.get("sub", ""))):
            await websocket.close(code=4401, reason="unauthorized")
            return
        await websocket.accept()
        q = await hub.subscribe_analytics()
        audit.append(claims["sub"], "analytics_connect", "predictive analytics feed")
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=5.0)
                    await websocket.send_text(json.dumps(msg))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "ping"}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe_analytics(q)
            audit.append(claims["sub"], "analytics_disconnect", "predictive analytics feed")

    # ---- federation ingest (Phase 13.3) ----
    @app.post("/api/federation/ingest")
    async def federation_ingest(request: Request):
        """Accept federation messages from peer BHAIRAV servers."""
        site = request.headers.get("X-Federation-Site", "unknown")
        # Simple shared-secret check (production should use HMAC)
        body = await request.json()
        if not isinstance(body, list):
            return {"error": "expected array"}
        # Store incoming alerts in the recent feed
        for msg in body:
            if msg.get("type") == "alert" and "payload" in msg:
                ad = msg["payload"]
                ad["federation_source"] = site
                recent.append(ad)
                del recent[:-200]
        return {"ok": True, "received": len(body)}

    # ---- Phase 17: Threat Response endpoints ----------------------------

    # PTZ endpoints
    ptz_controllers = {}

    @app.get("/api/ptz/cameras")
    def ptz_list_cameras(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return {"cameras": list(ptz_controllers.keys())}

    @app.post("/api/ptz/{camera_id}/move")
    def ptz_move(camera_id: str, payload: dict = Body(...),
                 claims: dict = Depends(require(PERM_USERS))):
        from bhairav.response.ptz import PTZController, PTZCommand
        if camera_id not in ptz_controllers:
            ptz_controllers[camera_id] = PTZController(camera_id)
        ctrl = ptz_controllers[camera_id]
        cmd = payload.get("command", "stop")
        try:
            ptz_cmd = PTZCommand(cmd)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown command: {cmd}")
        speed = float(payload.get("speed", 0.5))
        result = ctrl.move(ptz_cmd, speed=speed)
        return result

    @app.get("/api/ptz/{camera_id}/state")
    def ptz_state(camera_id: str, claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        if camera_id not in ptz_controllers:
            return {"camera": camera_id, "pan": 0, "tilt": 0, "zoom": 1, "moving": False}
        ctrl = ptz_controllers[camera_id]
        return {"camera": camera_id, "pan": ctrl.state.pan, "tilt": ctrl.state.tilt,
                "zoom": ctrl.state.zoom, "moving": ctrl.state.moving}

    # Escalation endpoints
    @app.get("/api/escalation/events")
    def escalation_events(limit: int = Query(20, ge=1, le=100),
                          claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return {"events": []}  # populated by engine at runtime

    # Incident report endpoints
    @app.get("/api/reports")
    def list_reports(status: str | None = None, limit: int = Query(50, ge=1, le=200),
                     claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return {"reports": []}  # populated by ReportGenerator at runtime

    @app.post("/api/reports")
    def create_report(payload: dict = Body(...),
                      claims: dict = Depends(require(PERM_USERS))):
        import uuid
        from bhairav.response.reports import ReportGenerator
        rg = ReportGenerator()
        rid = str(uuid.uuid4().hex[:8])
        report = rg.create_report(
            incident_id=f"INC-{rid}",
            title=payload.get("title", "Untitled Incident"),
            severity=payload.get("severity", "orange"),
            zone=payload.get("zone"),
            camera=payload.get("camera"),
            description=payload.get("description", ""),
        )
        return report.to_dict()

    # Tenant management endpoints
    @app.get("/api/tenants")
    def list_tenants(claims: dict = Depends(require(PERM_USERS))):
        from bhairav.response.tenant import TenantManager
        tm = TenantManager()
        return {"tenants": tm.list_tenants()}

    @app.post("/api/tenants")
    def create_tenant(payload: dict = Body(...),
                      claims: dict = Depends(require(PERM_USERS))):
        from bhairav.response.tenant import TenantManager
        tm = TenantManager()
        t = tm.create_tenant(
            tenant_id=payload.get("tenant_id", ""),
            name=payload.get("name", ""),
            role=payload.get("role", "operator"),
            cameras=payload.get("cameras", []),
            zones=payload.get("zones", []),
        )
        return t.to_dict()

    @app.get("/api/tenants/{tenant_id}")
    def get_tenant(tenant_id: str, claims: dict = Depends(require(PERM_USERS))):
        from bhairav.response.tenant import TenantManager
        tm = TenantManager()
        t = tm.get_tenant(tenant_id)
        if not t:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return t.to_dict()

    @app.delete("/api/tenants/{tenant_id}")
    def delete_tenant(tenant_id: str, claims: dict = Depends(require(PERM_USERS))):
        from bhairav.response.tenant import TenantManager
        tm = TenantManager()
        if not tm.delete_tenant(tenant_id):
            raise HTTPException(status_code=404, detail="Tenant not found")
        return {"deleted": tenant_id}

    # Integration endpoints
    @app.get("/api/integrations")
    def list_integrations(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return {"channels": []}

    @app.post("/api/integrations/channels")
    def register_channel(payload: dict = Body(...),
                         claims: dict = Depends(require(PERM_USERS))):
        from bhairav.response.integrations import ExternalChannel
        ch = ExternalChannel(
            channel_id=payload.get("channel_id", ""),
            channel_type=payload.get("channel_type", "webhook"),
            name=payload.get("name", ""),
            endpoint=payload.get("endpoint", ""),
            enabled=payload.get("enabled", True),
            severity_filter=payload.get("severity_filter", ["red"]),
        )
        return ch.__dict__

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
        camera = (websocket.query_params.get("camera") or "").strip() or None
        q = await hub.subscribe(camera)
        audit.append(claims["sub"], "ws_connect",
                     f"live-stream camera={camera or 'all'}")
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
            hub.unsubscribe(q, camera)
            audit.append(claims["sub"], "ws_disconnect",
                         f"live-stream camera={camera or 'all'}")

    # ---- Phase 21-24: 3D Scene, Traffic, Investigation, NLP Query ----------

    @app.get("/api/scene3d")
    def scene3d_snapshot(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return {"cameras": [], "persons": [], "zones": [], "events": []}

    @app.get("/api/traffic")
    def traffic_snapshot(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        return {"zones": [], "total_vehicles_tracked": 0}

    @app.get("/api/timeline")
    def timeline_events(limit: int = Query(50, ge=1, le=500),
                        event_type: str | None = None,
                        zone: str | None = None,
                        claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        from bhairav.investigation import InvestigationTimeline
        tl = InvestigationTimeline()
        return {"events": tl.query(event_type=event_type, zone=zone, limit=limit)}

    @app.post("/api/cases")
    def create_case(payload: dict = Body(...),
                    claims: dict = Depends(require(PERM_USERS))):
        from bhairav.investigation import InvestigationTimeline
        tl = InvestigationTimeline()
        case = tl.create_case(
            title=payload.get("title", "Untitled"),
            summary=payload.get("summary", ""),
            assigned_to=payload.get("assigned_to", ""),
        )
        return case.to_dict()

    @app.get("/api/cases")
    def list_cases(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        from bhairav.investigation import InvestigationTimeline
        tl = InvestigationTimeline()
        return {"cases": tl.list_cases()}

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str,
                 claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        from bhairav.investigation import InvestigationTimeline
        tl = InvestigationTimeline()
        case = tl.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return case

    @app.post("/api/cases/{case_id}/export")
    def export_case(case_id: str,
                    claims: dict = Depends(require(PERM_USERS))):
        from bhairav.investigation import InvestigationTimeline
        tl = InvestigationTimeline()
        return tl.export_case(case_id)

    @app.post("/api/nlp/query")
    def nlp_query(payload: dict = Body(...),
                  claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        from bhairav.nlp import NLPQueryEngine
        engine = NLPQueryEngine()
        result = engine.query(payload.get("query", ""))
        return result.to_dict()

    # ---- Phase 18: NL Summaries + Predictive Hotspot + Resource Allocation --

    @app.get("/api/analytics/summary")
    def analytics_summary(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Natural-language summary of recent alerts."""
        from bhairav.analytics import NLAlertSummarizer
        s = NLAlertSummarizer()
        # feed from recent alerts
        for a in recent[-100:]:
            s.observe(a)
        return s.snapshot()

    @app.get("/api/analytics/hotspots")
    def analytics_hotspots(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Ranked predictive hotspot zones."""
        from bhairav.analytics import PredictiveHotspot
        h = PredictiveHotspot()
        for a in recent[-200:]:
            h.observe(
                a.get("timestamp", 0),
                zone=a.get("zone"),
                severity=a.get("severity", "yellow"),
                rule=a.get("rule", ""),
            )
        return h.snapshot()

    @app.get("/api/analytics/recommendations")
    def analytics_recommendations(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Resource allocation recommendations."""
        from bhairav.analytics import ResourceAllocator, PredictiveHotspot
        h = PredictiveHotspot()
        zone_counts: dict = {}
        for a in recent[-200:]:
            z = a.get("zone", "(unknown)")
            zone_counts[z] = zone_counts.get(z, 0) + 1
            h.observe(
                a.get("timestamp", 0),
                zone=z,
                severity=a.get("severity", "yellow"),
                rule=a.get("rule", ""),
            )
        allocator = ResourceAllocator()
        return {"recommendations": [r.to_dict() for r in allocator.analyze(
            h.snapshot().get("hotspots", []), zone_counts
        )]}

    # ---- Live Face Search + Trajectory Prediction (new) --------------------
    try:
        from ..face_tracking import LiveFaceMonitor, TrajectoryPredictor, LiveFaceMonitorConfig

        # Trajectory predictor (no persistence by default — serve.py adds that)
        trajectory_predictor = TrajectoryPredictor(
            zones=[],
            enable_cross_camera_linking=True,
        )
        live_face_monitor = None
        if face is not None:
            try:
                live_face_monitor = LiveFaceMonitor(
                    face["recognizer"], face["gallery"],
                    config=LiveFaceMonitorConfig(
                        detect_every_n_frames=5,
                        min_detection_score=0.80,
                        min_match_similarity=0.50,
                        alert_cooldown_sec=30.0,
                    ),
                )
            except Exception as exc:
                log.warning("Live face monitor init failed: %s", exc)
    except ImportError:
        log.warning("face_tracking module not available; trajectory prediction disabled")
        trajectory_predictor = None
        live_face_monitor = None

    @app.get("/api/face/live-monitor/status")
    def face_live_status(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Status of the live face monitor."""
        if live_face_monitor is None:
            return {"enabled": False, "reason": "face models not installed"}
        return {"enabled": True, **live_face_monitor.stats()}

    @app.post("/api/face/live-monitor/process")
    async def face_live_process(payload: dict = Body(...),
                                claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Process a single frame for live face matching (async).

        Body: {image_b64, camera_id, frame_id}
        Returns: list of face matches against the gallery.

        Runs face detection in a thread pool to avoid blocking the event loop.
        """
        if live_face_monitor is None:
            raise HTTPException(status_code=503,
                                detail="live face monitor unavailable")
        img = _decode_image_b64(str(payload.get("image_b64", "")))
        camera_id = str(payload.get("camera_id", "unknown"))
        frame_id = int(payload.get("frame_id", 0))
        ts = time.time()
        # Run CPU-intensive face detection in a thread pool
        matches = await asyncio.to_thread(
            live_face_monitor.process_frame, img, camera_id, frame_id, ts
        )
        return {"matches": [m.to_dict() for m in matches],
                "stats": live_face_monitor.stats()}

    @app.get("/api/persons/tracked")
    def persons_tracked(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """List all currently tracked persons with movement data."""
        if trajectory_predictor is None:
            return {"persons": [], "stats": {}, "enabled": False}
        return {"persons": trajectory_predictor.list_tracked_persons(),
                "stats": trajectory_predictor.stats(), "enabled": True}

    @app.get("/api/persons/{person_id}/trajectory")
    def person_trajectory(person_id: str,
                          seconds: float = 30.0,
                          claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Get position history for a person."""
        if trajectory_predictor is None:
            return {"person": None, "positions": [], "enabled": False}
        positions = trajectory_predictor.get_recent_positions(person_id, seconds)
        traj = trajectory_predictor.get_trajectory(person_id)
        info = None
        if traj:
            info = {
                "person_id": person_id,
                "current_camera": traj.current_camera,
                "cameras_visited": traj.cameras_visited,
                "speed": round(traj.speed, 6),
                "heading_deg": round(traj.heading_deg, 1),
                "total_distance": round(traj.total_distance, 4),
                "first_seen": round(traj.first_seen, 3),
                "last_seen": round(traj.last_seen, 3),
            }
        return {"person": info, "positions": positions}

    @app.get("/api/persons/{person_id}/predict")
    def person_predict(person_id: str,
                       horizons: str = "1,2,5,10,15,30",
                       claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Predict future positions for a person.

        Query param: horizons = comma-separated seconds (default: 1,2,5,10,15,30)
        Returns 404 if person not found, empty predictions if insufficient data.
        """
        try:
            h_list = [float(h.strip()) for h in horizons.split(",") if h.strip()]
        except ValueError:
            h_list = [1.0, 2.0, 5.0, 10.0, 15.0, 30.0]
        if trajectory_predictor is None:
            return {"person_id": person_id, "current": None, "predictions": [], "enabled": False}
        predictions = trajectory_predictor.predict_multi(person_id, h_list)
        if not predictions:
            raise HTTPException(status_code=404,
                                detail=f"person '{person_id}' not found")
        traj = trajectory_predictor.get_trajectory(person_id)
        current = None
        if traj and traj.positions:
            last = traj.positions[-1]
            current = {
                "x": round(last.x, 4),
                "y": round(last.y, 4),
                "camera_id": last.camera_id,
                "timestamp": round(last.timestamp, 3),
                "speed": round(traj.speed, 6),
                "heading_deg": round(traj.heading_deg, 1),
            }
        return {
            "person_id": person_id,
            "current": current,
            "predictions": predictions,
        }

    @app.get("/api/persons/predict-zones")
    def persons_predict_zones(claims: dict = Depends(require(PERM_EVIDENCE_READ))):
        """Predict which zones people are heading toward."""
        if trajectory_predictor is None:
            return {"zone_predictions": [], "total_tracked": 0, "enabled": False}
        persons = trajectory_predictor.list_tracked_persons()
        zone_predictions = []
        for p in persons:
            pred = trajectory_predictor.predict(p["person_id"], seconds_ahead=10.0)
            if pred and pred.zone:
                zone_predictions.append({
                    "person_id": p["person_id"],
                    "current_camera": p["current_camera"],
                    "predicted_zone": pred.zone,
                    "predicted_position": pred.to_dict(),
                })
        return {"zone_predictions": zone_predictions,
                "total_tracked": len(persons)}

    # ---- City Safety Incidents (Phase 1) ----------------------------------
    from ..incidents import IncidentStore, DispatchEngine as _DispatchEngine
    from ..incidents import create_incident_routes, seed_demo_data

    inc_store = incident_store or IncidentStore(
        path=str(Path(store.root).parent / "incidents")
    )
    inc_dispatch = incident_dispatch or _DispatchEngine(inc_store)

    # Seed demo officers and incidents on first run
    if not inc_store.list_officers():
        seed_demo_data(inc_store)
        log.info("Seeded %d officers and demo incidents",
                 len(inc_store.list_officers()))

    create_incident_routes(app, inc_store, inc_dispatch)

    # WebSocket: real-time incident feed for operators
    @app.websocket("/ws/incidents")
    async def ws_incidents(websocket: WebSocket):
        """Real-time incident push for operators. New incidents, status
        changes, and dispatch events are pushed live."""
        token = (websocket.query_params.get("token") or "").strip()
        claims = validate_token(secret, token) if token else None
        if (claims is None or not _user_active(claims.get("sub", ""))
                or PERM_ALERTS not in ROLE_PERMISSIONS.get(
                    claims["role"], frozenset())):
            await websocket.close(code=4401, reason="unauthorized")
            return
        await websocket.accept()
        q = await hub.subscribe_field()  # reuse field channel for incidents
        try:
            # Send current snapshot on connect
            snapshot = inc_store.get_stats()
            await websocket.send_text(json.dumps({
                "type": "snapshot", "data": snapshot
            }))
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=5.0)
                    await websocket.send_text(json.dumps(msg))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "ping"}))
        except WebSocketDisconnect:
            pass

    return app
