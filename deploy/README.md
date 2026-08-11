# BHAIRAV - production deployment (Docker + nginx + TLS)

## Layout

```
deploy/
  Dockerfile              # app image (python:3.12-slim, non-root user)
  docker-compose.yml      # app + nginx (TLS) + PostgreSQL (evidence store)
  config.override.yaml    # mounted as /app/config.yaml (deep-merged over defaults)
  nginx/
    nginx.conf            # reverse proxy, TLS 1.2+, WS upgrade, edge rate limit
    generate_certs.sh     # self-signed certs for local/private deployments
```

## Quick start (Linux server with Docker)

```bash
# 1. secrets
cat > deploy/.env <<'EOF'
BHAIRAV_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
BHAIRAV_EVIDENCE_KEY=$(python -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())")
POSTGRES_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
EOF

# 2. TLS certs (replace with real ones for public exposure)
deploy/nginx/generate_certs.sh

# 3. point the pipeline at a real source (optional; default is the demo scene)
#    edit deploy/docker-compose.yml -> app.command, e.g.:
#      ["--source", "rtsp://user:pass@cam-ip:554/stream1", "--host", "0.0.0.0"]

# 4. build + run
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml logs -f app
```

Dashboard: `https://<host>/dashboard/`. API: `https://<host>/health`.

## What this gives you

- **TLS everywhere** (nginx terminates; HTTP redirects to HTTPS; TLS 1.2+).
- **Non-root app container**, health checks, restart policy.
- **Edge rate limiting** on `/api/` on top of the app's login limiter.
- **WebSocket upgrade** so the live stream works through the proxy.
- **Secrets via env** (`BHAIRAV_SECRET`, `BHAIRAV_EVIDENCE_KEY`); startup
  guards refuse default credentials on non-loopback interfaces.
- **Evidence on a named volume** (`bhairav-data`) - survives rebuilds.
- **PostgreSQL evidence store** (Phase 8) - the bundled `db` service is
  enabled by default via `DATABASE_URL`; remove that env var and the `db`
  service to go back to the file store.

## Honest notes

- **Evidence storage and the audit log default to PostgreSQL** (the bundled
  `db` service). Users and the plate watchlist remain file-based
  (JSON/JSONL on the volume) for now.
- **Single-replica.** For HA you would add a second app instance behind the
  same nginx (uvicorn workers per instance), move evidence to shared/object
  storage, and centralise the audit chain. Not included.
- **Self-signed certs** are fine for private deployments; for public use point
  nginx at a real certificate (Let's Encrypt).
- The ML models (YOLO/SFace/YuNet/pose/EasyOCR) are fetched at first run; make
  sure the container can reach the internet once, or bake `models/` into the
  image (they are gitignored - copy them into the build context if needed).
