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

# 4. build + run (--scale app=N for N replicas; default compose runs 2)
docker compose -f deploy/docker-compose.yml up -d --build --scale app=2
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
- **PostgreSQL backend** (Phase 8) - the bundled `db` service holds every
  persistent store: evidence (media as BYTEA), the hash-chained audit log,
  users, and the plate watchlist. Remove the `DATABASE_URL` env var and the
  `db` service to go back to file-based stores (single replica only).

## HA (Phase 8 M3)

- Run `docker compose up -d --scale app=N` (the compose file defaults to 2).
  All state lives in PostgreSQL, so replicas share evidence/audit/users/plates
  with no shared filesystem.
- nginx load-balances with `ip_hash`, pinning each client to one replica so
  the live WebSocket channel stays consistent; WebSockets upgrade through the
  proxy as before.
- Replicas are stateless beyond their in-memory live hub and recent-alert
  feed - a replica can be killed and replaced without losing anything
  persisted.

## Honest notes

- **In-memory per-replica state** (the WS hub and the recent-alert feed) is
  not shared across replicas; ip_hash keeps clients on one replica, but a
  replica restart reconnects viewers. A shared pub/sub (Redis) would remove
  even that - future work.
- **Single-replica caveats**: on the file store, run exactly one app
  instance; the file-based stores are not safe for concurrent writers.
- **Self-signed certs** are fine for private deployments; for public use point
  nginx at a real certificate (Let's Encrypt).
- The ML models (YOLO/SFace/YuNet/pose/EasyOCR) are fetched at first run; make
  sure the container can reach the internet once, or bake `models/` into the
  image (they are gitignored - copy them into the build context if needed).

## Phase 9 M3 - backups, metrics and health dashboards

### PostgreSQL backups (scheduled + on-demand)

The compose file runs a `backup` sidecar that dumps the whole public schema
to `backups/` every 6h with retention 14 and a post-write verify. The dump is
a pure-Python logical backup (gzip'd JSON, no `pg_dump` binary) that
round-trips BYTEA and JSONB columns; restore anywhere with:

```bash
python scripts/backup_db.py --url "$DATABASE_URL" --dir backups --retention 14 --verify
python scripts/backup_db.py --list
python scripts/restore_db.py --url "$TARGET_URL" --file backups/bhairav_....json.gz --wipe
```

Manually (and from the dashboard Status tab, admin only):
`POST /api/ops/backups`, list `GET /api/ops/backups`, download
`GET /api/ops/backups/<name>` (audited; filename-validated).

### Metrics + health

- `GET /ready` (public) - readiness for load balancers / compose healthchecks
  (DB ping in PostgreSQL mode).
- `GET /metrics` - Prometheus text exposition. Accepts an admin bearer token
  or the shared scrape token set via `BHAIRAV_METRICS_TOKEN` (same value in
  `deploy/prometheus-token`, no trailing newline:
  `printf '%s' "$BHAIRAV_METRICS_TOKEN" > deploy/prometheus-token`).
- Dashboard Status tab shows DB size/rows, backup age, a "Backup now" button
  (admin) and live sparklines of frames / fps / alerts / clients.
- `prometheus` + `grafana` compose services: `docker compose -f
  deploy/docker-compose.yml up -d` brings up Prometheus scraping `/metrics`
  and a pre-provisioned "BHAIRAV Ops" dashboard at http://<host>:3000
  (default login admin/admin - change `GRAFANA_ADMIN_PASSWORD` in .env).
