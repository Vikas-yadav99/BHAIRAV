# BHAIRAV Deployment Guide

## Quick Start (Docker)

```bash
# Clone the repository
git clone https://github.com/Vikas-yadav99/BHAIRAV.git
cd BHAIRAV

# Start all services
docker compose -f deploy/docker-compose.yml up -d

# Scale to 2 app replicas
docker compose -f deploy/docker-compose.yml up -d --scale app=2

# Check status
docker compose -f deploy/docker-compose.yml ps
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| app | 8000 | BHAIRAV API + WebSocket |
| nginx | 80/443 | Reverse proxy + TLS termination |
| postgres | 5432 | Primary database |
| redis | 6379 | HA clustering + caching |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Monitoring dashboards |

## Environment Variables

```bash
# Required
BHAIRAV_EVIDENCE_KEY=<base64-aes256-key>
BHAIRAV_JWT_SECRET=<random-hex-secret>

# Database
DATABASE_URL=postgresql://user:pass@postgres:5432/bhairav

# Redis (for HA)
REDIS_URL=redis://redis:6379/0

# Cameras (comma-separated)
CAMERAS=rtsp://cam1,rtsp://cam2,rtsp://cam3

# Optional
BHAIRAV_HOST=0.0.0.0
BHAIRAV_PORT=8000
BHAIRAV_DEBUG=false
```

## Production Checklist

- [ ] Generate AES-256 evidence key: `python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"`
- [ ] Generate JWT secret: `python -c "import secrets;print(secrets.token_hex(32))"`
- [ ] Set up TLS certificates in `deploy/nginx/ssl/`
- [ ] Configure camera RTSP/RTMP URLs
- [ ] Set up PostgreSQL with SSL
- [ ] Configure Redis authentication
- [ ] Set up Prometheus alerting rules
- [ ] Configure Grafana dashboards
- [ ] Set up automated backup schedule
- [ ] Run security scan: `python -m bhairav.backend.security scan`
- [ ] Run load test: `python -m tests.load_test`
- [ ] Review DR runbook: `cat docs/dr_runbook.json`

## Backup & Recovery

```bash
# Manual backup
python -c "from bhairav.backend.backups import BackupService; b=BackupService(db_url, './backups'); print(b.create())"

# List backups
python -c "from bhairav.backend.backups import BackupService; b=BackupService(db_url, './backups'); print(b.list())"

# Restore
python -c "from bhairav.backend.backups import restore; restore(db_url, './backups/bhairav_YYYYMMDD_HHMMSS.backup.json.gz')"
```

## Monitoring

- **Prometheus**: http://localhost:9090 — scrape metrics from /metrics
- **Grafana**: http://localhost:3000 — pre-configured dashboards
- **Health check**: `curl http://localhost:8000/api/status`

## Scaling

```bash
# Horizontal scaling
docker compose up -d --scale app=4

# Edge deployment (per-camera agent)
python -m bhairav.edge.agent --source rtsp://cam1 --upstream https://bhairav.city.gov/api/edge/alerts

# Federation (multi-site)
# Configure federation.peers in config.yaml with peer server URLs
```
