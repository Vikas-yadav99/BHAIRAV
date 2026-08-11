# BHAIRAV - Architecture and Onboarding

BHAIRAV is a behavioral video-surveillance engine: it ingests camera feeds,
runs object detection, tracking and pose estimation, applies scenario rules
to raise behavior alerts (intrusion, loitering, fight, fall, crowd, zone
crossing, abandoned object and more), records pre/during/post-event
evidence, and exposes everything through a FastAPI backend with a React
dashboard. Phase 8 adds PostgreSQL persistence, multi-camera pipelines,
horizontally-scalable deployment, and an offline natural-language
investigation assistant.

## Repository layout

- `src/bhairav/` - the package. Root modules are pure logic (synthetic
  scene, pipeline, sources, alert log, types, geometry, config); `backend/`
  holds the server plus the file and PostgreSQL stores; `detectors/`,
  `behavior/`, `rules/`, `pose/` and `trackers/` hold the vision chain.
- `scripts/` - runnable helpers: `serve.py` (the server), `run_demo.py`
  (headless demo), `make_test_video.py`, `fetch_models.py`, `make_cert.py`,
  `report.py`, `eval_anpr.py`.
- `dashboard/index.html` - the single-file React dashboard served by the
  backend (no build step, no node_modules).
- `config.yaml` - runtime configuration: detector, model, zones, rules,
  evidence, backend, cameras.
- `deploy/` - Docker, nginx, TLS and docker-compose for production, plus
  the deploy README.
- `tests/` - the pytest suite: unit tests run everywhere, PostgreSQL
  integration tests are gated behind `BHAIRAV_TEST_DB_URL`.
- `models/` and `output/` - downloaded weights and run artifacts
  (gitignored).

## Data flow for one camera

1. `sources.py` opens the source (scripted synthetic scene, video file,
   webcam index, or RTSP/RTMP URL) and yields frames.
2. `pipeline.py` runs one pipeline per camera: detector (YOLO or blob),
   IoU tracker, optional pose estimation, then the rules engine.
3. Rules emit `Alert` objects; `alert_log.py` persists them as JSONL and
   keeps the bounded live feed shown in the dashboard.
4. `EventRecorder` (file or PostgreSQL) records pre/during/post clips plus
   searchable metadata for every alert.
5. `backend/server.py` exposes the live WebSocket frames (LiveHub),
   evidence search, alerts, status, users and RBAC, the audit log, and the
   assistant endpoint.
6. `dashboard/index.html` renders the live wall, evidence, status,
   users and assistant views.

## Phase map (where features live)

- Phase 1: scripted synthetic scene (`synthetic.py`), blob detector,
  tracking, zone rules.
- Phase 2: YOLO detection, pose estimation and the behavior modules
  (`behavior/`), plus the rule engine (`rules/`).
- Phase 3: evidence store with pre/during/post clips, FastAPI backend,
  WebSocket live stream.
- Phase 4: React dashboard, webhook notifications, HTML run report.
- Phase 5: privacy layer (face blur, AES-256-GCM encryption at rest),
  RBAC, tamper-evident audit log, evidence expiry and pruning.
- Phase 6: TLS, rate limiting, face search, ANPR with a stolen-vehicle
  watchlist.
- Phase 7: RTSP/RTMP/webcam sources with reconnect, EasyOCR plate
  backend, Docker + nginx + TLS deployment.
- Phase 8: PostgreSQL stores (evidence, audit, users, plates),
  multi-camera pipelines with per-camera WebSocket channels, HA
  replicas behind nginx, and the offline Investigation Assistant.
- Phase 9: real-footage validation harness (`src/bhairav/eval/`),
  CI pipeline with lint + PostgreSQL integration + smoke,
  PostgreSQL backups + Prometheus metrics / health endpoints,
  appearance-based person re-identification across cameras, and
  read-only police / public dashboards with a privacy-blurred
  public live stream.

## Phase 8 in detail

### PostgreSQL mode

Every persistent store implements the same interface as its file-based
twin, so the server picks the backend from config: set `backend.db` in
`config.yaml`, or export `BHAIRAV_DB_URL`, or pass `--db-url` to
`serve.py`. Tables are created idempotently at boot and startup fails
fast with one clear error if the database is unreachable. The audit log
keeps its SHA-256 hash chain byte-identical in SQL, so a PostgreSQL
deployment can pick up where a JSONL log left off. Media is stored as
BYTEA, AES-256-GCM encrypted when `evidence.encrypt` is true.

### Multi-camera

`config.yaml` -> `cameras:` list; `serve.py` starts one pipeline thread
per camera, each with its own tracker, rules engine and recorder, so
track IDs cannot collide. LiveHub routes WebSocket frames per camera
via `?camera=`; alerts broadcast to every channel. `/api/evidence?camera=`
filters evidence and `/api/status` reports per-camera telemetry plus a
camera registry. An empty list keeps the classic single `--source` mode.

### High availability

`deploy/docker-compose.yml` runs N stateless app replicas behind nginx
with `ip_hash` for WebSocket stickiness, one shared PostgreSQL, health
checks gating the replicas, and no shared filesystem requirement.

### Investigation Assistant

`POST /api/assistant/query` parses plain English fully offline (no
external LLM): severity and rule synonyms, zone and camera names,
relative time windows, plate tokens cross-referenced against the ANPR
read log, and audit intents with actor detection. Every response carries
the parser plan so results stay explainable to a human operator.

## Phase 9 in detail

### Validation harness (`src/bhairav/eval/`)

`scripts/validate_footage.py <video>` replays real footage through the
production detector + rules engine and produces a JSON + HTML report:
frame rate, dropped-frame detection (self-calibrating from the observed
median gap), track statistics, per-rule alert tallies and confidence
threshold checks. Rules run with a `None` registry so the harness works
without a watchlist. Used as a release gate: `scripts/lint.py` /
CI run the unit tests and a smoke replay.

### CI pipeline (`.github/workflows/ci.yml`)

Lint (ruff, rules pinned in `pyproject.toml`, helper `scripts/lint.py`),
unit tests, PostgreSQL integration tests against a `postgres:16`
service container (`BHAIRAV_TEST_DB_URL`), a smoke boot of the server
(`/health`, `/api/status`, `/metrics`, `/ready`), and a
`pip install .` / import / version consistency check.

### Backups + ops telemetry

`scripts/backup_db.py` / `scripts/restore_db.py` wrap `pg_dump` /
`pg_restore` as logical dumps with per-day retention, always
re-registering the jsonb/bytea adapters the stores rely on.
`src/bhairav/backend/backups.py` exposes `BackupService` (list /
latest / prune). `src/bhairav/backend/metrics.py` is a dependency-free
Prometheus registry; `serve.py` samples it every 5 s (pipeline stats,
evidence, audit health, DB size/reachability, backup age) and serves it
at `/metrics` with an optional `BHAIRAV_METRICS_TOKEN`. `/ready` is a
live readiness probe (evidence + DB). `deploy/` adds prometheus.yml,
Grafana provisioning with a BHAIRAV dashboard, and compose services for
prometheus + grafana. The Status tab shows DB, backups and live
sparklines.

### Person re-identification (`src/bhairav/reid.py`, `backend/pg_reid.py`)

Appearance-based, no ML deps: per-person HSV spatial-pyramid descriptors
with per-part normalization, adaptive-mean gallery subjects, cosine
matching with a configurable assignment threshold, and camera trails.
`ReidService.observe()` runs inside each camera pipeline; sightings and
subjects are shared so the same person is matched across CAM-01,
CAM-02, ... The REST API (`/api/reid/*`) and dashboard Re-ID tab show
subjects, per-camera trails, and recent sightings. File store
(`ReidStore`) and PostgreSQL twin (`PostgresReidStore`) share the same
interface; `backend.reid` config tunes threshold / sighting throttle.

### Police + public read-only dashboards

- **`police` role** (`rbac.py`): live stream, alerts, evidence browse +
  clip download; no export/delete/audit/users, no management actions.
  The dashboard shows Live / Evidence / Search / Status / Re-ID and
  hides every write button (delete, rename, export, users, audit).
- **Public monitor**: set `backend.public_token` (or
  `BHAIRAV_PUBLIC_TOKEN`) and the server exposes `/api/public/info`
  plus `/api/public/stream?token=...`. The pipeline publishes a
  sanitized copy of every frame (downscaled, person heads heavily
  blurred, no tracks/poses/alerts) to a dedicated `__public__` hub
  channel - authenticated clients never receive it. The dashboard's
  `?public=<token>` URL renders a no-login, read-only, blurred
  monitor. The token is validated with constant-time comparison and
  the feed is intentionally unauthenticated-and-unaudited.

## Running and testing

- Dev, file stores, synthetic demo:
  `python scripts/serve.py` then open http://127.0.0.1:8000
  (default admin credentials live in the backend README).
- PostgreSQL mode:
  `BHAIRAV_DB_URL=postgresql://postgres@127.0.0.1:55432/bhairav python scripts/serve.py`
- Tests: `python -m pytest -q`. Set `BHAIRAV_TEST_DB_URL` to also run the
  PostgreSQL integration tests (they skip cleanly without it).
- Deploy: see `deploy/README.md` - `docker compose up -d` brings up
  nginx + N app replicas + PostgreSQL with health checks.

## Testing strategy

- Unit tests run everywhere with no heavy dependencies.
- PostgreSQL integration tests are gated behind `BHAIRAV_TEST_DB_URL` and
  skip cleanly when the database is not reachable.
- The synthetic scene is fully scripted: it guarantees every alert type
  fires within its loop, so end-to-end checks are reproducible without a
  live camera.
- One honest caveat: the demo scene uses scene-relative timestamps, so
  time-windowed assistant queries return zero results in the demo until
  real cameras (wall-clock time) are attached.

## Conventions for contributors

- Keep rules, detectors and behavior modules pure and deterministic; any
  per-track state lives on the FrameState, never on the module.
- Every persistent store has a file twin and a PostgreSQL twin with the
  same interface; new stores must follow both.
- Do not add heavy dependencies to pure-logic modules; heavy imports
  (FastAPI, ultralytics, easyocr) stay lazy or in `backend/server.py`.
- Media and secrets stay out of git; `output/`, `models/` and `*.pem`
  are ignored.
