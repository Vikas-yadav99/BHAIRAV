# BHAIRAV — Phase 7: Cameras, Real Plates & Deployment

**B**ehavioral **H**azard **A**nalysis & **I**ntelligent **R**eal-time **A**ction **V**igilance

Working, end-to-end vision product — **video in → detection → tracking → pose →
behavior alerts → evidence (pre/during/post) → privacy → REST + WebSocket API →
React command center** — with zero ML dependencies required to run the demo.
Phase 1 delivered the core MVP; Phase 2 added behavior intelligence
(fight/fall/chase/trespass/anomaly + pose); Phase 3 added the product backend &
evidence (FastAPI + WebSocket live stream, pre/during/post evidence pipeline,
face blur, AES-GCM encryption at rest, RBAC, tamper-evident audit logs, evidence
expiry); Phase 4 added the browser dashboard; Phase 5 closes the security hole
in login (real password accounts, not pick-a-role), adds an evidence workflow
(status / analyst notes / ZIP export), an ops Status page, in-browser clip
playback, webhook notifications for red alerts, and hardens auth (token
revocation on lock, brute-force lockout, hash-free API responses).
Phase 6 hardening adds TLS, login rate limiting, request-size caps, forced
non-default secrets/passwords outside localhost, and vendored (offline) React;
the **face search** module (YuNet + SFace, ONNX, no new deps) lets you find a
person in evidence by uploading a photo; the **ANPR / stolen-vehicle watchlist**
reads license plates and alerts when a watched plate appears.
Phase 7 work adds a **camera source layer** (RTSP/RTMP/webcam with automatic
reconnect + backoff and feed health in `/api/status`), an **EasyOCR backend**
for reading *real* license plates (with an evaluation script), a **pinned,
CVE-scanned dependency manifest** (pip-audit clean), a CI workflow, and
**Docker + nginx + TLS deployment configs** under `deploy/`.

> Phase 0 (Python → NumPy → OpenCV → YOLO) is the learning track, done in parallel.
> When you finish it, install `ultralytics` and the exact same pipeline runs on real
> CCTV with YOLO + ByteTrack (and `mediapipe` for real pose). Everything here already
> works today on a deterministic synthetic scene.

## Quick start

```bash
pip install -r requirements.txt      # numpy, opencv, pyyaml, pytest, fastapi, uvicorn, cryptography

# Watch the synthetic scene live (boxes, skeletons, zones, behavior alerts on screen)
python scripts/run_demo.py

# Headless + capture evidence (pre/during/post clips, face-blurred)
python scripts/run_demo.py --source blob --headless --evidence output/evidence

# HTML run report (frames + alert timeline + evidence cards)
python scripts/report.py --evidence output/evidence   # -> output/report.html

# LIVE SERVER: pipeline -> WebSocket stream + REST API on :8000
python scripts/serve.py
# -> open http://localhost:8000/dashboard/  (React command center)
```

Run the tests:

```bash
python -m pytest -q                  # 163 tests (Phases 1-7)
```

### Face search, ANPR and models

```bash
# One-time: download YuNet (face detect), SFace (face embed) and
# pose_landmarker_full.task into models/ with SHA-256 verification
python scripts/fetch_models.py

# Generate a self-signed TLS cert (serves the dashboard over https://)
python scripts/make_cert.py --out-dir certs
```

Face search and plate reading work **without any extra pip packages** (ONNX
models run through OpenCV; plates default to template OCR). The real CCTV path
adds `pip install ultralytics mediapipe`, and reading *real* plates well adds
`pip install easyocr` + `rules.stolen_vehicle.backend: easyocr`.

## Phase 3-7 - API & evidence

`scripts/serve.py` runs the pipeline in a background thread and exposes:

| Endpoint | Method | Permission | Purpose |
|---|---|---|---|
| `/auth/login` | POST | public | `{"username","password"}` -> bearer token |
| `/health` | GET | public | service status + live clients |
| `/api/status` | GET | evidence_read | ops: pipeline stats, evidence counts, audit integrity |
| `/api/evidence` | GET | evidence_read | search (`?rule=&severity=&q=&t0=&t1=`) |
| `/api/evidence/export` | GET | evidence_export | analyst+: zip bundle of a search (audited) |
| `/api/evidence/{id}` | GET | evidence_read | one event's metadata |
| `/api/evidence/{id}/snapshot` | GET | evidence_read | blurred snapshot jpeg |
| `/api/evidence/{id}/clip` | GET | evidence_download | mp4 clip (audited) |
| `/api/evidence/{id}/status` | POST | evidence_download | operator+: new → acknowledged → resolved |
| `/api/evidence/{id}/notes` | POST | audit | analyst+: append investigation note |
| `/api/evidence/{id}` | DELETE | evidence_delete | delete (audited) |
| `/api/evidence/expire` | POST | evidence_delete | retention run |
| `/api/audit` | GET | audit | tamper-evident audit trail |
| `/api/alerts/recent` | GET | alerts | recent alert feed |
| `/api/search/register` | POST | evidence_read | add a person to the face gallery (photo upload + name) |
| `/api/search/subjects` | GET | evidence_read | list gallery subjects |
| `/api/search/subjects/{name}` | DELETE | audit | remove a subject (audited) |
| `/api/search/index` | POST | evidence_read | (re)build the face-index over evidence snapshots |
| `/api/search/query` | POST | evidence_read | find evidence frames matching a subject photo |
| `/api/search/status` | GET | evidence_read | index size / last build / model status |
| `/api/vehicles/watch` | GET/POST | audit | list / add a stolen-vehicle plate (watchlist) |
| `/api/vehicles/watch/{plate}` | DELETE | audit | remove from watchlist (audited) |
| `/api/vehicles/reads` | GET | evidence_read | recent plate reads (plate, vehicle track, time, confidence) |
| `/api/users` | GET/POST | users | admin: list / create accounts |
| `/api/users/{u}` | DELETE | users | admin: delete account |
| `/api/users/{u}/lock` | POST | users | admin: lock / unlock (revokes live tokens) |
| `/api/users/{u}/password` | POST | users | admin: reset password |
| `/ws/stream?token=` | WS | stream | live frames + alerts |

Roles: `viewer` < `operator` < `analyst` < `admin` (see `src/bhairav/backend/rbac.py`).

**Accounts (Phase 5).** Login now requires real credentials: users live in
`output/users.json` (or `backend.users_file` in `config.yaml`), passwords are
salted **PBKDF2-HMAC-SHA256** (200k iterations, stdlib only), verified with a
constant-time compare, and the response never contains password material.
Seeded on first run: `admin/admin123`, `operator/operator123`,
`analyst/analyst123`, `viewer/viewer123` (override the admin password with the
`BHAIRAV_ADMIN_PW` env var). Failed logins are throttled (5 strikes → 5-minute
lockout), unknown usernames cost the same PBKDF2 work (no timing-based
username enumeration), and **locking/deleting an account immediately revokes
its outstanding tokens** — no 12h grace window.

Tokens are HMAC-signed, 12h TTL. Evidence events are stored as
`<dir>/<event_id>/{metadata.json, snapshot.jpg, clip.mp4}`; with
`evidence.encrypt: true` the metadata and clip are AES-256-GCM encrypted at rest
(wrong key -> unreadable). Face blur uses the pose nose keypoint (falls back to
the bbox top band) so stored evidence is privacy-safe by default. Audit log
entries are hash-chained: any tamper or deletion is detected by `verify()`.

### Security hardening (Phase 6)

- **Encryption at rest actually works in the server now.** `serve.py` reads
  `BHAIRAV_EVIDENCE_KEY` / `--evidence-key` (base64 32-byte AES-256 key) and
  refuses to start with `evidence.encrypt: true` but no key — previously the
  server crashed or silently ran unencrypted.
- **TLS.** `python scripts/make_cert.py --out-dir certs` then
  `python scripts/serve.py --tls-cert certs/cert.pem --tls-key certs/key.pem`
  serves everything over HTTPS (tokens and evidence clips no longer travel in
  cleartext).
- **Startup guards.** Outside loopback (and without `BHAIRAV_ALLOW_DEFAULT_*`),
  the server refuses to start with the default `dev-secret-change-me` secret or
  default passwords — no more minting an admin token with known credentials.
- **Login rate limiting** (per-IP + per-account sliding window) and a request
  body size cap on top of the existing per-account brute-force lockout.
- **No CDN.** React + Babel are vendored under `dashboard/vendor/` (offline,
  no third-party supply chain at page load).
- **Pinned, CVE-scanned dependencies.** `requirements.txt` pins every package
  to the exact audited version; `python -m pip_audit` reports **no known
  vulnerabilities** (the only advisory found was `pip` itself — fixed by
  upgrading). The CI workflow (`.github/workflows/ci.yml`) re-runs the scan on
  every push. `SECURITY.md` documents the threat model and a manual
  penetration-test checklist.
- Local-only binding by default (`127.0.0.1`); pass `--host 0.0.0.0` to expose
  (then TLS + non-default credentials are required).

```bash
curl -s -X POST localhost:8000/auth/login -H "Content-Type: application/json"      -d '{"username":"admin","password":"admin123"}'
# -> {"token":"eyJ...", "role":"admin", ...}
```

Live server smoke test (what I verified end-to-end):
login -> evidence search -> viewer denied clip download (403) -> admin 200 ->
audit chain intact -> WebSocket received live frames (13 tracks + 13 skeletons)
and alerts (chase escalation, zone crossing, trespass, loitering).

## Phase 4-7 - Dashboard (`dashboard/index.html`)

A single-file **React 18 SPA** (CDN UMD + Babel, no build step) served straight
from the Phase 3 server, so `python scripts/serve.py` is a complete product:

- **Login** — real username + password (demo accounts shown on the card); the
  token is stored and sent as a bearer header on every call.
- **📡 Live** — WebSocket camera wall: the raw frame on a `<canvas>` with boxes,
  skeleton stick-figures, and per-track severity badges drawn as an overlay,
  plus HUD stats (persons/vehicles/tracks/alerts/FPS) and a live alert feed.
- **🗂 Evidence** — searchable grid (text / rule / severity) with face-blurred
  snapshots; the detail modal now has **in-browser clip playback** (operator+),
  a **status workflow** (new → acknowledged → resolved), **analyst notes**, a
  **ZIP export** button (analyst+), plus admin delete and retention run.
- **📊 Status** — ops page for any logged-in user: uptime, FPS, frames, alerts,
  live clients, evidence totals by severity/rule, storage, account count,
  webhook state, and audit-chain integrity.
- **🛡 Audit** (analyst+) — tamper-evident chain table with a verified / tampered
  banner; the tab itself is hidden for lower roles.
- **👥 Users** (admin) — create accounts (username/password/role), lock/unlock
  (revokes their live tokens), delete; never shows password material.
- **🔍 Search** (evidence_read+) — register a suspect photo (YuNet detects the
  face, SFace embeds it), then search the evidence index for matching frames:
  upload a face, get every evidence snapshot it appears in ranked by
  similarity, with per-frame confidence. The gallery is stored locally and
  deleted subjects are audited.
- **🚗 Vehicles** (audit+ for watchlist edits) — live ANPR: every plate read
  appears in a table with its vehicle track and confidence; add a plate to the
  stolen-vehicle watchlist and any subsequent read raises a red
  **STOLEN-VEHICLE** alert (rule `stolen_vehicle`), captures evidence, and
  badges the read in the UI.

Role gating is enforced **twice**: the UI hides what the role can't do (Audit
/ Users tabs, status changes, download / export buttons) *and* the API returns
401/403 server-side. Evidence snapshots and clips are fetched with the bearer
token and rendered as blob URLs (plain `<img>`/`<a>` tags can't attach auth
headers); blob URLs are revoked when the modal closes.

## Face search: find a person in evidence by photo

`src/bhairav/backend/face_search.py` implements recognition with two ONNX
models that run on plain OpenCV — no torch/tf dependency:

- **YuNet** (`face_detection_yunet_2023mar.onnx`, 232 KB) — face detection +
  the 5 landmarks used for alignment.
- **SFace** (`face_recognition_sface_2021dec.onnx`, 38 MB) — 128-d embedding.

The embedding is computed from a deterministic box-fraction alignment (the
same math `alignCrop` uses internally but without OpenCV's stateful jitter,
which we measured flipping the score between runs). Match score is cosine
similarity: same person ≈ 1.0, different person ≈ 0.1–0.2, threshold 0.5.

The pipeline: register photo → detect + embed → face index built over every
evidence snapshot (upscaled faces only, min size, one pass) → query a photo →
ranked matches with per-frame confidence. Index and gallery live under
`output/faces/` and are rebuilt on demand (`POST /api/search/index`).

## Stolen-vehicle tracking: ANPR + watchlist

`src/bhairav/backend/anpr.py`:

- **PlateReader** — detects the plate rectangle inside a vehicle track's
  bbox, binarizes, segments glyphs by ink projection, and template-matches
  against a bundled OCR glyph set (trained on the same font the renderer
  draws, so the demo is deterministic). Two backends:
  - `template` (default) — zero extra dependencies, exact on the demo scene.
  - `easyocr` (set `rules.stolen_vehicle.backend: easyocr` in `config.yaml`;
    `pip install easyocr`) — deep OCR that reads **real-world plates**.
    Evaluated on real Brazilian plates (UFPR-ALPR samples): template reads
    0/4, easyocr reads 3/4 (e.g. `L04ZI`, `BOMBEIROS`, `545`) — reproduce with
    `python scripts/fetch_real_plate_samples.py` + `python scripts/eval_anpr.py`.
    The easyocr backend gracefully falls back to template when not installed.
- **PlateRegistry / stolen_vehicle rule** — watched plates live in
  `config.yaml` (`rules.stolen_vehicle.watchlist`) and via the API; a read of
  a watched plate fires a red alert with the plate in `details`, records
  evidence, and is persisted to the reads log.

Verified end-to-end: the demo scene renders a vehicle with plate
`MH12AB1234`; the reader OCRs it with 100% accuracy across all 133 frames of
the clip, and adding it to the watchlist fires the alert live.

## What fires in the demo scene (24 s, deterministic)

| Time | Alert | Why |
|------|-------|-----|
| ~2.1 s | 🟠 CHASE #13 pursuing #12 | runner fleeing + follower aligned → escalates 🔴 ~4.1 s |
| ~3.0 s | 🟡 ANOMALY in `plaza` | 3 people vs learned baseline 0.1±0.4 (amber flag) |
| ~3.3 s | 🟠 Crowd of 4+ in `plaza` | crowd-density threshold |
| ~4.4 s | 🔴 `person #2` in `server_room` | restricted-zone crossing |
| ~5.9 s | 🔴 FIGHT #10 vs #11 | close pair, high speed, erratic wobble |
| ~6.9 s | 🟠 TRESPASS #2 in `server_room` | dwell > 2.5 s → escalates 🔴 ~9.4 s |
| ~7.7 s | 🟡 Loitering 5 s | monitored-zone dwell |
| ~12.5 s | 🟠 FALL track #9 | vy spike + bbox flattens → 🔴 when stays down |

Every alert carries a **confidence score** (0–1) and rich `details`, logged to
`output/alerts.jsonl`.

## The Phase 2 behavior layer

```
src/bhairav/
├── types.py            # + Keypoint / Pose (17 COCO kpts), Alert.confidence
├── pose/
│   ├── base.py             # PoseModel interface: tracks -> skeletons
│   ├── synthetic.py        # deterministic skeletons per actor role (offline path)
│   └── mediapipe_model.py  # real CCTV path (lazy `mediapipe` import)
├── behavior/
│   ├── kinematics.py       # MotionBuffer: velocity, speed, heading, wobble
│   ├── fall.py             # vy spike + horizontal body + stays down (orange → red)
│   ├── fight.py            # close pair, both moving, erratic wobble (red)
│   ├── chase.py            # follower pursues a fleeing runner (orange → red)
│   ├── trespass.py         # dwell inside restricted zone (orange → red)
│   └── anomaly.py          # learned-normal baseline, z-score outliers (yellow)
├── rules/               # + 5 new rules registered in the engine
├── detectors/scenario.py   # actors now carry roles: walk / stand / fall / fight / chase
└── viz.py                  # + skeleton stick-figures, behavior tags/links
```

Design notes:

- **Every classifier is rule-based and dependency-free**, so it runs today and
  is unit-testable from synthetic `FrameState`s. The real-CCTV path swaps in
  the same way Phase 1's YOLO detector does.
- **Per-step kinematics.** `MotionBuffer` computes velocity from per-sample
  steps, not net displacement — oscillatory scuffling has ~zero net motion, so
  window-averaged velocity would miss fights entirely. `mean_speed` and
  `peak_downward_vy` capture oscillation and fall spikes.
- **Erratic-motion (wobble) gate for fights.** Straight-line walkers and
  stationary bystanders can't trigger a fight even when boxes overlap — both
  parties must be genuinely moving *and* erratic.
- **Chase needs both pursuit and flight**: the follower's heading must point at
  the runner *and* the runner must be moving away. Head-on passers-by never
  fire it.
- **Fall confirms the person stays down** (grace window) before alerting, so a
  stumble-and-recover is ignored; escalation to red only when they remain down.
- **Anomaly is a learned-normal amber layer**: a per-zone baseline is frozen
  after a warmup window, then z-score outliers are flagged. It's the seam where
  an autoencoder drops in once torch lands.
- **Pose strengthens fall detection**: a horizontal torso (`shoulder-hip axis`)
  confirms a fall even when the bbox stays upright. The synthetic path renders
  per-role skeletons; MediaPipe provides real ones.

## Going live: real CCTV (after Phase 0 / `pip install ultralytics mediapipe`)

```bash
# Render the scripted scene to MP4 (handy YOLO test clip)
python scripts/make_test_video.py    # -> output/sample_scene.mp4

# Run YOLO + ByteTrack (+ MediaPipe pose) on a real clip, webcam, or the sample
python scripts/run_demo.py --source output/sample_scene.mp4
python scripts/run_demo.py --source 0
```

The YOLO path uses `ultralytics` built-in **ByteTrack** (`bytetrack.yaml`) and
detects COCO classes `[person, car, bus, truck]` (configurable in `config.yaml`
under `model.classes`). Real pose is wired through `MediaPipePoseModel` (Tasks
API — mediapipe 1.0 removed the legacy `solutions` API) and auto-enables in
`YoloDetector` when `models/pose_landmarker_full.task` is present.

### Live camera feeds (RTSP / RTMP / webcam)

Any source works with `--source`: a video file, a webcam index (`0`), or a
network stream URL. Live sources are opened with low-latency FFmpeg options
(TCP transport, no buffering), retried with **exponential backoff** when the
camera drops, and their connect/drop health is exposed in `/api/status` under
`pipeline.source`.

```bash
python scripts/serve.py --source rtsp://user:pass@10.0.0.5:554/stream1
python scripts/serve.py --source 0            # webcam
```

Source classification and reconnect logic live in `src/bhairav/sources.py`
(`classify_source`, `SourceMonitor`, `open_capture`), covered by 16 tests.

**Verified on real footage** (this repo, `output/real/vtest.avi` — a real CCTV
clip): YOLOv8n + ByteTrack tracked 592 person detections (7 simultaneous) and
240 vehicle detections with 16 stable track IDs across 120 frames; MediaPipe
pose produced skeletons on the same frames the raw landmarker does. Real
pose needs bodies roughly ≥100 px tall — low-res distant people are skipped,
as with any landmarker.

## Config highlights (`config.yaml`)

- `detector: blob | yolo | auto` — `auto` picks `blob` for the synthetic source
- `alert.cooldown_sec` — minimum gap before the same alert re-fires
- `rules.fall / fight / chase / trespass / anomaly` — per-rule thresholds
  (normalized to frame size so they transfer across resolutions)
- Severity ladder: green → yellow → orange → red, with escalation at 2× windows

## Deliverables of this milestone

- ✅ Phase 1: detection + tracking + loitering / crossing / crowd alerts
- ✅ Phase 2: fall / fight / chase / trespass classifiers + anomaly layer
- ✅ Phase 3: FastAPI + WebSocket live stream, evidence pipeline (pre/during/post),
     face blur, AES-GCM at rest, RBAC, tamper-evident audit, retention
- ✅ Phase 4: React command center (live wall, evidence search, audit chain,
     role-aware UI) served from the same `scripts/serve.py` process
- ✅ Phase 5: real user accounts (PBKDF2), evidence workflow (status / notes /
     ZIP export), ops Status page, clip playback, webhook notifications,
     token revocation on lock, brute-force lockout, hash-free API responses
- ✅ 163 passing tests (geometry, tracker, rules, classifiers, pose, privacy,
     evidence, RBAC/audit, users, server incl. dashboard route, hardening,
     face search, ANPR, camera sources)

## Phase 5 - what was added and why

The Phase 3/4 login accepted `{username, role}` and minted a token with
whatever role the client claimed — anyone could log in as admin. Phase 5 fixes
that and adds the workflow a real command center needs:

- **Real accounts** (`backend/users.py`) — PBKDF2 hashed passwords, seeded demo
  users, admin-managed lifecycle. The role is now granted by the server, never
  asserted by the client.
- **Token revocation** — locking or deleting an account kills its outstanding
  tokens immediately (checked on every request and the WS handshake), so a
  fired operator can't keep watching for 12 more hours.
- **Brute-force protection** — 5 consecutive failures lock the account for 5
  minutes; unknown usernames burn the same PBKDF2 cost so login timing can't
  be used to enumerate accounts.
- **Evidence workflow** — `state.json` sidecar (sealed media untouched):
  `new → acknowledged (operator+) → resolved (analyst+)`, plus analyst notes
  and a ZIP export (`manifest.json` + metadata + snapshot per event) that
  finally uses the previously-unused `evidence_export` permission.
- **Ops visibility** — `/api/status` + a dashboard Status tab: pipeline stats
  (frames / FPS / alerts / uptime), evidence by severity and rule, storage,
  audit-chain integrity, account and client counts.
- **Webhook notifications** — `backend.webhook_url` in `config.yaml`; red
  alerts are POSTed fire-and-forget (best-effort, never blocks the pipeline).
- **UX** — in-browser clip playback in the evidence modal and an export button.

## Phase 6 - hardening, face search, ANPR (what was added and why)

- **Hardening** (`backend/hardening.py`) — evidence-key resolution that
  actually works, TLS serving, loopback-aware startup guards, per-IP login
  rate limiting and body-size caps, vendored React. Motivation: the product
  was secure-by-design but shipped with a default secret, no TLS, and an
  encryption switch that crashed the server.
- **Face search** (`backend/face_search.py`) — the "find this person in the
  footage" capability the presentation promised but the code never had.
  Deterministic (fixed the OpenCV `alignCrop` nondeterminism that flipped
  match scores between runs), threshold 0.5, verified same-person ≈ 1.0 vs
  different-person ≈ 0.14.
- **Stolen-vehicle watchlist** (`backend/anpr.py`, rule `stolen_vehicle`) —
  plate OCR + watchlist alerts + dashboard tab. The tracker already followed
  vehicles; now you can search them by plate.
- **Real-footage verification** — installed `ultralytics` + `mediapipe` and
  ran the actual YOLO + ByteTrack + pose path on a real CCTV clip; ported the
  pose wrapper to mediapipe 1.0's Tasks API (the legacy API is gone) and
  pinned the model file's SHA-256 in `scripts/fetch_models.py`.

## Deployment (Docker + nginx + TLS)

`deploy/` contains a production layout: `Dockerfile` (non-root app image),
`docker-compose.yml` (app + nginx TLS terminator, health checks, secrets via
env), `nginx/nginx.conf` (TLS 1.2+, WS upgrade, edge rate limiting), and a
self-signed cert generator. See `deploy/README.md` for the quick start. The
app container can point straight at an RTSP camera with one command-line
change. Storage remains file-based on a named volume; PostgreSQL is the
roadmap milestone for scale-out (the compose file sketches the shape).

## Phase 8 (in progress) - PostgreSQL evidence store

The evidence store can now run on PostgreSQL instead of the file store, with
the exact same API, recorder and face-search wiring. Set one of:

```bash
export BHAIRAV_DB_URL=postgresql://bhairav:pass@localhost:5432/bhairav
# or in config.yaml:  backend.db: postgresql://...
# or:  python scripts/serve.py --db-url postgresql://...
```

- `src/bhairav/backend/pg_store.py` implements the full `EvidenceStore`
  interface (save/search/workflow/counts/expire/prune/export, media in BYTEA)
  with parameterized SQL, event-id validation and AES-256-GCM encryption of
  clips when `evidence.encrypt: true` (wrong key -> unreadable, as before).
- Schema is created automatically on first connect; the driver (`psycopg 3`)
  is optional and imported lazily - `pip install "psycopg[binary]==3.3.4"`.
- `deploy/docker-compose.yml` now runs a `db` service (postgres:16-alpine)
  with a health check and wires `DATABASE_URL` into the app; remove both to
  stay on the file store.
- Integration tests are gated behind `BHAIRAV_TEST_DB_URL` (see
  `tests/test_evidence_pg.py`); the pure-logic unit tests always run.
- The audit log now lives in Postgres too (same hash chain, `audit_log` table);
  only the user store and the plate watchlist remain file-based for now. Next:
  multi-camera and HA.

## Next: Phase 8 (scale & the wider roadmap)

Multi-camera support in the dashboard (the evidence store already tags
`camera`), swap the file store for PostgreSQL, HA (multi-replica app + shared
storage), the natural-language **Investigation Assistant**, abandoned-object /
accident / riot detection, and public/police dashboards.
