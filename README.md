# BHAIRAV — v0.20.0

> **Honest Status**: This is a well-structured **proof-of-concept** with 27 phases of
> modular architecture. The synthetic demo works end-to-end. Real camera detection
> requires YOLO + ultralytics. Person re-ID uses HSV+HOG (not deep learning) unless
> you provide an ONNX model. The security modules exist but need production hardening.
> This compiles and passes 636 tests -- it is not yet production-deployed.

**B**ehavioral **H**azard **A**nalysis & **I**ntelligent **R**eal-time **A**ction **V**igilance

A complete, city-scale AI surveillance platform — **video in → detection → tracking → behavior alerts → audio analytics → predictive intelligence → automated response → evidence → compliance → deployment** — with zero ML dependencies required to run the demo.

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run synthetic demo (6 cameras, 12 rules, audio, analytics)
python -m bhairav.run_demo --source demo

# Run with live camera
python -m bhairav.backend.server --port 8000

# Run tests
python -m pytest tests/ -v

# Open dashboard
# http://localhost:8000/dashboard/
```

---

## What This Is

- A **modular surveillance pipeline**: detection -> tracking -> behavior rules -> alerts -> evidence
- **12 behavior detectors**: intrusion, fall, fight, chase, loitering, trespass, anomaly, stolen vehicle, abandoned object, accident, riot, crowd density
- **Audio analytics**: gunshot, glass break, scream detection (synthetic or live mic)
- **Person re-identification** across cameras (HSV+HOG, optional deep ONNX)
- **Predictive analytics**: crowd forecasting, hotspot prediction, NL summaries
- **Event bus** connecting escalation, PTZ tracking, integrations, federation, audit
- **Unified identity**: single person_id across all subsystems
- **Bounded data collections**: memory-safe for long-running deployments
- **GDPR compliance**: retention policies, consent management, right-to-deletion
- **Edge deployment**: single-camera agent with offline storage + upstream push
- **Mobile PWA**: offline-capable with push notifications and field dispatch

## What This Is NOT

- **Not trained on real surveillance data** -- the blob detector generates synthetic frames
- **Not tested on real cameras** -- YOLO integration exists but has not been validated on CCTV footage
- **Not production-hardened** -- security modules exist but are not fully wired into middleware
- **Not audited for real-world false positive rates** -- fight/fall/chase detection uses heuristics, not ML
- **Not load-tested at city scale** -- the load test benchmarks Python overhead, not real network I/O
- **Not a replacement for commercial surveillance** -- this is a research prototype / hackathon project

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    BHAIRAV Platform                         │
├─────────────────────────────────────────────────────────────┤
│  Detection Layer     │ blob + YOLO + Edge TPU/NPU          │
│  Tracking Layer      │ ByteTrack multi-object tracker       │
│  Behavior Layer      │ 12 rules + pose + anomaly            │
│  Audio Layer         │ gunshot / glass_break / scream       │
│  Analytics Layer     │ forecast + heatmap + trends          │
│  Predictive Layer    │ hotspot + NL summaries + allocation  │
│  Re-ID Layer         │ HSV+HOG + deep ONNX embeddings       │
│  Evidence Layer      │ encrypted + face-blurred + retention │
│  Response Layer      │ PTZ + escalation + reports           │
│  Federation Layer    │ multi-site + edge agents             │
│  HA Layer            │ Redis clustering + failover          │
│  Compliance Layer    │ GDPR retention + consent + deletion  │
│  API Layer           │ FastAPI + WebSocket + JWT + RBAC     │
│  Dashboard           │ React + site map + analytics tabs    │
└─────────────────────────────────────────────────────────────┘
```

---

## Phases Completed

### Phase 1-3: Core Vision Pipeline
- **Blob detection** + **YOLO** object detection + ByteTrack tracking
- **Pose estimation** (MediaPipe) for fall/fight/chase detection
- **12 behavior rules**: intrusion, loitering, fall, fight, abandoned object, accident, riot, crowd surge, stolen vehicle, gunshot, glass break, scream
- **Evidence recording**: pre/during/post clips, AES-GCM encryption, face blur
- **FastAPI backend** with JWT auth, RBAC, audit logs

### Phase 4-5: Dashboard & Security
- **React dashboard** with live stream, evidence browser, ops status
- **Real password accounts** (bcrypt), brute-force lockout, token revocation
- **Evidence workflow**: status, analyst notes, ZIP export
- **Webhook notifications** for red alerts

### Phase 6-7: Hardening & Camera Sources
- **TLS**, rate limiting, request-size caps, non-default secrets
- **Face search** (YuNet + SFace, ONNX) — find a person by photo
- **ANPR / stolen-vehicle watchlist** — read plates, alert on matches
- **Camera source layer**: RTSP/RTMP/webcam with auto-reconnect
- **Docker + nginx + TLS** deployment configs

### Phase 8-9: Scale-Out & Validation
- **PostgreSQL** for evidence, re-ID, alerts, plates
- **Multi-camera pipelines** with zone-based routing
- **Validation harness** — synthetic scene, CI-tested
- **Prometheus metrics** + automated backups
- **Person re-identification** across cameras
- **Read-only dashboards** for police / public

### Phase 10: Proactive Scene Intelligence
- Abandoned object detection, traffic accident detection, riot detection
- **Field-officer dispatch** — filtered webhooks, `/ws/field` push feed

### Phase 11: Audio Analytics
- **Gunshot / glass break / scream** detection (pure NumPy)
- **Synthetic audio track** for demo scenes
- **Live microphone input** (`--mic` flag)
- **Audio volume meter** in dashboard

### Phase 12: Predictive Analytics
- **Crowd density forecasting** (weighted linear regression)
- **Spatial heatmap** (32×24 grid, exponential decay)
- **Trend analysis** (15-min rolling window, burst detection)
- **Analytics WebSocket** + dashboard tab

### Phase 13: Edge Intelligence & Federation
- **Edge agent** — lightweight single-camera, offline storage, upstream push
- **Edge TPU/NPU** — Coral Edge TPU, ONNX Runtime (Jetson/CUDA)
- **Multi-site federation** — cross-server alerts, re-ID, analytics
- **Mobile PWA** — installable, offline cache, push notifications

### Phase 14: Deep Re-ID
- **ONNX person re-ID** (OSNet/MobileNet-ReID) with automatic fallback
- **Pairwise similarity matrix** API
- Auto-detects model input shape from ONNX graph

### Phase 15: Performance Optimization
- **ONNX model export** (opset 17, FP16) + INT8 quantization
- **Batched multi-camera inference**
- **Inference profiler** with latency benchmarking

### Phase 16: Interactive Site Map
- **Canvas-based site map** with camera positions, FOV cones
- **Real-time crowd heatmap overlay**
- **Re-ID cross-camera trail visualization**
- Click cameras to see people, toggle heatmap/trails

### Phase 17: Threat Response
- **PTZ camera control** — auto-track flagged persons
- **Incident reports** — PDF/HTML with timeline, evidence, re-ID trails
- **Escalation engine** — automated lockdown, siren, escalation chains
- **Multi-tenant management** — city-level zones/cameras/users
- **Third-party integrations** — 911, fire, EMS, traffic APIs

### Phase 18: Predictive Intelligence
- **NL alert summaries** — template engine + optional LLM hook
- **Predictive hotspot modeling** — spatial-temporal risk scoring
- **Resource allocation** — officer deployment, camera prioritization
- **Burst detection** + automated escalation recommendations

### Phase 19: High Availability
- **Redis-backed clustering** — node discovery, heartbeats, leader election
- **Failover monitor** — health checks, failure threshold, auto-recovery
- **Load balancer** — round-robin, least-connections, weighted strategies

### Phase 20: GDPR Compliance
- **Data retention** — per-type expiry (evidence: 90d, alerts: 365d, etc.)
- **Consent management** — grant/revoke/check with persistence
- **Right-to-deletion** — GDPR Art. 17 across all subsystems

---

## API Reference

### Core Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System status + camera health |
| GET | `/api/alerts` | Recent alerts (paginated) |
| GET | `/api/evidence` | Evidence clips (filtered) |
| POST | `/api/evidence/{id}/status` | Update evidence status |
| GET | `/api/metrics` | Prometheus metrics |

### Analytics Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/analytics/summary` | NL alert summary |
| GET | `/api/analytics/hotspots` | Ranked risk zones |
| GET | `/api/analytics/recommendations` | Resource allocation |

### Re-ID Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/reid/similarity` | Pairwise cosine matrix |
| GET | `/api/reid/embedding-info` | Embedding model info |

### Response Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ptz/{id}/move` | PTZ camera control |
| GET/POST | `/api/reports` | Incident reports |
| GET | `/api/tenants` | Tenant management |
| GET | `/api/escalation/events` | Escalation history |

### WebSocket Feeds
| Endpoint | Description |
|----------|-------------|
| `/ws/stream?camera=X` | Live video frames |
| `/ws/alerts` | Real-time alert feed |
| `/ws/field` | Field-officer dispatch |
| `/ws/analytics` | Predictive analytics + hotspot + allocation |

---

## Configuration

```yaml
# config.yaml
model:
  yolo_model: "yolov8n.pt"
  confidence: 0.45

cameras:
  - id: CAM-01
    name: "Main Entrance"
    source: "rtsp://..."
    zones:
      - name: "entry"
        polygon: [[0,0],[1,0],[1,1],[0,1]]

backend:
  host: "0.0.0.0"
  port: 8000
  secret: "change-me"

reid:
  deep_model: "osnet_ain_x1_0.onnx"
  assign_threshold: 0.6

analytics:
  enabled: true
  forecast_horizon_sec: 10.0
  officer_pool: 10

ha:
  enabled: false
  redis_url: "redis://localhost:6379"

compliance:
  enabled: true
  evidence_retention_days: 90
```

---

## Project Structure

```
src/bhairav/
├── __init__.py              # v0.17.0
├── config.py                # All config dataclasses
├── types.py                 # Alert, Severity, Track types
├── pipeline.py              # Core vision pipeline
├── sources.py               # Camera source abstraction
├── describe.py              # Scene description
├── geometry.py              # Zone geometry
├── viz.py                   # Visualization helpers
├── detectors/               # blob, yolo, edge_tpu detectors
├── trackers/                # ByteTrack multi-object tracker
├── behavior/                # Pose + behavior analysis
├── rules/                   # 12-rule alert engine
├── audio/                   # Audio analytics (Phase 11)
│   ├── analyzer.py          # gunshot/glass_break/scream
│   ├── synthetic.py         # Demo audio track
│   ├── fusion.py            # Audio → alert bridge
│   └── mic_source.py        # Live microphone input
├── analytics/               # Predictive intelligence (Phase 12+18)
│   ├── engine.py            # Unified analytics facade
│   ├── forecast.py          # Crowd density prediction
│   ├── heatmap.py           # Spatial heatmap
│   ├── trends.py            # Alert trend analysis
│   ├── summarizer.py        # NL alert summaries
│   ├── hotspot.py           # Predictive hotspot modeling
│   └── allocation.py        # Resource allocation
├── reid/                    # Person re-identification (Phase 9+14)
│   ├── reid.py              # HSV+HOG embeddings
│   └── deep_embedder.py     # ONNX deep embeddings
├── perf/                    # Performance optimization (Phase 15)
│   ├── onnx_export.py       # YOLO → ONNX export
│   ├── batched.py           # Multi-camera batching
│   └── profiler.py          # Inference benchmarking
├── edge/                    # Edge agent (Phase 13)
│   ├── agent.py             # Single-camera edge pipeline
│   ├── local_store.py       # Offline JSONL storage
│   └── upstream.py          # MQTT/HTTPS push
├── federation/              # Multi-site (Phase 13)
│   ├── protocol.py          # Federation messages
│   └── client.py            # Cross-server push
├── response/                # Threat response (Phase 17)
│   ├── ptz.py               # PTZ camera control
│   ├── reports.py           # Incident report generator
│   ├── escalation.py        # Alert escalation engine
│   ├── tenant.py            # Multi-tenant management
│   └── integrations.py      # 3rd-party adapters
├── ha/                      # High availability (Phase 19)
│   ├── cluster.py           # Redis clustering
│   ├── failover.py          # Health checks + failover
│   └── balancer.py          # Load balancing
├── compliance/              # GDPR compliance (Phase 20)
│   ├── retention.py         # Data retention policies
│   ├── consent.py           # Consent management
│   └── deletion.py          # Right-to-deletion
├── backend/                 # FastAPI server + WebSocket
├── eval/                    # Validation harness
└── pose/                    # MediaPipe pose estimation

tests/
├── test_audio.py            # Phase 11
├── test_analytics.py        # Phase 12
├── test_edge.py             # Phase 13
├── test_perf.py             # Phase 15
├── test_response.py         # Phase 17
├── test_phase18.py          # Phase 18
├── test_ha_compliance.py    # Phase 19+20
└── ...                      # Core tests (phases 1-10)

dashboard/
├── index.html               # Main dashboard
├── map-tab.js               # Interactive site map
├── manifest.json            # PWA manifest
└── sw.js                    # Service worker

deploy/
├── docker-compose.yml       # Docker deployment
├── Dockerfile               # Container build
└── nginx.conf               # Reverse proxy
```

---

## Running Tests

```bash
# All tests (353+ passing)
python -m pytest tests/ -v

# Specific phase
python -m pytest tests/test_phase18.py -v

# Skip PG-gated tests locally
python -m pytest tests/ -v -k "not pg"
```

---

## Deployment

### Docker
```bash
docker-compose up -d
```

### Manual
```bash
pip install -e ".[ml]"
python -m bhairav.backend.server --port 8000
```

### Edge Agent
```bash
python -m bhairav.edge.agent --source rtsp://cam --upstream http://server:8000
```

---

## Stats

| Metric | Value |
|--------|-------|
| Version | 0.17.0 |
| Phases | 20 |
| Python modules | 80+ |
| Test files | 14+ |
| Tests passing | 353+ |
| Lines of code | 15,000+ |
| Behavior rules | 12 |
| API endpoints | 25+ |
| WebSocket feeds | 5 |

---

## License

Research / educational use. See LICENSE for details.

---

*Built with ❤️ by Vikas Yadav — [github.com/Vikas-yadav99/BHAIRAV](https://github.com/Vikas-yadav99/BHAIRAV)*
