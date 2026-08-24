# BHAIRAV Architecture Documentation

## System Overview

BHAIRAV (Behavioral Hazard Analysis & Intelligent Real-time Action Vigilance) is a
city-scale AI surveillance platform designed for real-time threat detection, person
tracking, and automated response across hundreds of cameras.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FIELD LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Camera 1  │  │ Camera 2  │  │ Camera N  │  │ Edge TPU │       │
│  │ (RTSP)    │  │ (RTMP)    │  │ (Webcam)  │  │ (Coral)  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       └──────────────┴──────────────┴──────────────┘             │
│                          │                                       │
│  ┌───────────────────────┴────────────────────────────────┐     │
│  │              Edge Agent (per-camera)                    │     │
│  │  Detection → Rules → Local Store → Upstream Push        │     │
│  └────────────────────────┬───────────────────────────────┘     │
└───────────────────────────┼─────────────────────────────────────┘
                            │ HTTPS / MQTT
┌───────────────────────────┼─────────────────────────────────────┐
│                    PROCESSING LAYER                             │
│  ┌────────────────────────┴───────────────────────────────┐     │
│  │                 BHAIRAV Core Pipeline                    │     │
│  │                                                         │     │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐       │     │
│  │  │  Detectors  │  │  Trackers   │  │  Analyzers  │      │     │
│  │  │  - Blob     │  │  - ByteTrack│  │  - Behavior │      │     │
│  │  │  - YOLO     │  │  - Simple   │  │  - Audio    │      │     │
│  │  │  - EdgeTPU  │  │             │  │  - Traffic  │      │     │
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘       │     │
│  │         └───────────────┼────────────────┘              │     │
│  │                         ▼                               │     │
│  │  ┌──────────────────────────────────────────────┐      │     │
│  │  │              Rules Engine (12 rules)           │     │     │
│  │  │  intrusion | fall | fight | chase | loiter     │     │     │
│  │  │  trespass | anomaly | abandoned_object         │     │     │
│  │  │  accident | riot | stolen_vehicle | person_count│     │     │
│  │  └──────────────────────┬───────────────────────┘      │     │
│  │                         ▼                               │     │
│  │  ┌──────────────────────────────────────────────┐      │     │
│  │  │  Response Layer                                │     │     │
│  │  │  - PTZ Auto-Tracking  - Alert Escalation       │     │     │
│  │  │  - Incident Reports   - Dispatch              │      │     │
│  │  │  - Evidence Recording - 3rd Party Integration  │      │     │
│  │  └──────────────────────┬───────────────────────┘      │     │
│  └─────────────────────────┼───────────────────────────────┘     │
└────────────────────────────┼─────────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────────┐
│                      DATA LAYER                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ PostgreSQL  │  │   Redis     │  │  Filesystem │               │
│  │ (evidence,  │  │ (HA, cache, │  │ (backups,   │               │
│  │  alerts,    │  │  sessions)  │  │  evidence)  │               │
│  │  audit)     │  │             │  │             │               │
│  └────────────┘  └────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────────┐
│                     API LAYER                                    │
│  ┌─────────────────────────┴───────────────────────────────┐     │
│  │  FastAPI REST API + WebSocket                            │     │
│  │  JWT Auth | RBAC | Rate Limiting | CSRF                  │     │
│  └─────────────────────────┬───────────────────────────────┘     │
│                             │                                     │
│  ┌─────────────────────────┴───────────────────────────────┐     │
│  │  Analytics WebSocket (live feed + recommendations)       │     │
│  └─────────────────────────┬───────────────────────────────┘     │
└────────────────────────────┼─────────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │  Desktop    │  │  Mobile PWA │  │  3D Scene   │              │
│  │  Dashboard  │  │  (offline)  │  │  (Three.js) │              │
│  └────────────┘  └────────────┘  └────────────┘               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │  Site Map   │  │  API Docs   │  │  Investigation│            │
│  │  (Canvas)   │  │  (Swagger)  │  │  Timeline    │             │
│  └────────────┘  └────────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
config.py ─────────────────────────────────────────────────┐
    │                                                       │
    ├─► detectors/ (blob, yolo, edge_tpu)                   │
    ├─► trackers/ (bytetrack, simple)                       │
    ├─► behavior/ (fall, fight, chase, loiter, anomaly)     │
    ├─► rules/ (engine, cooldown)                           │
    ├─► reid/ (deep_embedder, service)                      │
    ├─► audio/ (analyzer, synthetic, fusion)                │
    ├─► traffic/ (analyzer)                                 │
    ├─► viz3d/ (scene3d_manager)                            │
    ├─► investigation/ (timeline, case_file)                │
    ├─► nlp/ (query_engine)                                 │
    ├─► perf/ (onnx_export, batched, profiler)              │
    ├─► response/ (ptz, reports, escalation, tenant, integrations) │
    ├─► analytics/ (engine, forecast, heatmap, summarizer, hotspot, allocation) │
    ├─► federation/ (client, protocol)                      │
    ├─► ha/ (cluster, failover, balancer)                   │
    ├─► compliance/ (retention, consent, deletion)          │
    ├─► edge/ (agent, local_store, upstream_pusher)         │
    └─► backend/ (server, pg_store, evidence, users,        │
        security, backups, disaster_recovery, hardening)    │
                                                             │
pipeline.py ◄───────────────────────────────────────────────┘
sources.py  (camera capture: RTSP, RTMP, webcam, video file)
types.py    (shared dataclasses: Detection, Person, Alert, etc.)
```

## Data Flow

1. **Capture**: Camera sources provide frames (RTSP/RTMP/webcam/video file)
2. **Detection**: Detectors find objects (blob, YOLO, Edge TPU) → `Detection[]`
3. **Tracking**: ByteTrack assigns persistent IDs → `Person[]`
4. **Behavior**: Behavior analyzers flag anomalies → `Rule[]` triggers
5. **Audio**: Audio analyzers detect events → fused into alert pipeline
6. **Response**: PTZ tracking, escalation, dispatch, evidence recording
7. **Storage**: Alerts, evidence, audit logs → PostgreSQL + filesystem
8. **Analytics**: Predictive models, NL summaries, hotspot analysis
9. **Presentation**: Dashboard, mobile PWA, 3D scene, site map

## Security Architecture

- **Authentication**: PBKDF2 password hashing, JWT tokens, optional 2FA
- **Authorization**: Role-based access control (admin, operator, viewer, field)
- **Encryption**: AES-256 evidence encryption, TLS for all transport
- **Input Validation**: SQL injection prevention, XSS filtering, path traversal blocking
- **Rate Limiting**: Per-IP fixed-window rate limiting
- **CSRF**: Double-submit cookie pattern
- **Audit**: Complete action logging with timestamp, user, action, result
- **Secrets**: Environment-based secrets management, no hardcoded credentials

## High Availability Architecture

- **Primary/Standby**: Redis-backed leader election with heartbeat
- **Failover**: Automatic failover after configurable failure threshold
- **Load Balancing**: Round-robin, least-connections, weighted strategies
- **Recovery**: RTO < 4h, RPO < 1h with automated backup verification

## Scalability Targets

| Metric | Target |
|--------|--------|
| Concurrent cameras | 200+ |
| Alerts per second | 1000+ |
| Concurrent dashboard users | 50+ |
| Edge agents | 100+ |
| Federation peers | 10+ |
| Retention | 90 days evidence, 365 days alerts |
