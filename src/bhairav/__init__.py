"""BHAIRAV - Behavioral Hazard Analysis & Intelligent Real-time Action Vigilance.

Phase 1: Core Vision MVP - detection, tracking, rule-based alerts.
Phase 2: Behavior Intelligence - pose, fight/fall/chase/trespass, anomaly.
Phase 3: Evidence & Live API - FastAPI + WebSocket, pre/during/post evidence,
         privacy (face blur, encryption, RBAC, audit, expiry).
Phase 4: Browser command center - React dashboard served by the same process.
Phase 5: Real accounts & workflow - PBKDF2 login, evidence status/notes/export,
         token revocation, webhooks.
Phase 6: Hardening, face search & ANPR - TLS, rate limits, person search by
         photo, stolen-vehicle watchlist.
Phase 7: Camera sources & deployment - RTSP/RTMP/webcam with reconnect,
         EasyOCR plate backend, Docker + nginx + TLS.
Phase 8: Scale-out - PostgreSQL evidence/audit/users/plate stores, multi-camera
         pipelines with per-camera WS channels, HA replicas behind nginx, and
         the offline natural-language Investigation Assistant.
Phase 9: Validation & ops - real-footage harness, CI, PG backups + metrics,
         person re-id across cameras, police + public read-only dashboards.
Phase 10: Proactive scene intelligence - abandoned-object / accident / riot
          detection, and live field-officer dispatch.
Phase 11: Audio analytics - gunshot / glass-break / scream detection with
          live microphone input and volume meter.
Phase 12: Predictive analytics - crowd forecast, heatmap, trend analysis.
Phase 13: Edge intelligence - single-camera edge agent, Edge TPU / NPU,
          multi-site federation, mobile PWA.
Phase 14: Deep re-ID - ONNX person re-ID embeddings (OSNet/MobileNet).
Phase 15: Performance - ONNX export, batched inference, profiling.
Phase 16: Interactive site map - camera FOV cones, re-ID trails, heatmap overlay.
Phase 17: Threat response - PTZ tracking, incident reports, alert escalation,
          multi-tenant management, third-party integrations.
Phase 18: Intelligence - NL alert summaries, predictive hotspot modeling,
          automated resource allocation.
Phase 19: High availability - Redis clustering, failover, load balancing.
Phase 20: GDPR compliance - data retention, consent, right-to-deletion.
Phase 21: 3D scene visualization with Three.js.
Phase 22: Traffic analytics - vehicle counting, speed estimation, congestion.
Phase 23: Investigation timeline with case export.
Phase 24: NLP query interface for natural language search.
Phase 25: Load & stress testing framework.
Phase 26: Security audit module - input sanitization, CSRF, CORS, vulnerability scanning.
Phase 27: Disaster recovery - automated backups, failover drills, DR runbook.
"""
__version__ = "0.21.0"
