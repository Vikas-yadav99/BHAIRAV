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
         detection, and live field-officer dispatch (filtered webhook channels
         with retries + a push-only /ws/field feed).
"""
__version__ = "0.18.0"
