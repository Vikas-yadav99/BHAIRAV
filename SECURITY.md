# BHAIRAV Security Posture & Pen-Test Checklist

## Dependency audit (automated)

Scan: `python -m pip_audit` (pip-audit 2.10.1, advisory DB current as of 2026-08-10)

Result against the pinned environment in `requirements.txt`:

```
No known vulnerabilities found
```

The only advisory found in the original environment was `pip 25.1.1`
(PYSEC-2026-196 / -1795 / -1796 / -2875 / -2876) - resolved by upgrading to
`pip 26.2.1`. No project package (numpy, opencv, fastapi, uvicorn, websockets,
cryptography, ultralytics, torch, mediapipe) has a known CVE at this time.

**Process:** run `pip-audit` on every CI run and after any dependency change.
Add a `pip-audit` step to your CI pipeline (see `.github/workflows/ci.yml`).

## Controls already in place

| Layer | Control |
|---|---|
| Passwords | PBKDF2-HMAC-SHA256, per-account lockout after N failures |
| AuthN | Bearer tokens, server-side revocation, expiry |
| AuthZ | RBAC: viewer / analyst / admin, enforced per-route |
| Audit | Hash-chained audit log (tamper-evident), write-only |
| At-rest | AES-GCM evidence encryption, key from `BHAIRAV_EVIDENCE_KEY`; server refuses to start unencrypted when `encrypt: true` |
| In-transit | Optional TLS via `--tls-cert/--tls-key` (see `scripts/make_cert.py`) |
| Abuse | Per-IP + per-account login rate limit, request-size caps |
| Defaults | Refuses default secret/password outside loopback |
| Supply chain | Frontend React/Babel vendored locally (no CDN), ML models SHA-256 pinned |
| Privacy | Evidence dir is outside the repo; face embeddings searchable only with a registered subject photo |

## Threat model

- **Assumed attacker:** remote, unauthenticated (script kiddie to mid-level);
  plus a lower-trust authenticated user (viewer trying to escalate).
- **Out of scope for current build:** physical compromise of the host, malware
  already running as the same OS user, and an attacker with the evidence
  encryption key.
- **Known residual exposure:** default HTTP when TLS is not configured; evidence
  should live on a local (non-synced) drive; no OS-level hardening or IDS on the
  host.

## Manual pen-test checklist (run against a fresh instance)

Setup: `python scripts/make_cert.py --out-dir certs` then start with TLS and a
non-default secret, e.g. `BHAIRAV_ADMIN_PASSWORD='Strong#Passw0rd!'`.

### 1. Authentication & session
- [ ] `POST /api/auth/login` with wrong password 5x -> 401, then lockout -> 429 even with correct password
- [ ] Lockout keyed per-account *and* per-IP (different accounts from same IP also throttle)
- [ ] Token revocation: logout, then old token -> 401 on `/api/me`
- [ ] Token expiry: tampered/expired token -> rejected
- [ ] Login timing: valid vs invalid password responses are indistinguishable (no user enumeration)
- [ ] `Authorization` header required - no token in URL/query accepted

### 2. Authorization (RBAC)
- [ ] Viewer token: `GET /api/users`, `GET /api/audit`, `POST /api/vehicles/watchlist` -> 403
- [ ] Analyst token: `GET /api/users` -> 403; watchlist + search allowed
- [ ] Admin token: everything allowed
- [ ] No role escalation via crafted payloads (e.g. `"role": "admin"` in update)

### 3. Injection & input handling
- [ ] SQL injection attempts on any query param -> no error leak, no data change
- [ ] Oversized body (with or without `Content-Length`, incl. chunked)
      -> 413, connection closed; the app enforces a 2 MB cap at the ASGI layer
- [ ] Path traversal in evidence/file endpoints (`../`, encoded `%2e%2e`) -> 400
- [ ] Malformed JSON / YAML in config inputs -> clean 4xx, no stack trace
- [ ] WebSocket messages oversized or malformed -> socket closed, server alive

### 4. Transport & encryption
- [ ] HTTP (non-TLS) on loopback works, but from another host -> refused (host guard)
- [ ] TLS cert is valid, chain complete, TLS >= 1.2 only
- [ ] Evidence files at rest: with `encrypt: true`, file bytes are not readable plaintext
- [ ] Without `BHAIRAV_EVIDENCE_KEY` and `encrypt: true` -> server refuses to start
- [ ] Wrong key -> evidence read fails cleanly, no plaintext fallback

### 5. Supply chain & deployment
- [ ] `python -m pip_audit` -> "No known vulnerabilities found"
- [ ] Dashboard loads with zero external network requests (all assets local)
- [ ] `models/` files match pinned SHA-256 (run `python scripts/fetch_models.py --verify`)
- [ ] Default credentials rejected outside loopback (startup guard message shown)

### 6. Resilience
- [ ] Kill the pipeline thread -> server still answers `/health` and status
- [ ] Watchlist add while pipeline running -> takes effect without restart
- [ ] Evidence pruning keeps dir bounded; no crash on full disk (documented behavior)

## Known gaps (accepted risk, tracked)

1. No third-party external penetration test - checklist above is the closest substitute.
2. uvicorn dev server, not behind nginx/reverse proxy with hardening (see `deploy/` for the production layout).
3. No automated SAST (e.g. bandit) in CI - add: `pip install bandit && bandit -r src/`.
4. Evidence on a cloud-synced path (OneDrive) is a data-leakage surface - use a local drive.
