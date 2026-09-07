#!/usr/bin/env python3
"""BHAIRAV Stress Test — Find the breaking point.

Sends 500+ concurrent requests across all endpoints and reports:
- Max sustainable RPS (requests per second)
- Breaking point (error rate > 5%)
- Latency distribution (P50/P95/P99)
- Per-endpoint breakdown

Usage:
    python -m tests.stress_test --url http://localhost:8000 -c 100 -n 500
"""
from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import sys
import time
import urllib.request
import json
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@dataclass
class RequestResult:
    status: int
    latency_ms: float
    endpoint: str
    error: str = ""


@dataclass
class EndpointStats:
    endpoint: str
    count: int = 0
    errors: int = 0
    latencies: list = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.95)]

    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0


def make_request(base_url: str, method: str, path: str, body: dict = None) -> RequestResult:
    url = f"{base_url}{path}"
    start = time.perf_counter()
    try:
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed = (time.perf_counter() - start) * 1000
            return RequestResult(resp.status, elapsed, f"{method} {path}")
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(e.code, elapsed, f"{method} {path}")
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(0, elapsed, f"{method} {path}", str(e)[:80])


ENDPOINTS = [
    ("GET", "/health", None),
    ("GET", "/ready", None),
    ("GET", "/api/metrics", None),
    ("GET", "/api/analytics?hours=24", None),
    ("GET", "/api/analytics/patterns", None),
    ("GET", "/api/analytics/heatmap", None),
    ("GET", "/api/analytics/officers", None),
    ("GET", "/api/safety/dashboard", None),
    ("GET", "/api/safety/stats", None),
    ("GET", "/api/phone/reports", None),
    ("GET", "/api/analytics/export/json", None),
    ("GET", "/api/analytics/export/geojson", None),
    ("GET", "/health/deep", None),
    ("POST", "/api/safety/gps", {"officer_id": "STRESS", "lat": 26.85, "lng": 80.95, "speed": 30}),
    ("POST", "/api/phone/sms", {"phone": "+91-9999000000", "message": "stress test"}),
    ("POST", "/api/phone/whatsapp", {"phone": "+91-9999000000", "message": "stress test"}),
    ("POST", "/api/phone/ivr/start", {"phone": "+91-9999000000"}),
]


def run_stress_test(base_url: str, concurrent: int = 100, total: int = 500):
    # Verify server
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"Server not healthy: {resp.status}"); return
    except Exception as e:
        print(f"Cannot reach {base_url}: {e}"); return

    print(f"\n{'='*70}")
    print(f"  BHAIRAV STRESS TEST")
    print(f"  Server: {base_url}  |  Concurrent: {concurrent}  |  Total: {total}")
    print(f"{'='*70}\n")

    results = []
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = []
        sent = 0
        while sent < total:
            batch = min(concurrent, total - sent)
            for i in range(batch):
                m, p, b = ENDPOINTS[sent % len(ENDPOINTS)]
                futures.append(pool.submit(make_request, base_url, m, p, b))
                sent += 1
                if sent >= total:
                    break

        for f in concurrent.futures.as_completed(futures):
            try: results.append(f.result())
            except Exception as e: results.append(RequestResult(0, 0, "err", str(e)[:80]))

    elapsed = time.perf_counter() - start
    total_r = len(results)
    errors = sum(1 for r in results if r.status >= 500 or r.status == 0)
    success = sum(1 for r in results if 200 <= r.status < 300)
    client_err = sum(1 for r in results if 400 <= r.status < 500)
    rps = total_r / elapsed if elapsed > 0 else 0
    err_rate = errors / total_r if total_r else 0

    lats = sorted([r.latency_ms for r in results if r.latency_ms > 0])
    p50 = lats[len(lats)//2] if lats else 0
    p95 = lats[int(len(lats)*0.95)] if lats else 0
    p99 = lats[int(len(lats)*0.99)] if lats else 0
    avg = statistics.mean(lats) if lats else 0

    print(f"  RESULTS")
    print(f"  {'─'*66}")
    print(f"  Total requests:    {total_r}")
    print(f"  Duration:          {elapsed:.1f}s")
    print(f"  Throughput:        {rps:.1f} req/s")
    print(f"  Success (2xx):     {success} ({success/total_r*100:.1f}%)")
    print(f"  Client errors:     {client_err} ({client_err/total_r*100:.1f}%)")
    print(f"  Server errors:     {errors} ({errors/total_r*100:.1f}%)")
    print(f"\n  Latency:")
    print(f"    P50: {p50:.1f}ms  |  P95: {p95:.1f}ms  |  P99: {p99:.1f}ms  |  Avg: {avg:.1f}ms")

    if err_rate > 0.05:
        print(f"\n  *** BREAKING POINT *** error rate {err_rate*100:.1f}% > 5%")
    elif err_rate > 0.01:
        print(f"\n  ** DEGRADED ** error rate {err_rate*100:.1f}% > 1%")
    else:
        print(f"\n  ** HEALTHY ** error rate {err_rate*100:.1f}% < 1%")

    # Per-endpoint
    eps = {}
    for r in results:
        if r.endpoint not in eps:
            eps[r.endpoint] = EndpointStats(r.endpoint)
        eps[r.endpoint].count += 1
        if r.status >= 500 or r.status == 0: eps[r.endpoint].errors += 1
        if r.latency_ms > 0: eps[r.endpoint].latencies.append(r.latency_ms)

    print(f"\n  {'Method':8s} {'Path':40s} {'Reqs':>5s} {'Err%':>6s} {'Avg':>8s} {'P95':>8s}")
    print(f"  {'─'*8} {'─'*40} {'─'*5} {'─'*6} {'─'*8} {'─'*8}")
    for ep in sorted(eps.values(), key=lambda e: e.count, reverse=True)[:15]:
        parts = ep.endpoint.split(" ", 1)
        err_pct = f"{ep.error_rate*100:.1f}%"
        color = "\033[91m" if ep.error_rate > 0.05 else "\033[93m" if ep.error_rate > 0.01 else "\033[92m"
        print(f"  {parts[0]:8s} {parts[1]:40s} {ep.count:5d} {color}{err_pct:>6s}\033[0m {ep.avg_ms:7.1f}ms {ep.p95:7.1f}ms")

    # Latency histogram
    if lats:
        buckets = [0, 50, 100, 200, 500, 1000, 2000, 5000]
        counts = [0] * len(buckets)
        for lat in lats:
            for i in range(len(buckets)-1, -1, -1):
                if lat >= buckets[i]:
                    counts[i] += 1; break
        mx = max(counts) or 1
        print(f"\n  Latency Distribution:")
        for i in range(len(buckets)):
            label = f">{buckets[i]}ms" if i == len(buckets)-1 else f"{buckets[i]:>5d}-{buckets[i+1]}ms"
            bar = "█" * int(counts[i] / mx * 30)
            print(f"  {label:>15s} | {bar:30s} {counts[i]:>5d}")

    print(f"\n{'='*70}\n")
    return {"total": total_r, "rps": rps, "error_rate": err_rate, "p50": p50, "p95": p95, "p99": p99}


def main():
    p = argparse.ArgumentParser(description="BHAIRAV Stress Test")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("-c", "--concurrent", type=int, default=100)
    p.add_argument("-n", "--requests", type=int, default=500)
    args = p.parse_args()
    run_stress_test(args.url, args.concurrent, args.requests)


if __name__ == "__main__":
    main()
