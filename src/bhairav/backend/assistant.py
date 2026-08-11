"""Investigation Assistant (Phase 8 M4): plain-English queries over evidence.

``parse_query`` is a pure, offline, dependency-free parser: it turns a
sentence like "show all red fight alerts in the plaza last 7 days where
plate MH12AB1234 was seen" into a structured filter set that the evidence
search, the plate-read log and the audit trail can all apply. Everything is
keyword + regex based - no LLM, no network - so it is deterministic,
testable, and works on an air-gapped deployment.

Recognized vocabulary:
  severity  red/critical/urgent/high, orange/warning/medium, yellow/amber/low,
            green/normal
  rules     fight/fall/loiter(ing)/chase/trespass/crowd/anomaly/zone
            crossing/stolen vehicle (plate/watchlist)
  zones     any zone name known to the deployment (exact match)
  cameras   any camera id known to the deployment (exact match)
  time      "last N minutes/hours/days/weeks", "today", "yesterday"
  plates    uppercase alphanumeric tokens of 6-10 chars with at least one
            letter and one digit (e.g. MH12AB1234)
  audit     the words audit/log/who/login/accessed/changed request the
            audit trail; usernames (admin/operator/analyst/viewer or any
            registered user) become an actor filter
"""
from __future__ import annotations

import re
import time

SEVERITY_WORDS = {
    "red": "red", "critical": "red", "urgent": "red", "high": "red",
    "orange": "orange", "warning": "orange", "medium": "orange",
    "yellow": "yellow", "amber": "yellow", "low": "yellow",
    "green": "green", "normal": "green",
}

RULE_KEYWORDS = {
    "fight": "fight", "fighting": "fight", "brawl": "fight", "scuffle": "fight",
    "fall": "fall", "fell": "fall", "fallen": "fall", "collapse": "fall",
    "loiter": "loitering", "loitering": "loitering",
    "chase": "chase", "pursuit": "chase",
    "trespass": "trespass", "intrusion": "trespass", "intruder": "trespass",
    "crowd": "crowd_density", "crowded": "crowd_density", "gathering": "crowd_density",
    "anomaly": "anomaly", "anomalous": "anomaly", "unusual": "anomaly",
    "crossing": "zone_crossing",
    "stolen": "stolen_vehicle", "watchlist": "stolen_vehicle",
    "vehicle": "stolen_vehicle", "plates": "stolen_vehicle",
}

STOPWORDS = {
    "show", "me", "all", "any", "the", "a", "an", "and", "or", "of", "for",
    "from", "with", "without", "in", "on", "at", "to", "by", "is", "are",
    "was", "were", "that", "this", "these", "those", "events", "event",
    "alerts", "alert", "evidence", "happened", "happens", "occurred",
    "please", "list", "find", "get", "search", "query", "over", "about",
    "during", "between", "since", "after", "before", "last", "week",
    "month", "day", "hour", "minute", "today", "yesterday", "morning",
    "afternoon", "evening", "night", "seen", "spotted", "detected",
    "recorded", "where", "when", "who", "what", "which", "cameras",
    "camera", "zones", "zone", "area", "areas", "person", "people",
    "vehicle", "vehicles", "car", "cars", "plate", "stolen", "their",
    "been", "has", "have", "had", "did", "does", "do", "it", "its",
    "latest", "recent", "recently", "incident", "incidents", "activity",
}

PLATE_RE = re.compile(r"(?<![A-Z0-9])(?=[A-Z]*[0-9])(?=[0-9]*[A-Z])[A-Z0-9]{6,10}(?![A-Z0-9])")
TIME_LAST_RE = re.compile(r"last\s+(\d+)\s+(minute|hour|day|week)s?")
_UNITS = {"minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400}


def _start_of_day(now: float) -> float:
    lt = time.gmtime(now)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


def parse_query(query: str, ctx: dict | None = None) -> dict:
    """Parse a free-text query into structured filters (pure function).

    `ctx` (optional) supplies the deployment vocabulary:
      {"zones": ["plaza", ...], "cameras": ["CAM-01", ...],
       "users": ["admin", ...], "now": <epoch float>}
    Returns a dict with `plan` (what was understood), `search_kwargs` for
    EvidenceStore.search, `plates`, `want_audit`, `actor`, `warnings`.
    """
    ctx = ctx or {}
    zones = {str(z).lower(): str(z) for z in ctx.get("zones", [])}
    users = {str(u).lower(): str(u) for u in ctx.get("users", [])}
    now = float(ctx.get("now", time.time()))

    raw = (query or "").strip()
    low = raw.lower()
    plan: list[str] = []
    warnings: list[str] = []
    filters: dict = {"rule": None, "severity": None, "camera": None,
                     "q": None, "t0": None, "t1": None}
    plates: list[str] = []
    want_audit = any(w in low for w in ("audit", "log", "who ", "login",
                                        "accessed", "changed", "change",
                                        "signed", "password"))
    actor = None

    # severity
    for word, sev in SEVERITY_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            filters["severity"] = sev
            plan.append(f"severity = {sev}")
            break

    # rule
    rule = None
    for word, r in RULE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            if r == "stolen_vehicle" and word == "vehicle" and not any(
                    w in low for w in ("stolen", "watchlist", "plate")):
                continue  # bare "vehicle" is not a watchlist query
            rule = r
            break
    if rule:
        filters["rule"] = rule
        plan.append(f"rule = {rule}")

    # zone + camera exact matches. Zone names stay in the free-text query
    # (the stores match zone via q); cameras have their own filter and are
    # dropped from free text; usernames become the audit actor filter.
    dropped_tokens: set[str] = set()
    matched_zones: list[str] = []
    # camera ids may contain separators (CAM-01): match each id directly on
    # the raw text so "cam-02" is recognized as one id, then drop its
    # constituent tokens from the free text
    for cid in ctx.get("cameras", []):
        cl = str(cid).lower()
        if re.search(rf"(?<![a-z0-9]){re.escape(cl)}(?![a-z0-9])", low):
            if not filters["camera"]:
                filters["camera"] = str(cid)
                plan.append(f"camera = {str(cid)}")
            for tok in re.findall(r"[a-z0-9]+", cl):
                dropped_tokens.add(tok)
    # zone names may contain separators too ("server_room"): match each zone
    # directly on the raw text; the canonical name is re-added to the free
    # text below so the stores' zone match in q works
    for zid in ctx.get("zones", []):
        zl = str(zid).lower()
        if re.search(rf"(?<![a-z0-9_]){re.escape(zl)}(?![a-z0-9_])", low):
            matched_zones.append(str(zid))
            for tok in re.findall(r"[a-z0-9]+", zl):
                dropped_tokens.add(tok)
    for zname in matched_zones:
        plan.append(f"zone = {zname}")
    for token in re.findall(r"[a-z0-9]+", low):
        if token in zones:
            pass  # zone tokens already handled by the direct match above
        elif token in users and not actor:
            dropped_tokens.add(token)
            actor = users[token]
            plan.append(f"audit actor = {users[token]}")

    # time windows
    m = TIME_LAST_RE.search(low)
    if m:
        n = int(m.group(1))
        secs = n * _UNITS[m.group(2)]
        filters["t0"] = round(now - secs, 3)
        plan.append(f"time: last {n} {m.group(2)}(s)")
    if re.search(r"\btoday\b", low):
        filters["t0"] = _start_of_day(now)
        plan.append("time: since start of today")
    if re.search(r"\byesterday\b", low):
        today0 = _start_of_day(now)
        filters["t0"] = today0 - 86400
        filters["t1"] = today0
        plan.append("time: yesterday")

    # plates
    plates = [p for p in PLATE_RE.findall(raw.upper())]
    for p in plates:
        plan.append(f"plate = {p}")

    # free text: leftover words (drop stopwords, severity/rule vocab,
    # camera/user tokens and plate tokens; zone names are kept deliberately)
    excluded = (set(STOPWORDS) | set(SEVERITY_WORDS) | set(RULE_KEYWORDS)
                | dropped_tokens)
    words = [w for w in re.findall(r"[a-z0-9]+", low)
             if w not in excluded and w.upper() not in plates]
    words += [str(z).lower() for z in matched_zones]  # canonical zone names
    q = " ".join(dict.fromkeys(words))[:200] if words else None
    if q:
        plan.append(f"free text: {q}")

    if want_audit:
        plan.append("audit trail requested")
    if not any((filters["rule"], filters["severity"], filters["camera"],
                filters["t0"], filters["t1"], q, plates)):
        warnings.append("no filters recognised - returning recent evidence")

    return {"plan": plan, "filters": filters, "plates": plates,
            "want_audit": want_audit, "actor": actor, "warnings": warnings,
            "search_kwargs": {"rule": filters["rule"],
                              "severity": filters["severity"],
                              "camera": filters["camera"],
                              "q": q, "t0": filters["t0"], "t1": filters["t1"],
                              "limit": 100}}
