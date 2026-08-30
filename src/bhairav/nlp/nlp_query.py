"""Natural language query interface for searching alerts, evidence, and analytics.

Parses free-text queries like:
  "show me all fights in Zone A last week"
  "how many intrusions were detected on CAM-01"
  "what happened at 3pm yesterday"

Maps natural language to structured queries against the data stores.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Result of a natural language query."""
    query: str
    parsed: dict  # extracted intent, filters, time range
    results: list[dict]
    count: int = 0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query, "parsed": self.parsed,
            "results": self.results[:50],  # cap for display
            "count": self.count, "explanation": self.explanation,
        }


# --- keyword mappings ---------------------------------------------------

_RULE_KEYWORDS = {
    "fight": ["fight", "fights", "brawl", "altercation", "assault", "punch", "punching"],
    "fall": ["fall", "fell", "fallen", "trip", "tripped", "down"],
    "intrusion": ["intrusion", "intruder", "break-in", "break in", "trespass", "unauthorized"],
    "loitering": ["loiter", "loitering", "hanging around", "lingering"],
    "abandoned_object": ["abandoned", "object", "bag", "package", "suspicious object"],
    "accident": ["accident", "crash", "collision", "vehicle crash", "car crash"],
    "riot": ["riot", "crowd disturbance", "mob", "unrest", "protest"],
    "gunshot": ["gunshot", "gun fire", "shooting", "gun"],
    "glass_break": ["glass break", "broken glass", "window break", "smash"],
    "scream": ["scream", "screaming", "yell", "yelling", "cry"],
    "stolen_vehicle": ["stolen vehicle", "stolen car", "car theft", "vehicle theft"],
    "crowd_surge": ["crowd surge", "crowd", "stampede", "crush"],
}

_SEVERITY_KEYWORDS = {
    "red": ["critical", "red", "severe", "extreme", "urgent"],
    "orange": ["high", "orange", "serious", "important"],
    "yellow": ["moderate", "yellow", "medium", "warning"],
    "green": ["low", "green", "minor", "info"],
}


class NLPQueryEngine:
    """Natural language query engine.

    Parameters
    ----------
    alert_store : callable
        Function(filters: dict) -> list[dict] to query alerts.
    evidence_store : callable
        Function(filters: dict) -> list[dict] to query evidence.
    analytics_store : callable
        Function() -> dict to get analytics snapshot.
    """

    def __init__(self, alert_store=None, evidence_store=None,
                 analytics_store=None):
        self._alert_store = alert_store or (lambda f: [])
        self._evidence_store = evidence_store or (lambda f: [])
        self._analytics_store = analytics_store or (lambda: {})
        self._history: list[dict] = []

    def query(self, text: str) -> QueryResult:
        """Parse and execute a natural language query."""
        parsed = self._parse(text)
        results = self._execute(parsed)
        explanation = self._explain(parsed, len(results))

        result = QueryResult(
            query=text, parsed=parsed, results=results,
            count=len(results), explanation=explanation,
        )
        self._history.append(result.to_dict())
        self._history = self._history[-100:]
        return result

    def _parse(self, text: str) -> dict:
        """Extract intent, rules, severity, zones, time range from text."""
        lower = text.lower()
        parsed = {
            "intent": "search",
            "rules": [],
            "severity": None,
            "zones": [],
            "cameras": [],
            "time_range": None,
            "limit": 50,
        }

        # Intent detection
        if any(w in lower for w in ["count", "how many", "total", "number of"]):
            parsed["intent"] = "count"
        elif any(w in lower for w in ["summary", "summarize", "overview", "what happened"]):
            parsed["intent"] = "summary"
        elif any(w in lower for w in ["what", "when", "where", "who", "show"]):
            parsed["intent"] = "search"

        # Rule detection
        for rule, keywords in _RULE_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    parsed["rules"].append(rule)
                    break

        # Severity detection
        for sev, keywords in _SEVERITY_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + kw + r"\b", lower):
                    parsed["severity"] = sev
                    break

        # Zone detection
        zone_match = re.findall(r"zone\s+([a-z0-9]+)", lower)
        parsed["zones"] = [z.upper() for z in zone_match]

        # Camera detection
        cam_match = re.findall(r"(?:camera|cam)\s*([a-z0-9\-]+)", lower)
        parsed["cameras"] = [c.upper() for c in cam_match]

        # Time range
        if "today" in lower:
            now = time.time()
            parsed["time_range"] = {"start": now - 86400, "end": now}
        elif "yesterday" in lower:
            now = time.time()
            parsed["time_range"] = {"start": now - 172800, "end": now - 86400}
        elif "last week" in lower or "past week" in lower:
            now = time.time()
            parsed["time_range"] = {"start": now - 604800, "end": now}
        elif "last hour" in lower:
            now = time.time()
            parsed["time_range"] = {"start": now - 3600, "end": now}
        elif "last 24" in lower or "last 24 hours" in lower:
            now = time.time()
            parsed["time_range"] = {"start": now - 86400, "end": now}
        elif "last month" in lower:
            now = time.time()
            parsed["time_range"] = {"start": now - 2592000, "end": now}

        # Limit
        lim_match = re.search(r"(?:top|first|limit)\s*(\d+)", lower)
        if lim_match:
            parsed["limit"] = min(int(lim_match.group(1)), 200)

        return parsed

    def _execute(self, parsed: dict) -> list[dict]:
        """Execute the parsed query against data stores."""
        filters = {}
        if parsed["rules"]:
            filters["rules"] = parsed["rules"]
        if parsed["severity"]:
            filters["severity"] = parsed["severity"]
        if parsed["zones"]:
            filters["zones"] = parsed["zones"]
        if parsed["cameras"]:
            filters["cameras"] = parsed["cameras"]
        if parsed["time_range"]:
            filters["start_time"] = parsed["time_range"]["start"]
            filters["end_time"] = parsed["time_range"]["end"]

        results = []

        # Query alerts
        try:
            alerts = self._alert_store(filters)
            for a in alerts[:parsed["limit"]]:
                results.append({"type": "alert", **a})
        except Exception:
            pass

        # Query evidence
        try:
            evidence = self._evidence_store(filters)
            for e in evidence[:parsed["limit"]]:
                results.append({"type": "evidence", **e})
        except Exception:
            pass

        # Analytics summary for summary queries
        if parsed["intent"] == "summary":
            try:
                analytics = self._analytics_store()
                results.append({"type": "analytics_summary", **analytics})
            except Exception:
                pass

        return results[:parsed["limit"]]

    def _explain(self, parsed: dict, count: int) -> str:
        """Generate human-readable explanation of the query."""
        parts = []
        if parsed["intent"] == "count":
            parts.append(f"Found {count} results")
        elif parsed["intent"] == "summary":
            parts.append(f"Summary with {count} data points")
        else:
            parts.append(f"Found {count} matching results")

        if parsed["rules"]:
            parts.append(f"for rules: {', '.join(parsed['rules'])}")
        if parsed["severity"]:
            parts.append(f"with severity: {parsed['severity']}")
        if parsed["zones"]:
            parts.append(f"in zones: {', '.join(parsed['zones'])}")
        if parsed["time_range"]:
            parts.append("in the specified time range")
        return " ".join(parts) + "."

    def suggest(self, partial: str) -> list[str]:
        """Suggest completions for a partial query."""
        suggestions = []
        lower = partial.lower()
        if not lower:
            return ["show me all alerts today", "count intrusions this week",
                    "what fights happened in Zone A", "summary of last hour"]
        for phrase in [
            "show me all alerts", "count", "what happened", "summary",
            "fights in Zone A", "intrusions today", "gunshots last week",
        ]:
            if lower in phrase.lower():
                suggestions.append(phrase)
        return suggestions[:5]

    def snapshot(self) -> dict:
        return {
            "history_count": len(self._history),
            "last_queries": [h["query"] for h in self._history[-10:]],
        }
