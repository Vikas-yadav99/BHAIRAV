"""Natural-language alert summaries for operators.

Two modes:
  1. Template engine (zero-dependency, always works)
  2. Optional local LLM hook (if transformers/llama available)

Templates turn structured alert data into plain-English sentences
like:
  "3 red-severity intrusions detected in Zone A over the last 5 minutes.
   Camera CAM-02 flagged a person loitering near the east entrance."
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class AlertSummary:
    """A natural-language summary of recent alert activity."""
    text: str
    severity: str            # highest severity in the window
    alert_count: int
    top_rules: list[str]
    top_zones: list[str]
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # 0-1, template=1.0, LLM varies

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "severity": self.severity,
            "alert_count": self.alert_count,
            "top_rules": self.top_rules,
            "top_zones": self.top_zones,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


# --- severity helpers ---------------------------------------------------

_SEV_ORDER = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
_SEV_LABEL = {
    "green": "low", "yellow": "moderate",
    "orange": "high", "red": "critical",
}
_RULE_LABELS = {
    "intrusion": "intrusion detected",
    "loitering": "loitering detected",
    "fall": "person fall detected",
    "fight": "physical altercation detected",
    "abandoned_object": "abandoned object detected",
    "accident": "traffic accident detected",
    "riot": "crowd disturbance detected",
    "gunshot": "gunshot audio detected",
    "glass_break": "glass break audio detected",
    "scream": "scream audio detected",
    "stolen_vehicle": "stolen vehicle alert",
    "crowd_surge": "crowd surge detected",
}


def _highest_severity(severities: list[str]) -> str:
    if not severities:
        return "green"
    return max(severities, key=lambda s: _SEV_ORDER.get(s, 0))


def _time_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds"
    if seconds < 3600:
        return f"{int(seconds / 60)} minutes"
    return f"{int(seconds / 3600)} hours"


class NLAlertSummarizer:
    """Template-based natural-language alert summarizer.

    Parameters
    ----------
    window_sec : float
        How far back to summarise (default 300 = 5 min).
    llm_callback : callable | None
        Optional async/sync function(prompt: str) -> str for LLM-backed
        summaries.  When None the template engine is used.
    """

    def __init__(self, window_sec: float = 300.0, llm_callback=None):
        self.window_sec = window_sec
        self._llm = llm_callback
        self._buffer: list[dict] = []

    # --- ingestion -------------------------------------------------------

    def observe(self, alert_dict: dict) -> None:
        """Ingest a serialised alert (as from Alert.to_dict())."""
        self._buffer.append(alert_dict)
        self._prune()

    def observe_batch(self, alerts: list[dict]) -> None:
        for a in alerts:
            self._buffer.append(a)
        self._prune()

    def _prune(self) -> None:
        cutoff = time.time() - self.window_sec
        self._buffer = [a for a in self._buffer
                        if a.get("timestamp", 0) >= cutoff]

    # --- summarisation ---------------------------------------------------

    def summarize(self) -> AlertSummary:
        """Generate a natural-language summary of recent alerts."""
        self._prune()
        if not self._buffer:
            return AlertSummary(
                text="No alerts in the last "
                     f"{_time_ago(self.window_sec)}. All clear.",
                severity="green",
                alert_count=0,
                top_rules=[],
                top_zones=[],
            )
        if self._llm is not None:
            return self._summarize_llm()
        return self._summarize_template()

    def _summarize_template(self) -> AlertSummary:
        alerts = self._buffer
        severities = [a.get("severity", "green") for a in alerts]
        rules = [a.get("rule", "unknown") for a in alerts]
        zones = [a.get("zone") for a in alerts if a.get("zone")]

        highest = _highest_severity(severities)
        sev_label = _SEV_LABEL.get(highest, "unknown")
        count = len(alerts)
        top_rules = [r for r, _ in Counter(rules).most_common(3)]
        top_zones = [z for z, _ in Counter(zones).most_common(3)]

        # --- build sentence ---
        parts: list[str] = []

        if count == 1:
            a = alerts[0]
            rule_desc = _RULE_LABELS.get(a.get("rule", ""), a.get("rule", "alert"))
            zone_txt = f" in {a['zone']}" if a.get("zone") else ""
            cam_txt = f" (camera {a['camera']})" if a.get("camera") else ""
            parts.append(
                f"A {sev_label}-severity {rule_desc}{zone_txt}{cam_txt}."
            )
        else:
            rule_phrases = []
            for r in top_rules:
                label = _RULE_LABELS.get(r, r)
                rule_phrases.append(label)
            rules_str = ", ".join(rule_phrases)
            zone_txt = f" in {', '.join(top_zones)}" if top_zones else ""
            parts.append(
                f"{count} {sev_label}-severity alerts detected{zone_txt} "
                f"over the last {_time_ago(self.window_sec)}."
            )
            parts.append(f"Top events: {rules_str}.")

        # burst hint
        recent = [a for a in alerts
                  if a.get("timestamp", 0) >= time.time() - 10]
        if len(recent) >= 5:
            parts.append(
                f"⚠ Burst: {len(recent)} alerts in the last 10 seconds."
            )

        text = " ".join(parts)
        return AlertSummary(
            text=text,
            severity=highest,
            alert_count=count,
            top_rules=top_rules,
            top_zones=top_zones,
        )

    def _summarize_llm(self) -> AlertSummary:
        prompt = self._build_llm_prompt()
        try:
            raw = self._llm(prompt)
            sev = _highest_severity(
                [a.get("severity", "green") for a in self._buffer]
            )
            return AlertSummary(
                text=str(raw).strip(),
                severity=sev,
                alert_count=len(self._buffer),
                top_rules=list(Counter(
                    a.get("rule", "") for a in self._buffer
                ).keys())[:3],
                top_zones=list(Counter(
                    a.get("zone", "") for a in self._buffer if a.get("zone")
                ).keys())[:3],
                confidence=0.8,
            )
        except Exception:
            return self._summarize_template()

    def _build_llm_prompt(self) -> str:
        lines = [
            "You are a security operations assistant. Summarise the "
            "following alerts in 1-2 plain-English sentences for a "
            "non-technical operator:\n"
        ]
        for a in self._buffer[-20:]:
            lines.append(
                f"- [{a.get('severity','?')}] {a.get('rule','?')} "
                f"at {a.get('zone','unknown zone')} "
                f"(camera {a.get('camera','?')})"
            )
        return "\n".join(lines)

    # --- reset ------------------------------------------------------------

    def reset(self) -> None:
        self._buffer.clear()

    def snapshot(self) -> dict:
        s = self.summarize()
        return s.to_dict()
