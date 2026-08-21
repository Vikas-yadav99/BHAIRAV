"""Incident report generation (Phase 17.2).

Generates PDF and HTML incident reports with:
- Alert timeline
- Evidence clip references
- Re-ID trail data
- Analytics charts (heatmap snapshot, trend summary)
- Camera metadata
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IncidentReport:
    incident_id: str
    title: str
    severity: str
    created_at: float = field(default_factory=time.time)
    zone: str | None = None
    camera: str | None = None
    description: str = ""
    alerts: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    reid_trail: list[dict] = field(default_factory=list)
    heatmap_snapshot: dict | None = None
    trend_summary: dict | None = None
    status: str = "open"
    assigned_to: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "severity": self.severity,
            "created_at": round(self.created_at, 3),
            "zone": self.zone,
            "camera": self.camera,
            "description": self.description,
            "alerts": self.alerts,
            "evidence": self.evidence,
            "reid_trail": self.reid_trail,
            "heatmap_snapshot": self.heatmap_snapshot,
            "trend_summary": self.trend_summary,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "metadata": self.metadata,
        }


class ReportGenerator:
    """Generates incident reports in HTML and JSON formats."""

    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reports: dict[str, IncidentReport] = {}

    def create_report(self, incident_id: str, title: str, severity: str,
                      **kwargs) -> IncidentReport:
        report = IncidentReport(incident_id=incident_id, title=title,
                                severity=severity, **kwargs)
        self._reports[incident_id] = report
        return report

    def get_report(self, incident_id: str) -> IncidentReport | None:
        return self._reports.get(incident_id)

    def list_reports(self, status: str | None = None,
                     limit: int = 50) -> list[dict]:
        reports = list(self._reports.values())
        if status:
            reports = [r for r in reports if r.status == status]
        reports.sort(key=lambda r: r.created_at, reverse=True)
        return [r.to_dict() for r in reports[:limit]]

    def add_alert_to_report(self, incident_id: str, alert: dict) -> bool:
        report = self._reports.get(incident_id)
        if not report:
            return False
        report.alerts.append(alert)
        return True

    def close_report(self, incident_id: str, notes: str = "") -> bool:
        report = self._reports.get(incident_id)
        if not report:
            return False
        report.status = "closed"
        if notes:
            report.metadata["close_notes"] = notes
        return True

    def export_html(self, incident_id: str) -> str | None:
        report = self._reports.get(incident_id)
        if not report:
            return None
        html = self._render_html(report)
        path = self.output_dir / f"{incident_id}.html"
        path.write_text(html, encoding="utf-8")
        return str(path)

    def export_json(self, incident_id: str) -> str | None:
        report = self._reports.get(incident_id)
        if not report:
            return None
        path = self.output_dir / f"{incident_id}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return str(path)

    def _render_html(self, report: IncidentReport) -> str:
        sev_colors = {"red": "#ef4444", "orange": "#f97316", "yellow": "#eab308", "green": "#22c55e"}
        color = sev_colors.get(report.severity, "#64748b")
        alert_rows = ""
        for a in report.alerts:
            alert_rows += f"""<tr>
<td>{a.get('rule', '')}</td>
<td style="color:{sev_colors.get(a.get('severity', ''), '#fff')}">{a.get('severity', '')}</td>
<td>{a.get('zone', '')}</td>
<td>{a.get('message', '')}</td>
<td>{a.get('timestamp', '')}</td>
</tr>"""
        trail_html = ""
        for t in report.reid_trail:
            trail_html += f'<span class="trail-step">{t.get("camera", "")} @ {t.get("ts", "")}</span> '
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Incident {report.incident_id}</title>
<style>
body {{ font-family: system-ui; max-width: 900px; margin: 0 auto; padding: 20px; background: #0f172a; color: #e2e8f0; }}
h1 {{ color: {color}; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #1e293b; font-size: 13px; }}
th {{ color: #94a3b8; text-transform: uppercase; font-size: 11px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
.trail-step {{ background: #1e293b; padding: 4px 8px; border-radius: 4px; margin: 2px; font-size: 12px; }}
</style></head><body>
<h1>Incident Report: {report.title}</h1>
<p><span class="badge" style="background:{color}">{report.severity.upper()}</span>
ID: {report.incident_id} | Status: {report.status} | Zone: {report.zone or 'N/A'}</p>
<p>{report.description}</p>
<h2>Alert Timeline ({len(report.alerts)} alerts)</h2>
<table><thead><tr><th>Rule</th><th>Severity</th><th>Zone</th><th>Message</th><th>Time</th></tr></thead>
<tbody>{alert_rows}</tbody></table>
{"<h2>Re-ID Trail</h2><div>" + trail_html + "</div>" if trail_html else ""}
<h2>Evidence</h2>
{"<ul>" + "".join(f"<li>{e.get('path', '')}</li>" for e in report.evidence) + "</ul>" if report.evidence else "<p>No evidence attached.</p>"}
<hr><p style="color:#64748b;font-size:11px">Generated by BHAIRAV v0.15.0</p>
</body></html>"""
