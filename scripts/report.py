"""Build an HTML run report from demo output (snapshots + alert log + evidence).

Usage: python scripts/report.py --alerts output/alerts.jsonl --frames output/frames \
       --evidence output/evidence --out output/report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEVERITY_COLORS = {
    "green": "#22c55e", "yellow": "#eab308", "orange": "#f97316", "red": "#ef4444",
}
SEVERITY_ORDER = ["green", "yellow", "orange", "red"]


def img_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{path.stem}">'


def evidence_section(evidence_dir: str | None, key_b64: str | None = None) -> str:
    """HTML for the Phase 3 evidence section (snapshot cards from the store)."""
    if not evidence_dir or not Path(evidence_dir).exists():
        return ""
    import base64
    from bhairav.backend.evidence import EvidenceStore
    key = base64.b64decode(key_b64) if key_b64 else None
    store = EvidenceStore(evidence_dir, fps=15.0, encrypt=key is not None, key=key)
    recs = store.list_all()
    if not recs:
        return ""
    cards = []
    for r in recs:
        snap = store.snapshot_bytes(r.event_id)
        if snap:
            img = f'<img src="data:image/jpeg;base64,{base64.b64encode(snap).decode()}" alt="{r.event_id}">'
        else:
            img = ""
        sev = r.severity
        color = SEVERITY_COLORS.get(sev, "#94a3b8")
        flags = []
        if r.blurred:
            flags.append("🫥 face-blur")
        if r.encrypted:
            flags.append("🔒 encrypted")
        cards.append(
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;'
            f'overflow:hidden;min-width:230px;max-width:260px;flex:1">'
            f'{img}'
            f'<div style="padding:12px">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
            f'<span style="background:{color}22;color:{color};border:1px solid {color}66;'
            f'border-radius:999px;padding:2px 10px;font-size:11px;font-weight:600">{sev.upper()}</span>'
            f'<span style="font-size:13px;font-weight:600">{r.rule}</span></div>'
            f'<div style="color:#94a3b8;font-size:12px">t={r.start_ts:.1f}s · {r.frame_count} frames · {r.camera}</div>'
            f'<div style="color:#64748b;font-size:11px;margin-top:4px">{html.escape(r.message[:80])}'
            f'{" ".join(flags)}</div>'
            f'</div></div>')
    return (f'<h2>Evidence (Phase 3) · {len(recs)} events</h2>'
            f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:8px">{''.join(cards)}</div>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="output/alerts.jsonl")
    ap.add_argument("--frames", default="output/frames")
    ap.add_argument("--evidence", default=None, help="evidence dir (Phase 3)")
    ap.add_argument("--evidence-key", default=None,
                    help="base64 32-byte key to decrypt encrypted evidence")
    ap.add_argument("--out", default="output/report.html")
    args = ap.parse_args()

    alerts: list[dict] = []
    if Path(args.alerts).exists():
        alerts = [json.loads(line) for line in Path(args.alerts).read_text(encoding="utf-8").splitlines() if line.strip()]

    frames = sorted(Path(args.frames).glob("*.png")) if Path(args.frames).exists() else []

    by_rule = Counter(a["rule"] for a in alerts)
    by_sev = Counter(a["severity"] for a in alerts)

    def badge(sev: str) -> str:
        color = SEVERITY_COLORS.get(sev, "#94a3b8")
        return (f'<span style="background:{color}22;color:{color};border:1px solid {color}66;'
                f'border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600">'
                f'{sev.upper()}</span>')

    chips = "".join(
        f'<div style="flex:1;min-width:140px;background:#0f172a;border:1px solid #1e293b;'
        f'border-radius:12px;padding:14px"><div style="font-size:26px;font-weight:700;color:#e2e8f0">{v}</div>'
        f'<div style="color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:1px">{k}</div></div>'
        for k, v in [("total alerts", len(alerts)), *[(r, by_rule[r]) for r in by_rule],
                     *[(s, by_sev[s]) for s in SEVERITY_ORDER if by_sev.get(s)]]
    )

    rows = "".join(
        f"<tr><td>{html.escape(a['message'])}</td><td>{badge(a['severity'])}</td>"
        f"<td>{a['rule']}</td><td>{a.get('zone') or '-'}</td><td>{a.get('track_id') or '-'}</td>"
        f"<td>{a.get('confidence', 1.0):.2f}</td>"
        f"<td style='font-variant-numeric:tabular-nums'>{a['timestamp']:.1f}s</td></tr>"
        for a in alerts
    )

    gallery = "".join(f'<figure style="margin:0 0 24px">{img_tag(p)}'
                      f'<figcaption style="color:#64748b;font-size:12px;margin-top:6px">{p.stem}</figcaption></figure>'
                      for p in frames)

    ev_html = evidence_section(args.evidence, args.evidence_key)

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>BHAIRAV Phase 3 - Run Report</title>
<style>
body {{ background:#0b1120; color:#e2e8f0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; margin:0 }}
.wrap {{ max-width:1100px; margin:0 auto; padding:40px 24px }}
h1 {{ font-size:22px; margin:0 0 4px }}
.sub {{ color:#64748b; font-size:13px; margin-bottom:28px }}
.chips {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:32px }}
table {{ width:100%; border-collapse:collapse; background:#0f172a; border-radius:12px; overflow:hidden; font-size:13px }}
th {{ text-align:left; color:#64748b; font-size:11px; text-transform:uppercase; letter-spacing:1px;
     padding:12px 14px; border-bottom:1px solid #1e293b }}
td {{ padding:10px 14px; border-bottom:1px solid #16213a }}
h2 {{ font-size:15px; margin:36px 0 14px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px }}
</style></head><body><div class="wrap">
<h1>🛰 BHAIRAV — Phase 3 Evidence & Live API</h1>
<div class="sub">End-to-end run report · {len(frames)} annotated snapshots · alerts logged to JSONL · evidence captured</div>
<div class="chips">{chips}</div>
{ev_html}
<h2>Annotated frames</h2>{gallery}
<h2>Alert timeline</h2>
<table><thead><tr><th>Message</th><th>Severity</th><th>Rule</th><th>Zone</th><th>Track</th><th>Conf</th><th>Time</th></tr></thead>
<tbody>{rows}</tbody></table>
</div></body></html>"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
