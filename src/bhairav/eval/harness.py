"""Validation harness core: metrics collection, summarization, reporting.

The collector is intentionally deterministic and dependency-free (numpy +
stdlib). It observes every ``(FrameState, alerts)`` pair produced by the
pipeline and aggregates counters; ``run_validation`` drives a real
detector + rules engine over a source and returns a ``ValidationSummary``.
Thresholds can then be checked and rendered as markdown / self-contained
HTML, so the harness doubles as a regression gate.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field

from ..types import FrameState

# A track that appears for a single frame is a fragmentation artifact (the
# tracker lost the object immediately) - counted in the summary.
_HANDOVER_GAP_FRAMES = 5   # a new id appearing within this gap, overlapping
_HANDOVER_IOU = 0.4        # an ended track's last bbox, counts as a possible
                           # tracker id handover (one object, two ids).


def _iou(a, b):
    """Intersection-over-union of two pixel-space boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class MetricCollector:
    """Aggregates per-frame pipeline observations into validation metrics."""

    nominal_fps: float = 0.0
    frames: int = 0
    dropped_frames: int = 0
    detections_total: int = 0
    detections_by_class: Counter = field(default_factory=Counter)
    detections_per_frame: list = field(default_factory=list)
    person_frames: int = 0
    pose_person_frames: int = 0
    alerts_by_rule: Counter = field(default_factory=Counter)
    alerts_by_severity: Counter = field(default_factory=Counter)
    alerts_total: int = 0
    track_lengths: dict = field(default_factory=dict)
    track_first_frame: dict = field(default_factory=dict)
    track_last_frame: dict = field(default_factory=dict)
    track_last_box: dict = field(default_factory=dict)
    pose_seen: set = field(default_factory=set)
    max_simultaneous: int = 0
    possible_handovers: int = 0
    _ended: deque = field(default_factory=deque)
    _last_ts: float | None = None
    _gaps: list = field(default_factory=list)

    def observe(self, state: FrameState, alerts: list) -> None:
        """Record one pipeline frame (call once per processed frame)."""
        self.frames += 1

        # source continuity: a timestamp gap > 1.5x the typical frame period
        # means frames were dropped between reads. The period is the median of
        # observed gaps once enough frames exist (self-calibrating), falling
        # back to the detector's nominal fps before calibration so the
        # detector's reported fps need not be known in advance.
        if self._last_ts is not None and state.timestamp > self._last_ts:
            gap = state.timestamp - self._last_ts
            self._gaps.append(gap)
            # only judge once the median period is calibrated (>=8 gaps) so
            # the detector's pre-open nominal fps never causes false drops
            if len(self._gaps) >= 8:
                period = self._expected_period()
                if period > 0 and gap > 1.5 * period:
                    self.dropped_frames += 1
        self._last_ts = state.timestamp

        current = set()
        for tr in state.tracks:
            tid = tr.track_id
            current.add(tid)
            self.track_lengths[tid] = self.track_lengths.get(tid, 0) + 1
            self.track_first_frame.setdefault(tid, state.frame_id)
            self.track_last_frame[tid] = state.frame_id
            self.track_last_box[tid] = tuple(float(v) for v in tr.bbox)
            self.detections_total += 1
            self.detections_by_class[tr.label] += 1
            if tr.is_person:
                self.person_frames += 1

        # pose coverage: at least one person track has a skeleton this frame.
        if state.poses and current:
            self.pose_person_frames += 1
            self.pose_seen.update(p.track_id for p in state.poses)

        self.detections_per_frame.append(len(state.tracks))
        self.max_simultaneous = max(self.max_simultaneous, len(state.tracks))

        # handover estimate: a track ending near a fresh id's first box within
        # a few frames is one object tracked under two ids (id switch).
        for tid in list(self.track_lengths):
            if (tid not in current
                    and self.track_last_frame[tid] < state.frame_id
                    and tid not in {e[2] for e in self._ended}):
                if state.frame_id - self.track_last_frame[tid] <= _HANDOVER_GAP_FRAMES:
                    self._ended.append((self.track_last_frame[tid],
                                        self.track_last_box[tid], tid))
        for tid in current:
            if self.track_first_frame[tid] == state.frame_id:
                box = self.track_last_box[tid]
                stale = [e for e in self._ended
                         if state.frame_id - e[0] <= _HANDOVER_GAP_FRAMES
                         and _iou(e[1], box) >= _HANDOVER_IOU]
                self.possible_handovers += len(stale)
        self._ended = deque(
            [e for e in self._ended
             if state.frame_id - e[0] <= _HANDOVER_GAP_FRAMES])

        for a in alerts:
            self.alerts_total += 1
            self.alerts_by_rule[a.rule] += 1
            self.alerts_by_severity[a.severity.value] += 1

    def _expected_period(self) -> float:
        """Typical frame period: median observed gap, else 1/nominal_fps."""
        if len(self._gaps) >= 8:
            return sorted(self._gaps)[len(self._gaps) // 2]
        return 1.0 / self.nominal_fps if self.nominal_fps > 0 else 0.0

    def summary(self, wall_clock_sec: float = 0.0):
        """Finalize the collected counters into a summary dataclass."""
        tracks = len(self.track_lengths)
        one_frame = sum(1 for n in self.track_lengths.values() if n <= 1)
        return ValidationSummary(
            frames=self.frames,
            wall_clock_sec=round(wall_clock_sec, 3),
            effective_fps=round(self.frames / wall_clock_sec, 2)
            if wall_clock_sec > 0 else 0.0,
            nominal_fps=round(self.nominal_fps, 2),
            dropped_frames=self.dropped_frames,
            detections_total=self.detections_total,
            detections_per_frame_avg=round(
                self.detections_total / self.frames, 2) if self.frames else 0.0,
            detections_per_frame_max=max(self.detections_per_frame, default=0),
            detections_by_class=dict(self.detections_by_class),
            person_frames=self.person_frames,
            pose_coverage=round(self.pose_person_frames / self.person_frames, 3)
            if self.person_frames else 0.0,
            pose_person_tracks=len(self.pose_seen),
            total_tracks=tracks,
            one_frame_tracks=one_frame,
            fragmentation_rate=round(one_frame / tracks, 3) if tracks else 0.0,
            mean_track_len=round(sum(self.track_lengths.values()) / tracks, 2)
            if tracks else 0.0,
            longest_track=max(self.track_lengths.values(), default=0),
            max_simultaneous=self.max_simultaneous,
            track_churn_per_100=round(tracks / self.frames * 100, 2)
            if self.frames else 0.0,
            possible_handovers=self.possible_handovers,
            alerts_total=self.alerts_total,
            alerts_by_rule=dict(self.alerts_by_rule),
            alerts_by_severity=dict(self.alerts_by_severity),
        )


@dataclass
class ValidationSummary:
    """Validated metrics for one run over one source."""

    frames: int = 0
    wall_clock_sec: float = 0.0
    effective_fps: float = 0.0
    nominal_fps: float = 0.0
    dropped_frames: int = 0
    detections_total: int = 0
    detections_per_frame_avg: float = 0.0
    detections_per_frame_max: int = 0
    detections_by_class: dict = field(default_factory=dict)
    person_frames: int = 0
    pose_coverage: float = 0.0
    pose_person_tracks: int = 0
    total_tracks: int = 0
    one_frame_tracks: int = 0
    fragmentation_rate: float = 0.0
    mean_track_len: float = 0.0
    longest_track: int = 0
    max_simultaneous: int = 0
    track_churn_per_100: float = 0.0
    possible_handovers: int = 0
    alerts_total: int = 0
    alerts_by_rule: dict = field(default_factory=dict)
    alerts_by_severity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain JSON-safe dict for the CLI's --json output."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


def run_validation(detector, engine, source: str | None = None,
                   max_frames: int | None = None,
                   collector: MetricCollector | None = None):
    """Run the pipeline over `source` and return (summary, alerts).

    `detector` is any Detector (blob synthetic, YOLO on a real clip, or a
    stub in tests); `engine` a RulesEngine. Returns a tuple of the summary
    and the full list of alerts raised during the run.
    """
    from ..pipeline import run_pipeline

    collector = collector or MetricCollector(
        nominal_fps=getattr(detector, "fps", 0.0) or 0.0)
    t0 = time.time()

    def on_frame(state: FrameState, alerts: list):
        collector.observe(state, alerts)
        return None

    alerts = run_pipeline(detector, engine, source=source,
                          max_frames=max_frames, on_frame=on_frame)
    return collector.summary(wall_clock_sec=time.time() - t0), alerts


# ---------------------------------------------------------------------------
# Thresholds -> pass/fail
# ---------------------------------------------------------------------------
def parse_thresholds(text: str) -> dict:
    """Parse ``metric>=value,metric<value`` into {metric: (op, value)}."""
    out: dict = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        op = None
        for candidate in (">=", "<=", ">", "<", "=="):
            if candidate in part:
                op = candidate
                break
        if op is None:
            raise ValueError(f"threshold {part!r} needs an operator "
                             "(e.g. fps>=10)")
        metric, _, raw = part.partition(op)
        out[metric.strip()] = (op, float(raw))
    return out


def check_thresholds(summary: ValidationSummary,
                     thresholds: dict) -> list:
    """Check summary metrics against {metric: (op, value)}.

    Returns a list of {"metric", "ok", "expected", "actual"} rows; metrics
    missing from the summary are reported as failures so a mistyped name
    cannot silently pass.
    """
    rows = []
    values = summary.to_dict()
    # friendly aliases so CLI thresholds read naturally (fps -> effective_fps)
    values["fps"] = values.get("effective_fps")
    for metric, (op, want) in sorted(thresholds.items()):
        actual = values.get(metric)
        if actual is None:
            rows.append({"metric": metric, "ok": False,
                         "expected": f"{op} {want}",
                         "actual": "metric not collected"})
            continue
        ok = {"<=": actual <= want, ">=": actual >= want,
              "<": actual < want, ">": actual > want,
              "==": actual == want}.get(op, False)
        rows.append({"metric": metric, "ok": bool(ok),
                     "expected": f"{op} {want}", "actual": actual})
    return rows


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def render_markdown(summary: ValidationSummary, label: str | None = None,
                    checks: list = None) -> str:
    """A compact markdown validation report (CLI stdout + --json artifact)."""
    lines = [f"# BHAIRAV validation report{f' - {label}' if label else ''}",
             "",
             f"- Frames: {summary.frames} | effective FPS: "
             f"{summary.effective_fps} (nominal {summary.nominal_fps}) "
             f"| wall clock: {summary.wall_clock_sec}s",
             f"- Dropped frames: {summary.dropped_frames}",
             f"- Detections: {summary.detections_total} total "
             f"({summary.detections_per_frame_avg}/frame, "
             f"max {summary.detections_per_frame_max})",
             f"- Detections by class: {summary.detections_by_class or 'none'}",
             f"- Tracks: {summary.total_tracks} total | "
             f"mean length {summary.mean_track_len} frames | "
             f"longest {summary.longest_track} | "
             f"max simultaneous {summary.max_simultaneous}",
             f"- Track continuity: {summary.one_frame_tracks} one-frame tracks "
             f"(fragmentation {summary.fragmentation_rate}) | "
             f"churn {summary.track_churn_per_100}/100 frames | "
             f"possible id handovers {summary.possible_handovers}",
             f"- Pose coverage: {summary.pose_coverage} of person frames "
             f"({summary.pose_person_tracks} person tracks with skeletons)",
             f"- Alerts: {summary.alerts_total} total "
             f"{summary.alerts_by_rule or ''} "
             f"{summary.alerts_by_severity or ''}"]
    if checks is not None:
        lines += ["", "## Thresholds"]
        if not checks:
            lines += ["- (no thresholds configured)"]
        for row in checks:
            mark = "PASS" if row["ok"] else "FAIL"
            lines.append(f"- [{mark}] {row['metric']}: expected "
                         f"{row['expected']}, got {row['actual']}")
        passed = sum(1 for c in checks if c["ok"])
        lines += ["", f"**{passed}/{len(checks)} thresholds passed**"]
    return "\n".join(lines) + "\n"


def render_html(summary: ValidationSummary, label: str | None = None,
                checks: list = None) -> str:
    """Self-contained HTML report (no external assets) for a browser."""
    md = render_markdown(summary, label, checks)
    esc = (md.replace("&", "&amp;").replace("<", "&lt;")
           .replace(">", "&gt;"))
    body_lines = []
    for line in esc.splitlines():
        if not line:
            body_lines.append("<hr/>")
        elif line.startswith("##"):
            body_lines.append(f"<h2>{line.lstrip('#').strip()}</h2>")
        elif line.startswith("#"):
            body_lines.append(f"<h1>{line.lstrip('#').strip()}</h1>")
        else:
            body_lines.append(f"<p>{line}</p>")
    ok = all(c["ok"] for c in checks) if checks is not None else True
    badge = ("<div class='badge ok'>VALIDATION PASSED</div>" if ok
             else "<div class='badge fail'>VALIDATION FAILED</div>")
    body_html = "".join(body_lines)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BHAIRAV validation report</title>
<style>
body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 860px;
       padding: 0 1rem; color: #1c1e21; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
p {{ margin: 0.25rem 0; }}
.badge {{ display: inline-block; padding: .35rem .8rem; border-radius: 6px;
          font-weight: 600; color: #fff; margin-bottom: 1rem; }}
.ok {{ background: #1a7f37; }} .fail {{ background: #cf222e; }}
</style></head><body>{badge}{body_html}</body></html>
"""
