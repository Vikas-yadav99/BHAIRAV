"""Phase 9 M4/M5 - re-identification validation on real footage.

Two complementary checks, both deterministic and dependency-light:

1. ``score_track_consistency`` - real CCTV, no ground truth. The harness
   runs the production pipeline over any clip and the AppearanceExtractor
   embeds every person-track crop. A healthy re-id model makes the same
   physical track look similar to itself across time (high same-track
   similarity) while different tracks stay separable (low cross-track
   similarity). The gap (separation) is the headline number.

2. ``run_cross_camera_experiment`` - deterministic ground truth. The same
   synthetic people are rendered in two layouts ("cameras") via
   ``variant_scenario``; each person's identity is known, so we can score
   rank-1 matching accuracy of probe embeddings against a gallery. This
   validates the *matching* side of re-id across camera views.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# make the module importable from scripts/ without an install
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from bhairav.reid import AppearanceExtractor, cosine  # noqa: E402


def _clamp_box(bbox, w, h):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def collect_track_embeddings(detector, engine, source=None, max_frames=None,
                             extractor=None):
    """Run the pipeline and embed every person-track crop per frame.

    Returns ``{track_id: [embedding, ...]}`` (embeddings may be empty for
    tracks whose crops never cleared the extractor's size floor).
    """
    from bhairav.pipeline import run_pipeline

    extractor = extractor or AppearanceExtractor()
    per_track: dict[int, list] = defaultdict(list)

    def on_frame(state, alerts):
        if state.frame is None:
            return None
        h, w = state.frame.shape[:2]
        for tr in state.tracks:
            if not tr.is_person:
                continue
            box = _clamp_box(tr.bbox, w, h)
            if box is None:
                continue
            emb = extractor.extract_from_frame(state.frame, box)
            if emb is not None:
                per_track[tr.track_id].append(emb)
        return None

    run_pipeline(detector, engine, source=source, max_frames=max_frames,
                 on_frame=on_frame)
    return {tid: list(embs) for tid, embs in per_track.items()}


def _mean(vecs):
    return list(np.mean(np.asarray(vecs, dtype=np.float32), axis=0))


def score_track_consistency(per_track, min_embeddings=4):
    """Same-track vs cross-track cosine similarity over one clip.

    Returns a dict with ``same_mean``, ``cross_mean``, ``separation``,
    ``n_tracks`` (tracks with >= min_embeddings) and ``n_embeddings``.
    """
    usable = {tid: embs for tid, embs in per_track.items()
              if len(embs) >= min_embeddings}
    ids = sorted(usable)
    if len(ids) < 2:
        return {"same_mean": 0.0, "cross_mean": 0.0, "separation": 0.0,
                "n_tracks": len(ids),
                "n_embeddings": sum(len(v) for v in usable.values()),
                "note": "need >= 2 tracks with >= %d embeddings each"
                        % min_embeddings}

    same_sims, cross_sims = [], []
    for i in range(len(ids)):
        a = usable[ids[i]]
        # within-track: first half vs second half (time separated)
        if len(a) >= 2 * min_embeddings:
            mid = len(a) // 2
            same_sims.append(cosine(_mean(a[:mid]), _mean(a[mid:])))
        # cross-track: this track's mean vs every other track's mean
        ai = _mean(a)
        for j in range(i + 1, len(ids)):
            cross_sims.append(cosine(ai, _mean(usable[ids[j]])))

    same_mean = float(np.mean(same_sims)) if same_sims else 0.0
    cross_mean = float(np.mean(cross_sims)) if cross_sims else 0.0
    return {"same_mean": round(same_mean, 3),
            "cross_mean": round(cross_mean, 3),
            "separation": round(same_mean - cross_mean, 3),
            "n_tracks": len(ids),
            "n_same_pairs": len(same_sims),
            "n_cross_pairs": len(cross_sims),
            "n_embeddings": sum(len(v) for v in usable.values())}


# ---------------------------------------------------------------------------
# Deterministic cross-camera experiment (ground truth = person pid)
# ---------------------------------------------------------------------------
def _render_person_embeddings(scenario, fps=15.0, width=1280, height=720,
                              extractor=None, min_embeddings=3):
    """Render a scene and embed each KNOWN person (pid) from its ground-truth
    bbox. Returns {pid: [embeddings]} for person actors only."""
    from bhairav.detectors.blob_detector import BlobDetector

    extractor = extractor or AppearanceExtractor()
    det = BlobDetector(scenario, fps=fps, width=width, height=height)
    per_pid: dict[int, list] = defaultdict(list)
    for state in det.stream():
        if state.frame is None:
            continue
        for pos in scenario.positions_at(state.timestamp):
            if pos.person.size != "person":
                continue
            box = det._bbox_for(pos)  # same geometry the detector uses
            emb = extractor.extract_from_frame(state.frame, box)
            if emb is not None:
                per_pid[pos.person.pid].append(emb)
    return {pid: embs for pid, embs in per_pid.items()
            if len(embs) >= min_embeddings}


def run_cross_camera_experiment(cfg, extractor=None):
    """Score rank-1 re-id accuracy with two camera views of the same people.

    Camera A renders the default scripted scene; camera B renders a shifted
    layout of the SAME people (same pids/colors, different positions).
    Each camera-B person is matched against the camera-A gallery by cosine
    similarity; ``rank1`` is the fraction of probes whose best gallery
    match is the true identity.
    """
    from bhairav.detectors.scenario import default_scenario, variant_scenario

    # accept either the full AppConfig or the synthetic sub-config
    synth = getattr(cfg, "synthetic", cfg)
    gallery = _render_person_embeddings(default_scenario(synth),
                                        extractor=extractor)
    probes = _render_person_embeddings(variant_scenario(synth),
                                       extractor=extractor)
    shared = sorted(set(gallery) & set(probes))
    if len(shared) < 2:
        return {"rank1": 0.0, "n_people": len(shared), "note": "too few people"}

    gal_means = {pid: _mean(gallery[pid]) for pid in shared}
    correct, n = 0, 0
    confusions: list[dict] = []
    for pid in shared:
        probe = _mean(probes[pid])
        scores = {g: cosine(probe, gv) for g, gv in gal_means.items()}
        best = max(scores, key=scores.get)
        n += 1
        if best == pid:
            correct += 1
        else:
            confusions.append({"true": pid, "matched": best,
                               "score": round(scores[best], 3),
                               "true_score": round(scores[pid], 3)})
    return {"rank1": round(correct / n, 3), "n_people": n,
            "correct": correct, "confusions": confusions[:10]}


def run_reid_validation(detector, engine, cfg, source=None, max_frames=None):
    """Full re-id validation: real-footage consistency + cross-camera score.

    Returns a dict ready to embed in the validation report / JSON payload.
    """
    per_track = collect_track_embeddings(detector, engine, source=source,
                                         max_frames=max_frames)
    consistency = score_track_consistency(per_track)
    xcam = run_cross_camera_experiment(cfg)
    return {"consistency": consistency, "cross_camera": xcam}


def render_reid_markdown(reid: dict) -> str:
    c = reid.get("consistency", {})
    x = reid.get("cross_camera", {})
    lines = ["", "## Re-ID validation (Phase 9)",
             f"- Track self-consistency (real footage): same-track mean "
             f"{c.get('same_mean')} vs cross-track mean {c.get('cross_mean')} "
             f"-> separation {c.get('separation')} "
             f"({c.get('n_tracks')} tracks, {c.get('n_embeddings')} embeddings)",
             f"- Cross-camera rank-1 accuracy: {x.get('rank1')} "
             f"({x.get('correct')}/{x.get('n_people')} people matched "
             f"correctly across camera views)"]
    for conf in (x.get("confusions") or []):
        lines.append(f"  - confusion: person {conf['true']} matched as "
                     f"{conf['matched']} (score {conf['score']} vs true "
                     f"{conf['true_score']})")
    return "\n".join(lines) + "\n"


def check_reid_thresholds(reid: dict, text: str | None) -> list:
    """Check re-id metrics against ``metric>=value,...`` (e.g.
    ``reid_separation>=0.15,reid_rank1>=0.8``). Mirrors the harness's
    threshold machinery; returns [{"metric", "ok", "expected", "actual"}].
    """
    from bhairav.eval.harness import parse_thresholds

    flat = {f"reid_{k}": v for k, v in reid["consistency"].items()}
    flat["reid_rank1"] = reid["cross_camera"].get("rank1", 0.0)
    rows = []
    for metric, (op, want) in sorted(parse_thresholds(text or "").items()):
        actual = flat.get(metric)
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
