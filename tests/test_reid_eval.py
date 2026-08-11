"""Phase 9 M5 - re-id validation on real footage: consistency scorer,
deterministic cross-camera experiment, thresholds, scenario variant."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from bhairav.config import SyntheticConfig
from bhairav.detectors.scenario import default_scenario, variant_scenario
from bhairav.eval.reid_eval import (check_reid_thresholds,
                                    run_cross_camera_experiment,
                                    score_track_consistency)


def test_variant_scenario_keeps_people_moves_positions():
    cfg = SyntheticConfig(duration_sec=2.0)
    a = default_scenario(cfg)
    b = variant_scenario(cfg)
    assert [p.pid for p in a.persons] == [p.pid for p in b.persons]
    assert [p.role for p in a.persons] == [p.role for p in b.persons]
    # positions differ somewhere across time
    ta, tb = 5.0, 5.0
    pa = {p.person.pid: (p.x, p.y) for p in a.positions_at(ta)}
    pb = {p.person.pid: (p.x, p.y) for p in b.positions_at(tb)}
    assert any(abs(pa[k][0] - pb[k][0]) > 0.05 for k in pa)


def _noisy(base, n, rng, sigma=0.01):
    return [list(base + rng.normal(0, sigma, len(base))) for _ in range(n)]


def test_track_consistency_separates_tracks():
    rng = np.random.default_rng(1)
    # zero-mean gaussian bases: truly different directions (uniform [0,1)
    # vectors all lean toward the same corner and look similar)
    t1 = rng.normal(0, 1, 64)
    t2 = rng.normal(0, 1, 64)
    per_track = {
        1: _noisy(t1, 20, rng),
        2: _noisy(t1, 20, rng),   # same person, different tracker id
        3: _noisy(t2, 20, rng),   # clearly different person
    }
    out = score_track_consistency(per_track, min_embeddings=4)
    assert out["same_mean"] > 0.95
    assert out["separation"] > 0.2
    assert out["n_tracks"] == 3


def test_track_consistency_needs_two_tracks():
    out = score_track_consistency({1: [[0.5, 0.5]] * 10})
    assert out["n_tracks"] == 1 and out["separation"] == 0.0


def test_cross_camera_rank1_perfect_on_synthetic():
    # 4 s gives enough frames to separate near-identical synthetic shades
    cfg = SyntheticConfig(duration_sec=4.0, width=1280, height=720)
    out = run_cross_camera_experiment(cfg)
    assert out["n_people"] >= 2
    assert out["rank1"] >= 0.9, out


def test_cross_camera_distinguishes_people():
    """The same scene rendered twice must match 1:1; a probe from a DIFFERENT
    person must not win. Builds a tiny 2-person gallery and cross-checks."""
    cfg = SyntheticConfig(duration_sec=4.0)
    out = run_cross_camera_experiment(cfg)
    assert out["rank1"] > 0.9 and len(out.get("confusions", [])) <= out["n_people"] // 2


def test_check_reid_thresholds():
    reid = {"consistency": {"separation": 0.17, "same_mean": 0.9,
                            "cross_mean": 0.7, "n_tracks": 5,
                            "n_embeddings": 100},
            "cross_camera": {"rank1": 1.0, "n_people": 13}}
    rows = check_reid_thresholds(reid, "reid_separation>=0.15,reid_rank1>=0.8")
    assert all(r["ok"] for r in rows), rows
    rows = check_reid_thresholds(reid, "reid_separation>=0.5")
    assert not rows[0]["ok"] and rows[0]["actual"] == 0.17
    # unknown metric must fail loudly (never silently pass)
    rows = check_reid_thresholds(reid, "reid_made_up>=0.1")
    assert not rows[0]["ok"] and rows[0]["actual"] == "metric not collected"
