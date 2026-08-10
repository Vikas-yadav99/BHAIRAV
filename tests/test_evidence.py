"""Unit tests for the Phase 3 evidence pipeline (pre/during/post + store)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from bhairav.backend.evidence import EvidenceStore, EventRecorder, PreEventBuffer
from bhairav.backend.privacy import Encryptor
from bhairav.types import Alert, FrameState, Severity


def _alert(rule="fight", ts=10.0, track=1, zone=None, sev=Severity.RED):
    return Alert(rule=rule, zone=zone, track_id=track, severity=sev,
                 message=f"{rule} detected", frame_id=int(ts * 15),
                 timestamp=ts, confidence=0.9)


def _frame(ts: float, fid: int, color: int) -> FrameState:
    img = np.full((240, 320, 3), color, np.uint8)
    return FrameState(frame_id=fid, timestamp=ts, tracks=[], frame_w=320,
                      frame_h=240, frame=img)


def test_pre_event_buffer_keeps_recent_window():
    buf = PreEventBuffer(duration_sec=5.0, fps=10.0)
    for i in range(30):
        buf.push(i * 0.1, np.zeros((240, 320, 3), np.uint8))
    frames = buf.frames_before(3.0)
    assert len(frames) == 30  # all pushed (still inside 5s window)


def test_event_recorder_captures_pre_and_during(tmp_path):
    store = EvidenceStore(tmp_path, camera="CAM-01", fps=10.0, blur_faces=False)
    rec = EventRecorder(store, pre_sec=2.0, post_sec=1.0, min_gap_sec=10.0)
    # warm up the pre-event buffer for 3s
    for i in range(30):
        rec.observe(_frame(i * 0.1, i, 40))
    # alert fires at t=3.0
    rec.on_alert(_alert(ts=3.0))
    # more frames during the event
    for i in range(30, 40):
        rec.observe(_frame(i * 0.1, i, 90))
    done = rec.flush()
    assert len(done) == 1
    rec2 = store.get(done[0])
    assert rec2 is not None
    assert rec2.rule == "fight"
    assert rec2.frame_count >= 20  # ~2s pre + during
    assert store.snapshot_bytes(done[0]) is not None
    assert store.clip_bytes(done[0]) is not None


def test_event_extended_by_same_key_and_deduped(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False)
    rec = EventRecorder(store, pre_sec=1.0, post_sec=1.0, min_gap_sec=10.0)
    for i in range(10):
        rec.observe(_frame(i * 0.1, i, 40))
    eid1 = rec.on_alert(_alert(ts=1.0))
    eid2 = rec.on_alert(_alert(ts=1.5))  # same key -> same event
    assert eid1 == eid2
    assert rec.active_count() == 1
    rec.flush()
    assert store.list_all()[0].end_ts >= 1.5


def test_store_search_filters(tmp_path):
    store = EvidenceStore(tmp_path, camera="CAM-01", fps=10.0, blur_faces=False)
    for rule, ts in [("fight", 1.0), ("fall", 2.0), ("chase", 3.0)]:
        rec = EventRecorder(store, pre_sec=0.5, post_sec=0.5)
        rec.on_alert(_alert(rule=rule, ts=ts))
        rec.flush()
    assert len(store.list_all()) == 3
    assert len(store.search(rule="fight")) == 1
    assert len(store.search(severity="red")) == 3
    assert len(store.search(q="fight")) == 1
    assert len(store.search(t0=2.5)) == 1  # only chase at t=3.0
    assert len(store.search(t1=1.5)) == 1  # only fight at t=1.0


def test_store_delete_and_missing(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False)
    rec = EventRecorder(store, pre_sec=0.5, post_sec=0.5)
    rec.on_alert(_alert(ts=1.0))
    eid = rec.flush()[0]
    assert store.delete(eid) is True
    assert store.get(eid) is None
    assert store.delete("does-not-exist") is False


def test_store_expire_by_retention(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False)
    rec = EventRecorder(store, pre_sec=0.5, post_sec=0.5)
    rec.on_alert(_alert(ts=1.0))
    eid = rec.flush()[0]
    # backdate the metadata mtime beyond retention
    import os
    old = time.time() - 40 * 86400
    os.utime(tmp_path / eid / "metadata.json", (old, old))
    removed = store.expire(max_age_days=30)
    assert removed == 1
    assert store.get(eid) is None


def test_store_encrypted_at_rest(tmp_path):
    key = Encryptor.new_key()
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False,
                          encrypt=True, key=key)
    rec = EventRecorder(store, pre_sec=0.5, post_sec=0.5)
    for i in range(6):
        rec.observe(_frame(i * 0.1, i, 40))  # give the clip some frames
    rec.on_alert(_alert(ts=1.0))
    eid = rec.flush()[0]
    rec2 = store.get(eid)
    assert rec2 is not None and rec2.encrypted
    assert rec2.rule == "fight"  # metadata decryptable with the key
    # ciphertext on disk is not readable JSON
    raw = (tmp_path / eid / "metadata.enc").read_bytes()
    assert b"fight" not in raw
    # wrong key cannot read
    store2 = EvidenceStore(tmp_path, fps=10.0, blur_faces=False,
                           encrypt=True, key=b"\x00" * 32)
    assert store2.get(eid) is None
    # clip decrypts and is a valid mp4 (mp4v files start with 00 00 00 XX)
    clip = store.clip_bytes(eid)
    assert clip is not None and len(clip) > 100 and clip[:2] == b"\x00\x00"


def test_store_frames_are_blurred_when_configured(tmp_path):
    """Privacy: with blur_faces=True, the stored snapshot's head region is
    genuinely blurred vs the same region in a non-blurred run."""
    from bhairav.types import Track, Pose, Keypoint
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (240, 320, 3), np.uint8)
    state = FrameState(frame_id=0, timestamp=0.0, frame_w=320, frame_h=240,
                       tracks=[Track(1, (140, 40, 180, 200), "person", 0.9, 0)],
                       poses=[Pose(1, [Keypoint(0.5, 0.25, 0.95)]
                                   + [Keypoint(0.5, 0.5, 0.95)] * 16)],
                       frame=img)

    def capture(blur):
        store = EvidenceStore(tmp_path / ("b" if blur else "r"), fps=10.0,
                              blur_faces=blur)
        rec = EventRecorder(store, pre_sec=1.0, post_sec=0.5, blur_faces=blur)
        rec.observe(state)
        rec.on_alert(_alert(ts=1.0), state=state)
        eid = rec.flush()[0]
        return cv2.imdecode(np.frombuffer(store.snapshot_bytes(eid), np.uint8),
                            cv2.IMREAD_COLOR)

    blurred = capture(True)
    raw = capture(False)
    # head region (nose at ~0.5,0.25) differs between the two runs
    head_b = blurred[60:80, 150:170].astype(float)
    head_r = raw[60:80, 150:170].astype(float)
    assert not np.array_equal(head_b, head_r)
    # blurring kills high-frequency detail: variance drops sharply in the head
    assert head_b.std() < head_r.std() * 0.6
    # and the two snapshots agree on the background (no spurious changes)
    bg_b = blurred[210:230, 20:60].astype(float)
    bg_r = raw[210:230, 20:60].astype(float)
    assert np.abs(bg_b - bg_r).mean() < 8


def test_store_rejects_path_traversal(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False)
    rec = EventRecorder(store, pre_sec=0.5, post_sec=0.5, blur_faces=False)
    rec.on_alert(_alert(ts=1.0))
    rec.flush()
    # ".." and friends must never resolve outside the evidence root
    assert store.get("..") is None
    assert store.get("../../etc") is None
    assert store.delete("..") is False
    assert store.clip_bytes("..") is None
    assert store.snapshot_bytes("..") is None
    assert store.get("not-an-id") is None
    # a real event still works
    assert len(store.list_all()) == 1


def test_event_recorder_finalize_due_after_quiet(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False)
    rec = EventRecorder(store, pre_sec=0.5, post_sec=1.0, min_gap_sec=10.0,
                        blur_faces=False)
    for i in range(10):
        rec.observe(_frame(i * 0.1, i, 40))
    rec.on_alert(_alert(ts=1.0))
    assert rec.active_count() == 1
    assert rec.finalize_due(now=1.5) == []       # still within post_sec
    done = rec.finalize_due(now=2.1)             # quiet for >1s -> closed
    assert len(done) == 1
    assert rec.active_count() == 0
    assert len(store.list_all()) == 1


def test_event_recorder_reset_allows_replay(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False)
    rec = EventRecorder(store, pre_sec=0.5, post_sec=0.5, min_gap_sec=10.0,
                        blur_faces=False)
    for i in range(10):
        rec.observe(_frame(i * 0.1, i, 40))
    rec.on_alert(_alert(ts=1.0))
    rec.flush()
    # without reset, the cooldown timestamps from pass 1 block pass 2 (scene
    # clock restarts at 0) - this is exactly the serve.py replay bug
    blocked = rec.on_alert(_alert(ts=1.0))
    assert blocked is None
    rec.reset()
    assert rec.on_alert(_alert(ts=1.0)) is not None  # pass 2 captures again
    assert len(rec.flush()) == 1
    assert len(store.list_all()) == 2


# ---- Phase 5: ops index + max_events cap -----------------------------------
_SEV_MAP = {"green": Severity.GREEN, "yellow": Severity.YELLOW,
            "orange": Severity.ORANGE, "red": Severity.RED}


def _mk_event(store, rule="loitering", sev="orange", ts=1.0, track=1):
    rec = EventRecorder(store, pre_sec=0.3, post_sec=0.3, blur_faces=False)
    rec.observe(_frame(ts - 0.2, 1, 40))
    rec.on_alert(_alert(ts=ts, rule=rule, sev=_SEV_MAP[sev], track=track))
    return rec.flush()[0]


def test_counts_incremental_index(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False)
    # index is warmed lazily on first counts()
    assert store.counts()["total"] == 0
    _mk_event(store, rule="fight", sev="red")
    _mk_event(store, rule="fall", sev="orange")
    _mk_event(store, rule="fight", sev="red")
    c = store.counts()
    assert c["total"] == 3
    assert c["by_rule"] == {"fight": 2, "fall": 1}
    assert c["by_severity"] == {"red": 2, "orange": 1}
    assert c["storage_bytes"] > 0
    # delete keeps the index accurate
    eid = store.list_all()[0].event_id
    store.delete(eid)
    c = store.counts()
    assert c["total"] == 2
    assert sum(c["by_rule"].values()) == 2


def test_counts_status_moves_bucket(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False)
    eid = _mk_event(store)
    assert store.counts()["by_status"] == {"new": 1}
    store.update_status(eid, "acknowledged", "op")
    assert store.counts()["by_status"] == {"acknowledged": 1}
    store.update_status(eid, "resolved", "an")
    assert store.counts()["by_status"] == {"resolved": 1}
    store.delete(eid)
    assert store.counts()["by_status"] == {}


def test_max_events_prunes_oldest(tmp_path):
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False, max_events=5)
    ids = []
    for i in range(10):
        ids.append(_mk_event(store, ts=float(i)))
    c = store.counts()
    assert c["total"] == 5
    rows = store.list_all()
    # the newest 5 survived; the oldest 5 were pruned
    assert {r.event_id for r in rows} == set(ids[5:])
    assert store.counts()["by_status"] == {"new": 5}


def test_prune_retries_locked_dir(tmp_path, monkeypatch):
    """If rmtree fails once (transient lock), the event stays in the prune
    order and disk doesn't grow forever - the next pass removes it."""
    import shutil
    store = EvidenceStore(tmp_path, fps=10.0, blur_faces=False, max_events=5)
    ids = []
    for i in range(7):
        ids.append(_mk_event(store, ts=float(i)))
    assert store.counts()["total"] == 5
    # simulate a lock: the next rmtree call does nothing (dir still exists)
    real_rmtree = shutil.rmtree
    locked = {"active": True}

    def flaky_rmtree(path, *a, **k):
        if locked["active"]:
            locked["active"] = False
            return  # pretend the delete was blocked
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)
    # one more save crosses the cap again -> prune pass hits the locked dir
    _mk_event(store, ts=8.0)
    # the delete was blocked once: the event is still there (not lost from the
    # index) so the count reflects the real on-disk state
    assert store.counts()["total"] == 6
    # the NEXT save retries the locked event and removes it
    _mk_event(store, ts=9.0)
    assert store.counts()["total"] == 5
    assert len(store.list_all()) == 5
