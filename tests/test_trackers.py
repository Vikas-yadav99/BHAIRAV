from bhairav.trackers import IoUTracker
from bhairav.types import Detection


def det(x1, y1, x2, y2, label="person"):
    return Detection((x1, y1, x2, y2), 0.9, 0, label)


def test_identity_is_stable_across_frames():
    tr = IoUTracker(min_hits=1)
    frame1 = tr.update([det(10, 10, 50, 90)])
    frame2 = tr.update([det(12, 12, 52, 92)])  # moved slightly
    assert [t.track_id for t in frame1] == [t.track_id for t in frame2]
    assert frame2[0].track_id == 1


def test_new_object_gets_new_id():
    tr = IoUTracker(min_hits=1)
    tr.update([det(10, 10, 50, 90)])
    frame2 = tr.update([det(10, 10, 50, 90), det(300, 200, 340, 280)])
    ids = sorted(t.track_id for t in frame2)
    assert ids == [1, 2]


def test_lost_track_expires():
    tr = IoUTracker(min_hits=1, max_age=5)
    tr.update([det(10, 10, 50, 90)])
    for _ in range(5):
        tr.update([])  # object disappears
    assert tr.update([]) == []


def test_tracks_returned_only_after_min_hits():
    tr = IoUTracker(min_hits=2)
    assert tr.update([det(10, 10, 50, 90)]) == []
    out = tr.update([det(11, 11, 51, 91)])
    assert len(out) == 1 and out[0].track_id == 1


def test_swap_does_not_duplicate_ids():
    tr = IoUTracker(min_hits=1)
    tr.update([det(10, 10, 50, 90), det(200, 10, 240, 90)])
    frame2 = tr.update([det(200, 10, 240, 90), det(10, 10, 50, 90)])  # order swapped
    assert len({t.track_id for t in frame2}) == 2
