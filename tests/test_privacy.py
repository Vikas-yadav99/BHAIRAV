"""Unit tests for the Phase 3 privacy layer: face blur, encryption, expiry."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from bhairav.backend.privacy import Encryptor, FaceBlur, expire_evidence_dir
from bhairav.types import FrameState, Keypoint, Pose, Track


def _person_frame() -> tuple[np.ndarray, FrameState]:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 1280, 3), np.uint8)  # textured
    state = FrameState(frame_id=0, timestamp=0.0, frame_w=1280, frame_h=720,
                       tracks=[Track(1, (600, 100, 680, 500), "person", 0.9, 0)],
                       poses=[Pose(1, [Keypoint(0.5, 0.2, 0.95)]
                                   + [Keypoint(0.5, 0.5, 0.95)] * 16)])
    return frame, state


def test_face_blur_masks_head_region():
    frame, state = _person_frame()
    out = FaceBlur(strength=41).blur_frame(frame, state)
    # head region (around the nose at ~0.5,0.2) must differ from the source
    assert not np.array_equal(out[120:180, 610:670], frame[120:180, 610:670])
    # far-away background must be unchanged
    assert np.array_equal(out[500:560, 100:200], frame[500:560, 100:200])


def test_face_blur_disabled_strength_zero():
    frame, state = _person_frame()
    out = FaceBlur(strength=0).blur_frame(frame, state)
    assert np.array_equal(out, frame)


def test_face_blur_bbox_fallback_without_pose():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (720, 1280, 3), np.uint8)
    state = FrameState(frame_id=0, timestamp=0.0, frame_w=1280, frame_h=720,
                       tracks=[Track(1, (600, 100, 680, 500), "person", 0.9, 0)],
                       poses=[])
    out = FaceBlur(strength=41).blur_frame(frame, state)
    # bbox top band (no pose available) must be blurred
    assert not np.array_equal(out[100:160, 600:680], frame[100:160, 600:680])


def test_face_blur_ignores_vehicles():
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (720, 1280, 3), np.uint8)
    state = FrameState(frame_id=0, timestamp=0.0, frame_w=1280, frame_h=720,
                       tracks=[Track(1, (200, 400, 400, 500), "car", 0.9, 2)],
                       poses=[])
    out = FaceBlur(strength=41).blur_frame(frame, state)
    assert np.array_equal(out, frame)  # vehicles are not persons -> no blur


def test_encryptor_roundtrip_and_wrong_key():
    key = Encryptor.new_key()
    enc = Encryptor(key)
    blob = enc.encrypt_json({"rule": "fight", "severity": "red"})
    assert enc.decrypt_json(blob)["rule"] == "fight"
    # different ciphertext each time (random nonce)
    assert blob != enc.encrypt_json({"rule": "fight", "severity": "red"})
    # wrong key must fail
    try:
        Encryptor(b"\x00" * 32).decrypt(blob)
        raise AssertionError("wrong key should raise")
    except Exception:
        pass


def test_encryptor_rejects_bad_key_length():
    try:
        Encryptor(b"short")
        raise AssertionError("short key should raise")
    except ValueError:
        pass


def test_expire_removes_old_events(tmp_path):
    now = time.time()
    old = tmp_path / "old"
    new = tmp_path / "new"
    for d in (old, new):
        d.mkdir()
        (d / "metadata.json").write_text("{}")
        # backdate mtimes
        ts = now - 40 * 86400 if d == old else now - 1 * 3600
        (d / "metadata.json").touch()
        import os
        os.utime(d / "metadata.json", (ts, ts))
    removed = expire_evidence_dir(tmp_path, max_age_days=30, now=now)
    assert removed == 1
    assert not old.exists()
    assert new.exists()
