"""Phase 11 tests: audio analyzer, synthetic track, fusion bridge, integration."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from bhairav.audio.analyzer import AudioAnalyzer, AudioEvent
from bhairav.audio.synthetic import SyntheticAudioTrack, default_audio_events
from bhairav.audio.fusion import audio_events_to_alerts, AudioFusionProcessor
from bhairav.types import Severity


class TestAudioAnalyzer:

    def test_silence_produces_no_events(self):
        ana = AudioAnalyzer(frame_rate=16000, chunk_sec=0.02)
        silence = np.zeros(1600, dtype=np.float32)
        events = ana.feed(silence)
        assert events == []

    def test_gunshot_detection(self):
        ana = AudioAnalyzer(frame_rate=16000, chunk_sec=0.02, sensitivity=1.1)
        rng = np.random.default_rng(42)
        sr = 16000
        for _ in range(60):
            ana.feed(rng.normal(0, 0.001, 320).astype(np.float32))
        n = 320
        t = np.arange(n, dtype=np.float32) / sr
        onset = (0.8 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 400)).astype(np.float32)
        events = ana.feed(onset)
        assert len(events) == 0
        decay = rng.normal(0, 0.001, 320).astype(np.float32)
        events = ana.feed(decay)
        assert any(e.rule == "gunshot" for e in events)

    def test_cooldown_prevents_rapid_refire(self):
        ana = AudioAnalyzer(frame_rate=16000, cooldown_sec=5.0, sensitivity=1.0)
        rng = np.random.default_rng(99)
        sr = 16000
        for _ in range(60):
            ana.feed(rng.normal(0, 0.001, 320).astype(np.float32))
        n = 320
        t = np.arange(n, dtype=np.float32) / sr
        onset = (0.8 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 400)).astype(np.float32)
        ana.feed(onset)
        decay = rng.normal(0, 0.001, 320).astype(np.float32)
        e1 = ana.feed(decay)
        assert any(e.rule == "gunshot" for e in e1)
        ana.feed(onset)
        e2 = ana.feed(decay)
        assert not any(e.rule == "gunshot" for e in e2)

    def test_reset_position_preserves_cooldowns(self):
        ana = AudioAnalyzer(frame_rate=16000, cooldown_sec=5.0)
        ana._last_fire["gunshot"] = 10.0
        ana.reset_position()
        assert ana.pos == 0
        assert ana._last_fire["gunshot"] == 10.0


class TestSyntheticAudioTrack:

    def test_generate_returns_correct_length(self):
        synth = SyntheticAudioTrack(sample_rate=16000, seed=7)
        track = synth.generate(duration_sec=2.0)
        assert track.shape == (32000,)
        assert track.dtype == np.float32

    def test_generate_clips_to_unit_range(self):
        synth = SyntheticAudioTrack(sample_rate=16000, seed=7)
        track = synth.generate(duration_sec=5.0)
        assert float(np.max(track)) <= 1.0
        assert float(np.min(track)) >= -1.0

    def test_default_events_filter_by_duration(self):
        events = default_audio_events(duration_sec=5.0)
        assert len(events) == 1
        assert events[0].kind == "gunshot"

    def test_chunks_yield_correct_samples(self):
        synth = SyntheticAudioTrack(sample_rate=16000, seed=7)
        chunks = list(synth.chunks(duration_sec=1.0, chunk_samples=320))
        total = sum(c.size for c in chunks)
        assert total == 16000
        assert all(c.dtype == np.float32 for c in chunks)


class TestFusion:

    def test_audio_events_to_alerts(self):
        evts = [
            AudioEvent(rule="gunshot", severity=Severity.RED,
                       confidence=0.9, message="test", timestamp=2.0),
            AudioEvent(rule="scream", severity=Severity.ORANGE,
                       confidence=0.7, message="test2", timestamp=7.5),
        ]
        alerts = audio_events_to_alerts(evts, frame_id=42)
        assert len(alerts) == 2
        assert alerts[0].rule == "gunshot"
        assert alerts[0].frame_id == 42
        assert alerts[1].zone is None
        assert alerts[1].track_id is None

    def test_fusion_processor_integration(self):
        sr = 16000
        ana = AudioAnalyzer(frame_rate=sr, sensitivity=1.1, cooldown_sec=15.0)
        synth = SyntheticAudioTrack(sample_rate=sr, seed=7)
        track = synth.generate(duration_sec=4.0)
        fusion = AudioFusionProcessor(analyzer=ana, sample_rate=sr)
        fusion.load_track(track)
        all_alerts = []
        for i in range(60):
            ts = i / 15.0
            frame_alerts = fusion.process_video_frame(i, ts)
            all_alerts.extend(frame_alerts)
        rules = [a.rule for a in all_alerts]
        assert "gunshot" in rules, f"Expected gunshot in {rules}"

    def test_fusion_processor_empty_track(self):
        ana = AudioAnalyzer(frame_rate=16000)
        fusion = AudioFusionProcessor(analyzer=ana, sample_rate=16000)
        alerts = fusion.process_video_frame(0, 0.0)
        assert alerts == []


class TestFullSyntheticScene:

    def test_all_three_detectors_fire(self):
        sr = 16000
        ana = AudioAnalyzer(frame_rate=sr, sensitivity=1.1, cooldown_sec=15.0,
                            scream_min_dur_sec=0.4)
        synth = SyntheticAudioTrack(sample_rate=sr, seed=7)
        track = synth.generate(duration_sec=32.0)
        events = []
        chunk = 320
        for i in range(0, len(track), chunk):
            events.extend(ana.feed(track[i:i + chunk]))
        rules_found = {e.rule for e in events}
        assert "gunshot" in rules_found, f"Missing gunshot; found {rules_found}"
        assert "scream" in rules_found, f"Missing scream; found {rules_found}"
        assert "glass_break" in rules_found, f"Missing glass_break; found {rules_found}"



# ---------------------------------------------------------------------------
# MicSource tests (mock sounddevice)
# ---------------------------------------------------------------------------

class TestMicSource:
    """MicSource with mocked sounddevice."""

    def test_level_property(self):
        from bhairav.audio.mic_source import MicSource
        ana = AudioAnalyzer(frame_rate=16000)
        src = MicSource(analyzer=ana)
        level = src.level
        assert level["rms"] == 0.0
        assert level["peak"] == 0.0

    def test_drain_events_empty(self):
        from bhairav.audio.mic_source import MicSource
        ana = AudioAnalyzer(frame_rate=16000)
        src = MicSource(analyzer=ana)
        evts = src.drain_events()
        assert evts == []

    def test_analyzer_level_property(self):
        ana = AudioAnalyzer(frame_rate=16000)
        rng = np.random.default_rng(7)
        for _ in range(40):
            ana.feed(rng.normal(0, 0.001, 320).astype(np.float32))
        level = ana.level
        assert "rms" in level
        assert "peak" in level
        assert "floor" in level
        assert level["rms"] > 0
        assert level["peak"] > 0


# ---------------------------------------------------------------------------
# LiveHub audio_level method tests
# ---------------------------------------------------------------------------

class TestLiveHubAudioLevel:
    """Verify LiveHub.publish_audio_level exists and is callable."""

    def test_publish_audio_level_exists(self):
        import asyncio
        from bhairav.backend.server import LiveHub
        hub = LiveHub()
        hub._loop = asyncio.new_event_loop()
        # Should not raise
        hub.publish_audio_level({"rms": 0.01, "peak": 0.05, "floor": 0.001})
        hub._loop.close()
