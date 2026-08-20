"""Phase 11 - fuse audio events into the BHAIRAV alert pipeline.

Converts AudioEvent objects into Alert objects that the existing
evidence, dispatch, log, and live-feed systems already consume.

The bridge also supports live mode: AudioFusionProcessor wraps an
AudioAnalyzer + SyntheticAudioTrack and feeds audio alongside the
video pipeline, returning audio alerts from each video frame tick.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..types import Alert
from .analyzer import AudioAnalyzer, AudioEvent


def audio_events_to_alerts(events: list[AudioEvent],
                           frame_id: int = 0) -> list[Alert]:
    """Convert audio events to pipeline Alerts (frame_id=0 for audio-only)."""
    alerts: list[Alert] = []
    for ev in events:
        alerts.append(Alert(
            rule=ev.rule,
            zone=None,
            track_id=None,
            severity=ev.severity,
            message=ev.message,
            frame_id=frame_id,
            timestamp=ev.timestamp,
            confidence=ev.confidence,
            details=ev.details,
        ))
    return alerts


@dataclass
class AudioFusionProcessor:
    """Wraps an AudioAnalyzer for live-stream fusion with the video pipeline.

    Call process_video_frame(frame_id, timestamp) on each video frame
    tick; it feeds the pre-generated synthetic audio chunk corresponding to
    that timestamp and returns any audio alerts detected.

    For a real deployment, replace _track with a live mic stream.
    """

    analyzer: AudioAnalyzer
    sample_rate: int = 16000
    _chunk_sec: float = 0.02  # must match analyzer.chunk_sec
    _track: object | None = None  # numpy array or None
    _pos_samples: int = 0

    def load_track(self, track) -> None:
        """Load a pre-generated PCM track (numpy float32 array)."""
        import numpy as np
        self._track = np.asarray(track, dtype=np.float32)
        self._pos_samples = 0
        self.analyzer.reset_position()

    def process_video_frame(self, frame_id: int, timestamp: float) -> list[Alert]:
        """Feed audio up to timestamp and return any audio alerts.

        For each frame, we advance the audio stream by chunk_sec
        (matching the analyzer frame rate) and collect any detections.
        This is called once per video frame; audio runs at its own rate.
        """
        if self._track is None:
            return []
        # how many samples should have been played by now
        target_samples = int(timestamp * self.sample_rate)
        events: list[AudioEvent] = []
        while self._pos_samples < target_samples:
            chunk_len = self.analyzer.chunk_samples
            chunk = self._track[self._pos_samples:self._pos_samples + chunk_len]
            if chunk.size == 0:
                break
            events.extend(self.analyzer.feed(chunk))
            self._pos_samples += chunk_len
        return audio_events_to_alerts(events, frame_id=frame_id)
