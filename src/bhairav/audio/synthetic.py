"""Phase 11 - deterministic synthetic audio track for the demo scene.

Generates a mono float32 PCM stream (16 kHz) with exactly the events
needed to exercise every audio detector. The track duration matches the
vision scene (32 s default) so audio + video alerts appear together.

Events placed in the demo timeline
-----------------------------------
  2.0 s  gunshot   (sharp thump, fast decay)
  7.5 s  scream    (0.5 s sustained vocal-band energy)
 14.0 s  gunshot   (second shot, re-detection after cooldown)
 22.0 s  glass_break (broadband burst + ringing)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _PlacedEvent:
    """Internal: an event to inject into the synthetic track."""
    kind: str        # "gunshot" | "glass_break" | "scream"
    onset_sec: float


def default_audio_events(duration_sec: float = 32.0) -> list[_PlacedEvent]:
    """Return the fixed event placement for the demo scene."""
    events = [
        _PlacedEvent("gunshot", 2.0),
        _PlacedEvent("scream", 7.5),
        _PlacedEvent("gunshot", 14.0),
        _PlacedEvent("glass_break", 22.0),
    ]
    return [e for e in events if e.onset_sec < duration_sec]


class SyntheticAudioTrack:
    """Generates a deterministic PCM stream with placed audio events.

    Call generate(duration_sec) to get the full float32 array,
    or iterate chunks(n) to feed the analyzer frame-by-frame.
    """

    def __init__(self, sample_rate: int = 16000, seed: int = 7):
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def generate(self, duration_sec: float = 32.0,
                 events: list[_PlacedEvent] | None = None) -> np.ndarray:
        """Return a mono float32 array of duration_sec seconds with
        placed events mixed over quiet background noise."""
        if events is None:
            events = default_audio_events(duration_sec)
        n = int(self.sample_rate * duration_sec)
        track = self.rng.normal(0, 0.002, n).astype(np.float32)
        for ev in events:
            self._inject(track, ev)
        np.clip(track, -1.0, 1.0, out=track)
        return track

    def _inject(self, track: np.ndarray, ev: _PlacedEvent) -> None:
        start = int(ev.onset_sec * self.sample_rate)
        if ev.kind == "gunshot":
            self._inject_gunshot(track, start)
        elif ev.kind == "glass_break":
            self._inject_glass_break(track, start)
        elif ev.kind == "scream":
            self._inject_scream(track, start)

    def _inject_gunshot(self, track: np.ndarray, start: int) -> None:
        click_len = int(0.008 * self.sample_rate)
        t = np.arange(click_len, dtype=np.float32) / self.sample_rate
        thump = 0.8 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 400)
        click = 0.3 * self.rng.normal(0, 1, click_len).astype(np.float32)
        click *= np.exp(-t * 600)
        sig = thump + click
        end = min(start + len(sig), len(track))
        track[start:end] += sig[:end - start]

    def _inject_glass_break(self, track: np.ndarray, start: int) -> None:
        burst_len = int(0.015 * self.sample_rate)
        burst = 0.7 * self.rng.normal(0, 1, burst_len).astype(np.float32)
        ring_len = int(0.5 * self.sample_rate)
        t = np.arange(ring_len, dtype=np.float32) / self.sample_rate
        ring = 0.15 * np.sin(2 * np.pi * 5000 * t) * np.exp(-t * 8)
        ring += 0.08 * np.sin(2 * np.pi * 6500 * t) * np.exp(-t * 10)
        sig = np.concatenate([burst, ring])
        end = min(start + len(sig), len(track))
        track[start:end] += sig[:end - start]

    def _inject_scream(self, track: np.ndarray, start: int) -> None:
        dur = 0.5
        n = int(dur * self.sample_rate)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        freq = 1200 + 80 * np.sin(2 * np.pi * 5 * t)
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        sig = 0.5 * np.sin(phase).astype(np.float32)
        sig += 0.2 * np.sin(2 * phase).astype(np.float32)
        env = np.ones(n, dtype=np.float32)
        att = min(int(0.03 * self.sample_rate), n // 4)
        env[:att] = np.linspace(0, 1, att, dtype=np.float32)
        rel = min(int(0.05 * self.sample_rate), n // 4)
        env[-rel:] = np.linspace(1, 0, rel, dtype=np.float32)
        sig *= env
        end = min(start + len(sig), len(track))
        track[start:end] += sig[:end - start]

    def chunks(self, duration_sec: float = 32.0, chunk_samples: int = 320,
               events: list[_PlacedEvent] | None = None):
        """Yield consecutive chunks of chunk_samples float32 samples."""
        track = self.generate(duration_sec, events)
        for i in range(0, len(track), chunk_samples):
            yield track[i:i + chunk_samples]
