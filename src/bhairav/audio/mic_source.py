"""Phase 11 - live microphone audio source using sounddevice.

Captures mono float32 PCM from the system default input device and
feeds it into an AudioFusionProcessor on a background thread.

Usage:
    source = MicSource(analyzer)
    source.start()
    # ... later ...
    source.stop()
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MicSource:
    """Captures live audio from the system microphone and feeds an analyzer.

    The capture runs on a daemon thread; start() opens the stream,
    stop() closes it. level exposes the most recent RMS/peak
    for the dashboard meter (no queue - the GUI can poll at its own rate).
    """
    analyzer: object  # AudioAnalyzer instance
    sample_rate: int = 16000
    blocksize: int = 320  # 20 ms at 16 kHz
    device: int | str | None = None  # None = system default input

    _stream: object | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)
    _rms: float = field(default=0.0, repr=False)
    _peak: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _events: list = field(default_factory=list, repr=False)

    def start(self) -> None:
        """Open the microphone stream and begin feeding the analyzer."""
        if self._running:
            return
        import sounddevice as sd

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Close the microphone stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    @property
    def level(self) -> dict:
        """Most recent RMS/peak (thread-safe read)."""
        with self._lock:
            return {"rms": self._rms, "peak": self._peak}

    def drain_events(self) -> list:
        """Return and clear buffered audio events (called from the video frame tick)."""
        with self._lock:
            evts = list(self._events)
            self._events.clear()
        return evts

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        """sounddevice callback - runs on the audio thread."""
        samples = indata[:, 0].copy()  # mono
        rms = float(np.sqrt(np.mean(samples ** 2) + 1e-10))
        peak = float(np.max(np.abs(samples)))
        with self._lock:
            self._rms = rms
            self._peak = peak
        try:
            evts = self.analyzer.feed(samples)
            if evts:
                with self._lock:
                    self._events.extend(evts)
        except Exception:
            pass
