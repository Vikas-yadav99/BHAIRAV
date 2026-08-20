"""Phase 11 - rule-based audio analyzer for gunshot, glass-break, and scream.

All detectors are pure NumPy (no ML dependencies). The analyzer processes
mono float32 PCM at a given sample rate and emits AudioEvent objects
whenever a classification fires.

Detectors
---------
gunshot:  impulsive transient - sharp onset, low spectral centroid (thump),
          fast decay (no ringing), high crest factor.
glass_break: broadband burst - high spectral centroid + high flatness,
          followed by sustained ringing (energy stays elevated).
scream:   sustained high-vocal-band segment - energy concentrated in
          0.8-3 kHz, low spectral flatness (voiced/tonal), lasting
          >= min_dur_sec.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..types import Severity


@dataclass
class AudioEvent:
    """A single audio detection event."""
    rule: str            # "gunshot" | "glass_break" | "scream"
    severity: Severity
    confidence: float    # 0.0 - 1.0
    message: str
    timestamp: float     # seconds into the audio stream
    details: dict = field(default_factory=dict)


class AudioAnalyzer:
    """Frame-by-frame audio analyzer with per-rule cooldown.

    Call ``feed(samples)`` with consecutive chunks of mono float32 PCM
    (values in [-1, 1]). Returns a list of AudioEvent for any detections
    in the fed samples.
    """

    def __init__(self, frame_rate: int = 16000, chunk_sec: float = 0.02,
                 cooldown_sec: float = 15.0, sensitivity: float = 1.0,
                 scream_min_dur_sec: float = 0.4):
        self.frame_rate = frame_rate
        self.chunk_samples = max(1, int(frame_rate * chunk_sec))
        self.chunk_sec = chunk_sec
        self.cooldown_sec = cooldown_sec
        self.sensitivity = sensitivity
        self.scream_min_dur_sec = scream_min_dur_sec

        # internal clock
        self._pos = 0  # samples consumed so far
        self._t = 0.0  # timestamp of next chunk boundary
        self._buf = np.zeros(0, dtype=np.float32)  # leftover from previous feed

        # noise floor EMA
        self._floor = 1e-4
        self._floor_alpha = 0.005

        # per-rule last-fire timestamp
        self._last_fire: dict[str, float] = {}

        # scream: running vocal-chunk counter
        self._scream_run_start: float | None = None
        self._scream_run_count: int = 0

        # gunshot / glass_break: onset tracking
        self._onset_t: float | None = None
        self._onset_rms: float = 0.0
        self._onset_kind: str | None = None  # "gunshot" | "glass_break"

        # live audio level (exposed for the dashboard volume meter)
        self._last_rms: float = 0.0
        self._last_peak: float = 0.0

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def level(self) -> dict:
        """Current audio level for the dashboard volume meter."""
        return {"rms": round(self._last_rms, 6),
                "peak": round(self._last_peak, 6),
                "floor": round(self._floor, 6)}

    def reset_position(self) -> None:
        """Reset the clock and buffers but keep cooldowns."""
        self._pos = 0
        self._t = 0.0
        self._buf = np.zeros(0, dtype=np.float32)
        self._scream_run_start = None
        self._scream_run_count = 0
        self._onset_t = None
        self._onset_rms = 0.0
        self._onset_kind = None
        self._last_rms = 0.0
        self._last_peak = 0.0

    def feed(self, samples: np.ndarray) -> list[AudioEvent]:
        """Feed a chunk of mono float32 PCM. Returns any detected events."""
        if samples.size == 0:
            return []
        samples = np.asarray(samples, dtype=np.float32)
        data = np.concatenate([self._buf, samples])
        events: list[AudioEvent] = []
        while data.size >= self.chunk_samples:
            chunk = data[:self.chunk_samples]
            data = data[self.chunk_samples:]
            t = self._t
            self._t += self.chunk_sec
            self._pos += self.chunk_samples
            evts = self._analyze_chunk(chunk, t)
            events.extend(evts)
        self._buf = data
        return events

    # ------------------------------------------------------------------
    # feature extraction per chunk
    # ------------------------------------------------------------------
    def _analyze_chunk(self, chunk: np.ndarray, t: float) -> list[AudioEvent]:
        # Startup guard: no detections until noise floor stabilizes
        if self._pos < self.chunk_samples * 30:
            return []
        eps = 1e-10
        rms = float(np.sqrt(np.mean(chunk ** 2) + eps))
        peak = float(np.max(np.abs(chunk)) + eps)
        crest = peak / (rms + eps)
        self._last_rms = rms
        self._last_peak = peak

        # update noise floor EMA - slow alpha so sustained loud
        # sounds (scream, glass-ring) don't eat the threshold
        self._floor = (1 - self._floor_alpha) * self._floor + self._floor_alpha * rms

        # spectral features via rfft
        n = chunk.size
        windowed = chunk * np.hanning(n)
        fft = np.abs(np.fft.rfft(windowed)) + eps
        freqs = np.fft.rfftfreq(n, 1.0 / self.frame_rate)
        total_energy = float(np.sum(fft ** 2))
        if total_energy < eps:
            return []

        # band energies (frac of total)
        low_mask = freqs < 800
        mid_mask = (freqs >= 800) & (freqs <= 3000)
        high_mask = freqs > 4000

        low_frac = float(np.sum(fft[low_mask] ** 2) / total_energy) if low_mask.any() else 0.0
        mid_frac = float(np.sum(fft[mid_mask] ** 2) / total_energy) if mid_mask.any() else 0.0
        high_frac = float(np.sum(fft[high_mask] ** 2) / total_energy) if high_mask.any() else 0.0

        # spectral centroid (Hz)
        centroid = float(np.sum(freqs * fft) / np.sum(fft) + eps)

        # spectral flatness (geometric mean / arithmetic mean)
        log_fft = np.log(fft + eps)
        flatness = float(np.exp(np.mean(log_fft)) / (np.mean(fft) + eps))

        # dB above noise floor
        db_above = 20 * math.log10(rms / (self._floor + eps) + eps)

        events: list[AudioEvent] = []
        events.extend(self._check_gunshot(t, rms, crest, low_frac, high_frac, db_above))
        events.extend(self._check_glass_break(t, rms, crest, centroid, flatness, high_frac, db_above))
        events.extend(self._check_scream(t, rms, mid_frac, flatness, db_above))

        return events

    # ------------------------------------------------------------------
    # cooldown helpers
    # ------------------------------------------------------------------
    def _can_fire(self, rule: str, t: float) -> bool:
        last = self._last_fire.get(rule, -999.0)
        return (t - last) >= self.cooldown_sec

    def _record_fire(self, rule: str, t: float) -> None:
        self._last_fire[rule] = t

    # ------------------------------------------------------------------
    # gunshot detector
    # ------------------------------------------------------------------
    def _check_gunshot(self, t: float, rms: float, crest: float,
                       low_frac: float, high_frac: float,
                       db_above: float) -> list[AudioEvent]:
        onset_db = 20 * self.sensitivity  # ~22 dB with sens=1.1

        # Phase 1: detect onset - loud, thumpy, high crest, low ring
        is_onset = (db_above >= onset_db and crest >= 3.0
                    and low_frac >= 0.45 and high_frac < 0.30)
        if is_onset and self._onset_t is None:
            self._onset_t = t
            self._onset_rms = rms
            self._onset_kind = "gunshot"
            return []

        # Phase 2: confirm via fast decay
        if self._onset_kind == "gunshot" and self._onset_t is not None:
            elapsed = t - self._onset_t
            if elapsed <= 0.25:
                if rms < self._onset_rms * 0.40 and self._can_fire("gunshot", t):
                    self._onset_t = None
                    self._onset_kind = None
                    conf = min(0.95, 0.6 + 0.15 * min(low_frac / 0.6, 1.0)
                               + 0.1 * min(db_above / 30, 1.0)
                               + 0.1 * min(crest / 15, 1.0))
                    self._record_fire("gunshot", t)
                    return [AudioEvent(
                        rule="gunshot", severity=Severity.RED,
                        confidence=round(conf, 3),
                        message=f"Gunshot detected (dB above floor: {db_above:.0f}, crest: {crest:.1f})",
                        timestamp=round(t, 3),
                        details={"db_above_floor": round(db_above, 1),
                                 "crest_factor": round(crest, 2),
                                 "low_band_frac": round(low_frac, 3),
                                 "confidence": round(conf, 3)})]
            else:
                # decay didn't happen fast enough - reset
                self._onset_t = None
                self._onset_kind = None

        return []

    # ------------------------------------------------------------------
    # glass-break detector
    # ------------------------------------------------------------------
    def _check_glass_break(self, t: float, rms: float, crest: float,
                           centroid: float, flatness: float,
                           high_frac: float, db_above: float) -> list[AudioEvent]:
        onset_db = 15 * self.sensitivity

        # Phase 1: broadband burst - loud, high centroid, high flatness, high-frequency
        is_onset = (db_above >= onset_db and centroid > 2000
                    and flatness > 0.20 and high_frac > 0.30
                    and self._floor > 3e-4)
        if is_onset and self._onset_kind != "gunshot":
            if self._onset_t is None:
                self._onset_t = t
                self._onset_rms = rms
                self._onset_kind = "glass_break"
                return []

        # Phase 2: confirm via sustained energy (ringing)
        if self._onset_kind == "glass_break" and self._onset_t is not None:
            elapsed = t - self._onset_t
            if elapsed <= 0.60:
                # ringing: energy stays above half the onset level
                if rms >= self._onset_rms * 0.15 and self._can_fire("glass_break", t):
                    self._onset_t = None
                    self._onset_kind = None
                    conf = min(0.93, 0.55 + 0.15 * min(high_frac / 0.5, 1.0)
                               + 0.13 * min(flatness / 0.4, 1.0)
                               + 0.10 * min(db_above / 25, 1.0))
                    self._record_fire("glass_break", t)
                    return [AudioEvent(
                        rule="glass_break", severity=Severity.RED,
                        confidence=round(conf, 3),
                        message=f"Glass break detected (centroid: {centroid:.0f} Hz, flatness: {flatness:.2f})",
                        timestamp=round(t, 3),
                        details={"centroid_hz": round(centroid, 0),
                                 "spectral_flatness": round(flatness, 3),
                                 "high_band_frac": round(high_frac, 3),
                                 "confidence": round(conf, 3)})]
            else:
                # ringing window expired - reset
                self._onset_t = None
                self._onset_kind = None

        return []

    # ------------------------------------------------------------------
    # scream detector (sustained vocal-band energy)
    # ------------------------------------------------------------------
    def _check_scream(self, t: float, rms: float, mid_frac: float,
                      flatness: float, db_above: float) -> list[AudioEvent]:
        onset_db = 10 * self.sensitivity

        # A chunk is "vocal" if mid-band (0.8-3 kHz) dominates and
        # spectral flatness is low (tonal / voiced quality).
        is_vocal = (db_above >= onset_db and mid_frac >= 0.35
                    and flatness < 0.50)

        if is_vocal:
            if self._scream_run_start is None:
                self._scream_run_start = t
                self._scream_run_count = 1
            else:
                self._scream_run_count += 1

            duration = t - self._scream_run_start + self.chunk_sec
            if duration >= self.scream_min_dur_sec and self._can_fire("scream", t):
                conf = min(0.90, 0.50 + 0.20 * min(mid_frac / 0.6, 1.0)
                           + 0.10 * min(db_above / 25, 1.0)
                           + 0.10 * min(duration / 1.0, 1.0))
                self._record_fire("scream", t)
                evt = AudioEvent(
                    rule="scream", severity=Severity.ORANGE,
                    confidence=round(conf, 3),
                    message=f"Scream detected (duration: {duration:.1f}s, mid-band: {mid_frac:.0%})",
                    timestamp=round(self._scream_run_start, 3),
                    details={"duration_sec": round(duration, 2),
                             "mid_band_frac": round(mid_frac, 3),
                             "spectral_flatness": round(flatness, 3),
                             "confidence": round(conf, 3)})
                # reset run after firing (allows re-detection after cooldown)
                self._scream_run_start = None
                self._scream_run_count = 0
                return [evt]
        else:
            # gap in vocal energy - reset
            self._scream_run_start = None
            self._scream_run_count = 0

        return []
