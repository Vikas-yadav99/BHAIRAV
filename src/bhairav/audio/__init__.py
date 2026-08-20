"""Phase 11 - audio analytics: gunshot / glass-break / scream detection.

Pure-NumPy rule-based analyzers that run on mono PCM (16 kHz, float32).
No ML dependencies; a deterministic synthetic track exercises every
detector path for the demo scene.
"""
from .analyzer import AudioAnalyzer, AudioEvent
from .synthetic import SyntheticAudioTrack, default_audio_events
from .fusion import audio_events_to_alerts, AudioFusionProcessor
from .mic_source import MicSource

__all__ = [
    "AudioAnalyzer", "AudioEvent",
    "SyntheticAudioTrack", "default_audio_events",
    "audio_events_to_alerts", "AudioFusionProcessor",
    "MicSource",
]
