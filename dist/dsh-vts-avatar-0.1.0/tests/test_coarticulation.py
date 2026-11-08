import numpy as np

from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.mixer import PerformanceMixer
from dsh_vts_avatar.tts.timeline import Timeline, VisemeSegment


def test_coarticulation_blends_into_neighbouring_vowels():
    rate = 48000
    segments = [
        VisemeSegment(0, 24000, "a", "A", "vowel", 0.95, 0.0, 0.9),
        VisemeSegment(24000, 48000, "i", "I", "vowel", 0.55, 0.38, 0.8),
    ]
    timeline = Timeline(
        sample_rate=rate,
        audio=np.zeros(48000, dtype=np.float32),
        spans=[],
        segments=segments,
        rms=np.zeros(10, dtype=np.float32),
        rms_hop=4800,
    )
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="vowel"), rate)
    before = mixer._speech_coarticulated(timeline, 23800, segments[0], 0.8)
    after = mixer._speech_coarticulated(timeline, 24200, segments[1], 0.8)
    assert before["open"] < 0.95
    assert after["open"] > 0.55
