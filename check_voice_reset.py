import time

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.performance.mixer import PerformanceMixer
from dsh_vts_avatar.tts.piper_tts import PiperTTS
from dsh_vts_avatar.vts.profile import ModelProfile, profile_path

cfg = load_config("dsh-vts-avatar/config.toml")
tts = PiperTTS(cfg.tts)
timeline = tts.synthesize("妈妈爸爸")
profile = ModelProfile.load(
    profile_path(cfg.profile_dir, "8cb985f5b9514021a5c5d941b482e57b")
)
mixer = PerformanceMixer(cfg.avatar, timeline.sample_rate, profile=profile)
mixer.set_state("speaking")
now = time.perf_counter()
for i in range(80):
    sample = int(i / 30 * timeline.sample_rate)
    segment = timeline.at(sample)
    out = mixer.compute(sample, timeline, now + i / 30)
    if segment.phoneme in {"m", "p", "ɑ"}:
        print(
            f"{i/30:.2f}s {segment.phoneme!r} "
            f"MouthOpen={out.get('MouthOpen', 0):.2f} "
            f"VoiceA={out.get('VoiceA', 0):.2f}"
        )
