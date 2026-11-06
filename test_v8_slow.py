"""Play a v8 sample at 2x the current duration (length_scale 1.6)."""

from dataclasses import replace
from pathlib import Path
import wave

import numpy as np
import sounddevice as sd

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"
OUT = Path("dsh-vts-avatar/data/audit/v8_slow")
cfg = load_config("dsh-vts-avatar/config.toml")
cfg.tts = replace(cfg.tts, length_scale=1.60)
tts = PiperTTS(cfg.tts)
tts.load()
timeline = tts.synthesize(TEXT)
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "v8_length160.wav"
with wave.open(str(path), "wb") as fh:
    fh.setnchannels(1)
    fh.setsampwidth(2)
    fh.setframerate(timeline.sample_rate)
    fh.writeframes((np.clip(timeline.audio, -1, 1) * 32767).astype("<i2").tobytes())
print(
    "duration", round(timeline.duration_s, 3),
    "rms", round(float(np.sqrt(np.mean(timeline.audio ** 2))), 4),
    "peak", round(float(np.max(np.abs(timeline.audio))), 4),
    path,
)
print("playing on WH-1000XM5")
sd.play(timeline.audio, timeline.sample_rate, device=5)
sd.wait()
print("played")
