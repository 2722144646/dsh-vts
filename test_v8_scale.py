"""Play v8 without atempo: scale the model itself to the reference duration."""

from dataclasses import replace
from pathlib import Path
import wave

import numpy as np
import sounddevice as sd

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"
OUT = Path("dsh-vts-avatar/data/audit/v8_scale")
cfg = load_config("dsh-vts-avatar/config.toml")
cfg.tts = replace(
    cfg.tts,
    voice_model=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx"),
    voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx.json"),
    length_scale=0.80,
    postprocess=False,
)
tts = PiperTTS(cfg.tts)
tts.load()
timeline = tts.synthesize(TEXT)
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "v8_scale080.wav"
peak = float(np.max(np.abs(timeline.audio))) or 1.0
audio = timeline.audio * (10 ** (-1 / 20) / peak)
with wave.open(str(path), "wb") as fh:
    fh.setnchannels(1)
    fh.setsampwidth(2)
    fh.setframerate(timeline.sample_rate)
    fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
print("duration", round(timeline.duration_s, 3), "peak_before", round(peak, 3), path)
sd.play(audio, timeline.sample_rate, device=5)
sd.wait()
print("played")
