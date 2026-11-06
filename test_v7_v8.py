"""Play v7 (production) and v8 (native scale, no atempo) back to back."""

from dataclasses import replace
from pathlib import Path
import gc
import wave

import numpy as np
import sounddevice as sd

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.distill import DistilledPiperTTS
from dsh_vts_avatar.tts.piper_tts import PiperTTS

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"
OUT = Path("dsh-vts-avatar/data/audit/v7_v8")


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio * (10 ** (-1 / 20) / peak)).astype(np.float32)


cfg = load_config("dsh-vts-avatar/config.toml")
base = replace(cfg.tts, length_scale=1.15, postprocess=False)

v7_cfg = replace(
    base,
    distill_model=Path("dsh-vts-avatar/models/feiyu_distill_v7.onnx"),
    distill_voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v7.onnx.json"),
    distill_length_scale=0.61,
)
v7 = DistilledPiperTTS(v7_cfg)
v7.load()
v7_timeline = v7.synthesize(TEXT)
v7_audio = normalize(v7_timeline.audio)
v7_rate = v7_timeline.sample_rate
print("v7", round(v7_timeline.duration_s, 3), "factor", round(v7.last_factor, 3))
del v7
gc.collect()

v8_cfg = replace(
    base,
    voice_model=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx"),
    voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx.json"),
    length_scale=0.80,
)
v8 = PiperTTS(v8_cfg)
v8.load()
v8_timeline = v8.synthesize(TEXT)
v8_audio = normalize(v8_timeline.audio)
v8_rate = v8_timeline.sample_rate
print("v8", round(v8_timeline.duration_s, 3))
assert v7_rate == v8_rate
OUT.mkdir(parents=True, exist_ok=True)
write(OUT / "v7.wav", v7_audio, v7_rate)
write(OUT / "v8.wav", v8_audio, v8_rate)
gap = np.zeros(int(v7_rate * 0.5), dtype=np.float32)
sequence = np.concatenate([v7_audio, gap, v8_audio])
write(OUT / "sequence.wav", sequence, v7_rate)
print("sequence", OUT / "sequence.wav", "duration", round(sequence.size / v7_rate, 2))
print("playing v7, gap, v8 on WH-1000XM5")
sd.play(sequence, v7_rate, device=5)
sd.wait()
print("played")
