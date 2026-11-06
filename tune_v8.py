"""A small v8 tuning sweep: noise scale / noise_w / light postprocess."""

from dataclasses import replace
from pathlib import Path
import wave

import numpy as np
import sounddevice as sd

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS
from dsh_vts_avatar.tts.postprocess import AudioPostProcessor

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"
OUT = Path("dsh-vts-avatar/data/audit/v8_tuning")
VARIANTS = [
    ("baseline 4", 0.667, 0.80, False),
    ("noise .50 / w .70", 0.50, 0.70, False),
    ("noise .40 / w .60", 0.40, 0.60, False),
    ("noise .33 / w .50", 0.33, 0.50, False),
    ("noise .40 / w .60 + post", 0.40, 0.60, True),
    ("noise .50 / w .70 + post", 0.50, 0.70, True),
]


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio * (10 ** (-1 / 20) / peak)).astype(np.float32)


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


cfg = load_config("dsh-vts-avatar/config.toml")
cfg.tts = replace(
    cfg.tts,
    length_scale=0.80,
    postprocess=False,
    noise_scale=0.667,
    noise_w_scale=0.80,
)
tts = PiperTTS(cfg.tts)
tts.load()
OUT.mkdir(parents=True, exist_ok=True)
parts = []
rate = None
for label, noise, noise_w, post in VARIANTS:
    tts.config.noise_scale = noise
    tts.config.noise_w_scale = noise_w
    timeline = tts.synthesize(TEXT)
    if post:
        timeline = AudioPostProcessor(replace(cfg.tts, postprocess=True)).process(timeline)
    audio = normalize(timeline.audio)
    if rate is None:
        rate = timeline.sample_rate
    assert rate == timeline.sample_rate
    print(label, round(timeline.duration_s, 3), "s")
    parts.append(audio)
    parts.append(np.zeros(int(rate * 0.6), dtype=np.float32))
sequence = np.concatenate(parts)
write(OUT / "sequence.wav", sequence, rate)
print("sequence", OUT / "sequence.wav", round(len(sequence) / rate, 2), "s")
print("playing on WH-1000XM5")
sd.play(sequence, rate, device=5)
sd.wait()
print("played")
