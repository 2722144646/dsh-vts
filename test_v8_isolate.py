"""Isolate v8 artifacts: raw model vs reference-timed atempo vs normalized."""

from dataclasses import replace
from pathlib import Path
import wave

import numpy as np
import sounddevice as sd

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS
from dsh_vts_avatar.tts.distill import DistilledPiperTTS

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"
OUT = Path("dsh-vts-avatar/data/audit/v8_isolate")


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0, fh.getframerate()


cfg = load_config("dsh-vts-avatar/config.toml")
cfg.tts = replace(cfg.tts, length_scale=1.15, postprocess=False)
base = PiperTTS(cfg.tts).synthesize(TEXT)
reference = replace(
    cfg.tts,
    distill_model=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx"),
    distill_voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx.json"),
    distill_length_scale=0.61,
)
distilled = DistilledPiperTTS(reference)
distilled.load()
timed = distilled.synthesize(TEXT)
v8_cfg = replace(
    cfg.tts,
    voice_model=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx"),
    voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx.json"),
    length_scale=0.61,
)
raw = PiperTTS(v8_cfg).synthesize(TEXT)
OUT.mkdir(parents=True, exist_ok=True)
write(OUT / "v8_raw.wav", raw.audio, raw.sample_rate)
write(OUT / "v8_timed.wav", timed.audio, timed.sample_rate)
peak = float(np.max(np.abs(timed.audio))) or 1.0
write(OUT / "v8_timed_norm.wav", timed.audio * (10 ** (-1 / 20) / peak), timed.sample_rate)
print("raw", round(raw.duration_s, 3), "peak", round(float(np.max(np.abs(raw.audio))), 3))
print("timed", round(timed.duration_s, 3), "factor", round(distilled.last_factor, 3), "peak", round(peak, 3))
print("reference", round(base.duration_s, 3))

gap = np.zeros(int(timed.sample_rate * 0.5), dtype=np.float32)
tracks = []
for name in ("v8_raw.wav", "v8_timed.wav", "v8_timed_norm.wav"):
    audio, rate = read(OUT / name)
    tracks.extend([audio, gap])
combined = np.concatenate(tracks)
write(OUT / "sequence.wav", combined, timed.sample_rate)
print("sequence", OUT / "sequence.wav", "duration", round(combined.size / timed.sample_rate, 2))
print("playing on WH-1000XM5")
sd.play(combined, timed.sample_rate, device=5)
sd.wait()
print("played")
