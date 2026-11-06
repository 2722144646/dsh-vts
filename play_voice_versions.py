"""Play all recent voice variants in one labeled-by-order sequence."""

from pathlib import Path
import wave

import numpy as np
import sounddevice as sd

BASE = Path("dsh-vts-avatar/data/audit")
TRACKS = [
    ("v7 reference-timed", BASE / "v7_v8/v7.wav"),
    ("v8 raw 0.61", BASE / "v8_isolate/v8_raw.wav"),
    ("v8 reference-timed/atempo", BASE / "v8_isolate/v8_timed.wav"),
    ("v8 scale 0.80 no atempo", BASE / "v8_scale/v8_scale080.wav"),
    ("v8 scale 0.80 + postprocess", BASE / "v8_post/v8_post.wav"),
]
OUT = BASE / "voice_versions.wav"


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        audio = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return audio, rate


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio * (10 ** (-1 / 20) / peak)).astype(np.float32)


parts = []
rate = None
for label, path in TRACKS:
    audio, sr = read(path)
    if rate is None:
        rate = sr
    assert sr == rate, (label, sr, rate)
    audio = normalize(audio)
    print(label, round(audio.size / rate, 3), "s")
    parts.append(audio)
    parts.append(np.zeros(int(rate * 0.7), dtype=np.float32))
combined = np.concatenate(parts)
with wave.open(str(OUT), "wb") as fh:
    fh.setnchannels(1)
    fh.setsampwidth(2)
    fh.setframerate(rate)
    fh.writeframes((np.clip(combined, -1, 1) * 32767).astype("<i2").tobytes())
print("sequence", OUT, round(combined.size / rate, 2), "s")
print("playing on WH-1000XM5")
sd.play(combined, rate, device=5)
sd.wait()
print("played")
