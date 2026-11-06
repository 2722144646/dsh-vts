"""Play the same five voice variants, slowed uniformly without pitch shift."""

from pathlib import Path
import subprocess
import tempfile
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
OUT = BASE / "voice_versions_slow.wav"
SPEED = 0.88  # about 14% slower


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        audio = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return audio, rate


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def slow(audio: np.ndarray, rate: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="slow-voice-") as tmp:
        source = Path(tmp) / "in.wav"
        target = Path(tmp) / "out.wav"
        write(source, audio, rate)
        subprocess.run(
            [
                r"D:\ruanjian\oopz\ffmpeg.exe", "-y", "-i", str(source),
                "-filter:a", f"atempo={SPEED:.3f}",
                "-ar", str(rate), "-ac", "1", str(target),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        result, _ = read(target)
    return result


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio * (10 ** (-1 / 20) / peak)).astype(np.float32)


parts = []
rate = None
for label, path in TRACKS:
    audio, sr = read(path)
    if rate is None:
        rate = sr
    slow_audio = slow(audio, sr)
    slow_audio = normalize(slow_audio)
    print(label, round(len(slow_audio) / rate, 3), "s", f"(x{SPEED:.2f})")
    parts.append(slow_audio)
    parts.append(np.zeros(int(rate * 0.7), dtype=np.float32))
combined = np.concatenate(parts)
write(OUT, combined, rate)
print("sequence", OUT, round(len(combined) / rate, 2), "s")
print("playing on WH-1000XM5")
sd.play(combined, rate, device=5)
sd.wait()
print("played")
