from pathlib import Path
import wave

import numpy as np

OUT = Path("dsh-vts-avatar/data/audit/voice_ab")
GAP = 0.8
RATE = 22050


def read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as fh:
        if fh.getframerate() != RATE:
            raise ValueError(f"{path}: rate {fh.getframerate()}")
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def write(path: Path, chunks: list[np.ndarray]) -> None:
    gap = np.zeros(int(RATE * GAP), dtype=np.float32)
    parts = [np.zeros(int(RATE * 0.3), dtype=np.float32)]
    for chunk in chunks:
        parts.append(chunk)
        parts.append(gap)
    audio = np.concatenate(parts)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    print(path, len(audio) / RATE, "s")


recommended = [
    OUT / "02_115_raw.wav",
    OUT / "03_115_processed.wav",
    OUT / "04_118_processed.wav",
]
all_versions = [
    OUT / "01_current_122.wav",
    OUT / "02_115_raw.wav",
    OUT / "03_115_processed.wav",
    OUT / "04_118_processed.wav",
    OUT / "05_122_processed.wav",
    OUT / "06_112_processed.wav",
]
write(OUT / "recommended_sequence.wav", [read(p) for p in recommended])
write(OUT / "all_sequence.wav", [read(p) for p in all_versions])
