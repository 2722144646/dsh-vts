"""Generate A/B voice samples: Piper speed vs light audio post-processing."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
import wave

import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS

OUT = Path("dsh-vts-avatar/data/audit/voice_ab")
FFMPEG = Path(r"D:\ruanjian\oopz\ffmpeg.exe")
TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"


def normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-9:
        return audio
    target = 10 ** (target_db / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        samples = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate


def process(source: Path, destination: Path) -> None:
    filters = (
        "highpass=f=75,"
        "equalizer=f=300:t=q:w=1.0:g=-1.5,"
        "equalizer=f=3200:t=q:w=1.2:g=2.5,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=10:release=120:makeup=2"
    )
    subprocess.run(
        [
            str(FFMPEG), "-y", "-i", str(source),
            "-af", filters, "-ar", "22050", "-ac", "1",
            str(destination),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    audio, rate = read_wav(destination)
    write_wav(destination, normalize(audio), rate)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_config("dsh-vts-avatar/config.toml")
    variants = [
        ("01_current_122", 1.22, False),
        ("02_115_raw", 1.15, False),
        ("03_115_processed", 1.15, True),
        ("04_118_processed", 1.18, True),
        ("05_122_processed", 1.22, True),
        ("06_112_processed", 1.12, True),
    ]
    rows = []
    for name, scale, processed in variants:
        cfg = dataclasses.replace(base, tts=dataclasses.replace(base.tts, length_scale=scale))
        tts = PiperTTS(cfg.tts)
        timeline = tts.synthesize(TEXT)
        raw = OUT / f"{name}_raw.wav"
        audio = np.asarray(timeline.audio, dtype=np.float32)
        write_wav(raw, normalize(audio), timeline.sample_rate)
        final = OUT / f"{name}.wav"
        if processed:
            process(raw, final)
        else:
            final.write_bytes(raw.read_bytes())
        rows.append((name, scale, processed, timeline.duration_s, final))
        print(name, "scale", scale, "processed", processed, "seconds", round(timeline.duration_s, 3))

    html = ["<!doctype html><meta charset='utf-8'><title>Piper A/B</title>"]
    html.append("<style>body{font:16px system-ui;max-width:900px;margin:30px auto;line-height:1.5}audio{width:100%}</style>")
    html.append("<h1>Piper 语速与轻后处理 A/B</h1>")
    html.append(f"<p>{TEXT}</p>")
    for name, scale, processed, duration, path in rows:
        html.append(f"<h3>{name}</h3><p>length_scale={scale} · processed={processed} · {duration:.2f}s</p>")
        html.append(f"<audio controls src='{path.name}'></audio>")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    print("saved", OUT / "index.html")


if __name__ == "__main__":
    main()
