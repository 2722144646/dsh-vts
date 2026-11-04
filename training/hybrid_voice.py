"""Hybrid timing: huayan controls phoneme timing, distilled audio changes timbre.

The candidate audio is time-stretched to the huayan duration without changing
pitch, then fitted to the exact sample count. Mouth sync uses the huayan
timeline, so VTS sees the same phoneme boundaries as the current voice.
"""

from __future__ import annotations

from dataclasses import replace
import subprocess
import tempfile
from pathlib import Path
import wave

import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS
from dsh_vts_avatar.tts.timeline import Timeline

FFMPEG = Path(r"D:\ruanjian\oopz\ffmpeg.exe")
TEXTS = [
    "你好，我是肥鱼。今天天气不错，我们开始吧。",
    "早上好，昨晚睡得怎么样？",
    "我刚泡了一杯热茶，你要不要也来一杯？",
    "这个周末你有什么安排吗？",
    "别着急，慢慢说，我在听。",
]
OUT = Path("dsh-vts-avatar/data/audit/voice_hybrid")


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        samples = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate


def write(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def stretch(source: np.ndarray, rate: int, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 0.01:
        return source
    with tempfile.TemporaryDirectory(prefix="dsh-hybrid-") as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        write(src, source, rate)
        subprocess.run(
            [
                str(FFMPEG), "-y", "-i", str(src),
                "-filter:a", f"atempo={factor:.5f}",
                "-ar", str(rate), "-ac", "1", str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        result, _ = read(dst)
    return result


def fit(samples: np.ndarray, size: int) -> np.ndarray:
    if samples.size >= size:
        return samples[:size].astype(np.float32, copy=False)
    return np.pad(samples, (0, size - samples.size)).astype(np.float32, copy=False)


def replace_audio(timeline: Timeline, audio: np.ndarray) -> Timeline:
    hop = max(1, int(timeline.sample_rate * 0.01))
    count = max(1, audio.size // hop)
    frames = audio[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
    return Timeline(
        sample_rate=timeline.sample_rate,
        audio=audio.astype(np.float32, copy=False),
        spans=timeline.spans,
        segments=timeline.segments,
        rms=rms,
        rms_hop=hop,
        language=timeline.language,
    )


def main() -> None:
    cfg = load_config("dsh-vts-avatar/config.toml")
    timing_cfg = replace(cfg.tts, postprocess=False, length_scale=1.15)
    voice_cfg = replace(
        timing_cfg,
        voice_model=Path("dsh-vts-avatar/models/feiyu_distill_v3.onnx"),
        voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v3.onnx.json"),
        length_scale=0.61,
    )
    timing_tts = PiperTTS(timing_cfg)
    voice_tts = PiperTTS(voice_cfg)
    for index, text in enumerate(TEXTS, 1):
        timing = timing_tts.synthesize(text)
        voice = voice_tts.synthesize(text)
        factor = timing.duration_s / max(0.001, voice.duration_s)
        stretched = stretch(voice.audio, voice.sample_rate, factor)
        audio = fit(stretched, timing.audio.size)
        hybrid = replace_audio(timing, audio)
        path = OUT / f"{index:02d}_hybrid.wav"
        write(path, hybrid.audio, hybrid.sample_rate)
        print(
            index,
            "timing", round(timing.duration_s, 3),
            "voice", round(voice.duration_s, 3),
            "factor", round(factor, 3),
            "hybrid", round(hybrid.duration_s, 3),
            path,
        )


if __name__ == "__main__":
    main()
