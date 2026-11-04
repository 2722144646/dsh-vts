"""Fit the candidate to a reference duration and scale its own timeline.

The candidate's own phoneme spans are scaled with the audio, so mouth timing
stays internally consistent instead of borrowing the reference voice's phone
boundaries.
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
from dsh_vts_avatar.tts.timeline import PhonemeSpan, Timeline, VisemeSegment

FFMPEG = Path(r"D:\ruanjian\oopz\ffmpeg.exe")
TEXTS = [
    "你好，我是肥鱼。今天天气不错，我们开始吧。",
    "早上好，昨晚睡得怎么样？",
    "我刚泡了一杯热茶，你要不要也来一杯？",
    "这个周末你有什么安排吗？",
    "别着急，慢慢说，我在听。",
]
OUT = Path("dsh-vts-avatar/data/audit/voice_stretch")
CANDIDATE = Path("dsh-vts-avatar/models/feiyu_distill_v4.onnx")


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0, rate


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def atempo(audio: np.ndarray, rate: int, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 0.005:
        return audio
    with tempfile.TemporaryDirectory(prefix="dsh-stretch-") as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        write(src, audio, rate)
        subprocess.run(
            [str(FFMPEG), "-y", "-i", str(src), "-filter:a", f"atempo={factor:.6f}", "-ar", str(rate), "-ac", "1", str(dst)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        result, _ = read(dst)
    return result


def fit(samples: np.ndarray, size: int) -> np.ndarray:
    if samples.size >= size:
        return samples[:size].astype(np.float32, copy=False)
    return np.pad(samples, (0, size - samples.size)).astype(np.float32, copy=False)


def scale_timeline(timeline: Timeline, factor: float, audio: np.ndarray) -> Timeline:
    spans = [
        PhonemeSpan(
            span.phoneme,
            int(round(span.start * factor)),
            min(audio.size, int(round(span.end * factor))),
        )
        for span in timeline.spans
    ]
    segments = [
        VisemeSegment(
            int(round(seg.start * factor)),
            min(audio.size, int(round(seg.end * factor))),
            seg.phoneme, seg.viseme, seg.kind, seg.open, seg.form, seg.vowel_gain,
        )
        for seg in timeline.segments
    ]
    hop = max(1, int(timeline.sample_rate * 0.01))
    count = max(1, audio.size // hop)
    frames = audio[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
    return Timeline(
        sample_rate=timeline.sample_rate,
        audio=audio,
        spans=spans,
        segments=segments,
        rms=rms,
        rms_hop=hop,
        language=timeline.language,
    )


def main() -> None:
    cfg = load_config("dsh-vts-avatar/config.toml")
    reference_cfg = replace(cfg.tts, postprocess=False, length_scale=1.15)
    candidate_cfg = replace(
        reference_cfg,
        voice_model=CANDIDATE,
        voice_config=CANDIDATE.with_suffix(".onnx.json"),
        length_scale=0.61,
    )
    reference = PiperTTS(reference_cfg)
    candidate = PiperTTS(candidate_cfg)
    for index, text in enumerate(TEXTS, 1):
        target = reference.synthesize(text)
        voice = candidate.synthesize(text)
        factor = target.duration_s / max(0.001, voice.duration_s)
        speed = 1.0 / factor
        stretched = atempo(voice.audio, voice.sample_rate, speed)
        audio = fit(stretched, target.audio.size)
        timeline = scale_timeline(voice, factor, audio)
        print(
            index,
            "target", round(target.duration_s, 3),
            "voice", round(voice.duration_s, 3),
            "speed", round(speed, 3),
            "result", round(timeline.duration_s, 3),
            "spans", len(timeline.spans),
            "last_span", timeline.spans[-1].end / timeline.sample_rate if timeline.spans else None,
        )
        write(OUT / f"{index:02d}_target.wav", target.audio, target.sample_rate)
        write(OUT / f"{index:02d}_candidate_stretched.wav", timeline.audio, timeline.sample_rate)


if __name__ == "__main__":
    main()
